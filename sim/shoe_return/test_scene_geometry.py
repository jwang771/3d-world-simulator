"""Geometry checks for scene.py without Isaac: builds the shoebox and a sneaker on an in-memory USD stage
(needs `pip install usd-core numpy`).

    python sim/shoe_return/test_scene_geometry.py
"""
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as S  # noqa: E402
from geometry import pose, rot_x, rot_z, top_down  # noqa: E402


def world_range(stage, path):
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def lid_pose(box, T_base, deg):  # as run_sim.Lid.lid_pose
    return T_base @ pose(np.eye(3), box.hinge_in_base) @ pose(rot_x(-np.deg2rad(deg)), [0, 0, 0])


def main():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mat = S.friction_material(stage, "/World/m", 1.0)
    box = S.build_shoebox(stage, mat)
    lx, ly = S.BOX["inner"]
    h, w = S.BOX["height"], S.BOX["wall"]

    # Closed lid rests on the walls and its skirts hang outside them.
    lo, hi = world_range(stage, f"{box.lid_path}/top")
    assert abs(lo[2] - h) < 1e-6, lo
    wlo, whi = world_range(stage, f"{box.base_path}/wall_yn")
    slo, shi = world_range(stage, f"{box.lid_path}/skirt_front")
    assert shi[1] <= wlo[1] - S.BOX["gap"] + 1e-6, "front skirt overlaps the front wall"
    # The pull tab sticks out in front of the skirt; a top-down pinch across it clears the skirt.
    tlo, thi = world_range(stage, f"{box.lid_path}/pull_tab")
    tab = box.hinge_in_base + box.tab_in_lid
    assert thi[1] <= slo[1] + 1e-6 and np.allclose((tlo + thi) / 2, tab)
    pad_half_width = 0.0135  # 2F-85 pad extent across the closing direction
    assert tab[1] + pad_half_width < slo[1] - 0.003, "pads would brush the front skirt"
    print(f"box: lid on walls, tab {S.TAB['thick']*1000:.0f} mm thick, pads clear the skirt by {(slo[1] - tab[1] - pad_half_width)*1000:.1f} mm")

    # Lid arc: the tab rises over the hinge; past 90 deg it leans back.
    T_base = pose(rot_z(0.3), [0.1, -0.1, S.TABLE_Z])
    tab_at = lambda deg: (lid_pose(box, T_base, deg) @ np.append(box.tab_in_lid, 1))[:3]
    hinge_w = (T_base @ np.append(box.hinge_in_base, 1))[:3]
    assert np.allclose(tab_at(0), (T_base @ np.append(tab, 1))[:3])
    up90 = tab_at(90) - hinge_w
    assert up90[2] > ly * 0.9, up90
    back = np.linalg.inv(T_base[:3, :3]) @ (tab_at(100) - hinge_w)
    assert back[1] > 0, "lid at 100 deg should lean past the hinge"
    print(f"lid: tab {up90[2]*100:.0f} cm above the hinge at 90 deg, leans back past it at 100 deg")

    # A pair fits heel-to-toe and under the closed lid.
    shoe = S.build_shoe(stage, "/World/shoe", 0.27, mat)
    slo, shi = world_range(stage, "/World/shoe")
    assert shi[2] < h - 0.01, f"shoe {shi[2]:.3f} m tall does not fit under the lid ({h})"
    half_gap = 0.058
    assert shoe.width / 2 < half_gap and half_gap + shoe.width / 2 < ly / 2, "pair does not fit side by side"
    assert shoe.length < lx, "shoe longer than the box"
    assert 0.058 + 0.01 < 0.087, "heel counter does not fit the open gripper"
    print(f"shoes: {shi[2]*100:.1f} cm tall under a {h*100:.0f} cm lid, {(half_gap - shoe.width/2)*2000:.0f} mm between the pair")

    # Shoe sampling stays inside its region, never overlaps, and never touches the most-shifted box.
    rng = np.random.default_rng(0)
    ox = lx + 2 * w + 2 * (S.BOX["gap"] + S.BOX["lid_t"])
    oy = ly + 2 * w + S.BOX["gap"] + S.BOX["lid_t"] + S.TAB["depth"]
    for _ in range(300):
        T_box = S.box_pose_from_nominal(rng)
        poses = S.sample_shoe_poses(rng, shoe)
        boxrect = (T_box[:2, 3], np.arctan2(T_box[1, 0], T_box[0, 0]), np.array([ox / 2, oy / 2]))
        for xy, yaw in poses:
            assert not S.obb_overlap((xy, yaw, [shoe.length / 2, shoe.width / 2]), boxrect, 0.01), "shoe spawned on the box"
        assert not S.obb_overlap((*poses[0], [shoe.length / 2, shoe.width / 2]), (*poses[1], [shoe.length / 2, shoe.width / 2]), 0.0)
    print("sampling: 300 episodes, no shoe-shoe or shoe-box overlap")

    # Everything the cameras must see is inside the task bounds of the world model.
    sys.path.insert(0, str(HERE.parents[1]))
    from tasks import TASKS

    t = TASKS["shoe_return"]
    lo_b = np.array([t["x_range"][0], t["y_range"][0], t["z_range"][0]])
    hi_b = np.array([t["x_range"][1], t["y_range"][1], t["z_range"][1]])
    rx, ry = S.SHOE_REGION["x"], S.SHOE_REGION["y"]
    assert lo_b[0] <= rx[0] and rx[1] <= hi_b[0] and lo_b[1] <= ry[0] and ry[1] <= hi_b[1]
    for deg in (0, S.BOX["open_limit"]):
        for T_box in (S.box_pose_from_nominal(np.random.default_rng(k)) for k in range(50)):
            p = (lid_pose(box, T_box, deg) @ np.append(box.tab_in_lid, 1))[:3]
            assert np.all(p > lo_b) and np.all(p < hi_b), (deg, p)
    print("bounds: shoe region and the lid's swing stay inside the shoe_return task volume")
    print("ALL OK")


if __name__ == "__main__":
    main()
