"""Checks that EpisodeWriter output loads through the repo's own readers. Needs numpy, h5py, opencv, torch; no Isaac.

    python sim/shoe_return/test_writer_compat.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from datasets.aloha_packed import AlohaPackedPCDataset  # noqa: E402
from datasets.frames import Frames, warp_depth  # noqa: E402
from geometry import look_at_ros, pose  # noqa: E402
from models.background_bank import build_background_init  # noqa: E402
from models.cached_robot_labels import load_frame  # noqa: E402
from tasks import TASKS  # noqa: E402
from writer import EpisodeWriter, level_shapes  # noqa: E402

TABLE_Z, W, H = -0.07, 160, 120
TASK = TASKS["shoe_return"]
SHOE = (np.array([-0.1, -0.05, TABLE_Z]), np.array([-0.1 + 0.26, 0.04, TABLE_Z + 0.08]))


def render(E, K, shoe_shift):
    """Ray-cast a table plane plus one axis-aligned 'shoe' box, through pixel centres like an RTX camera."""
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u + 0.5 - W / 2) / K[0, 0], (v + 0.5 - H / 2) / K[1, 1], np.ones_like(u, float)], -1)
    d = d_cam @ E[:3, :3].T
    o = E[:3, 3]
    t_tab = (TABLE_Z - o[2]) / d[..., 2]
    t_tab[t_tab <= 0] = np.inf
    lo, hi = SHOE[0] + shoe_shift, SHOE[1] + shoe_shift
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
    hit = (tn < tf) & (tn > 0)
    t = np.where(hit & (tn < t_tab), tn, t_tab)
    depth = np.where(np.isfinite(t), t, 0.0)  # d_cam has z = 1, so t is the optical-axis depth
    obj = (hit & (tn < t_tab)).astype(np.uint8)
    rgb = np.full((H, W, 3), 120, np.uint8)
    rgb[obj > 0] = (200, 40, 40)
    grip = np.zeros((H, W), bool)
    grip[:10, :10] = True
    return dict(rgb=rgb, depth=depth.astype(np.float32), obj_mask=obj, gripper_mask=grip, K=K, E_cam2world=E)


def main():
    K = np.array([[W * 0.9, 0, (W - 1) / 2], [0, W * 0.9, (H - 1) / 2], [0, 0, 1]], np.float32)  # as scene.define_camera
    eyes = {0: (-0.38, -0.88, 0.52), 1: (0.38, -0.88, 0.52), 2: (0.0, 0.8, 0.7)}
    Es = {c: pose(look_at_ros(e, (0, 0, -0.04)), e) for c, e in eyes.items()}
    hand = (pose(np.eye(3), [0.0, 0.0, 0.15]), np.array([0.03, 0.1, 0.03]))
    link = (pose(np.eye(3), [-0.4, 0.0, 0.05]), np.array([0.06, 0.06, 0.15]))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "shoe_return_sim"
        for ep, split in ((0, "train"), (1, "val")):
            w = EpisodeWriter(root, split, ep, sorted(Es), TASK)
            for t in range(5):
                cams = {c: render(E, K, np.array([0.01 * t, 0, 0])) for c, E in Es.items()}
                w.add_frame(np.arange(14, dtype=np.float32), cams, [link, hand], [hand])
            w.close(dict(shoe_condition="none"))
        aborted = EpisodeWriter(root, "train", 7, sorted(Es), TASK)
        aborted.discard()
        assert not list(root.rglob("*episode_7*")), "discard left files behind"

        bounds = [[TASK["x_range"][0], TASK["y_range"][0], TASK["z_range"][0]], [TASK["x_range"][1], TASK["y_range"][1], TASK["z_range"][1]]]
        ds = AlohaPackedPCDataset([root / "train"], n_points=20000, point_bounds=bounds)
        s = ds[2]
        n_obj = int(s["obj_label"].sum())
        assert s["xyz"].shape == (20000, 3) and n_obj > 0, n_obj
        xyz = s["xyz"].numpy()
        shoe_pts = xyz[s["obj_label"].numpy() > 0]
        assert np.all(shoe_pts[:, 2] > TABLE_Z - 0.01) and np.all(shoe_pts[:, 2] < TABLE_Z + 0.09), "lifted shoe points off the box"
        table = xyz[s["obj_label"].numpy() == 0]
        assert np.abs(table[:, 2] - TABLE_Z).max() < 0.003, "table points off the plane: extrinsics or depth convention wrong"
        print(f"dataset: {len(ds)} frames, {n_obj}/20000 shoe points, table z err {np.abs(table[:, 2] - TABLE_Z).max() * 1000:.2f} mm")

        fr = Frames(ds.files, "cpu")
        rc = fr.render(0, 1)
        dc = fr.depth(0, 1)
        assert len(rc) == 3 and rc[0]["gripper"].any() and rc[0]["obj"].any()
        warped = warp_depth(dc, rc)
        for c, wd in zip(rc, warped):
            dep = torch.from_numpy(render(Es[c["cid"]], K, np.array([0.01, 0, 0]))["depth"])
            m = (wd > 0) & (dep > 0)
            err = (wd[m] - dep[m]).abs().median().item()
            assert err < 0.01, f"camera {c['cid']}: stereo depth disagrees with rendered depth ({err})"
        print("frames: render + stereo depth round trip ok")

        grid = TASK["grid"]
        shapes = [(s[2], s[1], s[0]) for s in level_shapes(grid)]
        labels = load_frame(root / "robot_label_cache", 0, 3, shapes, torch.tensor(bounds[0]), torch.tensor(bounds[1]), "cpu")
        for kind in ("robot", "gripper"):
            assert [tuple(g.shape[2:]) for g in labels[kind]] == level_shapes(grid)
            assert all(g.any() for g in labels[kind])
        assert labels["robot"][1].sum() > labels["gripper"][1].sum()
        print("robot_label_cache: shapes", [tuple(g.shape[2:]) for g in labels["robot"]], "ok")

        bb = (bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1], bounds[0][2], bounds[1][2])
        xyz, rgb, _ = build_background_init(ds.files, bb, n_near=2000, n_far=1000, max_eps=2)
        assert len(xyz) > 0
        print("background bank init ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
