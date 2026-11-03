"""Checks the shoebox lid kinematics: set the hinge to an angle, step, and compare the simulated lid pose to
run_sim.Lid.lid_pose. Also lists the box's bodies so we know which one is the root.

    cd ../IsaacLab && OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/tools/lid_probe.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.argv += ["--out", "/tmp/claude-1000/lid_probe", "--episodes", "0", "--no-wrist-cams"]
sys.path.insert(0, str(HERE))
import run_sim as R  # noqa: E402

import numpy as np  # noqa: E402


def main():
    import isaaclab.sim as sim_utils
    import torch

    sim_cfg = sim_utils.SimulationCfg(dt=1 / R.PHYSICS_HZ, device=R.args.device)
    R.args.require_kit, R.args.enable_cameras, R.args.physics = True, True, "isaacsim_physx"
    with R.launch_simulation(sim_cfg, R.args):
        import scene as S
        from isaaclab.utils import instantiate

        from repack_scene import make_scene_cfg

        R.S, R.T_W3_TCP = S, S.T_W3_GB @ S.T_GB_TCP
        assets = R.prepare_assets(Path("/tmp/claude-1000/lid_probe/_assets"), np.random.default_rng(0))
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(make_scene_cfg(assets, 64, 48, wrist_cams=False))
        R.fix_robot_prims()
        sim.reset()
        box = scene["shoebox"]
        print(f"[lid] bodies {box.body_names} joints {box.joint_names} fixed {box.is_fixed_base} "
              f"limits {box.data.joint_pos_limits.torch[0].cpu().numpy() if hasattr(box.data.joint_pos_limits, 'torch') else box.data.joint_pos_limits}", flush=True)
        cell = R.Cell(scene, assets["shoe_meta"])
        lid = R.Lid(cell, assets["shoebox"])
        li = box.body_names.index("lid")
        for q_deg in (0.0, -30.0, -60.0, 30.0):
            box.write_joint_position_to_sim_index(position=torch.tensor([[np.deg2rad(q_deg)]], dtype=torch.float32, device=sim.device))
            box.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, 1, device=sim.device))
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            d = box.data.body_pose_w.torch[0, li].cpu().numpy()
            T_sim = R.pose7(d[:3], d[3:])
            T_model = lid.lid_pose(cell.hinge_deg())
            tab_sim = (T_sim @ np.append(lid.tab, 1))[:3]
            tab_model = (T_model @ np.append(lid.tab, 1))[:3]
            print(f"[lid] set {q_deg:+.0f} read hinge_deg {cell.hinge_deg():+.1f}  tab sim {np.round(tab_sim, 3)} model {np.round(tab_model, 3)}  "
                  f"lid z-axis sim {np.round(T_sim[:3, 2], 2)}", flush=True)


if __name__ == "__main__":
    main()
