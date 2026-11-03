"""Probe the arms' TCP tracking on the T1 scene with no task: hold home, then step through yaw and roll targets
and print commanded-vs-actual error and joint angles. Used to debug the differential IK.

    cd ../IsaacLab && OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/tools/ik_probe.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLOSE_ONLY = "--close-only" in sys.argv
sys.argv = [a for a in sys.argv if a != "--close-only"] + ["--out", "/tmp/claude-1000/ik_probe", "--episodes", "0", "--no-wrist-cams"]
sys.path.insert(0, str(HERE))
import run_sim as R  # noqa: E402  (parses the launcher args)

import numpy as np  # noqa: E402


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=1 / R.PHYSICS_HZ, device=R.args.device)
    R.args.require_kit, R.args.enable_cameras, R.args.physics = True, True, "isaacsim_physx"
    with R.launch_simulation(sim_cfg, R.args):
        import scene as S
        from isaaclab.utils import instantiate

        from repack_scene import make_scene_cfg

        R.S, R.T_W3_TCP = S, S.T_W3_GB @ S.T_GB_TCP
        assets = R.prepare_assets(Path("/tmp/claude-1000/ik_probe/_assets"), np.random.default_rng(0))
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(make_scene_cfg(assets, 64, 48, wrist_cams=False))
        R.fix_robot_prims()
        sim.reset()
        robots = {"left": scene["robot_left"], "right": scene["robot_right"]}
        arms = {s: R.Arm(s, r, sim.device) for s, r in robots.items()}

        def run(n):
            for _ in range(n):
                for a in arms.values():
                    a.plan_step()
                    a.apply()
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())

        def report(tag):
            for s, a in arms.items():
                T = a.tcp()
                e = np.linalg.norm(T[:3, 3] - a.p) * 100
                r = np.rad2deg(np.linalg.norm(R.mat_to_rotvec(T[:3, :3].T @ a.R)))
                q = robots[s].data.joint_pos.torch[0, a.joint_ids].cpu().numpy()
                fq = np.rad2deg(robots[s].data.joint_pos.torch[0].cpu().numpy())
                names = robots[s].joint_names
                fingers = {n: round(float(v), 1) for n, v in zip(names, fq) if n not in R.S_UR_JOINTS}
                print(f"[probe {tag:>14s} {s:5s}] err {e:5.1f}cm {r:6.1f}deg  cmd_yaw {np.rad2deg(R.tcp_yaw(a.R)):7.1f} act_yaw {np.rad2deg(R.tcp_yaw(T[:3, :3])):7.1f}  q {np.round(np.rad2deg(q), 0)} {fingers}", flush=True)

        run(240)
        report("home")
        for a in arms.values():
            a.grip = 0.0
        for i in range(6):
            run(60)
            report(f"closed +{(i + 1) * 60}")
        for a in arms.values():
            a.grip = 1.0
        run(240)
        report("reopened")
        if CLOSE_ONLY:
            return
        for dyaw in (30, 60, 90, -60):
            for s, a in arms.items():
                a.queue.append(R.Move(lambda arm, d=dyaw: (R.HOME[arm.side], R.top_down(arm.base_yaw + np.deg2rad(d)))))
            run(360)
            report(f"yaw {dyaw:+d}")
        for roll in (30, 60, 90):
            for s, a in arms.items():
                a.queue.append(R.Move(lambda arm, d=roll: (R.HOME[arm.side] + [0, -0.1, 0], R.top_down(arm.base_yaw) @ R.rot_x(np.deg2rad(d)))))
            run(480)
            report(f"roll {roll}")


if __name__ == "__main__":
    main()
