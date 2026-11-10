"""Physics-only gripper check, no IK: hold every arm joint at its start angle, close the 2F-85 and watch whether
the wrist drifts. Runs our UR5e config and Isaac Lab's own UR10e + 2F-85 (gear-assembly settings) side by side.

    cd ../IsaacLab && OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/tools/gripper_hold_probe.py
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "envs"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--strip", action="store_true", help="strip the 2F-85's nested articulation root first")
parser.add_argument("--close", type=float, default=0.6, help="finger_joint target when closed (rad)")
parser.add_argument("--mode", default="finger", choices=["finger", "commanded", "measured"],
                    help="linkage joint targets: held at start (finger), sign * close (commanded), sign * measured finger angle")
parser.add_argument("--drive-right", action="store_true", help="drive right_outer_knuckle_joint like finger_joint on our UR10e")
parser.add_argument("--finger-stiffness", type=float, default=None, help="override the inner-finger (mimic) joint drive on our UR10e")
add_launcher_args(parser)
args = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
    args.require_kit, args.physics = True, "isaacsim_physx"
    with launch_simulation(sim_cfg, args):
        import numpy as np
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass, instantiate, replace
        from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG

        from repack_scene import ur_2f85_cfg

        nv = replace(UR10e_ROBOTIQ_2F_85_CFG, prim_path="{ENV_REGEX_NS}/NvUR10e")
        nv.init_state.pos = (1.5, 0.0, 0.0)
        nv.actuators["gripper_finger"] = ImplicitActuatorCfg(joint_names_expr=[".*_inner_finger_joint"], joint_effort_limit=10.0, joint_velocity_limit=10.0, stiffness=10.0, damping=0.05)
        nv.actuators["gripper_drive"] = ImplicitActuatorCfg(joint_names_expr=["finger_joint"], joint_effort_limit=10.0, joint_velocity_limit=1.0, stiffness=40.0, damping=1.0)
        nv.articulation_root_prim_path = "/root_joint"

        @configclass
        class ProbeCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(), init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.0)))
            light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=500.0))
            a = ur_2f85_cfg("left", ur_limits=False)
            b = ur_2f85_cfg("right", contact_offsets=False)
            c = ur_2f85_cfg("left", usd="ur10e")
            theirs = nv

        cfg = ProbeCfg(num_envs=1, env_spacing=4.0)
        if args.drive_right:
            from repack_scene import drive_both_knuckles
            drive_both_knuckles(cfg.c)
        if args.finger_stiffness is not None:
            cfg.c.actuators["gripper_finger"].stiffness = args.finger_stiffness
            cfg.c.actuators["gripper_finger"].damping = 0.0 if args.finger_stiffness == 0 else cfg.c.actuators["gripper_finger"].damping
        cfg.a.prim_path, cfg.b.prim_path, cfg.c.prim_path = "{ENV_REGEX_NS}/A", "{ENV_REGEX_NS}/B", "{ENV_REGEX_NS}/C"
        cfg.a.init_state.pos, cfg.b.init_state.pos, cfg.c.init_state.pos = (-1.5, 0, 0), (0, 1.5, 0), (0, -1.5, 0)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(cfg)
        if args.strip:
            from isaaclab.sim.utils import get_current_stage
            from pxr import Usd, UsdPhysics

            n = 0
            for prim in Usd.PrimRange(get_current_stage().GetPseudoRoot()):
                if prim.GetName() == "Robotiq_2F_85" and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    for api in ("PhysxArticulationAPI", "NewtonArticulationRootAPI"):
                        prim.RemoveAppliedSchema(api)
                    n += 1
            print(f"[hold] stripped {n} nested roots", flush=True)
        sim.reset()
        robots = {"ur5e no-UR-limits": scene["a"], "ur5e no-offsets": scene["b"], "ur10e-usd our cfg": scene["c"], "ur10e isaac lab": scene["theirs"]}
        for name in ("ur5e no-UR-limits", "ur10e isaac lab"):
            r = robots[name]
            lim = r.data.joint_pos_limits
            lim = (lim.torch if hasattr(lim, "torch") else lim)[0].cpu().numpy()
            print(f"[dofs] {name}: " + ", ".join(f"{n}[{np.rad2deg(lo):.0f},{np.rad2deg(hi):.0f}]" for n, (lo, hi) in zip(r.joint_names, lim)), flush=True)
            print(f"[dofs] {name} root view dof names: {getattr(r.root_view, 'dof_names', None) or getattr(r.root_view.shared_metatype, 'dof_names', None)}", flush=True)
        hold = {k: r.data.joint_pos.torch.clone() for k, r in robots.items()}
        fj = {k: r.find_joints(["finger_joint"])[0] for k, r in robots.items()}
        w3 = {k: r.find_joints(["wrist_3_joint"])[0] for k, r in robots.items()}
        signs = dict(finger_joint=1, right_outer_knuckle_joint=1, left_inner_finger_joint=-1, right_inner_finger_joint=1,
                     left_inner_finger_knuckle_joint=-1, right_inner_finger_knuckle_joint=-1)
        link = {k: (r.find_joints(list(signs), preserve_order=True)[0], torch.tensor(list(signs.values()), dtype=torch.float32, device=r.device)) for k, r in robots.items()}
        for k in range(720):
            for name, r in robots.items():
                tgt = hold[name].clone()
                if k >= 120:
                    ids, sg = link[name]
                    if args.mode != "finger":
                        now = r.data.joint_pos.torch[0, fj[name][0]].item()
                        tgt[0, ids] = sg * (args.close if args.mode == "commanded" else now)
                    tgt[:, fj[name]] = args.close
                r.set_joint_position_target_index(target=tgt)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            if k % 60 == 0 or k == 719:
                line = []
                for name, r in robots.items():
                    q = r.data.joint_pos.torch[0].cpu().numpy()
                    line.append(f"{name}: wrist_3 {np.rad2deg(q[w3[name][0]]):8.1f} finger {np.rad2deg(q[fj[name][0]]):6.1f} links {np.round(np.rad2deg(q[link[name][0][1:]])).astype(int)}")
                print(f"[hold k={k:3d}{' closed' if k >= 120 else ' open  '}] " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
