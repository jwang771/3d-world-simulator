"""Lists the 2F-85 joints on the live stage: type, bodies, articulation membership, drives and mimic settings."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.argv += ["--out", "/tmp/claude-1000/gj_probe", "--episodes", "0", "--no-wrist-cams"]
sys.path.insert(0, str(HERE))
import run_sim as R  # noqa: E402

import numpy as np  # noqa: E402


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=1 / R.PHYSICS_HZ, device=R.args.device)
    R.args.require_kit, R.args.enable_cameras, R.args.physics = True, True, "isaacsim_physx"
    with R.launch_simulation(sim_cfg, R.args):
        import scene as S
        from isaaclab.sim.utils import get_current_stage
        from isaaclab.utils import instantiate
        from pxr import Usd, UsdPhysics

        from repack_scene import make_scene_cfg

        R.S, R.T_W3_TCP = S, S.T_W3_GB @ S.T_GB_TCP
        assets = R.prepare_assets(Path("/tmp/claude-1000/gj_probe/_assets"), np.random.default_rng(0))
        sim = sim_utils.SimulationContext(sim_cfg)
        instantiate(make_scene_cfg(assets, 64, 48, wrist_cams=False))
        stage = get_current_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot_left")
        vs = root.GetVariantSets()
        print("[gj] variants", {n: vs.GetVariantSet(n).GetVariantSelection() for n in vs.GetNames()}, flush=True)
        g = stage.GetPrimAtPath("/World/envs/env_0/Robot_left/Gripper/Robotiq_2F_85")
        if g.IsValid():
            gv = g.GetVariantSets()
            print("[gj] gripper variants", {n: gv.GetVariantSet(n).GetVariantSelection() for n in gv.GetNames()}, flush=True)
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            attrs = {a.GetName(): a.Get() for a in p.GetAttributes() if "mimic" in a.GetName().lower() or a.GetName().startswith("drive:") and "stiffness" in a.GetName()}
            print(f"[gj] {p.GetName():34s} {p.GetTypeName():22s} excl={j.GetExcludeFromArticulationAttr().Get()} "
                  f"b0={[t.name for t in j.GetBody0Rel().GetTargets()]} b1={[t.name for t in j.GetBody1Rel().GetTargets()]} "
                  f"apis={[a for a in p.GetAppliedSchemas() if 'Mimic' in a or 'Drive' in a]} {attrs}", flush=True)


if __name__ == "__main__":
    main()
