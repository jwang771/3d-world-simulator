"""Diffs the composed 2F-85 physics (joints, drives, mimic, mass) between the UR5e and UR10e Gripper variants."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "envs"))
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()


def dump(stage, root):
    from pxr import Usd

    out = {}
    base = stage.GetPrimAtPath(root)
    for p in Usd.PrimRange(base, Usd.TraverseInstanceProxies()):
        path = str(p.GetPath())
        if "/visuals" in path or "finger" not in path and "knuckle" not in path and "Robotiq" not in path and "gripper" not in path.lower():
            continue
        rel = path[len(root) + 1:]
        for a in p.GetAttributes():
            n = a.GetName()
            if n.startswith(("physics:", "physx", "drive:", "state:", "mimic", "xformOp:", "limit:")):
                v = a.Get()
                out[f"{rel}.{n}"] = v
        for r in p.GetRelationships():
            if r.GetName().startswith(("physics:", "physx")):
                out[f"{rel}.{r.GetName()}"] = [t.pathString.rsplit("/", 1)[-1] for t in r.GetTargets()]
        out[f"{rel}.apis"] = tuple(sorted(str(s) for s in p.GetAppliedSchemas()))
    return out


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
    args.require_kit, args.physics = True, "isaacsim_physx"
    with launch_simulation(sim_cfg, args):
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sim.utils import get_current_stage
        from isaaclab.utils import configclass, instantiate

        from repack_scene import ur_2f85_cfg

        @configclass
        class C(InteractiveSceneCfg):
            a = ur_2f85_cfg("left")
            b = ur_2f85_cfg("right", usd="ur10e")

        cfg = C(num_envs=1, env_spacing=4.0)
        cfg.a.prim_path, cfg.b.prim_path = "{ENV_REGEX_NS}/UR5e", "{ENV_REGEX_NS}/UR10e"
        sim_utils.SimulationContext(sim_cfg)
        instantiate(cfg)
        stage = get_current_stage()
        a, b = dump(stage, "/World/envs/env_0/UR5e"), dump(stage, "/World/envs/env_0/UR10e")
        import re

        def joints(d):
            out = {}
            for k, v in d.items():
                m = re.match(r"(.*?)([^/]*_joint|FixedJoint)\.(physics:(axis|body0|body1|localRot0|localRot1|localPos0|localPos1|lowerLimit|upperLimit|excludeFromArticulation)|apis|physxMimicJoint:.*:(gearing|referenceJointAxis)|drive:angular:physics:(stiffness|maxForce))$", k)
                if m:
                    out.setdefault(m.group(2) + " @" + m.group(1), {})[m.group(3)] = v
            return out

        from pxr import Usd, UsdPhysics

        for name, root in (("UR5e", "/World/envs/env_0/UR5e"), ("UR10e", "/World/envs/env_0/UR10e")):
            for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if p.HasAPI(UsdPhysics.RigidBodyAPI) and ("wrist_3" in p.GetName() or "finger" in p.GetName() or "knuckle" in p.GetName() or p.GetName() in ("base_link", "ee_link")):
                    m = UsdPhysics.MassAPI(p)
                    get = lambda attr: attr.Get() if attr and attr.HasAuthoredValue() else None
                    print(f"[mass] {name:6s} {str(p.GetPath())[len(root):]:60s} mass={get(m.GetMassAttr())} I={get(m.GetDiagonalInertiaAttr())} "
                          f"com={get(m.GetCenterOfMassAttr())} density={get(m.GetDensityAttr())} hasMassAPI={p.HasAPI(UsdPhysics.MassAPI)}", flush=True)
        return
        for name, d in (("UR5e", a), ("UR10e", b)):
            print(f"[joints] ===== {name}", flush=True)
            for j, attrs in sorted(joints(d).items()):
                print(f"[joints] {j}: " + ", ".join(f"{k.split(':')[-1] if 'physics:' in k else k}={v}" for k, v in sorted(attrs.items()) if k != "apis") + f"  apis={[x for x in attrs.get('apis', ()) if 'Mimic' in x or 'Drive' in x]}", flush=True)
        return
        for k in sorted(set(a) | set(b)):
            va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
            if str(va) != str(vb):
                print(f"[diff] {k}\n        ur5e : {va}\n        ur10e: {vb}", flush=True)
        print(f"[diff] done: {len(a)} ur5e attrs, {len(b)} ur10e attrs", flush=True)


if __name__ == "__main__":
    main()
