"""Turns Google Scanned Objects sneakers into simulation-ready USD shoe pairs.

For every scan: orient it toe +x / sole on z=0, find the heel-counter pinch (one finger inside the collar,
one behind the heel, closing along the shoe's length), decompose it into convex colliders with CoACD that
keep the collar open for the inner finger, and write a textured USD plus its mirror image (the other foot).

    pip install usd-core trimesh shapely coacd scipy numpy pillow
    python sim/shoe_return/tools/build_shoe_assets.py --src <dir of GSO model folders> --out sim/shoe_return/assets/shoes

GSO scans are CC-BY 4.0, Google LLC; every output folder keeps the source name in meta.json for attribution.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from shapely.geometry import LineString

# Plausible adult/kids shoe envelope; each pair gets a box sized to it (see scene.py)
MIN_LEN, MAX_LEN, MAX_WIDTH, MAX_HEIGHT = 0.18, 0.33, 0.125, 0.16
# The heel pinch opens the 2F-85 to 60 mm, not its full 85 mm: the inner finger then sits 30 mm inside the
# heel wall, clear of collar padding. FINGER_THICK and PAD_WIDTH are measured from the 5.1/6.x asset.
PINCH_OPEN_WIDTH = 0.060
GRIP_HALF_GAP, FINGER_THICK, PAD_WIDTH = PINCH_OPEN_WIDTH / 2, 0.031, 0.027
PINCH_BELOW_COLLAR = 0.022  # TCP height below the collar top at the heel


def canonicalize(mesh):
    """Sole on z=0, centred in x/y, toe toward +x. The heel end is the one with the higher collar."""
    v = mesh.vertices.copy()
    ext = v.max(0) - v.min(0)
    long_axis = int(np.argmax(ext[:2]))
    if long_axis == 1:  # length along y -> rotate onto x
        v = v[:, [1, 0, 2]] * [1, -1, 1]
    v[:, :2] -= (v[:, :2].max(0) + v[:, :2].min(0)) / 2
    v[:, 2] -= v[:, 2].min()
    L = v[:, 0].max() - v[:, 0].min()
    back = v[v[:, 0] < v[:, 0].min() + 0.15 * L, 2].max()
    front = v[v[:, 0] > v[:, 0].max() - 0.15 * L, 2].max()
    if front > back:  # heel is at +x: turn half round
        v[:, :2] *= -1
    m = mesh.copy()
    m.vertices = v  # both steps are proper rotations, so face winding stays outward
    return m


def heel_pinch(mesh):
    """Grasp point on the heel counter: crossings of the section at grasp height along y=0."""
    v = mesh.vertices
    xmin = v[:, 0].min()
    back = v[(v[:, 0] < xmin + 0.04) & (np.abs(v[:, 1]) < 0.02)]
    z_collar = back[:, 2].max()
    z = z_collar - PINCH_BELOW_COLLAR
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return None, "no section at grasp height"
    xs = []
    for poly in sec.discrete:
        ls = LineString(poly[:, :2])
        hit = ls.intersection(LineString([(xmin - 0.05, 0), (v[:, 0].max() + 0.05, 0)]))
        pts = [hit] if hit.geom_type == "Point" else list(getattr(hit, "geoms", []))
        xs += [p.x for p in pts if p.geom_type == "Point"]
    xs = sorted(xs)
    if len(xs) < 3:
        return None, f"collar not open at the heel ({len(xs)} crossings)"
    x_out, x_in, x_far = xs[0], xs[1], xs[2]
    wall = x_in - x_out
    if not 0.002 < wall < 0.03:
        return None, f"heel wall {wall*1000:.0f} mm"
    mid = (x_out + x_in) / 2
    if x_far < mid + GRIP_HALF_GAP + FINGER_THICK + 0.005:
        return None, f"collar opening too short for the inner finger ({(x_far - x_in)*1000:.0f} mm)"
    return dict(grasp=[float(mid), 0.0, float(z)], wall=float(wall), collar_z=float(z_collar), opening=float(x_far - x_in)), None


def heel_clamp(mesh, collar_z):
    """Whole-heel grasp from outside: just below the collar the heel counter is narrow (median 59 mm across the
    library), so the 2F-85 can close across it along the shoe's width. Returns the grasp point and heel width."""
    v = mesh.vertices
    xmin = v[:, 0].min()
    best = None
    for dz in (0.015, 0.025, 0.035):
        z = collar_z - dz
        sl = v[(np.abs(v[:, 2] - z) < 0.004) & (v[:, 0] > xmin + 0.02) & (v[:, 0] < xmin + 0.06)]
        if len(sl) < 10:
            continue
        w = sl[:, 1].max() - sl[:, 1].min()
        if best is None or w < best[1]:
            best = ([float(xmin + 0.04), float((sl[:, 1].max() + sl[:, 1].min()) / 2), float(z)], float(w))
    return best


def finger_probe_points(g):
    """Points the open inner finger sweeps on the way down into the collar; colliders must leave them free."""
    mid, _, z = g["grasp"]
    xs = np.linspace(mid + GRIP_HALF_GAP + 0.004, mid + GRIP_HALF_GAP + FINGER_THICK - 0.004, 4)
    ys = np.linspace(-PAD_WIDTH / 2 + 0.003, PAD_WIDTH / 2 - 0.003, 3)
    zs = np.linspace(z - 0.02, z + 0.04, 5)
    return np.array([[x, y, zz] for x in xs for y in ys for zz in zs])


def decompose(mesh, threshold):
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, max_convex_hull=60, preprocess_resolution=80, resolution=3000)
    return [trimesh.Trimesh(v, f) for v, f in parts]


def blocked(hulls, pts):
    for h in hulls:
        if np.any(h.contains(pts)):
            return True
    return False


def write_usd(path, mesh, uv, hulls, meta, texture_name, mirror):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Shoe").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(meta["mass"])
    s = np.array([1, -1, 1]) if mirror else np.ones(3)
    flip = (lambda f: f[:, ::-1]) if mirror else (lambda f: f)

    vis = UsdGeom.Mesh.Define(stage, "/Shoe/visual")
    vis.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((mesh.vertices * s).astype(np.float32)))
    vis.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    vis.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(flip(mesh.faces).reshape(-1).astype(np.int32)))
    vis.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy((mesh.vertex_normals * s).astype(np.float32)))
    vis.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    vis.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(vis).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))

    mat = UsdShade.Material.Define(stage, "/Shoe/Looks/scan")
    sh = UsdShade.Shader.Define(stage, "/Shoe/Looks/scan/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, "/Shoe/Looks/scan/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, "/Shoe/Looks/scan/albedo")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{texture_name}")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(mat)

    UsdGeom.Scope.Define(stage, "/Shoe/collision")
    for i, h in enumerate(hulls):
        c = UsdGeom.Mesh.Define(stage, f"/Shoe/collision/hull_{i:02d}")
        c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((h.vertices * s).astype(np.float32)))
        c.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(h.faces)))
        c.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(flip(h.faces).reshape(-1).astype(np.int32)))
        c.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(c.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    stage.GetRootLayer().Save()


def build_one(src, out_root):
    obj = src / "meshes" / "model.obj"
    tex = src / "materials" / "textures" / "texture.png"
    if not obj.exists() or not tex.exists():
        return None, "missing mesh or texture"
    raw = trimesh.load(obj, force="mesh", process=False)
    uv = np.asarray(raw.visual.uv) if getattr(raw.visual, "uv", None) is not None else None
    if uv is None or len(uv) != len(raw.vertices):
        return None, "no per-vertex UVs"
    mesh = canonicalize(raw)
    L, W, H = mesh.vertices.max(0) - mesh.vertices.min(0)
    if not MIN_LEN <= L <= MAX_LEN or W > MAX_WIDTH or H > MAX_HEIGHT:
        return None, f"outside the shoe envelope ({L:.3f} x {W:.3f} x {H:.3f})"
    g, why = heel_pinch(mesh)
    if g is None:
        return None, why
    probe = finger_probe_points(g)
    collider = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    for thr in (0.03, 0.02):
        hulls = decompose(collider, thr)
        if not blocked(hulls, probe):
            break
    else:
        return None, "convex colliders fill the collar"
    name = src.name
    mass = float(0.32 * (L / 0.28) ** 2)  # adult sneakers weigh 0.25-0.40 kg
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(tex, out / "texture.png")
    clamp = heel_clamp(mesh, g["collar_z"])
    if clamp is not None:
        g["heel_grasp"], g["heel_width"] = clamp
    meta = dict(pinch_open_width=PINCH_OPEN_WIDTH, source=name, license="CC-BY-4.0 Google LLC (Google Scanned Objects)", length=float(L), width=float(W), height=float(H), mass=mass, hulls=len(hulls), **g)
    write_usd(out / "left.usd", mesh, uv, hulls, meta, "texture.png", mirror=False)
    write_usd(out / "right.usd", mesh, uv, hulls, meta, "texture.png", mirror=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder of GSO model folders, or one model folder with --one")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--one", action="store_true", help="build the single model folder --src (used per subprocess)")
    p.add_argument("--update-meta", action="store_true", help="recompute grasp metadata of already-built shoes only")
    a = p.parse_args()
    out = Path(a.out)
    if a.update_meta:
        for d in sorted(x for x in out.iterdir() if (x / "meta.json").exists()):
            meta = json.loads((d / "meta.json").read_text())
            mesh = canonicalize(trimesh.load(Path(a.src) / meta["source"] / "meshes" / "model.obj", force="mesh", process=False))
            clamp = heel_clamp(mesh, meta["collar_z"])
            if clamp is not None:
                meta["heel_grasp"], meta["heel_width"] = clamp
                (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print("metadata updated")
        return
    if a.one:
        meta, why = _safe_build(Path(a.src), out)
        report(Path(a.src), meta, why)
        sys.exit(0 if meta else 2)
    # One subprocess per scan: a CoACD crash on one mesh must not take the batch down.
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    srcs = sorted(d for d in Path(a.src).iterdir() if (d / "meshes").is_dir())[: a.limit]
    todo = [d for d in srcs if not (out / d.name / "meta.json").exists()]
    print(f"{len(srcs) - len(todo)} already built, {len(todo)} to go", flush=True)

    def run(d):
        r = subprocess.run([sys.executable, "-W", "ignore", __file__, "--src", str(d), "--out", str(out), "--one"], capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or [f"skip {d.name[:48]:48s} crashed (exit {r.returncode})"])[-1]
        print(line, flush=True)

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(run, todo))
    ok = sum((out / d.name / "meta.json").exists() for d in srcs)
    print(f"{ok}/{len(srcs)} shoes built into {out}")
    sys.exit(0 if ok else 1)


def _safe_build(d, out):
    try:
        return build_one(d, out)
    except Exception as e:  # report a bad scan instead of a traceback
        return None, f"error: {e}"


def report(d, meta, why):
    if meta:
        print(f"OK   {d.name[:48]:48s} L={meta['length']:.3f} wall={meta['wall']*1000:.0f}mm opening={meta['opening']*1000:.0f}mm hulls={meta['hulls']}", flush=True)
    else:
        print(f"skip {d.name[:48]:48s} {why}", flush=True)


if __name__ == "__main__":
    main()
