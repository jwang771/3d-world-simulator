"""Layout and assets for bimanual shoe-box repacking. Pure USD (pxr) and numpy: no Isaac imports, so the
asset builders and the geometry tests run without Isaac. The Isaac Lab scene lives in envs/.

World frame: table centre, z up, table top at TABLE_Z so the repo's table cut (`heights <= -0.06` in
scripts/train_vae.py) catches the table voxels. Both arms sit behind the table (+y) facing -y.
"""
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from geometry import look_at_ros, pose, rot_z

TABLE_Z = -0.07
UR5E_USD = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"  # its "Gripper" variant "Robotiq_2f_85" adds the 2F-85
# UR10e + Robotiq 2F-85 (the spec allows any UR arm). The UR5e asset's 2F-85 variant is physically unstable in
# Isaac Sim 6.2: closing the gripper spins the wrist without bound (tools/gripper_hold_probe.py), while the UR10e
# asset, which mounts the gripper through ee_link, is stable with the same settings.
ROBOT_MODEL = "ur10e"
ROBOT_BASES = {"left": ((-0.45, 0.42, TABLE_Z), -np.pi / 2), "right": ((0.45, 0.42, TABLE_Z), -np.pi / 2)}
UR_HOME_Q = np.array([0.0, -1.9, 1.9, -1.57, -1.57, 0.0])  # shoulder_pan .. wrist_3: elbow up, tool down
# UR10e datasheet: torque 330 Nm (base, shoulder), 150 Nm (elbow), 56 Nm (wrists); speed 120 deg/s (base,
# shoulder), 180 deg/s (elbow, wrists).
UR_LIMITS = {"shoulder_pan_joint": (330.0, 2.094), "shoulder_lift_joint": (330.0, 2.094), "elbow_joint": (150.0, np.pi),
             "wrist_1_joint": (56.0, np.pi), "wrist_2_joint": (56.0, np.pi), "wrist_3_joint": (56.0, np.pi)}
GRIPPER_KEY = "Robotiq_2F_85"  # path component of every 2F-85 prim
# Close target past the thinnest thing gripped: a 6 mm lid tab is reached at ~0.74 rad, and a target short of contact
# squeezes with no force. Stays off the 47 deg (0.82 rad) limit.
FINGER_JOINT, GRIP_OPEN, GRIP_CLOSED = "finger_joint", 0.0, 0.80
# 2F-85 geometry (measured from the asset meshes): in the gripper base_link frame the pads span z 0.091..0.148
# and close along y (open inner gap 0.087). TCP = pad centre: gripper base axes, 0.125 along z.
T_GB_TCP = pose(np.eye(3), [0, 0, 0.125])
# wrist_3_link -> 2F-85 base_link in the UR10e asset (6.2): a quarter turn about the tool axis. Checked at runtime.
T_W3_GB = pose(np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]), [0, 0, 0])
PAD_BELOW_TCP = 0.023  # how far the fingertips reach past the TCP along the approach axis

# yaw = pi: hinge on the far side, front (pull tab) toward the robots, as a box faces a packer. The box sits in a
# nest on the table (fixed base): a free box gets dragged or lifted by any arm-lid mismatch.
# Hinge toward the left arm (box +y -> world -x): the left arm lifts the tab on the box's right edge and swings the
# lid up and over toward itself, close in, and the open lid then leans away from the right arm's shoe traffic. (With the
# hinge on the far side, opening pushed the lid away from the robots at the reach limit and often stalled short of
# vertical.)
BOX_NOMINAL = dict(center=(-0.16, -0.20), yaw=np.pi / 2)
BOX = dict(inner=(0.31, 0.24), height=0.11, wall=0.006, lid_t=0.006, skirt=0.035, gap=0.006, open_limit=98.0)
# Pull tab on the lid front (a ribbon loop on many shoeboxes), pinched across its thickness, top-down.
TAB = dict(depth=0.035, height=0.03, thick=0.006)
SHOE_REGION = dict(x=(0.16, 0.62), y=(-0.50, 0.05))

STATIC_CAMS = {0: ((-0.02, -0.32, 1.05), (0.0, -0.12, -0.07)), 1: ((0.05, -1.05, 0.55), (0.0, -0.10, 0.0))}
WRIST_CAMS = {"left": 3, "right": 4}
STATIC_FOCAL_MM, WRIST_FOCAL_MM, H_APERTURE_MM = 18.0, 11.0, 20.955

UPPER_PALETTE = [(0.92, 0.92, 0.90), (0.08, 0.08, 0.09), (0.55, 0.10, 0.10), (0.12, 0.22, 0.50), (0.35, 0.42, 0.30), (0.60, 0.60, 0.62), (0.85, 0.55, 0.20)]
SOLE_PALETTE = [(0.96, 0.96, 0.94), (0.10, 0.10, 0.10), (0.45, 0.30, 0.18)]
BOX_PALETTE = [(0.70, 0.52, 0.32), (0.85, 0.30, 0.12), (0.12, 0.12, 0.13), (0.90, 0.90, 0.88), (0.15, 0.30, 0.55)]


def label(prim, cls):
    """Semantic class label in the UsdSemantics schema the Replicator annotators read (Isaac Sim 5+)."""
    prim.ApplyAPI("SemanticsLabelsAPI", "class")
    prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray).Set([cls])


def friction_material(stage, path, mu):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    api.CreateRestitutionAttr(0.0)
    return mat


def bind_physics_material(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")


def cube(stage, path, center, size, color=None, collide=True, mat=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    c.CreateDisplayColorAttr([Gf.Vec3f(*(color or (0.7, 0.7, 0.7)))])
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if mat is not None:
            bind_physics_material(c.GetPrim(), mat)
    return c


def rigid_xform(stage, path, mass, com=None):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(mass)
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    return prim


# ---------------------------------------------------------------- shoebox: articulation of base + hinged lid


@dataclass
class Shoebox:
    root: str
    base_path: str
    lid_path: str
    hinge_in_base: np.ndarray  # hinge-axis point in the base frame; the lid frame sits there, base-aligned, when closed
    tab_in_lid: np.ndarray  # pull-tab centre in the lid frame
    parts: list


def build_shoebox(stage, mat, root="/Shoebox"):
    """Base (heavy, floating so it can be re-posed each episode) and a lid on a damped revolute hinge along the
    back (+y) top edge. Hinge angle 0 = closed, negative = open, down to -BOX['open_limit'] degrees."""
    UsdGeom.Xform.Define(stage, root)
    lx, ly = BOX["inner"]
    w, h, t, sk, gap = BOX["wall"], BOX["height"], BOX["lid_t"], BOX["skirt"], BOX["gap"]
    ox, oy = lx + 2 * w, ly + 2 * w
    base = f"{root}/base"
    rigid_xform(stage, base, 3.0, com=(0, 0, 0.02))
    # Root on the base body so the base can be fixed to the table (a nest), see envs/repack_scene.py.
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(base))
    parts = [
        cube(stage, f"{base}/floor", (0, 0, w / 2), (ox, oy, w), mat=mat),
        cube(stage, f"{base}/wall_xn", (-ox / 2 + w / 2, 0, h / 2), (w, oy, h), mat=mat),
        cube(stage, f"{base}/wall_xp", (ox / 2 - w / 2, 0, h / 2), (w, oy, h), mat=mat),
        cube(stage, f"{base}/wall_yn", (0, -oy / 2 + w / 2, h / 2), (ox - 2 * w, w, h), mat=mat),
        cube(stage, f"{base}/wall_yp", (0, oy / 2 - w / 2, h / 2), (ox - 2 * w, w, h), mat=mat),
    ]
    # Lid frame: origin on the hinge axis at the back top edge; the closed lid extends toward -y.
    lid = f"{root}/lid"
    hinge = np.array([0.0, oy / 2, h])
    rigid_xform(stage, lid, 0.15)
    UsdGeom.Xformable(stage.GetPrimAtPath(lid)).AddTranslateOp().Set(Gf.Vec3d(*map(float, hinge)))
    Lx, Ly = ox + 2 * (gap + t), oy + gap + t
    parts += [
        cube(stage, f"{lid}/top", (0, -Ly / 2, t / 2), (Lx, Ly, t)),
        cube(stage, f"{lid}/skirt_front", (0, -Ly + t / 2, -sk / 2), (Lx, t, sk)),
        cube(stage, f"{lid}/skirt_xn", (-Lx / 2 + t / 2, -Ly / 2, -sk / 2), (t, Ly, sk)),
        cube(stage, f"{lid}/skirt_xp", (Lx / 2 - t / 2, -Ly / 2, -sk / 2), (t, Ly, sk)),
    ]
    tab = np.array([0.0, -Ly - TAB["depth"] / 2, t - TAB["height"] / 2])
    parts.append(cube(stage, f"{lid}/pull_tab", tab, (TAB["thick"], TAB["depth"], TAB["height"])))
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/lid_hinge")
    j.CreateBody0Rel().SetTargets([base])
    j.CreateBody1Rel().SetTargets([lid])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, hinge)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLowerLimitAttr(-BOX["open_limit"])
    j.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.02)  # a little hinge friction so the lid does not slam
    for p in parts:
        label(p.GetPrim(), "box")
    return Shoebox(root, base, lid, hinge, tab, parts)


def new_stage(path, default_prim):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, default_prim).GetPrim())
    return stage


def write_shoebox_usd(path, rng=None):
    """The shoebox as a standalone USD (default prim /Shoebox) for Isaac Lab's UsdFileCfg."""
    stage = new_stage(path, "/Shoebox")
    box = build_shoebox(stage, friction_material(stage, "/Shoebox/Looks/cardboard_physics", 0.6), "/Shoebox")
    if rng is not None:
        style_box(box, rng)
    stage.GetRootLayer().Save()
    return box


def write_table_usd(path, size=(1.5, 1.1, 0.04), color=(0.52, 0.47, 0.40), mu=0.5):
    """Static table slab whose top face sits at the asset origin (place it at z = TABLE_Z). Laminate on rubber
    soles and cardboard is roughly mu 0.4-0.6."""
    stage = new_stage(path, "/Table")
    mat = friction_material(stage, "/Table/Looks/laminate_physics", mu)
    top = cube(stage, "/Table/top", (0, 0, -size[2] / 2), size, color=color, mat=mat)
    label(top.GetPrim(), "table")
    stage.GetRootLayer().Save()


def write_procedural_shoe_usd(path, length, rng=None):
    """A procedural sneaker as a standalone USD (default prim /Shoe). It is symmetric in y, so it serves as either
    foot; scanned shoes get a true mirrored right foot from tools/build_shoe_assets.py."""
    stage = new_stage(path, "/Shoe")
    shoe = build_shoe(stage, "/Shoe", length, friction_material(stage, "/Shoe/Looks/rubber_physics", 1.0))
    if rng is not None:
        style_pair([shoe], rng)
    stage.GetRootLayer().Save()
    return shoe


def style_box(box, rng):
    c = np.clip(np.array(BOX_PALETTE[rng.integers(len(BOX_PALETTE))]) + rng.normal(0, 0.03, 3), 0, 1)
    for p in box.parts:
        p.GetDisplayColorAttr().Set([Gf.Vec3f(*c.tolist())])


def size_box_for(length, width, height):
    """The box fits the pair heel-to-toe with play (updates BOX in place). 6 cm of length play: a shoe is set
    down 2 cm toward its toe, leaving room behind the heel for the gripper finger that pinches the heel counter.
    6 cm of width play across the pair: a heel-held shoe can turn a few degrees as its toe touches down and
    pivots flat; with less play it landed partly on the other shoe."""
    BOX["inner"] = (round(length + 0.06, 3), round(2 * width + 0.06, 3))
    BOX["height"] = round(height + 0.025, 3)  # 2.5 cm headroom: with 1.2 cm a shoe resting slightly propped up stopped the lid


# 2F-85 inner gap is ~85 mm at finger_joint 0 and closes at ~0.8 rad; roughly linear in between.
def finger_angle_for_gap(gap):
    return float(np.clip(0.8 * (1.0 - gap / 0.085), 0.0, 0.75))


SHOE_LIBRARY = "assets/shoes"  # relative to this file; built by tools/build_shoe_assets.py


def box_pose_from_nominal(rng):
    """Spec randomisation: box position +-10 cm, yaw +-20 degrees."""
    c = np.array(BOX_NOMINAL["center"]) + rng.uniform(-0.10, 0.10, 2)
    yaw = BOX_NOMINAL["yaw"] + np.deg2rad(rng.uniform(-20, 20))
    return pose(rot_z(yaw), [c[0], c[1], TABLE_Z])


# ---------------------------------------------------------------- sneakers


@dataclass
class Shoe:
    path: str
    length: float
    width: float
    grasp_local: np.ndarray  # heel-counter grasp point; shoe frame origin at the sole bottom centre, +x toe
    upper_prims: list
    sole_prims: list
    accent_prims: list
    body: object = None


def build_shoe(stage, path, length, mat):
    """A rigid sneaker from box colliders. The heel counter is narrow enough for the 2F-85 (0.087 m) to pinch."""
    L, W, HW, HS = length, 0.33 * length, 0.058, 0.024
    heel_len, heel_h, vamp_h = 0.085, 0.07, 0.05  # everything stays under 0.095 so the lid closes over the pair
    rigid_xform(stage, path, 0.30, com=(-0.15 * L, 0.0, 0.035))
    x_heel = -L / 2 + heel_len / 2
    vamp_len = L - heel_len
    sole = [cube(stage, f"{path}/sole", (0, 0, HS / 2), (L, W, HS), mat=mat)]
    upper = [
        cube(stage, f"{path}/heel_counter", (x_heel, 0, HS + heel_h / 2), (heel_len, HW, heel_h), mat=mat),
        cube(stage, f"{path}/vamp", (-L / 2 + heel_len + vamp_len / 2, 0, HS + vamp_h / 2), (vamp_len, W * 0.92, vamp_h), mat=mat),
        cube(stage, f"{path}/toe_cap", (L / 2 - 0.03, 0, HS + 0.02), (0.06, W * 0.86, 0.04), mat=mat),
        cube(stage, f"{path}/tongue", (-L / 2 + heel_len + 0.03, 0, HS + vamp_h + 0.008), (0.06, W * 0.55, 0.016), mat=mat),
    ]
    accents = [
        cube(stage, f"{path}/laces", (-L / 2 + heel_len + vamp_len * 0.35, 0, HS + vamp_h + 0.002), (vamp_len * 0.45, W * 0.35, 0.004), collide=False),
        cube(stage, f"{path}/stripe_l", (0.0, W * 0.46 + 0.002, HS + vamp_h * 0.45), (L * 0.45, 0.004, 0.018), collide=False),
        cube(stage, f"{path}/stripe_r", (0.0, -W * 0.46 - 0.002, HS + vamp_h * 0.45), (L * 0.45, 0.004, 0.018), collide=False),
    ]
    label(stage.GetPrimAtPath(path), "shoe")
    grasp = np.array([x_heel, 0.0, HS + heel_h - 0.04])  # pads 7 mm clear of the wider sole
    return Shoe(path, L, W, grasp, upper, sole, accents)


def style_pair(pair, rng):
    pick = lambda pal: np.array(pal[rng.integers(len(pal))])
    upper, sole, accent = pick(UPPER_PALETTE), pick(SOLE_PALETTE), pick(UPPER_PALETTE)
    cols = [np.clip(c + rng.normal(0, 0.04, 3), 0, 1) for c in (upper, sole, accent)]
    for shoe in pair:  # a pair shares its colours
        for prims, c in zip((shoe.upper_prims, shoe.sole_prims, shoe.accent_prims), cols):
            for p in prims:
                p.GetDisplayColorAttr().Set([Gf.Vec3f(*c.tolist())])


def park_position(i):
    """Unused shoes wait on the floor under the table, out of every camera's view."""
    return np.array([-0.6 + 0.15 * i, 0.0, -0.75])


def obb_overlap(a, b, margin):
    """2-D separating-axis test for (centre xy, yaw, half extents) rectangles."""

    def axes(r):
        c, s = np.cos(r[1]), np.sin(r[1])
        return np.array([[c, s], [-s, c]])

    d = np.asarray(b[0]) - np.asarray(a[0])
    for ax in np.concatenate([axes(a), axes(b)]):
        ra = sum(abs(ax @ v) * h for v, h in zip(axes(a), a[2]))
        rb = sum(abs(ax @ v) * h for v, h in zip(axes(b), b[2]))
        if abs(ax @ d) > ra + rb + margin:
            return False
    return True


def sample_shoe_poses(rng, shoe, tries=500):
    """Two non-overlapping upright poses (xy, yaw) inside SHOE_REGION, beside the box."""
    half = np.array([shoe.length / 2, shoe.width / 2])
    rx, ry = SHOE_REGION["x"], SHOE_REGION["y"]
    for _ in range(tries):
        out = []
        for _ in range(2):
            yaw = rng.uniform(-np.pi, np.pi)
            c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
            ext = np.array([c * half[0] + s * half[1], s * half[0] + c * half[1]])
            if np.any(2 * ext > [rx[1] - rx[0], ry[1] - ry[0]]):
                break
            xy = np.array([rng.uniform(rx[0] + ext[0], rx[1] - ext[0]), rng.uniform(ry[0] + ext[1], ry[1] - ext[1])])
            out.append((xy, yaw, half))
        if len(out) == 2 and not obb_overlap(out[0], out[1], 0.04):
            return [(xy, yaw) for xy, yaw, _ in out]
    raise RuntimeError("could not place two shoes without overlap; widen SHOE_REGION")


# ---------------------------------------------------------------- robots


def rigid_link_boxes(stage, root_path, gripper_key=GRIPPER_KEY):
    """Per rigid link under root_path: (path, is_gripper, local box centre, half extents) from its untransformed bound."""
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    out = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        out.append((str(prim.GetPath()), gripper_key in str(prim.GetPath()), (lo + hi) / 2, (hi - lo) / 2))
    return out


def local_transform(stage, child, ancestor):
    xc = UsdGeom.XformCache()
    A = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(ancestor))).T
    C = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(child))).T
    return np.linalg.inv(A) @ C


# ---------------------------------------------------------------- cameras


def static_cam_pose(eye, target):
    return pose(look_at_ros(eye, target), eye)


# Wrist camera in the gripper base frame: beside the gripper body (off the closing axis), looking down the fingers.
T_GB_WRISTCAM = pose(look_at_ros((0.075, 0.0, 0.02), (0.0, 0.0, 0.20), up=(1.0, 0.0, 0.0)), (0.075, 0.0, 0.02))
