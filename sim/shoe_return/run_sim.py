"""T1 bimanual shoe-box repacking on Isaac Lab 3.0 / Isaac Sim 6.2, recorded in this repo's training format.

Two UR5e + Robotiq 2F-85 arms:
  1. the left arm pinches the lid's pull tab and swings the hinged lid open past vertical, then lets go,
  2. the right arm picks shoe 1 and sets it in the box,
  3. picks shoe 2 and sets it beside the first in the opposite direction (heel-to-toe),
  4. the left arm swings the lid closed.
Success: both shoes inside the box, hinge < 5 deg, no shoe off the table, wrists never closer than ARM_CLEARANCE.

    cd ../IsaacLab && uv run python ../3d-world-simulator/sim/shoe_return/run_sim.py \
        --out ../3d-world-simulator/data/shoe_return_sim --episodes 20    # add --viz kit for the GUI
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "envs"))
sys.path.insert(0, str(HERE.parents[1]))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="dataset root, e.g. data/shoe_return_sim")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--val-every", type=int, default=10, help="every n-th episode goes to val/")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--record-every", type=int, default=8, help="physics steps (120 Hz) per recorded frame")
parser.add_argument("--no-wrist-cams", dest="wrist_cams", action="store_false")
parser.add_argument("--keep-failures", action="store_true", help="also write episodes that fail the success check")
parser.add_argument("--max-steps", type=int, default=12000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--grasp", default="collar", choices=["collar", "clamp"], help="shoe grasp: collar pinch or whole-heel clamp")
parser.add_argument("--shoe", default="random", help="scanned shoe model name from the library, 'random', or 'procedural'")
parser.add_argument("--debug", action="store_true", help="print TCP tracking error twice a second")
parser.add_argument("--video", default=None, help="write the overhead camera of every episode to <dir>/episode_<E>.mp4")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np  # noqa: E402

import ur_kin  # noqa: E402
from geometry import mat_to_quat, mat_to_rotvec, pose, quat_to_mat, rot_x, rot_z, slerp_mat, tcp_yaw, top_down, wrap  # noqa: E402

PHYSICS_HZ = 120
HOME = {"left": np.array([-0.45, 0.02, 0.35]), "right": np.array([0.40, 0.02, 0.35])}
RIGHT_WAIT = np.array([0.30, -0.18, 0.30])
LID_OPEN_DEG = 100.0  # past vertical: the lid then rests open on its 105 deg stop by itself
LID_CLOSED_DEG = 2.0
ARM_CLEARANCE = 0.12  # wrist-to-wrist distance below which the episode counts as an arm collision
S = None  # scene.py imports pxr, which must load after Kit's own USD libraries (see main)
T_W3_TCP = None


def pose7(p, q_xyzw):
    """Pose from Isaac Lab (position, quaternion x, y, z, w)."""
    x, y, z, w = q_xyzw
    return pose(quat_to_mat([w, x, y, z]), p)


def pose_rot_y(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, 0, s_], [0, 1.0, 0], [-s_, 0, c]])


def rot_z_pose(a):
    return pose(rot_z(a), [0, 0, 0])


def to_xyzw(R):
    w, x, y, z = mat_to_quat(R)
    return [x, y, z, w]


def prepare_assets(cache, rng, shoe="random"):
    """Writes the table and a shoebox sized to the chosen pair; returns USD paths and the pair's grasp metadata."""
    cache.mkdir(parents=True, exist_ok=True)
    out = dict(table=str(cache / "table.usda"), shoebox=str(cache / "shoebox.usda"))
    S.write_table_usd(out["table"])
    if shoe == "procedural":
        length = float(rng.uniform(0.23, 0.27))
        for k in range(2):
            out[f"shoe_{k}"] = str(cache / f"shoe_{k}.usda")
            built = S.write_procedural_shoe_usd(out[f"shoe_{k}"], length, np.random.default_rng(7))
        meta = dict(name="procedural", grasp_local=built.grasp_local.tolist(), closing_axis="y", open=S.GRIP_OPEN,
                    length=length, width=built.width, height=0.094, mass=0.3)
    else:
        lib = HERE / S.SHOE_LIBRARY
        names = sorted(d.name for d in lib.iterdir() if (d / "meta.json").exists())
        name = names[rng.integers(len(names))] if shoe == "random" else shoe
        m = json.loads((lib / name / "meta.json").read_text())
        out["shoe_0"], out["shoe_1"] = str(lib / name / "left.usd"), str(lib / name / "right.usd")
        # Collar pinch by default: one finger inside the collar, one behind the heel, closing along the shoe's
        # length, so a toe-down swing is resisted by the pads' 57 mm contact along the heel wall. (A clamp across
        # the heel closes along the width: the swing then acts about the closing axis and only torsional friction
        # resists it, and the shoe slipped out in testing.)
        if args.grasp == "clamp" and "heel_grasp" in m and m["heel_width"] < 0.078:
            meta = dict(name=name, grasp_local=m["heel_grasp"], closing_axis="y",
                        open=S.finger_angle_for_gap(min(0.085, m["heel_width"] + 0.025)))
        else:
            meta = dict(name=name, grasp_local=m["grasp"], closing_axis="x", open=S.finger_angle_for_gap(m["pinch_open_width"]))
        meta.update(length=m["length"], width=m["width"], height=m["height"], mass=m["mass"])
    S.size_box_for(meta["length"], meta["width"], meta["height"])
    S.write_shoebox_usd(out["shoebox"], rng)
    out["shoe_meta"] = meta
    print(f"[assets] shoe {meta['name']}  L={meta['length']:.3f} W={meta['width']:.3f} H={meta['height']:.3f}  "
          f"box inner {S.BOX['inner']} x {S.BOX['height']}", flush=True)
    return out


# ---------------------------------------------------------------- motion primitives


class Move:
    """Straight-line move of the tool centre point (slerp on orientation), then hold until it converges."""

    def __init__(self, goal, grip=None, speed=0.30, turn=1.0, min_steps=30, settle=120, tol=0.01):
        self.goal, self.grip, self.speed, self.turn = goal, grip, speed, turn
        self.min_steps, self.settle, self.tol = min_steps, settle, tol


class Track:
    """Closed-loop move: the goal is re-evaluated every step (e.g. from a measured hinge angle) and approached at
    bounded speed; it ends when `done(arm)` holds or after `timeout` steps."""

    def __init__(self, goal, done, grip=None, speed=0.15, turn=0.8, timeout=2400, compliance=None):
        self.goal, self.done, self.grip, self.speed, self.turn, self.timeout = goal, done, grip, speed, turn, timeout
        self.compliance = compliance  # arm joint stiffness scale while tracking, like a UR in force mode


class MoveJ(Move):
    """Joint-space move (UR "movej"): solve IK for the goal once, seeded from the current configuration, then
    interpolate joint angles. Used for transits, where Cartesian paths can pass near singularities."""

    def __init__(self, goal, grip=None, joint_speed=0.9, min_steps=60):
        super().__init__(goal, grip=grip, min_steps=min_steps)
        self.joint_speed = joint_speed


class Branch:
    """Runs `moves` (in front of the rest of the queue) when `cond(arm)` is False; e.g. regrasp after a slip."""

    def __init__(self, cond, moves):
        self.cond, self.moves = cond, moves


def hold(steps, grip):
    return Move(lambda arm: (arm.p, arm.R), grip=grip, min_steps=steps, settle=0)


def up(dz):
    return lambda arm: (arm.p + [0, 0, dz], arm.R)


def back_off(d):
    """Retreat along the approach axis (TCP +z)."""
    return lambda arm: (arm.p - arm.R[:, 2] * d, arm.R)


def to(p):
    return lambda arm: (np.asarray(p, float), arm.R)


def home(arm):
    return HOME[arm.side], top_down(arm.comfy(arm.base_yaw))


def home_j():
    return MoveJ(home)


class Arm:
    """One UR5e tracking a TCP target with damped-least-squares differential IK on wrist_3_link."""

    def __init__(self, side, robot, device):
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.torch = torch
        self.side, self.robot, self.device = side, robot, device
        self.base_pos, self.base_yaw = S.ROBOT_BASES[side]
        self.joint_ids = robot.find_joints(S_UR_JOINTS, preserve_order=True)[0]
        self.finger_id = robot.find_joints([S.FINGER_JOINT])[0]
        # The 2F-85 linkage joints are mimic-constrained to finger_joint with these signs (measured, and the same
        # pattern Isaac Lab's gear-assembly task writes). Driving them anywhere else fights the constraint, stalls
        # the fingers and spins the wrist, so every gripper joint gets a consistent target.
        signs = dict(finger_joint=1, right_outer_knuckle_joint=1, left_inner_finger_joint=-1, right_inner_finger_joint=1,
                     left_inner_finger_knuckle_joint=-1, right_inner_finger_knuckle_joint=-1)
        names = [n for n in signs if n in robot.joint_names]
        self.grip_ids = robot.find_joints(names, preserve_order=True)[0]
        self.grip_signs = torch.tensor([[float(signs[n]) for n in names]], device=device)
        self.w3 = robot.find_bodies(["wrist_3_link"])[0][0]
        self.jac_idx = self.w3 - 1 if robot.is_fixed_base else self.w3
        self.k0 = robot.data.joint_stiffness.torch[:, self.joint_ids].clone()
        self.d0 = robot.data.joint_damping.torch[:, self.joint_ids].clone()
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=device)
        self.reset()

    def set_compliance(self, scale):
        """Scale the arm's joint stiffness (damping by sqrt) so the hand yields to a constraint, e.g. a hinged lid."""
        self.robot.write_joint_stiffness_to_sim_index(stiffness=self.k0 * scale, joint_ids=self.joint_ids)
        self.robot.write_joint_damping_to_sim_index(damping=self.d0 * scale**0.5, joint_ids=self.joint_ids)

    def reset(self):
        if hasattr(self, "k0"):
            self.set_compliance(1.0)
        self.p, self.R, self.grip = HOME[self.side].copy(), top_down(self.base_yaw), 1.0
        self.q_cmd, self.q_prev = None, None
        self.open_pos = getattr(self, "pinch_open", S.GRIP_OPEN)
        self.queue, self.move = [], None
        self.ik.reset()

    def joint_pos(self):
        return self.robot.data.joint_pos.torch[0, self.joint_ids].cpu().numpy().astype(float)

    def root_pose(self):
        r = self.robot.data.root_pose_w.torch[0].cpu().numpy()
        return pose7(r[:3], r[3:])

    def calibrate_kinematics(self):
        """Fit the DH model to the simulated robot: candidate base conventions A (URDF base_link is the DH base
        turned half round about z), flange offset B measured at the current configuration."""
        q = self.joint_pos()
        T = np.linalg.inv(self.root_pose()) @ self.w3_pose()
        self.kin_candidates = []
        for A in (rot_z_pose(np.pi), np.eye(4)):
            B = np.linalg.inv(A @ ur_kin.fk(q, S.ROBOT_MODEL)) @ T
            self.kin_candidates.append((A, B))
        self.kin = self.kin_candidates[0]

    def check_kinematics(self):
        """At a second configuration, keep the candidate whose prediction matches the simulator."""
        q = self.joint_pos()
        T = np.linalg.inv(self.root_pose()) @ self.w3_pose()
        errs = [np.linalg.norm((A @ ur_kin.fk(q, S.ROBOT_MODEL) @ B)[:3, 3] - T[:3, 3]) for A, B in self.kin_candidates]
        self.kin = self.kin_candidates[int(np.argmin(errs))]
        print(f"[kin {self.side}] DH model vs sim at q={np.round(q, 2)}: errors {np.round(np.array(errs) * 1000, 2)} mm", flush=True)
        return min(errs)

    def model_tcp(self, q):
        A, B = self.kin
        return self.root_pose() @ A @ ur_kin.fk(q, S.ROBOT_MODEL) @ B @ T_W3_TCP

    def solve_ik(self, p, R, seed):
        A, B = self.kin
        T_dh = np.linalg.inv(A) @ np.linalg.inv(self.root_pose()) @ pose(R, p) @ np.linalg.inv(T_W3_TCP) @ np.linalg.inv(B)
        q, res = ur_kin.ik_best(T_dh, seed, S.UR_HOME_Q, S.ROBOT_MODEL)
        if res > 5e-3:
            print(f"[kin {self.side}] no valid IK within joint limits (residual {res:.3f}) for target {np.round(p, 3)}", flush=True)
        return q

    def w3_pose(self):
        d = self.robot.data.body_pose_w.torch[0, self.w3].cpu().numpy()
        return pose7(d[:3], d[3:])

    def tcp(self):
        return self.w3_pose() @ T_W3_TCP

    def comfy(self, yaw):
        """The gripper is symmetric under a half turn; take the yaw nearer the current wrist yaw."""
        ref = tcp_yaw(self.R)
        return yaw if abs(wrap(yaw - ref)) <= np.pi / 2 else wrap(yaw + np.pi)

    @property
    def idle(self):
        return self.move is None and not self.queue

    def plan_step(self):
        if isinstance(self.move, MoveJ):
            m = self.move
            m.k += 1
            s = min(1.0, m.k / m.dur)
            s = s * s * (3 - 2 * s)
            self.q_cmd = m.q0 + (m.q1 - m.q0) * s
            T = self.model_tcp(self.q_cmd)
            self.p, self.R = T[:3, 3], T[:3, :3]
            if m.k >= m.dur + 30:
                self.q_cmd, self.move = None, None
                self.p, self.R = m.p1, m.R1
            return
        while self.move is None and self.queue and isinstance(self.queue[0], Branch):
            b = self.queue.pop(0)
            if not b.cond(self):
                self.queue[0:0] = b.moves()
        if self.move is None and self.queue and isinstance(self.queue[0], Track):
            self.move = self.queue.pop(0)
            self.move.k = 0
            if self.move.grip is not None:
                self.grip = self.move.grip
            if self.move.compliance is not None:
                self.set_compliance(self.move.compliance)
        if isinstance(self.move, Track):
            m = self.move
            m.k += 1
            p1, R1 = (np.asarray(v, float) for v in m.goal(self))
            dp = p1 - self.p
            n = np.linalg.norm(dp)
            self.p = self.p + (dp if n <= m.speed / PHYSICS_HZ else dp / n * m.speed / PHYSICS_HZ)
            ang = np.linalg.norm(mat_to_rotvec(self.R.T @ R1))
            self.R = R1 if ang <= m.turn / PHYSICS_HZ else slerp_mat(self.R, R1, m.turn / PHYSICS_HZ / ang)
            if m.done(self) or m.k >= m.timeout:
                if m.compliance is not None:
                    self.set_compliance(1.0)
                self.p, self.R = self.tcp()[:3, 3].copy(), self.tcp()[:3, :3].copy()  # continue from where the hand is
                self.move = None
            return
        if self.move is None and self.queue and isinstance(self.queue[0], MoveJ):
            m = self.move = self.queue.pop(0)
            m.p1, m.R1 = (np.asarray(v, float) for v in m.goal(self))
            if m.grip is not None:
                self.grip = m.grip
            m.q0 = self.joint_pos()
            m.q1 = self.solve_ik(m.p1, m.R1, m.q0)
            m.dur = int(max(m.min_steps, np.abs(m.q1 - m.q0).max() / m.joint_speed * PHYSICS_HZ))
            if args.debug:
                print(f"[movej {self.side}] to {np.round(m.p1, 3)} max|dq| {np.rad2deg(np.abs(m.q1 - m.q0).max()):.0f}deg "
                      f"q0 {np.round(np.rad2deg(m.q0)).astype(int)} q1 {np.round(np.rad2deg(m.q1)).astype(int)}", flush=True)
            m.k = 0
            return self.plan_step()
        if self.move is None and self.queue:
            m = self.move = self.queue.pop(0)
            m.p0, m.R0 = self.p.copy(), self.R.copy()
            m.p1, m.R1 = (np.asarray(v, float) for v in m.goal(self))
            if m.grip is not None:
                self.grip = m.grip
            ang = np.linalg.norm(mat_to_rotvec(m.R0.T @ m.R1))
            m.dur = int(max(m.min_steps, np.linalg.norm(m.p1 - m.p0) / m.speed * PHYSICS_HZ, ang / m.turn * PHYSICS_HZ))
            m.k = 0
        if self.move is not None:
            m = self.move
            m.k += 1
            s = min(1.0, m.k / m.dur)
            s = s * s * (3 - 2 * s)
            self.p, self.R = m.p0 + (m.p1 - m.p0) * s, slerp_mat(m.R0, m.R1, s)
            err = np.linalg.norm(self.tcp()[:3, 3] - m.p1)
            if m.k >= m.dur and (err < m.tol or m.k >= m.dur + m.settle):
                self.move = None

    def apply(self):
        torch = self.torch
        if self.q_cmd is not None:
            self.q_prev = self.q_cmd.copy()
            q = torch.as_tensor(self.q_cmd[None], dtype=torch.float32, device=self.robot.device)
            self.robot.set_joint_position_target_index(target=q, joint_ids=self.joint_ids)
            g = self.open_pos if self.grip > 0.5 else S.GRIP_CLOSED
            self.robot.set_joint_position_target_index(target=self.grip_signs * float(g), joint_ids=self.grip_ids)
            return
        # Cartesian target -> joints with the calibrated UR model, seeded from the last command so the solution
        # stays on one branch (Isaac's Jacobian DLS veered onto other branches near wrist singularities).
        A, B = self.kin
        T_dh = np.linalg.inv(A) @ np.linalg.inv(self.root_pose()) @ pose(self.R, self.p) @ np.linalg.inv(T_W3_TCP) @ np.linalg.inv(B)
        q_now = self.joint_pos()
        seed = self.q_prev if self.q_prev is not None else q_now
        q_t, _ = ur_kin.ik(T_dh, seed, S.ROBOT_MODEL, iters=12, tol=1e-5, damping=1e-3)
        q_t = ur_kin.nearest_equivalent(q_t, seed)
        q_t = seed + np.clip(q_t - seed, -0.025, 0.025)  # UR speed: <= 3 rad/s at 120 Hz
        q_t = np.clip(q_t, -(ur_kin.LIMITS - 0.02), ur_kin.LIMITS - 0.02)
        self.q_prev = q_t
        q_des = torch.as_tensor(q_t[None], dtype=torch.float32, device=self.robot.device)
        self.robot.set_joint_position_target_index(target=q_des, joint_ids=self.joint_ids)
        g = self.open_pos if self.grip > 0.5 else S.GRIP_CLOSED
        self.robot.set_joint_position_target_index(target=self.grip_signs * float(g), joint_ids=self.grip_ids)

    def action(self):
        return np.concatenate([self.p, mat_to_rotvec(self.R), [self.grip]])


S_UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


# ---------------------------------------------------------------- task state and skills


class Cell:
    """numpy view of the scene state the skills need."""

    def __init__(self, scene, shoe_meta):
        self.scene, self.meta = scene, shoe_meta
        self.box = scene["shoebox"]

    def shoe_pose(self, k):
        d = self.scene[f"shoe_{k}"].data.root_pose_w.torch[0].cpu().numpy()
        return pose7(d[:3], d[3:])

    def box_pose(self):
        d = self.box.data.root_pose_w.torch[0].cpu().numpy()
        return pose7(d[:3], d[3:])

    def hinge_deg(self):
        return float(-np.rad2deg(self.box.data.joint_pos.torch[0, 0].item()))


def grasp_point(cell, k):
    T = cell.shoe_pose(k)
    g = np.asarray(cell.meta["grasp_local"], float)
    if k == 1 and cell.meta["name"] != "procedural":
        g = g * [1, -1, 1]  # shoe 1 is the mirrored (right) foot
    return T[:3, :3] @ g + T[:3, 3]


def heel_grasp(cell, k, dz=0.0):
    """Top-down pinch at the shoe's grasp point, closing along the axis its metadata names."""

    def goal(arm):
        T = cell.shoe_pose(k)
        p = grasp_point(cell, k)
        psi = 0.0 if cell.meta["closing_axis"] == "y" else -np.pi / 2
        return p + [0, 0, dz], top_down(arm.comfy(tcp_yaw(T[:3, :3] @ top_down(psi))))

    return goal


class Placement:
    """Sets the held shoe down at a box-frame pose. A heel-held shoe hangs toe-down, so place it by the heel: keep the
    hand top-down, put the grasped heel over its spot in the box, close the loop on the measured heel position and
    shoe yaw, then lower until the open fingers just clear the rim and let go; the toe swings in first."""

    def __init__(self, cell, k, xy_yaw):
        self.cell, self.k, self.xy_yaw = cell, k, xy_yaw

    def target(self):
        (x, y), yaw = self.xy_yaw
        return self.cell.box_pose() @ pose(rot_z(yaw), [x, y, S.BOX["wall"] + 0.004])

    def heel_target(self):
        g = np.asarray(self.cell.meta["grasp_local"], float)
        if self.k == 1 and self.cell.meta["name"] != "procedural":
            g = g * [1, -1, 1]
        return (self.target() @ np.append(g, 1.0))[:3]

    def rim_tcp_z(self):
        return self.cell.box_pose()[2, 3] + S.BOX["height"] + S.PAD_BELOW_TCP + 0.012

    def errors(self):
        Sh = self.cell.shoe_pose(self.k)
        e = self.heel_target() - grasp_point(self.cell, self.k)
        G = self.target()
        dyaw = wrap(np.arctan2(G[1, 0], G[0, 0]) - np.arctan2(Sh[1, 0], Sh[0, 0]))
        return e, dyaw

    def __call__(self, dz):
        def goal(arm):
            e, dyaw = self.errors()
            R = rot_z(dyaw) @ arm.R
            return np.array([arm.p[0] + e[0], arm.p[1] + e[1], self.rim_tcp_z() + dz]), R

        return goal

    def align(self):
        def goal(arm):
            e, dyaw = self.errors()
            return arm.p + [e[0], e[1], 0.0], rot_z(dyaw) @ arm.R

        def done(arm):
            e, dyaw = self.errors()
            return np.linalg.norm(e[:2]) < 0.012 and abs(dyaw) < 0.10

        return Track(goal, done, speed=0.06, turn=0.5, timeout=720)

    def descend(self):
        return Move(lambda arm: (np.array([arm.p[0], arm.p[1], self.rim_tcp_z()]), arm.R), speed=0.08)


class Lid:
    """Lid kinematics from the box base pose and the hinge angle, and the pull-tab grasp on it."""

    def __init__(self, cell, box_usd):
        self.cell, self.T_lid_tcp = cell, None
        lx, ly = S.BOX["inner"]
        w, h, t, gap = S.BOX["wall"], S.BOX["height"], S.BOX["lid_t"], S.BOX["gap"]
        oy = ly + 2 * w
        self.hinge = np.array([0.0, oy / 2, h])
        Ly = oy + gap + t
        self.Ly, self.skirt = Ly, S.BOX["skirt"]
        self.tab = np.array([0.0, -Ly - S.TAB["depth"] / 2, t - S.TAB["height"] / 2])

    def lid_pose(self, deg):
        return self.cell.box_pose() @ pose(np.eye(3), self.hinge) @ pose(rot_x(-np.deg2rad(deg)), [0, 0, 0])

    def lid_body_pose(self):
        box = self.cell.box
        i = box.body_names.index("lid")
        d = box.data.body_pose_w.torch[0, i].cpu().numpy()
        return pose7(d[:3], d[3:])

    def capture_grasp(self, arm):
        """After the fingers close: re-anchor the swing on the *actual* hand pose relative to the lid. Closing the
        pads on the tab pulls the hand a few cm off its command; anchoring on the command would bake that offset into
        every arc target and drag the box. The command is re-seated on the actual pose so nothing jumps."""
        T = arm.tcp()
        # Freeze the hinge axis in the world at grasp time: targets then depend only on the measured hinge angle,
        # so if the box creeps the arm resists instead of following it (following made the drift self-sustaining).
        self.T_hinge0 = self.cell.box_pose() @ pose(np.eye(3), self.hinge)
        theta = np.deg2rad(self.cell.hinge_deg())
        self.T_lid_tcp = np.linalg.inv(self.T_hinge0 @ pose(rot_x(-theta), [0, 0, 0])) @ T
        arm.p, arm.R = T[:3, 3].copy(), T[:3, :3].copy()
        return True

    def tcp_at(self, deg, arm, standoff=0.0):
        if self.T_lid_tcp is None:
            # Pinch the tab across its thickness (lid x); of the two equivalent wrist yaws take the nearer one.
            opts = [pose(top_down(psi), self.tab) for psi in (-np.pi / 2, np.pi / 2)]
            ref = tcp_yaw(arm.R)
            self.T_lid_tcp = min(opts, key=lambda T: abs(wrap(tcp_yaw((self.lid_pose(0.0) @ T)[:3, :3]) - ref)))
        T = self.lid_pose(deg) @ self.T_lid_tcp @ pose(np.eye(3), [0, 0, -standoff])
        return T[:3, 3], T[:3, :3]

    def swing(self, to_deg, lead_deg=6.0, tol_deg=3.0):
        """Swing the held lid toward to_deg by targeting the lid pose a few degrees past the *measured* hinge angle,
        so the arm follows the lid wherever the box actually is instead of dragging it along a fixed arc."""

        def goal(arm):
            now = self.cell.hinge_deg()
            step = np.clip(to_deg - now, -lead_deg, lead_deg)
            # Lid pose `step` degrees on from the measured angle, about the hinge axis frozen at grasp time.
            T = self.T_hinge0 @ pose(rot_x(-np.deg2rad(now + step)), [0, 0, 0]) @ self.T_lid_tcp
            return T[:3, 3], T[:3, :3]

        return [Track(goal, lambda arm: abs(self.cell.hinge_deg() - to_deg) < tol_deg, speed=0.12, turn=0.6, compliance=0.15)]

    def grasp(self):
        state = {}

        def at(standoff):
            def goal(arm):
                return self.tcp_at(state.setdefault("deg", self.cell.hinge_deg()), arm, standoff)

            return goal

        return [MoveJ(at(0.08), grip=1.0), Move(at(0.0), speed=0.08), hold(60, 0.0), Branch(self.capture_grasp, list)]

    def push_closed(self, lead_deg=8.0, end_deg=3.0):
        """Close an open lid the way a person does: with the fingers closed, press on its outer face near the free
        edge from the robot side and follow it down, compliant, as the measured hinge angle falls."""
        lx, ly = S.BOX["inner"]
        Ly = ly + 2 * S.BOX["wall"] + S.BOX["gap"] + S.BOX["lid_t"]
        contact = np.array([0.0, -0.7 * Ly, S.BOX["lid_t"]])
        state = {}

        def tool(deg, arm, standoff):
            T_h = state.setdefault("hinge", self.cell.box_pose() @ pose(np.eye(3), self.hinge))
            T_l = T_h @ pose(rot_x(-np.deg2rad(deg)), [0, 0, 0])
            n = T_l[:3, :3] @ [0, 0, 1.0]  # outer face normal
            z = -n
            y = T_l[:3, 0]
            R = np.stack([np.cross(y, z), y, z], axis=1)
            if abs(wrap(tcp_yaw(R) - tcp_yaw(arm.R))) > np.pi / 2 and abs(z[2]) > 0.7:
                R = R @ rot_z(np.pi)  # the closed gripper is symmetric under a half turn
            p = (T_l @ np.append(contact, 1.0))[:3] + n * standoff
            return p, R

        approach = MoveJ(lambda arm: tool(state.setdefault("start", self.cell.hinge_deg()), arm, 0.07), grip=0.0)
        touch = Move(lambda arm: tool(state["start"], arm, 0.0), speed=0.05)
        follow = Track(lambda arm: tool(max(self.cell.hinge_deg() - lead_deg, 0.0), arm, -0.004),
                       lambda arm: self.cell.hinge_deg() < end_deg, speed=0.10, turn=0.8, compliance=0.15)
        return [approach, touch, follow, Move(lambda arm: tool(0.0, arm, 0.08), speed=0.1)]


def script(cell, lid):
    """Phases; each is (name, {side: [Move]}) and the next starts once both arms are idle."""
    half_gap = cell.meta["width"] / 2 + 0.006
    place1 = Placement(cell, 0, ((0.0, -half_gap), np.pi))  # toe toward -x
    place2 = Placement(cell, 1, ((0.0, half_gap), 0.0))  # toe toward +x: heel-to-toe
    def pick(k, retries=2):
        # Grasp, lift, and if the shoe did not come up with the gripper, open and try again from above.
        moves = [MoveJ(heel_grasp(cell, k, 0.12), grip=1.0), Move(heel_grasp(cell, k), speed=0.10), hold(60, 0.0), Move(up(0.20), speed=0.2)]
        if retries:
            # A heel-held shoe pivots (toe on the table), so test that the grasped heel came up, not the shoe origin.
            lifted = lambda arm: grasp_point(cell, k)[2] > S.TABLE_Z + cell.meta["grasp_local"][2] + 0.08
            moves.append(Branch(lifted, lambda: [hold(30, 1.0), Move(up(0.10)), home_j(), *pick(k, retries - 1)]))
        return moves

    place = lambda pl: [MoveJ(pl(0.10)), pl.align(), pl.descend(), hold(50, 1.0), Move(up(0.12))]
    return [
        # Left arm pinches the pull tab and swings the lid past vertical; it stays open on its stop, so the arm
        # lets go and clears the box. Right arm fetches shoe 1 meanwhile.
        ("open_lid", {"left": [*lid.grasp(), *lid.swing(LID_OPEN_DEG), hold(40, 1.0), Move(back_off(0.08)), home_j()],
                      "right": [*pick(0), MoveJ(to(RIGHT_WAIT))]}),
        # Long transits go through the home pose (joint space).
        ("place_shoes", {"right": [*place(place1), home_j(), *pick(1), *place(place2), home_j()]}),
        # Re-grasp the tab where the lid rests and swing it shut, then let go and clear.
        ("close_lid", {"left": [*lid.grasp(), *lid.swing(LID_CLOSED_DEG), hold(50, 1.0), Move(back_off(0.08)), home_j()]}),
    ]


def shoe_in_box(cell, k):
    T = np.linalg.inv(cell.box_pose()) @ cell.shoe_pose(k)
    (lx, ly), h = S.BOX["inner"], S.BOX["height"]
    return abs(T[0, 3]) < lx / 2 and abs(T[1, 3]) < ly / 2 and T[2, 3] < h and T[2, 2] > 0.7


# ---------------------------------------------------------------- recording


class Recorder:
    def __init__(self, scene, cam_ids, robots):
        from isaaclab.sim.utils import get_current_stage

        self.scene, self.cam_ids, self.robots = scene, cam_ids, robots
        stage = get_current_stage()
        self.boxes = {}
        for side, r in robots.items():
            root = r.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_[^/]+", "env_0").replace("env_.*", "env_0")
            links = S.rigid_link_boxes(stage, root)
            names = r.body_names
            self.boxes[side] = [(names.index(p.rsplit("/", 1)[-1]), g, c, h) for p, g, c, h in links if p.rsplit("/", 1)[-1] in names]
            assert self.boxes[side], f"no link boxes matched for {root}"

    def label_boxes(self):
        robot, grip = [], []
        for side, r in self.robots.items():
            bp = r.data.body_pose_w.torch[0].cpu().numpy()
            for i, g, c, h in self.boxes[side]:
                b = (pose7(bp[i, :3], bp[i, 3:]) @ pose(np.eye(3), c), h)
                robot.append(b)
                if g:
                    grip.append(b)
        return robot, grip

    def frames(self):
        out = {}
        for cid in self.cam_ids:
            cam = self.scene[f"camera_{cid}"]
            o = {k: (v.torch if hasattr(v, "torch") else v) for k, v in cam.data.output.items()}
            seg = o["semantic_segmentation"][0].cpu().numpy()
            info = cam.data.info["semantic_segmentation"] or {}
            info = info.get("idToLabels", info)
            t = lambda x: (x.torch if hasattr(x, "torch") else x)[0].cpu().numpy()
            E = pose7(t(cam.data.pos_w), t(cam.data.quat_w_ros))
            out[cid] = dict(
                rgb=o["rgb"][0, ..., :3].cpu().numpy(),
                depth=o["distance_to_image_plane"][0, ..., 0].cpu().numpy(),
                obj_mask=seg_mask(seg, info, "shoe").astype(np.uint8),
                gripper_mask=seg_mask(seg, info, "gripper"),
                K=t(cam.data.intrinsic_matrices),
                E_cam2world=E,
            )
        return out


def seg_mask(seg, id_to_labels, word):
    """Pixels whose semantic label contains `word`. Label keys are either integer ids or "(r, g, b, a)" colours;
    a one-channel image then holds the colour packed little-endian into a uint32 (r | g<<8 | b<<16 | a<<24)."""
    mask = np.zeros(seg.shape[:2], bool)
    for k, v in id_to_labels.items():
        if word not in str(v).lower():
            continue
        k = str(k).strip()
        if k.startswith("("):
            rgba = [int(c) for c in k.strip("()").split(",")]
            if seg.shape[-1] == 4:
                mask |= np.all(seg == np.array(rgba, seg.dtype), axis=-1)
            else:
                packed = rgba[0] | rgba[1] << 8 | rgba[2] << 16 | rgba[3] << 24
                mask |= seg[..., 0].astype(np.int64) & 0xFFFFFFFF == packed
        else:
            mask |= seg[..., 0] == int(k)
    return mask


# ---------------------------------------------------------------- episode loop


def main():
    global S, T_W3_TCP
    import isaaclab.sim as sim_utils

    rng = np.random.default_rng(args.seed)
    sim_cfg = sim_utils.SimulationCfg(dt=1 / PHYSICS_HZ, device=args.device)
    # The scene (with its RTX cameras) is built after launch, so tell the launcher up front: Kit, Isaac Sim
    # PhysX and camera rendering.
    args.require_kit, args.enable_cameras, args.physics = True, True, "isaacsim_physx"
    with launch_simulation(sim_cfg, args):
        import scene as S

        T_W3_TCP = S.T_W3_GB @ S.T_GB_TCP
        from isaaclab.utils import instantiate

        from repack_scene import make_scene_cfg
        from tasks import TASKS
        from writer import EpisodeWriter

        assets = prepare_assets(Path(args.out) / "_assets", rng, args.shoe)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([0.0, -1.6, 1.0], [0.0, -0.1, 0.0])
        scene = instantiate(make_scene_cfg(assets, args.width, args.height, args.wrist_cams))
        fix_robot_prims()
        fix_shoe_prims()
        check_flange_offset()
        sim.reset()
        robots = {"left": scene["robot_left"], "right": scene["robot_right"]}
        arms = {side: Arm(side, r, sim.device) for side, r in robots.items()}
        arms["right"].pinch_open = assets["shoe_meta"]["open"]
        for arm in arms.values():
            arm.calibrate_kinematics()
        cell = Cell(scene, assets["shoe_meta"])
        cam_ids = sorted(int(n.split("_")[1]) for n in scene.keys() if n.startswith("camera_"))
        recorder = Recorder(scene, cam_ids, robots)
        task = TASKS["shoe_return"]
        out = Path(args.out)
        stats = []

        def step(render):
            for arm in arms.values():
                arm.apply()
            scene.write_data_to_sim()
            sim.step(render=render)
            scene.update(sim.get_physics_dt())

        for ep in range(args.start_episode, args.start_episode + args.episodes):
            t0 = time.time()
            reset_episode(scene, robots, cell, rng)
            for arm in arms.values():
                arm.reset()
            lid = Lid(cell, assets["shoebox"])
            for _ in range(240):  # settle, arms to home, renderer warm-up
                for arm in arms.values():
                    arm.plan_step()
                step(render=True)
            if ep == args.start_episode:
                for arm in arms.values():
                    assert arm.check_kinematics() < 0.005, "DH model does not match the simulated arm"

            split = "val" if args.val_every and ep % args.val_every == 0 else "train"
            writer = EpisodeWriter(out, split, ep, cam_ids, task)
            video = []
            phases, k, dropped, min_gap, phase_name = script(cell, lid), 0, False, np.inf, "setup"
            while k < args.max_steps:
                if all(a.idle for a in arms.values()):
                    if not phases:
                        break
                    phase_name, moves = phases.pop(0)
                    for side, m in moves.items():
                        arms[side].queue.extend(m)
                for arm in arms.values():
                    arm.plan_step()
                record = k % args.record_every == 0
                step(render=record)
                wl, wr = (robots[s].data.body_pose_w.torch[0, arms[s].w3, :3].cpu().numpy() for s in ("left", "right"))
                min_gap = min(min_gap, float(np.linalg.norm(wl - wr)))
                if args.debug and k % 60 == 0:
                    errs = {s: np.linalg.norm(a.tcp()[:3, 3] - a.p) for s, a in arms.items()}
                    rerr = {s: np.rad2deg(np.linalg.norm(mat_to_rotvec(a.tcp()[:3, :3].T @ a.R))) for s, a in arms.items()}
                    print(f"[dbg k={k} {phase_name}] tcp err L {errs['left']*100:.1f}cm/{rerr['left']:.0f}deg R {errs['right']*100:.1f}cm/{rerr['right']:.0f}deg "
                          f"hinge {cell.hinge_deg():.0f} box {np.round(cell.box_pose()[:3, 3], 3)} "
                          f"fingers L {np.rad2deg(robots['left'].data.joint_pos.torch[0, arms['left'].finger_id].item()):.0f} "
                          f"R {np.rad2deg(robots['right'].data.joint_pos.torch[0, arms['right'].finger_id].item()):.0f}deg", flush=True)
                if record:
                    frames = recorder.frames()
                    robot_boxes, grip_boxes = recorder.label_boxes()
                    writer.add_frame(np.concatenate([arms["left"].action(), arms["right"].action()]), frames, robot_boxes, grip_boxes)
                    if args.video:
                        video.append(frames[0]["rgb"])
                if any(cell.shoe_pose(i)[2, 3] < S.TABLE_Z - 0.05 for i in range(2)):
                    dropped = True
                    break
                k += 1
            finished = not phases and all(a.idle for a in arms.values())
            for _ in range(60):
                step(render=False)
            inside = [shoe_in_box(cell, i) for i in range(2)]
            hinge = cell.hinge_deg()
            ok = finished and all(inside) and hinge < 5.0 and not dropped and min_gap > ARM_CLEARANCE
            reason = None if ok else ("shoe_dropped" if dropped else "timeout" if not finished else "arm_collision" if min_gap <= ARM_CLEARANCE else "shoe_outside_box" if not all(inside) else "lid_open")
            wall = time.time() - t0
            print(f"[ep {ep}] phase={phase_name} in_box={inside} hinge={hinge:.1f}deg gap={min_gap:.2f}m -> {'ok' if ok else 'FAILED: ' + reason} ({writer.T} frames, {wall:.0f}s)", flush=True)
            stats.append(dict(episode=ep, success=ok, failure=reason, phase=phase_name, hinge_deg=hinge, frames=writer.T, wall_s=wall))
            if ok or args.keep_failures:
                writer.close(dict(success=ok, failure_label=reason or "", failure_phase="" if ok else phase_name, hinge_deg=hinge, seed=args.seed, schema_version=1))
            else:
                writer.discard()
            if args.video and video:
                save_video(Path(args.video) / f"episode_{ep}.mp4", video)
        n_ok = sum(s["success"] for s in stats)
        print(f"success {n_ok}/{len(stats)}")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"run_stats_seed{args.seed}_ep{args.start_episode}.json").write_text(json.dumps(stats, indent=1))


def fix_shoe_prims():
    """Scanned shoes: rubber/textile friction and the 'shoe' semantic class (the asset build authors neither)."""
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    rubber = S.friction_material(stage, "/World/Materials/shoe_rubber", 0.9)
    for k in range(2):
        root = stage.GetPrimAtPath(f"/World/envs/env_0/Shoe_{k}")
        S.label(root, "shoe")
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                S.bind_physics_material(prim, rubber)


def check_flange_offset():
    """T_W3_GB (wrist_3 -> 2F-85 base) is measured from the asset; confirm it on the loaded robot."""
    from pxr import Usd

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    root = "/World/envs/env_0/Robot_left"
    gb = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
          if p.GetName() == "base_link" and S.GRIPPER_KEY in str(p.GetPath())]
    assert gb, "no 2F-85 base_link under the robot"
    T = S.local_transform(stage, gb[0], root + "/wrist_3_link")
    err = np.abs(T - S.T_W3_GB).max()
    print(f"[check] wrist_3 -> gripper base matches scene.T_W3_GB to {err:.2e}", flush=True)
    assert err < 1e-4, f"wrist_3 -> gripper base differs from scene.T_W3_GB:\n{T}"


def strip_nested_gripper_roots(stage):
    """The UR5e asset's Gripper variant leaves the 2F-85 with its own ArticulationRootAPI (and a disabled
    PhysxArticulationAPI); the UR10e asset's does not. With it, closing the gripper drives the wrist into a runaway
    spin (tools/gripper_hold_probe.py). Strip it everywhere on the stage, clone templates included, before physics
    starts."""
    from pxr import Usd, UsdPhysics

    n = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.GetName() == "Robotiq_2F_85" and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            for api in ("PhysxArticulationAPI", "NewtonArticulationRootAPI"):
                prim.RemoveAppliedSchema(api)
            n += 1
    return n


def fix_robot_prims():
    """Before physics starts: strip the 2F-85's nested articulation root, tag its links 'gripper' (the robot
    spawns as 'robot') and give its inner fingers high-friction pads."""
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    print(f"[fix] stripped {strip_nested_gripper_roots(stage)} nested gripper articulation roots", flush=True)
    pads = S.friction_material(stage, "/World/Materials/finger_pads", 1.2)
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if S.GRIPPER_KEY in path and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            S.label(prim, "gripper")
            if "inner_finger" in prim.GetName():
                S.bind_physics_material(prim, pads)  # silicone fingertip pads


def reset_episode(scene, robots, cell, rng):
    import torch

    dev = scene.device
    for r in robots.values():
        r.write_joint_position_to_sim_index(position=r.data.default_joint_pos.torch.clone())
        r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(r.data.default_joint_vel.torch))
        r.reset()
    origin = scene.env_origins[0].cpu().numpy()
    T_box = S.box_pose_from_nominal(rng)
    box = scene["shoebox"]
    box.write_root_pose_to_sim_index(root_pose=torch.tensor([[*(T_box[:3, 3] + origin), *to_xyzw(T_box[:3, :3])]], dtype=torch.float32, device=dev))
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))
    box.write_joint_position_to_sim_index(position=torch.zeros(1, 1, device=dev))
    box.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, 1, device=dev))
    box.reset()

    class _Dims:
        length, width = cell.meta["length"], cell.meta["width"]

    for k, (xy, yaw) in enumerate(S.sample_shoe_poses(rng, _Dims)):
        shoe = scene[f"shoe_{k}"]
        p = [xy[0] + origin[0], xy[1] + origin[1], S.TABLE_Z + 0.003 + origin[2]]
        shoe.write_root_pose_to_sim_index(root_pose=torch.tensor([[*p, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]], dtype=torch.float32, device=dev))
        shoe.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))
        shoe.reset()


def save_video(path, frames, fps=15):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR))
    vw.release()


if __name__ == "__main__":
    main()
