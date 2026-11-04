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
parser.add_argument("--max-steps", type=int, default=24000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--grasp", default="collar", choices=["collar", "clamp"], help="shoe grasp: collar pinch or whole-heel clamp")
parser.add_argument("--shoe", default="random", help="scanned shoe model name, 'random', 'random:train' / 'random:test' (held-out split), or 'procedural'")
parser.add_argument("--debug", action="store_true", help="print TCP tracking error twice a second")
parser.add_argument("--skill", default=None, choices=["pick", "carry"], help="debug: run one skill instead of the task (pick: lift shoe 0 and hold it; carry: also swing it over the box and back)")
parser.add_argument("--grasp-depth", type=float, default=0.015, help="lower the shoe grasp point by this much (m) so the pads cover the heel counter, not just the collar edge")
parser.add_argument("--grip-effort", type=float, default=None, help="override the 2F-85 finger_joint effort limit (Nm)")
parser.add_argument("--grip-close", type=float, default=None, help="finger_joint close target (rad); default S.GRIP_CLOSED")
parser.add_argument("--grip-stiffness", type=float, default=None, help="override the 2F-85 finger_joint drive stiffness")
parser.add_argument("--policy", default=None, help="evaluate a trained policy checkpoint (policy/train.py) instead of the scripted expert")
parser.add_argument("--no-write", action="store_true", help="do not write episode files (policy evaluation)")
parser.add_argument("--video", default=None, help="write the overhead camera of every episode to <dir>/episode_<E>.mp4")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np  # noqa: E402

import collision  # noqa: E402
import ur_kin  # noqa: E402
from geometry import mat_to_quat, mat_to_rotvec, pose, quat_to_mat, rot_x, rot_z, slerp_mat, tcp_yaw, top_down, wrap  # noqa: E402

PHYSICS_HZ = 120
HOME = {"left": np.array([-0.45, 0.02, 0.35]), "right": np.array([0.40, 0.02, 0.35])}
RIGHT_WAIT = np.array([0.30, -0.18, 0.30])
LID_HOLD_DEG = 95.0  # held open by the left arm during packing: past vertical, so the lid's weight presses it open
# against the (compliant) hand and the 98 deg stop instead of sagging over the box
LID_OPEN_DEG = 95.0  # past vertical: the lid then rests open on its 98 deg stop by itself. (A 105 deg stop put the
# tab out of comfortable reach for closing. The swing can stall short of vertical for some box poses: it is
# time-limited and followed by a push on the lid's inner face if needed.)
LID_CLOSED_DEG = 2.0
CARRY_JOINT_SPEED = 0.35  # rad/s, joint-space speed with a shoe in hand (a heel-held shoe swings like a pendulum; at 0.6 one was flung out)
LID_BUFFER = 0.05  # m, extra clearance around the lid for collision-checked transits
RELEASE = 2.0  # grip command: open fully (85 mm). Used once a shoe rests flat in the box: at pinch width the pads
# still dragged a thick heel wall up with the hand.
TOE_SHIFT = 0.02  # m, a placed shoe sits this far toward its toe from the box's centre line
# A heel-held shoe hangs toe-down, its toe up to a shoe length (~0.3 m) below the hand: lift and carry high enough
# that it clears the box walls (a toe that caught the box corner knocked the shoe out of the fingers).
LIFT = 0.35  # m, lift after the grasp
CARRY_DZ = 0.35  # m, TCP above the rim when the shoe arrives over the box
MAX_CORRECTION = 0.08  # m, largest closed-loop placement correction
HELD_DIST = 0.06  # m, grasp point within this of the TCP counts as still held
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
        if shoe.startswith("random"):
            # "random:train" / "random:test": every 5th model (alphabetical) is held out for policy evaluation.
            split = shoe.partition(":")[2]
            if split:
                names = [n for i, n in enumerate(names) if (i % 5 == 4) == (split == "test")]
            name = names[rng.integers(len(names))]
        else:
            name = shoe
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
            meta = dict(name=name, grasp_local=(np.asarray(m["grasp"]) - [0, 0, args.grasp_depth]).tolist(), closing_axis="x",
                        open=S.finger_angle_for_gap(m["pinch_open_width"]))
        meta.update(length=m["length"], width=m["width"], height=m["height"], mass=m["mass"],
                    wall=m.get("wall", 0.02), opening=m.get("opening", 0.09))
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

    def __init__(self, goal, done, grip=None, speed=0.15, turn=0.8, timeout=2400, compliance=None, keep_compliance=False):
        self.goal, self.done, self.grip, self.speed, self.turn, self.timeout = goal, done, grip, speed, turn, timeout
        self.compliance = compliance  # arm joint stiffness scale while tracking, like a UR in force mode
        self.keep_compliance = keep_compliance  # stay compliant afterwards (e.g. while holding a lid open)


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
        self.cell = None
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

    def held_shoe(self):
        """Index of the shoe in this hand (grip closed and its grasp point at the TCP), else None."""
        if self.grip >= 0.5 or self.cell is None:
            return None
        tcp = self.tcp()[:3, 3]
        d = [np.linalg.norm(grasp_point(self.cell, k) - tcp) for k in range(2)]
        return int(np.argmin(d)) if min(d) < HELD_DIST else None

    def attached(self):
        """Spheres along the held shoe's length, in the TCP frame at its current, measured pose in the hand."""
        k = self.held_shoe()
        if k is None:
            return None
        m = self.cell.meta
        T = np.linalg.inv(self.tcp()) @ self.cell.shoe_pose(k)
        xs = np.linspace(-m["length"] / 2 + 0.04, m["length"] / 2 - 0.04, 5)
        pts = [(T @ [x, 0.0, 0.45 * m["height"], 1.0])[:3] for x in xs]
        return np.array(pts), 0.55 * max(m["width"], m["height"])

    def spheres_at(self, q, attached=None):
        A, B = self.kin
        return collision.arm_spheres(self.root_pose() @ A, q, B @ T_W3_TCP, S.ROBOT_MODEL, attached=attached)

    def plan_transit(self, q0, q1, p1, R1):
        """Joint waypoints to q1 that keep this arm clear of the table, the box, the lid and the other arm. Falls back
        to the straight path (and says so) if no clear path is found."""
        scene = self.transit_scene(self)
        att = self.attached()
        spheres_at = lambda q: self.spheres_at(q, att)
        if collision.path_hit(spheres_at, scene, q0, q1) is None:
            return [q1]
        p0, R0 = self.tcp()[:3, 3], self.tcp()[:3, :3]
        vias = []
        for p, R, seed in ((p0, R0, q0), (p1, R1, q1)):
            # Low detours first: the same height, pulled toward the robot (around the open box side, away from the
            # lid); then raised ones. A shoe swung over a high via fell out of the pinch.
            inward = np.append(self.base_pos[:2] - p[:2], 0.0)
            inward /= max(np.linalg.norm(inward), 1e-6)
            for d in (0.12, 0.22):
                vias.append(self.solve_ik(p + inward * d, R, seed))
            for z in (0.45, 0.60, 0.70):
                vias.append(self.solve_ik(np.array([p[0], p[1], max(p[2], z)]), R, seed))
        vias.append(self.solve_ik(HOME[self.side] + [0, 0, 0.10], top_down(self.comfy(self.base_yaw)), q0))
        path = collision.plan(spheres_at, scene, q0, q1, vias=vias, rng=np.random.default_rng(len(vias)))
        if path is None:
            print(f"[plan {self.side}] no collision-free transit to {np.round(p1, 3)} "
                  f"(direct path hits {collision.path_hit(spheres_at, scene, q0, q1)}); moving directly", flush=True)
            return [q1]
        return path

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
            a, b, dur = m.segs[m.i]
            s = min(1.0, m.k / dur)
            s = s ** 3 * (10 - 15 * s + 6 * s * s)  # minimum jerk: a hanging shoe is not jolted at the ends
            self.q_cmd = a + (b - a) * s
            T = self.model_tcp(self.q_cmd)
            self.p, self.R = T[:3, 3], T[:3, :3]
            if m.k >= dur and m.i + 1 < len(m.segs):
                m.i, m.k = m.i + 1, 0
            elif m.k >= dur + 30:
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
                if m.compliance is not None and not m.keep_compliance:
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
            m.path = self.plan_transit(m.q0, m.q1, m.p1, m.R1)
            pts = [m.q0, *m.path]
            speed = m.joint_speed if self.held_shoe() is None else min(m.joint_speed, CARRY_JOINT_SPEED)
            m.segs = [(a, b, int(max(m.min_steps, np.abs(b - a).max() / speed * PHYSICS_HZ))) for a, b in zip(pts, pts[1:])]
            if args.debug:
                print(f"[movej {self.side}] to {np.round(m.p1, 3)} via {len(m.path) - 1} max|dq| {np.rad2deg(np.abs(m.q1 - m.q0).max()):.0f}deg "
                      f"q0 {np.round(np.rad2deg(m.q0)).astype(int)} q1 {np.round(np.rad2deg(m.q1)).astype(int)}", flush=True)
            m.k, m.i = 0, 0
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
            self.apply_grip()
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
        self.apply_grip()

    def apply_grip(self):
        g = S.GRIP_OPEN if self.grip > 1.5 else self.open_pos if self.grip > 0.5 else (args.grip_close or S.GRIP_CLOSED)
        self.robot.set_joint_position_target_index(target=self.grip_signs * float(g), joint_ids=self.grip_ids)

    def action(self):
        return np.concatenate([self.p, mat_to_rotvec(self.R), [min(self.grip, 1.0)]])  # RELEASE is recorded as open


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
    """Sets the held shoe down at a box-frame pose the way a person lays down a shoe held by the heel: it hangs toe-down,
    so put the heel above where the *toe* will end up, lower until the toe touches the floor, then swing the heel back
    and down on an arc about that toe contact until the shoe lies flat with its heel on its spot. Lowering the heel
    straight onto its spot instead made the toe slide a shoe length along the floor; rubber on cardboard often did not,
    and the shoe stayed leaning (its top up to 9 cm above the rim, blocking the lid). Positions and heading are
    closed-loop on the measured shoe."""

    def __init__(self, cell, k, xy_yaw):
        self.cell, self.k, self.xy_yaw = cell, k, xy_yaw

    def target(self):
        (x, y), yaw = self.xy_yaw
        # Shifted toe-ward, leaving room behind the heel for the outer finger (see scene.size_box_for).
        return self.cell.box_pose() @ pose(rot_z(yaw), [x, y, S.BOX["wall"] + 0.004]) @ pose(np.eye(3), [TOE_SHIFT, 0, 0])

    def grasp_local(self):
        g = np.asarray(self.cell.meta["grasp_local"], float)
        return g * [1, -1, 1] if self.k == 1 and self.cell.meta["name"] != "procedural" else g

    def heel_target(self):
        return (self.target() @ np.append(self.grasp_local(), 1.0))[:3]

    def pivot(self):
        """(toe direction, horizontal and vertical grasp-to-toe distances) of the shoe lying flat."""
        g = self.grasp_local()
        return self.target()[:3, 0], self.cell.meta["length"] / 2 - g[0], g[2] - 0.005

    def floor_z(self):
        return self.cell.box_pose()[2, 3] + S.BOX["wall"]

    def start_xy(self):
        """Heel above the toe's final contact point."""
        x, rh, _ = self.pivot()
        return (self.heel_target() + x * rh)[:2]

    def rim_tcp_z(self):
        return self.cell.box_pose()[2, 3] + S.BOX["height"] + S.PAD_BELOW_TCP + 0.012

    def errors(self, cap=None):
        """Heel position error (to the pivot start) and yaw error of the held shoe. `cap` bounds the horizontal
        correction (fine alignment: a shoe lost from the hand cannot send the arm across the cell)."""
        Sh = self.cell.shoe_pose(self.k)
        e = np.append(self.start_xy() - grasp_point(self.cell, self.k)[:2], 0.0)
        n = np.linalg.norm(e[:2])
        if cap is not None and n > cap:
            e[:2] *= cap / n
        G = self.target()
        # Heading from the shoe's lateral (y) axis: a heel-held shoe hangs toe-down, pivoting about y, so its long
        # axis points nearly straight down (heading from it was noise) while y stays horizontal.
        dyaw = wrap(np.arctan2(G[1, 1], G[0, 1]) - np.arctan2(Sh[1, 1], Sh[0, 1]))
        return e, dyaw

    def __call__(self, dz):
        def goal(arm):
            e, dyaw = self.errors()
            R = rot_z(dyaw) @ arm.R
            return np.array([arm.p[0] + e[0], arm.p[1] + e[1], self.rim_tcp_z() + dz]), R

        return goal

    def align(self):
        def goal(arm):
            e, dyaw = self.errors(cap=MAX_CORRECTION)
            return arm.p + [e[0], e[1], 0.0], rot_z(dyaw) @ arm.R

        def done(arm):
            # 0.04 rad: a 0.3 m shoe 6 deg off swings its toe ~3 cm sideways, more than its share of the box play.
            e, dyaw = self.errors()
            return np.linalg.norm(e[:2]) < 0.008 and abs(dyaw) < 0.04

        return Track(goal, done, speed=0.06, turn=0.5, timeout=1200)

    def rest_tcp_z(self):
        """TCP height with the shoe lying flat (heel grasp point ~1 cm above its resting height)."""
        return self.floor_z() + self.grasp_local()[2] + 0.010

    def descend(self, n=8):
        """Down until the hanging toe touches the floor, then the heel swings back and down on an arc about the toe."""
        x, rh, rz = self.pivot()
        r, phi1 = np.hypot(rh, rz), np.arctan2(rz, rh)
        state = {}

        def arc_point(phi):
            def goal(arm):
                if "toe" not in state:  # toe contact fixed where the heel came down (closed loop up to here)
                    state["toe"] = np.append(arm.p[:2], self.floor_z())
                p = state["toe"] - x * r * np.cos(phi) + np.array([0, 0, r * np.sin(phi) + 0.010])
                return p, arm.R
            return goal

        touch_z = lambda: self.floor_z() + r + 0.015
        moves = [Move(lambda arm: (np.array([arm.p[0], arm.p[1], touch_z() + 0.08]), arm.R), speed=0.15),
                 Move(lambda arm: (np.array([arm.p[0], arm.p[1], touch_z()]), arm.R), speed=0.04)]
        moves += [Move(arc_point(phi), speed=0.04, min_steps=20, settle=30) for phi in np.linspace(np.pi / 2, phi1, n + 1)[1:]]
        return moves


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

    def obstacles(self):
        """Box base and lid (at the measured hinge angle) as oriented boxes [(T_world, half extents)]."""
        lx, ly = S.BOX["inner"]
        w, h, t, sk = S.BOX["wall"], S.BOX["height"], S.BOX["lid_t"], S.BOX["skirt"]
        ox, oy = lx + 2 * w, ly + 2 * w
        base = (self.cell.box_pose() @ pose(np.eye(3), [0, 0, h / 2]), np.array([ox / 2, oy / 2, h / 2]))
        Lx, depth = ox + 2 * (S.BOX["gap"] + t), self.Ly + S.TAB["depth"]
        # The open lid gets a wide buffer: a hand-held shoe swings, and the real arm lags its planned path. A shoe
        # brushing the lid tipped it back past vertical and it fell shut.
        lid = (self.lid_pose(self.cell.hinge_deg()) @ pose(np.eye(3), [0, -depth / 2, (t - sk) / 2]),
               np.array([Lx / 2, depth / 2, (t + sk) / 2]) + LID_BUFFER)
        return [base, lid]

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

    def swing(self, to_deg, lead_deg=12.0, tol_deg=3.0, timeout=4800, keep_compliance=False):
        """Swing the held lid toward to_deg by targeting the lid pose a few degrees past the *measured* hinge angle,
        so the arm follows the lid wherever the box actually is instead of dragging it along a fixed arc."""

        def goal(arm):
            now = self.cell.hinge_deg()
            step = np.clip(to_deg - now, -lead_deg, lead_deg)
            # Lid pose `step` degrees on from the measured angle, about the hinge axis frozen at grasp time.
            T = self.T_hinge0 @ pose(rot_x(-np.deg2rad(now + step)), [0, 0, 0]) @ self.T_lid_tcp
            return T[:3, 3], T[:3, :3]

        # Compliance 0.35: the hand yields to the hinge constraint. (Stiffer, 0.6 with an 18 deg lead, let the lid
        # stall near vertical and slide back shut inside the pinch.)
        return [Track(goal, lambda arm: abs(self.cell.hinge_deg() - to_deg) < tol_deg, speed=0.12, turn=0.6, timeout=timeout, compliance=0.35,
                      keep_compliance=keep_compliance)]

    def grasp(self):
        state = {}

        def at(standoff):
            def goal(arm):
                if "deg" not in state:  # a new grasp (possibly by the other arm) picks its own tab-pinch yaw
                    state["deg"], self.T_lid_tcp = self.cell.hinge_deg(), None
                return self.tcp_at(state["deg"], arm, standoff)

            return goal

        return [MoveJ(at(0.10), grip=1.0), Move(at(0.0), speed=0.08), hold(60, 0.0), Branch(self.capture_grasp, list)]

    def push_open(self, lead_deg=8.0, end_deg=95.0):
        """Finish opening a lid that stalled short of vertical (the pinch-swing runs out of reach for some box
        poses): with the fingers closed, press on the lid's inner face near its free edge from the robot side and
        follow it open, compliant, until it is past vertical and rests on its stop."""
        contact = np.array([0.0, -0.7 * self.Ly, 0.0])  # on the inner face, clear of the front skirt
        state = {}

        def tool(deg, arm, standoff):
            T_h = state.setdefault("hinge", self.cell.box_pose() @ pose(np.eye(3), self.hinge))
            T_l = T_h @ pose(rot_x(-np.deg2rad(deg)), [0, 0, 0])
            n_in = -T_l[:3, 2]  # inner-face normal
            z, y = -n_in, T_l[:3, 0]
            R = np.stack([np.cross(y, z), y, z], axis=1)
            if abs(wrap(tcp_yaw(R) - tcp_yaw(arm.R))) > np.pi / 2:
                R = R @ rot_z(np.pi)  # the closed gripper is symmetric under a half turn
            p = (T_l @ np.append(contact, 1.0))[:3] + n_in * (S.PAD_BELOW_TCP + standoff)
            return p, R

        # The lid keeps falling (below vertical gravity closes it) while the hand comes round, so after a quick
        # approach the push tracks the *measured* angle: target = lid surface a few degrees further open.
        approach = MoveJ(lambda arm: tool(self.cell.hinge_deg(), arm, 0.06), grip=0.0, joint_speed=1.5)
        push = Track(lambda arm: tool(min(self.cell.hinge_deg() + lead_deg, S.BOX["open_limit"]), arm, -0.006),
                     lambda arm: self.cell.hinge_deg() > end_deg, speed=0.15, turn=1.0, timeout=1200, compliance=0.5)
        return [approach, push, Move(lambda arm: tool(self.cell.hinge_deg(), arm, 0.10), speed=0.15)]

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
            p = (T_l @ np.append(contact, 1.0))[:3] + n * (S.PAD_BELOW_TCP + standoff)
            return p, R

        approach = MoveJ(lambda arm: tool(self.cell.hinge_deg(), arm, 0.06), grip=0.0, joint_speed=1.5)
        press = Track(lambda arm: tool(max(self.cell.hinge_deg() - lead_deg, -2.0), arm, -0.006),
                      lambda arm: self.cell.hinge_deg() < end_deg, speed=0.10, turn=0.8, timeout=900, compliance=0.8)
        return [approach, press, Move(lambda arm: tool(self.cell.hinge_deg(), arm, 0.08), speed=0.1)]


def script(cell, lid):
    """Phases; each is (name, {side: [Move]}) and the next starts once both arms are idle."""
    half_gap = cell.meta["width"] / 2 + 0.015  # each shoe centred in its half of the box (see scene.size_box_for)
    place1 = Placement(cell, 0, ((0.0, -half_gap), np.pi))  # toe toward -x
    place2 = Placement(cell, 1, ((0.0, half_gap), 0.0))  # toe toward +x: heel-to-toe
    def pick(k, retries=2):
        # Grasp, lift, and if the shoe did not come up with the gripper, open and try again from above.
        moves = [MoveJ(heel_grasp(cell, k, 0.12), grip=1.0), Move(heel_grasp(cell, k), speed=0.10), hold(60, 0.0), Move(up(LIFT), speed=0.2)]
        if retries:
            # A heel-held shoe pivots (toe on the table), so test that the grasped heel came up, not the shoe origin.
            lifted = lambda arm: grasp_point(cell, k)[2] > S.TABLE_Z + cell.meta["grasp_local"][2] + 0.08
            moves.append(Branch(lifted, lambda: [hold(30, 1.0), Move(up(0.10)), home_j(), *pick(k, retries - 1)]))
        return moves

    def held(k):
        # Still between the fingers, or already in the box (a shoe that slipped and landed inside counts as placed;
        # re-picking it from inside the box only made things worse).
        return lambda arm: np.linalg.norm(grasp_point(cell, k) - arm.tcp()[:3, 3]) < HELD_DIST or shoe_in_box(cell, k)

    def place(pl, retries=2):
        # A heel-held shoe creeps out of the pinch if held for long, so it is carried straight in; if it is no longer
        # between the fingers when the hand arrives over the box, fetch it again before aligning and lowering.
        def recheck(r):
            return Branch(held(pl.k), lambda: [hold(30, 1.0), Move(up(0.05)), home_j(), *pick(pl.k), MoveJ(pl(CARRY_DZ)),
                                               *([recheck(r - 1)] if r > 1 else [])])

        in_hand = lambda arm: np.linalg.norm(grasp_point(cell, pl.k) - arm.tcp()[:3, 3]) < HELD_DIST
        dropped_in = lambda arm: shoe_in_box(cell, pl.k) and not in_hand(arm)
        # Let go fully before rising, and rise slowly at first so the finger inside the collar slides out instead
        # of hooking the shoe (a fast lift carried a released shoe back out of the box).
        def unhook():
            # The heel came up with the hand: the finger inside the collar caught it. Set it back down, shift the
            # hand toward the toe so that finger moves off the heel wall into the opening, and lift again.
            def toe_shift(arm):
                x = cell.shoe_pose(pl.k)[:3, 0]
                return arm.p - [0, 0, 0.05] + np.array([x[0], x[1], 0.0]) / max(np.linalg.norm(x[:2]), 1e-6) * 0.015, arm.R
            return [Move(lambda arm: (arm.p - [0, 0, 0.05], arm.R), speed=0.05), hold(30, RELEASE), Move(toe_shift, speed=0.03),
                    Move(up(0.05), speed=0.04)]

        def centre_in_collar(arm):
            # Fully open, the inside finger's centre is ~47.5 mm toe-ward of the TCP; move it to the middle of the
            # collar opening so neither finger touches the shoe on the way up (boot shafts are short, low collars are
            # narrow: at a fixed offset one or the other hooked).
            m = cell.meta
            d = float(np.clip(m.get("wall", 0.02) / 2 + m.get("opening", 0.09) / 2 - 0.0475, 0.0, 0.02))
            x = cell.shoe_pose(pl.k)[:3, 0]
            x = np.array([x[0], x[1], 0.0]) / max(np.linalg.norm(x[:2]), 1e-6)
            return arm.p + x * d, arm.R

        released = lambda arm: grasp_point(cell, pl.k)[2] < pl.rest_tcp_z() + 0.025
        set_down = lambda: [pl.align(), *pl.descend(), hold(90, RELEASE), Move(centre_in_collar, speed=0.02),
                            Move(up(0.05), speed=0.05), Branch(released, unhook), Move(up(0.10), speed=0.15)]
        # After letting go, check the shoe lies flat in the box (heel within 2.5 cm of resting on the floor); a shoe
        # left propped on the end wall or on the other shoe stops the lid. If not, pick it up and place it again, once.
        # Flat = heel near the floor *and* sole roughly horizontal (a shoe resting on its heel with the toe up passed
        # the heel check alone and stood 9 cm above the rim, blocking the lid).
        flat = lambda arm: (shoe_in_box(cell, pl.k) and grasp_point(cell, pl.k)[2] < pl.rest_tcp_z() + 0.015
                            and cell.shoe_pose(pl.k)[2, 2] > 0.95)
        redo = [Branch(flat, lambda: [home_j(), *pick(pl.k, retries=1), *place(pl, retries=0)])] if retries else []
        return [MoveJ(pl(CARRY_DZ)), *([recheck(retries)] if retries else []), Branch(dropped_in, set_down), *redo]

    if args.skill == "pick":
        return [("pick", {"right": [*pick(0, retries=0), hold(480, 0.0)]})]
    if args.skill == "carry":
        return [("carry", {"right": [*pick(0, retries=0), MoveJ(place1(CARRY_DZ)), MoveJ(heel_grasp(cell, 1, 0.25)), hold(240, 0.0)]})]
    return [
        # Left arm pinches the pull tab, swings the lid up past vertical, lets go and clears the box (the creased
        # hinge holds the lid open). Holding it instead put the left hand right above the second shoe's slot.
        # Right arm waits, empty, above shoe 1.
        ("open_lid", {"left": [*lid.grasp(), *lid.swing(LID_OPEN_DEG, timeout=1600), hold(40, 1.0), Move(back_off(0.08)), home_j()],
                      "right": [MoveJ(heel_grasp(cell, 0, 0.12), grip=1.0)]}),
        ("place_shoes", {"right": [*pick(0), *place(place1), home_j(), *pick(1), *place(place2), home_j()]}),
        # Re-grasp the tab where the lid stands and swing it shut; if it does not seat, press it down from above.
        # (Tried with the right arm: it could not swing the lid down from vertical.)
        ("close_lid", {"left": [*lid.grasp(), *lid.swing(LID_CLOSED_DEG, timeout=2400), hold(50, 1.0), Move(back_off(0.08)),
                                Branch(lambda arm: cell.hinge_deg() < 4.0, lid.push_closed), home_j()]}),
    ]


def shoe_in_box(cell, k):
    T = np.linalg.inv(cell.box_pose()) @ cell.shoe_pose(k)
    (lx, ly), h = S.BOX["inner"], S.BOX["height"]
    return abs(T[0, 3]) < lx / 2 and abs(T[1, 3]) < ly / 2 and T[2, 3] < h and T[2, 2] > 0.7


# ---------------------------------------------------------------- learned policy


class PolicyRunner:
    """Closed-loop execution of a policy/train.py checkpoint: every STRIDE recorded frames, predict an action chunk
    from the cameras and the previous command, and execute its first `n_exec` steps as TCP targets (receding horizon)."""

    def __init__(self, path, n_exec=8):
        import torch

        sys.path.insert(0, str(HERE / "policy"))
        import data as pdata
        from models import build

        self.torch, self.pdata, self.n_exec = torch, pdata, n_exec
        ck = torch.load(path, map_location="cuda", weights_only=False)
        cfg = ck["cfg"]
        self.model = build(cfg["kind"], **cfg.get("model", {}), chunk=cfg["chunk"], pretrained=False).cuda().eval()
        self.model.load_state_dict(ck["state"])
        self.lo, self.hi = np.array(ck["stats"]["lo"], np.float32), np.array(ck["stats"]["hi"], np.float32)
        self.no_proprio = cfg.get("proprio") == "none"  # trained without the previous command (see policy/train.py)
        self.period = args.record_every * pdata.STRIDE  # physics steps per policy step
        self.reset()

    def reset(self):
        self.chunk, self.i = None, 0

    def step(self, frames, arms):
        pd, torch = self.pdata, self.torch
        if self.chunk is None or self.i >= self.n_exec:
            import cv2

            imgs = np.stack([cv2.resize(frames[c]["rgb"], (pd.IMG_HW[1], pd.IMG_HW[0]), interpolation=cv2.INTER_AREA) for c in pd.CAMS])
            prev = pd.to_policy_action(np.concatenate([arms["left"].action(), arms["right"].action()])[None])[0]
            prev = 2 * (prev - self.lo) / np.maximum(self.hi - self.lo, 1e-4) - 1
            if self.no_proprio:
                prev = np.zeros_like(prev)
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model.act(x, torch.from_numpy(prev).float().unsqueeze(0).cuda())
            self.chunk = (out[0].float().cpu().numpy() + 1) / 2 * (self.hi - self.lo) + self.lo
            self.i = 0
        a = self.chunk[self.i]
        self.i += 1
        for side, (p, R, g) in zip(("left", "right"), self.pdata.from_policy_action(a)):
            arms[side].p, arms[side].R, arms[side].grip = np.asarray(p, float), np.asarray(R, float), 1.0 if g > 0.5 else 0.0


def task_done(cell):
    return all(shoe_in_box(cell, i) for i in range(2)) and cell.hinge_deg() < 5.0


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
        scene_cfg = make_scene_cfg(assets, args.width, args.height, args.wrist_cams)
        if args.grip_stiffness is not None:
            for r in (scene_cfg.robot_left, scene_cfg.robot_right):
                r.actuators["gripper_drive"].stiffness = args.grip_stiffness
        if args.grip_effort is not None:
            for r in (scene_cfg.robot_left, scene_cfg.robot_right):
                r.actuators["gripper_drive"].joint_effort_limit = args.grip_effort
        scene = instantiate(scene_cfg)
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
        for arm in arms.values():
            arm.cell = cell
        cam_ids = sorted(int(n.split("_")[1]) for n in scene.keys() if n.startswith("camera_"))
        recorder = Recorder(scene, cam_ids, robots)
        task = TASKS["shoe_return"]
        out = Path(args.out)
        stats = []
        policy = PolicyRunner(args.policy) if args.policy else None

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

            def transit_scene(arm, lid=lid):
                other = arms["left" if arm.side == "right" else "right"]
                att = other.attached()
                others = [other.spheres_at(other.joint_pos(), att)]
                if isinstance(other.move, MoveJ):
                    others += [other.spheres_at(q, att) for q in other.move.path]
                return collision.Scene(boxes=lid.obstacles(), table_z=S.TABLE_Z, others=others)

            for arm in arms.values():
                arm.transit_scene = transit_scene
            for _ in range(240):  # settle, arms to home, renderer warm-up
                for arm in arms.values():
                    arm.plan_step()
                step(render=True)
            if ep == args.start_episode:
                for arm in arms.values():
                    assert arm.check_kinematics() < 0.005, "DH model does not match the simulated arm"
                check_sphere_model(arms, robots, recorder)

            split = "val" if args.val_every and ep % args.val_every == 0 else "train"
            writer = EpisodeWriter(out, split, ep, cam_ids, task)
            video = []
            phases, k, dropped, min_gap, phase_name = script(cell, lid), 0, False, np.inf, "setup"
            policy_done = False
            if policy is not None:
                phases, phase_name = [], "policy"
                policy.reset()
            while k < args.max_steps:
                if policy is None and all(a.idle for a in arms.values()):
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
                          f"R {np.rad2deg(robots['right'].data.joint_pos.torch[0, arms['right'].finger_id].item()):.0f}deg "
                          f"heel z {grasp_point(cell, 0)[2]:.3f}/{grasp_point(cell, 1)[2]:.3f}", flush=True)
                if record:
                    frames = recorder.frames()
                    if not args.no_write:
                        robot_boxes, grip_boxes = recorder.label_boxes()
                        writer.add_frame(np.concatenate([arms["left"].action(), arms["right"].action()]), frames, robot_boxes, grip_boxes)
                    if args.video:
                        video.append(frames[0]["rgb"])
                    if policy is not None and k % policy.period == 0:
                        if k > 0 and task_done(cell):
                            policy_done = True
                            break
                        policy.step(frames, arms)
                if any(cell.shoe_pose(i)[2, 3] < S.TABLE_Z - 0.05 for i in range(2)):
                    dropped = True
                    break
                k += 1
            finished = policy_done if policy is not None else (not phases and all(a.idle for a in arms.values()))
            for _ in range(60):
                step(render=False)
            inside = [shoe_in_box(cell, i) for i in range(2)]
            hinge = cell.hinge_deg()
            if args.debug:
                rim = cell.box_pose()[2, 3] + S.BOX["height"]
                tops = []
                for i in range(2):
                    T = cell.shoe_pose(i)
                    corners = np.array([[sx * cell.meta["length"] / 2, sy * cell.meta["width"] / 2, sz * cell.meta["height"], 1.0]
                                        for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
                    tops.append(float((T @ corners.T)[2].max()))
                print(f"[end] shoe tops above rim {np.round(np.array(tops) - rim, 3)} m; left arm q "
                      f"{np.round(np.rad2deg(arms['left'].joint_pos())).astype(int)}", flush=True)
            ok = finished and all(inside) and hinge < 5.0 and not dropped and min_gap > ARM_CLEARANCE
            reason = None if ok else ("shoe_dropped" if dropped else "timeout" if not finished else "arm_collision" if min_gap <= ARM_CLEARANCE else "shoe_outside_box" if not all(inside) else "lid_open")
            wall = time.time() - t0
            print(f"[ep {ep}] phase={phase_name} in_box={inside} hinge={hinge:.1f}deg gap={min_gap:.2f}m -> {'ok' if ok else 'FAILED: ' + reason} ({writer.T} frames, {wall:.0f}s)", flush=True)
            stats.append(dict(episode=ep, success=ok, failure=reason, phase=phase_name, hinge_deg=hinge, frames=writer.T, wall_s=wall, shoe=assets["shoe_meta"]["name"], seed=args.seed, expert=args.policy or "scripted"))
            if args.no_write:
                writer.discard()
            elif ok or args.keep_failures:
                writer.close(dict(success=ok, failure_label=reason or "", failure_phase="" if ok else phase_name, hinge_deg=hinge, seed=args.seed, schema_version=1,
                                  shoe=assets["shoe_meta"]["name"], expert=args.policy or "scripted"))
            else:
                writer.discard()
            if args.video and video:
                save_video(Path(args.video) / f"episode_{ep}.mp4", video)
        n_ok = sum(s["success"] for s in stats)
        print(f"success {n_ok}/{len(stats)}")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"run_stats_seed{args.seed}_ep{args.start_episode}.json").write_text(json.dumps(stats, indent=1))


def check_sphere_model(arms, robots, recorder):
    """Each simulated link's bounding-box centre should lie inside (or within a few cm of) the collision spheres."""
    for side, arm in arms.items():
        c, r = arm.spheres_at(arm.joint_pos())
        bp = robots[side].data.body_pose_w.torch[0].cpu().numpy()
        worst = []
        for i, _, cen, half in recorder.boxes[side]:
            x = (pose7(bp[i, :3], bp[i, 3:]) @ np.append(cen, 1.0))[:3]
            worst.append((float(np.min(np.linalg.norm(c - x, axis=1) - r)), robots[side].body_names[i]))
        worst.sort(reverse=True)
        print(f"[check {side}] link-centre distance outside the sphere model, worst: "
              + ", ".join(f"{n} {d * 100:.1f}cm" for d, n in worst[:4]), flush=True)


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
    pads = S.friction_material(stage, "/World/Materials/finger_pads", 1.5)  # silicone fingertip pads
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
