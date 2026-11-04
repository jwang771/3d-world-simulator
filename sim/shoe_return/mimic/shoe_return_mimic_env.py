"""Isaac Lab Mimic environment for the shoe-return T1 task (bimanual: left arm handles the lid, right arm the shoes).

Mimic generates new demonstrations from a few source demonstrations by splitting each into object-centric
subtasks and replaying every segment transformed to the new object poses:

    left arm : open_lid (relative to the lid)                          | close_lid (lid)
    right arm: pick shoe 0 (shoe_0) | place shoe 0 (box) | pick shoe 1 (shoe_1) | place shoe 1 (box)

with ordering constraints: shoe 0 is placed only after the lid is open, and the lid is closed only after shoe 1 is
placed. Source demonstrations come from the scripted expert (`run_sim.py --mimic-record`), recorded with the
datagen annotations already attached (no separate annotation pass).

Action (16): per arm [TCP position (3), TCP quaternion (x, y, z, w) (4)], then per arm a gripper command:
0 closed, 1 open to the pinch width, 2 fully open. Poses are in the env frame. The action term solves IK with the
calibrated UR model in ur_kin.py, as the expert does.

Everything here imports pxr (via scene.py) lazily, so the module can be imported before Kit starts; the scene
itself is built after launch (see `configure`).
"""
import json
import os
import sys
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]  # sim/shoe_return
for p in (HERE, HERE / "envs", HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gymnasium as gym  # noqa: E402

import isaaclab.utils.math as PoseUtils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv  # noqa: E402
from isaaclab.envs import mdp  # noqa: E402
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig, SubTaskConstraintConfig, SubTaskConstraintType  # noqa: E402
from isaaclab.managers import ActionTerm, ActionTermCfg  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import RecorderTerm, RecorderTermCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import ur_kin  # noqa: E402
from geometry import pose, rot_z  # noqa: E402

TASK_ID = "Isaac-ShoeReturn-T1-Mimic-v0"
PHYSICS_HZ = 120
ARMS = ("left", "right")
UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
GRIP_SIGNS = dict(finger_joint=1, right_outer_knuckle_joint=1, left_inner_finger_joint=-1, right_inner_finger_joint=1,
                  left_inner_finger_knuckle_joint=-1, right_inner_finger_knuckle_joint=-1)

# Run-time configuration (set by the launching script before the env cfg is built).
_CONF = dict(shoe=os.environ.get("SHOE_MIMIC_SHOE", "SUPERSTAR_CLR"), cache=os.environ.get("SHOE_MIMIC_CACHE", "/tmp/shoe_mimic"),
             cameras=os.environ.get("SHOE_MIMIC_CAMERAS", "0") == "1", width=320, height=240, grasp_depth=0.015)


def configure(**kw):
    _CONF.update(kw)


def S():
    import scene  # pxr: only after Kit is up

    return scene


# ---------------------------------------------------------------- assets and geometry


def prepare_assets():
    """Table, the shoe pair and a box sized to it (as run_sim.prepare_assets for a scanned pair)."""
    sc = S()
    cache = Path(_CONF["cache"]) / _CONF["shoe"]
    cache.mkdir(parents=True, exist_ok=True)
    lib = HERE / sc.SHOE_LIBRARY / _CONF["shoe"]
    m = json.loads((lib / "meta.json").read_text())
    meta = dict(name=_CONF["shoe"], grasp_local=(np.asarray(m["grasp"]) - [0, 0, _CONF["grasp_depth"]]).tolist(), closing_axis="x",
                open=sc.finger_angle_for_gap(m["pinch_open_width"]), length=m["length"], width=m["width"], height=m["height"],
                mass=m["mass"], wall=m.get("wall", 0.02), opening=m.get("opening", 0.09))
    sc.size_box_for(meta["length"], meta["width"], meta["height"])
    out = dict(table=str(cache / "table.usda"), shoebox=str(cache / "shoebox.usda"), shoe_0=str(lib / "left.usd"), shoe_1=str(lib / "right.usd"),
               shoe_meta=meta)
    sc.write_table_usd(out["table"])
    sc.write_shoebox_usd(out["shoebox"], np.random.default_rng(0))
    return out


def tcp_offset():
    sc = S()
    return sc.T_W3_GB @ sc.T_GB_TCP


def pose7_to_mat(p, q_xyzw):
    return PoseUtils.make_pose(p, PoseUtils.matrix_from_quat(q_xyzw))


def body_pose(asset, body_idx):
    d = asset.data.body_pose_w.torch[:, body_idx]
    return pose7_to_mat(d[:, :3], d[:, 3:7])


def root_pose(asset):
    d = asset.data.root_pose_w.torch
    return pose7_to_mat(d[:, :3], d[:, 3:7])


def env_relative(env, T):
    T = T.clone()
    T[:, :3, 3] -= env.scene.env_origins
    return T


# ---------------------------------------------------------------- action term: TCP pose + gripper per arm


class UrTcpPoseAction(ActionTerm):
    """Absolute TCP pose for one UR10e + 2F-85, tracked with per-step IK on the calibrated UR model (rate-limited,
    seeded from the last command, like the scripted expert), plus a gripper command."""

    cfg: "UrTcpPoseActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.joint_ids = self.robot.find_joints(UR_JOINTS, preserve_order=True)[0]
        names = [n for n in GRIP_SIGNS if n in self.robot.joint_names]
        self.grip_ids = self.robot.find_joints(names, preserve_order=True)[0]
        self.grip_signs = torch.tensor([[float(GRIP_SIGNS[n]) for n in names]], device=self.device)
        self.w3 = self.robot.find_bodies(["wrist_3_link"])[0][0]
        self._raw = torch.zeros(self.num_envs, 8, device=self.device)
        self.kin = None
        self.q_prev = [None] * self.num_envs

    @property
    def action_dim(self):
        return 8

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._raw

    def _root_np(self, i):
        return root_pose(self.robot)[i].cpu().numpy().astype(float)

    def calibrate(self):
        """Flange offset fitted to the simulated robot at its current pose; the DH base is the URDF base turned half
        round about z (measured: the model then matches the simulator to 0.00 mm, see Arm.check_kinematics)."""
        q = self.robot.data.joint_pos.torch[0, self.joint_ids].cpu().numpy().astype(float)
        T = np.linalg.inv(self._root_np(0)) @ body_pose(self.robot, self.w3)[0].cpu().numpy().astype(float)
        A = pose(rot_z(np.pi), [0, 0, 0])
        self.kin = (A, np.linalg.inv(A @ ur_kin.fk(q, S().ROBOT_MODEL)) @ T)

    def process_actions(self, actions):
        self._raw[:] = actions

    def reset(self, env_ids=None):
        self.q_prev = [None] * self.num_envs

    def apply_actions(self):
        if self.kin is None:
            self.calibrate()
        A, B = self.kin
        T_w3_tcp = tcp_offset()
        origins = self._env.scene.env_origins.cpu().numpy()
        act = self._raw.cpu().numpy().astype(float)
        q_now = self.robot.data.joint_pos.torch[:, self.joint_ids].cpu().numpy().astype(float)
        q_cmd = np.zeros_like(q_now)
        for i in range(self.num_envs):
            p = act[i, :3] + origins[i]
            R = PoseUtils.matrix_from_quat(torch.tensor(act[i, 3:7])[None]).numpy()[0]
            if not np.all(np.isfinite(R)):
                R = np.eye(3)
            T_dh = np.linalg.inv(A) @ np.linalg.inv(self._root_np(i)) @ pose(R, p) @ np.linalg.inv(T_w3_tcp) @ np.linalg.inv(B)
            seed = self.q_prev[i] if self.q_prev[i] is not None else q_now[i]
            q, _ = ur_kin.ik(T_dh, seed, S().ROBOT_MODEL, iters=12, tol=1e-5, damping=1e-3)
            q = ur_kin.nearest_equivalent(q, seed)
            q = seed + np.clip(q - seed, -0.025, 0.025)
            q = np.clip(q, -(ur_kin.LIMITS - 0.02), ur_kin.LIMITS - 0.02)
            self.q_prev[i] = q
            q_cmd[i] = q
        self.robot.set_joint_position_target_index(target=torch.as_tensor(q_cmd, dtype=torch.float32, device=self.device),
                                                   joint_ids=self.joint_ids)
        g = act[:, 7]
        width = np.where(g > 1.5, S().GRIP_OPEN, np.where(g > 0.5, self.cfg.pinch_open, S().GRIP_CLOSED))
        targets = self.grip_signs * torch.as_tensor(width, dtype=torch.float32, device=self.device)[:, None]
        self.robot.set_joint_position_target_index(target=targets, joint_ids=self.grip_ids)


@configclass
class UrTcpPoseActionCfg(ActionTermCfg):
    class_type: type = UrTcpPoseAction
    asset_name: str = MISSING
    pinch_open: float = 0.0  # finger_joint angle for "open to the pinch width"


# ---------------------------------------------------------------- task state helpers (env frame)


def tcp_pose(env, side):
    r = env.scene[f"robot_{side}"]
    w3 = r.find_bodies(["wrist_3_link"])[0][0]
    T = body_pose(r, w3) @ torch.as_tensor(tcp_offset(), dtype=torch.float32, device=env.device)
    return env_relative(env, T)


def hinge_deg(env):
    return -torch.rad2deg(env.scene["shoebox"].data.joint_pos.torch[:, 0])


def lid_pose(env):
    box = env.scene["shoebox"]
    return env_relative(env, body_pose(box, box.body_names.index("lid")))


def box_pose(env):
    return env_relative(env, root_pose(env.scene["shoebox"]))


def shoe_pose(env, k):
    return env_relative(env, root_pose(env.scene[f"shoe_{k}"]))


def grasp_point(env, k):
    g = np.asarray(env.cfg.shoe_meta["grasp_local"], float)
    if k == 1:
        g = g * [1, -1, 1]
    T = shoe_pose(env, k)
    return T[:, :3, :3] @ torch.as_tensor(g, dtype=torch.float32, device=env.device) + T[:, :3, 3]


def shoe_in_box(env, k):
    sc = S()
    T = torch.linalg.inv(box_pose(env)) @ shoe_pose(env, k)
    (lx, ly), h = sc.BOX["inner"], sc.BOX["height"]
    return (T[:, 0, 3].abs() < lx / 2) & (T[:, 1, 3].abs() < ly / 2) & (T[:, 2, 3] < h) & (T[:, 2, 2] > 0.7)


def finger_deg(env, side):
    r = env.scene[f"robot_{side}"]
    return torch.rad2deg(r.data.joint_pos.torch[:, r.find_joints(["finger_joint"])[0][0]])


def task_success(env):
    return shoe_in_box(env, 0) & shoe_in_box(env, 1) & (hinge_deg(env) < 5.0)


# observation terms
def obs_tcp_pos(env, side):
    return tcp_pose(env, side)[:, :3, 3]


def obs_tcp_quat(env, side):
    return PoseUtils.quat_from_matrix(tcp_pose(env, side)[:, :3, :3])


def obs_joint_pos(env, side):
    return env.scene[f"robot_{side}"].data.joint_pos.torch


def obs_hinge(env):
    return hinge_deg(env)[:, None]


def obs_shoe_pose(env, k):
    T = shoe_pose(env, k)
    return torch.cat([T[:, :3, 3], PoseUtils.quat_from_matrix(T[:, :3, :3])], -1)


# ---------------------------------------------------------------- events


def prestartup_fixups(env, env_ids):
    """Before physics starts (as run_sim.fix_robot_prims / fix_shoe_prims): strip the 2F-85's nested articulation
    roots, label gripper links, silicone pads, rubber shoes with the 'shoe' class."""
    from pxr import Usd, UsdPhysics

    sc = S()
    stage = env.sim.stage
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.GetName() == sc.GRIPPER_KEY and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            for api in ("PhysxArticulationAPI", "NewtonArticulationRootAPI"):
                prim.RemoveAppliedSchema(api)
    pads = sc.friction_material(stage, "/World/Materials/finger_pads", 1.5)
    rubber = sc.friction_material(stage, "/World/Materials/shoe_rubber", 0.9)
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if sc.GRIPPER_KEY in path and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            sc.label(prim, "gripper")
            if "inner_finger" in prim.GetName():
                sc.bind_physics_material(prim, pads)
        if "/Shoe_" in path:
            if prim.GetName().startswith("Shoe_"):
                sc.label(prim, "shoe")
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                sc.bind_physics_material(prim, rubber)


def reset_cell(env, env_ids):
    """Random box pose (+-10 cm, +-20 deg) and shoe poses in their region; arms to their default pose."""
    sc = S()
    rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = range(env.num_envs)
    for i in env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids):
        o = env.scene.env_origins[i].cpu().numpy()
        T_box = sc.box_pose_from_nominal(rng)
        q = PoseUtils.quat_from_matrix(torch.tensor(T_box[:3, :3], dtype=torch.float32)[None])[0].tolist()
        box = env.scene["shoebox"]
        ids = torch.tensor([i], device=env.device)
        box.write_root_pose_to_sim_index(root_pose=torch.tensor([[*(T_box[:3, 3] + o), *q]], dtype=torch.float32, device=env.device), env_ids=ids)
        box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
        box.write_joint_position_to_sim_index(position=torch.zeros(1, 1, device=env.device), env_ids=ids)
        box.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, 1, device=env.device), env_ids=ids)

        class _Dims:
            length, width = env.cfg.shoe_meta["length"], env.cfg.shoe_meta["width"]

        for k, (xy, yaw) in enumerate(sc.sample_shoe_poses(rng, _Dims)):
            p = [xy[0] + o[0], xy[1] + o[1], sc.TABLE_Z + 0.003 + o[2]]
            shoe = env.scene[f"shoe_{k}"]
            shoe.write_root_pose_to_sim_index(root_pose=torch.tensor([[*p, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]], dtype=torch.float32,
                                                                     device=env.device), env_ids=ids)
            shoe.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
        for side in ARMS:
            r = env.scene[f"robot_{side}"]
            r.write_joint_position_to_sim_index(position=r.data.default_joint_pos.torch[ids].clone(), env_ids=ids)
            r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(r.data.default_joint_vel.torch[ids]), env_ids=ids)
            r.set_joint_position_target_index(target=r.data.default_joint_pos.torch[ids].clone(), env_ids=ids)


# ---------------------------------------------------------------- recorder terms (datagen annotations while recording)


class DatagenInfoRecorder(RecorderTerm):
    def record_pre_step(self):
        eef = {s: self._env.get_robot_eef_pose(s) for s in self._env.cfg.subtask_configs}
        return "obs/datagen_info", {"object_pose": self._env.get_object_poses(), "eef_pose": eef,
                                    "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action)}


@configclass
class DatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type = DatagenInfoRecorder


class SubtaskTermRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class SubtaskTermRecorderCfg(RecorderTermCfg):
    class_type: type = SubtaskTermRecorder


@configclass
class ShoeMimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = DatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = SubtaskTermRecorderCfg()


# ---------------------------------------------------------------- env cfg


@configclass
class ActionsCfg:
    left = UrTcpPoseActionCfg(asset_name="robot_left")
    right = UrTcpPoseActionCfg(asset_name="robot_right")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_eef_pos = ObsTerm(func=obs_tcp_pos, params={"side": "left"})
        left_eef_quat = ObsTerm(func=obs_tcp_quat, params={"side": "left"})
        right_eef_pos = ObsTerm(func=obs_tcp_pos, params={"side": "right"})
        right_eef_quat = ObsTerm(func=obs_tcp_quat, params={"side": "right"})
        left_joint_pos = ObsTerm(func=obs_joint_pos, params={"side": "left"})
        right_joint_pos = ObsTerm(func=obs_joint_pos, params={"side": "right"})
        shoe_0_pose = ObsTerm(func=obs_shoe_pose, params={"k": 0})
        shoe_1_pose = ObsTerm(func=obs_shoe_pose, params={"k": 1})
        lid_hinge = ObsTerm(func=obs_hinge)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    fixups = EventTerm(func=prestartup_fixups, mode="prestartup")
    reset = EventTerm(func=reset_cell, mode="reset")


def _success(env):
    return task_success(env)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success)


@configclass
class RewardsCfg:
    pass


@configclass
class ShoeReturnMimicEnvCfg(ManagerBasedRLEnvCfg, MimicEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    shoe_meta: dict = None
    mimic_recorder_config: RecorderManagerBaseCfg = ShoeMimicRecorderManagerCfg()

    def __post_init__(self):
        from repack_scene import make_scene_cfg

        assets = prepare_assets()
        self.shoe_meta = assets["shoe_meta"]
        self.scene = make_scene_cfg(assets, _CONF["width"], _CONF["height"], _CONF["cameras"])
        self.scene.replicate_physics = False  # the prestartup USD fix-ups edit prims; Isaac Lab requires this
        if not _CONF["cameras"]:
            self.scene.camera_0 = None
            self.scene.camera_1 = None
        self.decimation = 1
        self.sim.dt = 1 / PHYSICS_HZ
        self.sim.render_interval = 8
        self.episode_length_s = 24000 / PHYSICS_HZ
        self.actions.right.pinch_open = float(self.shoe_meta["open"])
        self.actions.left.pinch_open = 0.0

        d = self.datagen_config
        d.name = f"shoe_return_t1_{_CONF['shoe']}"
        d.generation_guarantee = True
        d.generation_keep_failed = False
        d.generation_num_trials = 10
        d.generation_select_src_per_subtask = False
        d.generation_select_src_per_arm = False
        d.generation_relative = False
        d.generation_joint_pos = False
        d.generation_transform_first_robot_pose = False
        d.generation_interpolate_from_last_target_pose = True
        d.max_num_failures = 50
        d.seed = 1

        def sub(obj, signal, interp=60):
            return SubTaskConfig(object_ref=obj, subtask_term_signal=signal, subtask_term_offset_range=(0, 0),
                                 selection_strategy="nearest_neighbor_object", selection_strategy_kwargs={"nn_k": 3},
                                 action_noise=0.0, num_interpolation_steps=interp, num_fixed_steps=0,
                                 apply_noise_during_interpolation=False)

        self.subtask_configs = {
            "left": [sub("lid", "lid_opened", 0), sub("lid", None)],
            "right": [sub("shoe_0", "shoe_0_grasped", 0), sub("box", "shoe_0_placed"), sub("shoe_1", "shoe_1_grasped"), sub("box", None)],
        }
        self.task_constraint_configs = [
            # shoe 0 is set into the box only after the lid is open
            SubTaskConstraintConfig(eef_subtask_constraint_tuple=[("left", 0), ("right", 1)], constraint_type=SubTaskConstraintType.SEQUENTIAL,
                                    sequential_min_time_diff=-1),
            # the lid is closed only after shoe 1 is in
            SubTaskConstraintConfig(eef_subtask_constraint_tuple=[("right", 3), ("left", 1)], constraint_type=SubTaskConstraintType.SEQUENTIAL,
                                    sequential_min_time_diff=-1),
        ]


# ---------------------------------------------------------------- env


class ShoeReturnMimicEnv(ManagerBasedRLMimicEnv):
    def get_robot_eef_pose(self, eef_name, env_ids=None):
        T = tcp_pose(self, eef_name)
        return T if env_ids is None else T[env_ids]

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        parts = []
        for side in ARMS:
            p, R = PoseUtils.unmake_pose(target_eef_pose_dict[side])
            q = PoseUtils.quat_unique(PoseUtils.quat_from_matrix(R))
            if action_noise_dict is not None and action_noise_dict.get(side):
                p = p + action_noise_dict[side] * torch.randn_like(p)
            parts += [p, q]
        return torch.cat([*parts[0:2], gripper_action_dict["left"], *parts[2:4], gripper_action_dict["right"]], dim=0)

    def action_to_target_eef_pose(self, action):
        out = {}
        for i, side in enumerate(ARMS):
            a = action[:, 8 * i:8 * i + 7]
            out[side] = PoseUtils.make_pose(a[:, :3], PoseUtils.matrix_from_quat(a[:, 3:7]))
        return out

    def actions_to_gripper_actions(self, actions):
        return {"left": actions[:, 7:8], "right": actions[:, 15:16]}

    def get_object_poses(self, env_ids=None):
        poses = {"shoe_0": shoe_pose(self, 0), "shoe_1": shoe_pose(self, 1), "box": box_pose(self), "lid": lid_pose(self)}
        return poses if env_ids is None else {k: v[env_ids] for k, v in poses.items()}

    def get_subtask_term_signals(self, env_ids=None):
        sc = S()
        floor = box_pose(self)[:, 2, 3] + sc.BOX["wall"]
        lifted = lambda k: grasp_point(self, k)[:, 2] > sc.TABLE_Z + self.cfg.shoe_meta["grasp_local"][2] + 0.08
        right_closed = finger_deg(self, "right") > 30.0
        right_open = finger_deg(self, "right") < 20.0
        sig = {
            # lid past vertical and the left hand let go of the tab
            "lid_opened": (hinge_deg(self) > 85.0) & (finger_deg(self, "left") < 10.0),
            "shoe_0_grasped": lifted(0) & right_closed,
            "shoe_0_placed": shoe_in_box(self, 0) & right_open & (tcp_pose(self, "right")[:, 2, 3] > floor + 0.15),
            "shoe_1_grasped": lifted(1) & right_closed,
        }
        sig = {k: v.float() for k, v in sig.items()}
        return sig if env_ids is None else {k: v[env_ids] for k, v in sig.items()}


def register():
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point=f"{__name__}:ShoeReturnMimicEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{__name__}:ShoeReturnMimicEnvCfg"})
    return []
