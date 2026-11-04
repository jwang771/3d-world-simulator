"""Isaac Lab (3.0) scene for T1, bimanual shoe-box repacking. Import after the simulation app is launched.

Our own assets (table, shoebox articulation, shoes) carry their physics in USD written by scene.py; Isaac Lab
only spawns them. The robots and cameras use Isaac Lab configs. Quaternions here are (x, y, z, w).
"""
import re

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import clone, configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

import scene as S
from geometry import mat_to_quat, pose

UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def xyzw(q_wxyz):
    w, x, y, z = q_wxyz
    return (float(x), float(y), float(z), float(w))


def yaw_quat(yaw):
    return (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))


def ur_2f85_cfg(side, ur_limits=True, contact_offsets=True, usd=None):
    """UR + Robotiq 2F-85 from the UR asset's Gripper variant (scene.ROBOT_MODEL), based on Isaac Lab's
    UR10e + 2F-85 config."""
    cfg = clone(UR10e_ROBOTIQ_2F_85_CFG)
    cfg.prim_path = "{ENV_REGEX_NS}/Robot_" + side
    usd = usd or S.ROBOT_MODEL
    cfg.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/{usd}/{usd}.usd"
    cfg.spawn.semantic_tags = [("class", "robot")]
    # The 2F-85 inside the Gripper variant carries a second ArticulationRootAPI; name the UR5e's own root so
    # Isaac Lab does not have to pick between them.
    cfg.articulation_root_prim_path = "/root_joint"
    pos, yaw = S.ROBOT_BASES[side]
    cfg.init_state.pos = tuple(float(v) for v in pos)
    cfg.init_state.rot = yaw_quat(yaw)
    for name, q in zip(UR_JOINTS, S.UR_HOME_Q):
        cfg.init_state.joint_pos[name] = float(q)
    if ur_limits:  # datasheet torque and speed per joint; an uncapped PD drive launches whatever it hits
        for group, joints in (("shoulder", ["shoulder_.*"]), ("elbow", ["elbow_joint"]), ("wrist", ["wrist_.*"])):
            act = cfg.actuators[group]
            act.joint_names_expr = joints
            names = [n for n in S.UR_LIMITS if re.fullmatch(joints[0], n)]
            act.joint_effort_limit = {n: S.UR_LIMITS[n][0] for n in names}
            act.joint_velocity_limit = {n: S.UR_LIMITS[n][1] for n in names}
    # 2F-85 tuning from Isaac Lab's own UR10e + 2F-85 gear-assembly task (the closed-loop finger linkage goes
    # unstable with the soft inner-finger drive of the base config): finger_joint 40/1, inner fingers 10/0.05
    # to keep the pads parallel, 5 mm contact offset.
    # Effort raised from Isaac Lab's 10 to 30: at 10 a lifted shoe's tapered heel wedged the fingers open (the real
    # 2F-85 grips with 20-235 N).
    cfg.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"], joint_effort_limit=30.0, joint_velocity_limit=1.0, stiffness=120.0, damping=3.0
    )
    cfg.actuators["gripper_finger"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"], joint_effort_limit=10.0, joint_velocity_limit=10.0, stiffness=10.0, damping=0.05
    )
    drive_both_knuckles(cfg)
    if contact_offsets:
        cfg.spawn.collision_props = PhysxCollisionCfg(contact_offset=0.005, rest_offset=0.0)
    return cfg


def drive_both_knuckles(cfg):
    """Drive right_outer_knuckle_joint with the same PD as finger_joint instead of leaving it to the mimic
    constraint alone. The two knuckles turn in mirrored directions, so a squeeze carried by the mimic constraint
    reacts on the gripper base with a net torque that twists the wrist; two equal drives react and cancel."""
    d = cfg.actuators["gripper_drive"]
    cfg.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint", "right_outer_knuckle_joint"], joint_effort_limit=d.joint_effort_limit,
        joint_velocity_limit=d.joint_velocity_limit, stiffness=d.stiffness, damping=d.damping,
    )
    p = cfg.actuators.get("gripper_passive")
    if p is not None:
        p.joint_names_expr = [e for e in p.joint_names_expr if e != "right_outer_knuckle_joint"]
    return cfg


def ros_offset(T):
    return CameraCfg.OffsetCfg(pos=tuple(float(v) for v in T[:3, 3]), rot=xyzw(mat_to_quat(T[:3, :3])), convention="ros")


def camera_cfg(prim_path, T_parent_cam, focal_mm, width, height):
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal_mm, horizontal_aperture=S.H_APERTURE_MM, clipping_range=(0.02, 10.0)),
        offset=ros_offset(T_parent_cam),
    )


_OBJECT_PHYSX = [PhysxRigidBodyCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0)]


@configclass
class RepackSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(), init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.8)))
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=350.0, color=(0.95, 0.95, 1.0)))
    key = AssetBaseCfg(prim_path="/World/key", spawn=sim_utils.DistantLightCfg(intensity=500.0, angle=1.0))
    robot_left: ArticulationCfg = ur_2f85_cfg("left")
    robot_right: ArticulationCfg = ur_2f85_cfg("right")


def make_scene_cfg(assets, width, height, wrist_cams=True, num_envs=1):
    """assets: dict with table, shoebox and shoe_0/shoe_1 USD paths (see run_sim.prepare_assets)."""
    cfg = RepackSceneCfg(num_envs=num_envs, env_spacing=3.0)
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table", spawn=sim_utils.UsdFileCfg(usd_path=assets["table"]), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, S.TABLE_Z))
    )
    cfg.shoebox = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Shoebox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=assets["shoebox"],
            fix_root_link=True,  # the box sits in a nest/jig on the table
            rigid_props=_OBJECT_PHYSX,
            articulation_props=[PhysxArticulationCfg(enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(*S.BOX_NOMINAL["center"], S.TABLE_Z), joint_pos={"lid_hinge": 0.0}),
        # Cardboard hinge: enough damping that a released lid settles rather than slams (lid gravity torque <= 0.2 Nm).
        # A paperboard hinge is stiff: a lid nudged near vertical creeps rather than flops shut. Heavy damping models
        # that (with 0.25 an open lid brushed by a carried shoe fell closed); gravity near vertical is ~0.02 Nm.
        actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=["lid_hinge"], stiffness=0.0, damping=2.0)},
    )
    for k in range(2):
        setattr(
            cfg,
            f"shoe_{k}",
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shoe_" + str(k),
                spawn=sim_utils.UsdFileCfg(usd_path=assets[f"shoe_{k}"], rigid_props=_OBJECT_PHYSX),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30 + 0.12 * k, -0.25, S.TABLE_Z + 0.005)),
            ),
        )
    for cid, (eye, target) in S.STATIC_CAMS.items():
        setattr(cfg, f"camera_{cid}", camera_cfg("{ENV_REGEX_NS}/Camera_" + str(cid), S.static_cam_pose(eye, target), S.STATIC_FOCAL_MM, width, height))
    if wrist_cams:
        for side, cid in S.WRIST_CAMS.items():
            T = S.T_W3_GB @ S.T_GB_WRISTCAM
            setattr(cfg, f"camera_{cid}", camera_cfg("{ENV_REGEX_NS}/Robot_" + side + "/wrist_3_link/wrist_cam", T, S.WRIST_FOCAL_MM, width, height))
    return cfg
