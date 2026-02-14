# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg  # keep if you plan joints later

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # robot prefab
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp


@configclass
class TwistLidSceneCfg(InteractiveSceneCfg):
    """Scene with two Frankas and two cylinders (bottle + lid)."""

    # Robots (filled in __post_init__)
    robot_bottle: ArticulationCfg = MISSING
    robot_lid: ArticulationCfg = MISSING

    # Sensors (filled in __post_init__)
    ee_frame_bottle: FrameTransformerCfg = MISSING
    ee_frame_lid: FrameTransformerCfg = MISSING

    # Objects
    bottle: RigidObjectCfg | DeformableObjectCfg = MISSING
    lid: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Tables - make prim paths unique
    table_bottle: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableBottle",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    table_lid: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLid",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 1.5], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground + light
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for both robots (pose targets at the gripper)."""

    rlid_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_lid",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

    rbottle_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_bottle",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Separate action specs for each Franka."""

    # Arm joints for each robot
    arm_bottle = mdp.JointPositionActionCfg(
        asset_name="robot_bottle", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    arm_lid = mdp.JointPositionActionCfg(
        asset_name="robot_lid", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )

    # Grippers for each robot
    gripper_bottle = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_bottle",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.09},
        close_command_expr={"panda_finger_.*": 0.0},
    )
    gripper_lid = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_lid",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.09},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications (kept minimal; extend as needed)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)  # typically uses SceneEntityCfg("object")
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "rbottle_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle", body_names="Bottle"),
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.11}, weight=15.0)
    bonus_lift_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.16}, weight=1.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.11, "command_name": "rbottle_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.11, "command_name": "rbottle_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot_bottle")})
    upright_penalty = RewTerm(func=mdp.object_uprightness, weight=-1e-4)

    object_lin_vel_penalty = RewTerm(
        func=mdp.root_lin_vel_l2,
        params={"object_cfg": SceneEntityCfg("bottle")},
        weight=-1e-4,
    )
    object_ang_vel_penalty = RewTerm(
        func=mdp.root_ang_vel_l2,
        params={"object_cfg": SceneEntityCfg("bottle")},
        weight=-1e-4,
    )
    bonus_lift_penality = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.25}, weight=-1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("bottle")},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1, "num_steps": 10000})
    joint_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.5, "num_steps": 10000})
    upright_penalty = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "upright_penalty", "weight": -1, "num_steps": 11000})


@configclass
class TwistLidEnvCfg(ManagerBasedRLEnvCfg):
    scene: TwistLidSceneCfg = TwistLidSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # Two robots with unique prim paths
        self.scene.robot_bottle = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotBottle")
        self.scene.robot_lid = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotLid")
        self.scene.robot = self.scene.robot_bottle
        # Two cylinders: bottle (green), lid (blue)
        self.scene.bottle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bottle",
            spawn=sim_utils.CylinderCfg(
                radius=0.03, height=0.2,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), roughness=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.0, 0.1], rot=[1.0, 0.0, 0.0, 0.0]
            ),
        )

        self.scene.lid = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Lid",
            spawn=sim_utils.CylinderCfg(
                radius=0.03, height=0.05,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0), roughness=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.0, 1.6], rot=[1.0, 0.0, 0.0, 0.0]
            ),
        )

        # End-effector frame transformers for both robots
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame_bottle = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RobotBottle/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/RobotBottle/panda_hand",
                    name="ee_bottle",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.08]),
                ),
            ],
        )

        self.scene.ee_frame_lid = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RobotLid/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/RobotLid/panda_hand",
                    name="ee_lid",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.08]),
                ),
            ],
        )

        # General sim settings
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # PhysX tuning
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625