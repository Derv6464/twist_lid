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
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp
from . import utils



@configclass
class TwistLidSceneCfg(InteractiveSceneCfg):
    """Scene with two Frankas and two cylinders (bottle + lid)."""

    robot_bottle: ArticulationCfg = MISSING
    robot_lid: ArticulationCfg = MISSING

    ee_frame_bottle: FrameTransformerCfg = MISSING
    ee_frame_lid: FrameTransformerCfg = MISSING

    bottle: RigidObjectCfg | DeformableObjectCfg = MISSING
    lid: RigidObjectCfg | DeformableObjectCfg = MISSING

    table_bottle: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableBottle",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    table_middle: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableMiddle",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.75, 0.0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    table_lid: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLid",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 1.5, 0.0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.8]),
        spawn=GroundPlaneCfg(),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for robots (pose targets at the gripper)."""

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

    rbottle_pose =  mdp.UniformPoseCommandCfg(
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
    arm_bottle = mdp.JointPositionActionCfg(
        asset_name="robot_bottle", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    arm_lid = mdp.JointPositionActionCfg(
        asset_name="robot_lid", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )

    gripper_bottle = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_bottle",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.08},
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

    @configclass
    class PolicyCfg(ObsGroup):
        b_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg" :SceneEntityCfg("robot_bottle")})
        l_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg" :SceneEntityCfg("robot_lid") })

        b_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot_bottle")})
        l_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot_lid")})
        
        b_object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"robot_cfg":SceneEntityCfg("robot_bottle"), "object_cfg":SceneEntityCfg("bottle")})
        l_object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"robot_cfg":SceneEntityCfg("robot_lid"), "object_cfg":SceneEntityCfg("lid")})

        b_target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "rbottle_pose"})
        l_target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "rlid_pose"})
        
        actions = ObsTerm(func=mdp.last_action)
       

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    b_reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.005), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle", body_names="bottle"),
        },
    )

    l_reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("lid", body_names="cap_001"),
        },
    )


@configclass
class RewardsCfg:
    # Reaching rewards 
    b_reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1, "object_cfg": SceneEntityCfg("bottle"), "ee_frame_cfg" : SceneEntityCfg("ee_frame_bottle") }, weight=1.0)
    l_reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.08, "object_cfg": SceneEntityCfg("lid"), "ee_frame_cfg" : SceneEntityCfg("ee_frame_lid") }, weight=1.0)

    # Lifting rewards
    b_lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.3, "object_cfg": SceneEntityCfg("bottle")}, weight=20.0)
    l_lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.05, "object_cfg": SceneEntityCfg("lid")}, weight=20.0)

    b_object_goal_tracking = RewTerm(func=mdp.object_goal_distance, params={"std": 0.3, "minimal_height": 0.3, "command_name": "rbottle_pose", "robot_cfg" : SceneEntityCfg("robot_bottle"),  "object_cfg": SceneEntityCfg("bottle")}, weight=16.0)
    l_object_goal_tracking = RewTerm(func=mdp.object_goal_distance, params={"std": 0.1, "minimal_height": 0.05, "command_name": "rlid_pose", "robot_cfg" : SceneEntityCfg("robot_lid"),  "object_cfg": SceneEntityCfg("lid")}, weight=16.0)
    
    b_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.3,"command_name": "rbottle_pose", "robot_cfg" : SceneEntityCfg("robot_bottle"),  "object_cfg": SceneEntityCfg("bottle")},
        weight=5.0,
    )
    l_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.01, "minimal_height":  0.05,"command_name": "rlid_pose", "robot_cfg" : SceneEntityCfg("robot_lid"),  "object_cfg": SceneEntityCfg("lid")},
        weight=5.0,
    )

    b_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot_bottle")})
    l_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot_lid")})

    #b_object_lin_vel_penalty = RewTerm(func=mdp.root_lin_vel_l2, params={"object_cfg": SceneEntityCfg("bottle")}, weight=-1e-3)
    #l_object_lin_vel_penalty = RewTerm(func=mdp.root_lin_vel_l2, params={"object_cfg": SceneEntityCfg("lid")}, weight=-1e-3)

    #b_object_ang_vel_penalty = RewTerm(func=mdp.root_ang_vel_l2, params={"object_cfg": SceneEntityCfg("bottle")}, weight=-1e-3)
    #l_object_ang_vel_penalty = RewTerm(func=mdp.root_ang_vel_l2, params={"object_cfg": SceneEntityCfg("lid")}, weight=-1e-3)

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)



@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    b_object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("bottle")})
    l_object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("lid")})

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 10000}
    )

    b_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "b_joint_vel", "weight": -1e-3, "num_steps": 20000}
    )

    l_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "l_joint_vel", "weight": -1e-3, "num_steps": 20000}
    )


@configclass
class TwistLidEnvCfg(ManagerBasedRLEnvCfg):
    scene: TwistLidSceneCfg = TwistLidSceneCfg(num_envs=128, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.scene.robot_bottle = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_Bottle")
        self.scene.robot_lid = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotLid")
        #self.scene.robot = self.scene.robot_bottle

        self.scene.bottle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bottle",
            spawn=sim_utils.UsdFileCfg(
                usd_path='/home/dgargan2/twist_lid/assets/bottle.usdc',
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg()
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        )

        self.scene.lid = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/lid",
            spawn=sim_utils.UsdFileCfg(
                usd_path='/home/dgargan2/twist_lid/assets/cap.usdc',
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg()
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 1.5, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        )

        self.scene.robot_bottle.init_state.pos = [0.0, 0.0, 0.0]
        self.scene.robot_lid.init_state.pos    = [0.0, 1.5, 0.0]

        # End-effector frames
        marker_cfg_bottle = FRAME_MARKER_CFG.copy()
        marker_cfg_bottle.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_lid = FRAME_MARKER_CFG.copy()
        marker_cfg_lid.markers["frame"].scale = (0.1, 0.1, 0.1)

        marker_cfg_bottle.prim_path = "{ENV_REGEX_NS}/Visuals/FrameBottle"
        marker_cfg_lid.prim_path    = "{ENV_REGEX_NS}/Visuals/FrameLid"

        self.scene.ee_frame_bottle = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_Bottle/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_bottle,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_Bottle/panda_hand",
                    name="ee_frame_bottle",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.08]),
                ),
            ],
        )


        self.scene.ee_frame_lid = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RobotLid/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_lid,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/RobotLid/panda_hand",
                    name="ee_frame_lid",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.08]),
                ),
            ],
        )

        # General sim settings
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # PhysX tuning
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625