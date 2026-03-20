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
USD_BOTTLE_WITH_LID = '/home/dgargan2/bottle_open/assets/bottle.usda'

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.8]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    table_middle: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableMiddle",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.75, 0.8]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    table_lid: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLid",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 1.5, 0.8]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    # Ground + light
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
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
        b_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg" :SceneEntityCfg("robot_bottle")})
        l_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg" :SceneEntityCfg("robot_lid") })

        b_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot_bottle")})
        l_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot_lid")})
        
        b_object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"robot_cfg":SceneEntityCfg("robot_bottle"), "object_cfg":SceneEntityCfg("bottle")})  # typically uses SceneEntityCfg("object")
        l_object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"robot_cfg":SceneEntityCfg("robot_lid"), "object_cfg":SceneEntityCfg("lid")})  # typically uses SceneEntityCfg("object")

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
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle", body_names="Bottle"),
        },
    )

    l_reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("lid", body_names="Lid"),
        },
    )


@configclass
class RewardsCfg:
    b_reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.3, "object_cfg": SceneEntityCfg("bottle"), "ee_frame_cfg" : SceneEntityCfg("ee_frame_bottle") }, weight=5.0)
    l_reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1, "object_cfg": SceneEntityCfg("lid"), "ee_frame_cfg" : SceneEntityCfg("ee_frame_lid") }, weight=2.0)
    
    b_lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": (0.8+0.11), "object_cfg": SceneEntityCfg("bottle")}, weight=15.0)
    l_lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": (0.8+0.07), "object_cfg": SceneEntityCfg("lid")}, weight=15.0)
    
    b_bonus_lift_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": (0.8+0.16), "object_cfg": SceneEntityCfg("bottle")}, weight=1.0)
    l_bonus_lift_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": (0.8+0.16), "object_cfg": SceneEntityCfg("lid")}, weight=1.0)

    b_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": (0.8+0.11), "command_name": "rbottle_pose", "robot_cfg" : SceneEntityCfg("robot_bottle"),  "object_cfg": SceneEntityCfg("bottle")},
        weight=16.0,
    )
    b_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": (0.8+0.11), "command_name": "rbottle_pose", "robot_cfg" : SceneEntityCfg("robot_bottle"),  "object_cfg": SceneEntityCfg("bottle")},
        weight=7.0,
    )

    l_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height":(0.8+0.07), "command_name": "rlid_pose", "robot_cfg" : SceneEntityCfg("robot_lid"),  "object_cfg": SceneEntityCfg("lid")},
        weight=16.0,
    )
    l_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": (0.8+0.07), "command_name": "rlid_pose", "robot_cfg" : SceneEntityCfg("robot_lid"),  "object_cfg": SceneEntityCfg("lid")},
        weight=7.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    b_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5, params={"asset_cfg": SceneEntityCfg("robot_bottle")})
    l_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5, params={"asset_cfg": SceneEntityCfg("robot_lid")})
    
    #b_upright_penalty = RewTerm(func=mdp.object_uprightness, weight=-1e-4, params={"object_cfg": SceneEntityCfg("bottle")})
    #l_upright_penalty = RewTerm(func=mdp.object_uprightness, weight=-1e-4, params={"object_cfg": SceneEntityCfg("lid")})

    b_object_lin_vel_penalty = RewTerm(
        func=mdp.root_lin_vel_l2,
        params={"object_cfg": SceneEntityCfg("bottle")},
        weight=-1e-3,
    )
    l_object_lin_vel_penalty = RewTerm(
        func=mdp.root_lin_vel_l2,
        params={"object_cfg": SceneEntityCfg("lid")},
        weight=-1e-3,
    )

    b_object_ang_vel_penalty = RewTerm(
        func=mdp.root_ang_vel_l2,
        params={"object_cfg": SceneEntityCfg("bottle")},
        weight=-1e-3,
    )
    l_object_ang_vel_penalty = RewTerm(
        func=mdp.root_ang_vel_l2,
        params={"object_cfg": SceneEntityCfg("lid")},
        weight=-1e-3,
    )

   # b_lift_penality = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.25, "object_cfg": SceneEntityCfg("bottle")}, weight=-1.0)
   # l_lift_penality = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.25, "object_cfg": SceneEntityCfg("lid")}, weight=-1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    b_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.1, "asset_cfg": SceneEntityCfg("bottle")},
    )
    l_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.1, "asset_cfg": SceneEntityCfg("lid")},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight":  -1e-1, "num_steps": 10000})
    #b_joint_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "b_joint_vel", "weight": -1e-1, "num_steps": 10000})
    #l_joint_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "l_joint_vel", "weight":  -1e-1, "num_steps": 10000})
    #b_upright_penalty = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "b_upright_penalty", "weight": -0.5, "num_steps": 11000})
    #l_upright_penalty = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "l_upright_penalty", "weight": -0.5, "num_steps": 11000})

from omni.isaac.core.utils.stage import get_current_stage

def print_all_prim_paths():
    stage = get_current_stage()
    for prim in stage.Traverse():
        print(prim.GetPath().pathString)

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

        self.scene.bottle_asset = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/BottleAsset",
            spawn=sim_utils.UsdFileCfg(usd_path=USD_BOTTLE_WITH_LID),
        )
        print_all_prim_paths()
        # Bottle → reference the child prim
        self.scene.bottle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BottleAsset/bottle014/body",
        )
        
        # Lid → reference the child prim
        self.scene.lid = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BottleAsset/bottle014/lid",
        )


        self.scene.robot_bottle.init_state.pos = [0.0, 0.0, 0.8]
        #self.scene.robot_bottle.init_state.rot = [-0.707, 0.0, 0.0, -0.707]
        self.scene.robot_lid.init_state.pos    = [0.0, 1.5, 0.8]

        #self.scene.table_bottle.init_state.pos = [0.6, -0.6, 0.0]

        self.scene.bottle.init_state.pos = [0.5, 0.0, 0.8]
        self.scene.lid.init_state.pos = [0.5, 1.5, 0.8]

        # End-effector frame transformers for both robots
        marker_cfg_bottle = FRAME_MARKER_CFG.copy()
        marker_cfg_bottle.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_lid = FRAME_MARKER_CFG.copy()
        marker_cfg_bottle.markers["frame"].scale = (0.1, 0.1, 0.1)
        
        marker_cfg_bottle.prim_path = "{ENV_REGEX_NS}/Visuals/FrameBottle"
        marker_cfg_lid.prim_path    = "{ENV_REGEX_NS}/Visuals/FrameLid"


        self.scene.ee_frame_bottle = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RobotBottle/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_bottle,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/RobotBottle/panda_hand",
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
        self.episode_length_s = 10.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # PhysX tuning
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625