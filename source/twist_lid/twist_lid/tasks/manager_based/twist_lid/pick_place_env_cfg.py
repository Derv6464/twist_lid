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
    robot_bottle: ArticulationCfg = MISSING
    robot_lid: ArticulationCfg = MISSING

    ee_frame_bottle: FrameTransformerCfg = MISSING
    ee_frame_lid: FrameTransformerCfg = MISSING

    bottle: RigidObjectCfg = MISSING
    lid: RigidObjectCfg = MISSING

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
    rbottle_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_bottle",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5), pos_y=(-0.2, 0.2), pos_z=(0.15, 0.35),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

    rlid_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_lid",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5), pos_y=(-0.2, 0.2), pos_z=(0.15, 0.35),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

    meeting_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_bottle",  
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.4),
            pos_y=(0.7, 0.8), 
            pos_z=(0.2, 0.3),
            roll=(0, 0),
            pitch=(0, 0),
            yaw=(0, 0),
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    arm_bottle = mdp.JointPositionActionCfg(
        asset_name="robot_bottle", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    arm_lid = mdp.JointPositionActionCfg(
        asset_name="robot_lid", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )

    gripper_bottle = mdp.JointPositionActionCfg(
        asset_name="robot_bottle",
        joint_names=["panda_finger.*"],
        scale=0.04, 
        use_default_offset=False,
    )
    gripper_lid = mdp.JointPositionActionCfg(
        asset_name="robot_lid",
        joint_names=["panda_finger.*"],
        scale=0.03, 
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # bottle robot obs
        b_joint_pos = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": SceneEntityCfg("robot_bottle")})
        b_joint_vel = ObsTerm(func=mdp.joint_vel_rel,params={"asset_cfg": SceneEntityCfg("robot_bottle")})

        l_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_lid")})
        l_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_lid")})

        b_object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot_bottle"),
                "object_cfg": SceneEntityCfg("bottle"),
            }
        )
        l_object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot_lid"),
                "object_cfg": SceneEntityCfg("lid"),
            }
        )

        lid_bottle_rel_pose = ObsTerm(
            func=mdp.lid_bottle_relative_pose,
            params={
                "lid_cfg": SceneEntityCfg("lid"),
                "bottle_cfg": SceneEntityCfg("bottle"),
            }
        )

        b_target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "rbottle_pose"})
        l_target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "rlid_pose"})
        meeting_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "meeting_pose"})

        b_ee_to_meeting = ObsTerm(
            func=mdp.ee_goal_distance_obs,
            params={
                "command_name": "meeting_pose",
                "robot_cfg": SceneEntityCfg("robot_bottle"),
            }
        )
        l_ee_to_meeting = ObsTerm(
            func=mdp.ee_goal_distance_obs,
            params={
                "command_name": "meeting_pose",
                "robot_cfg": SceneEntityCfg("robot_lid"),
            }
        )

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
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle", body_names="bottle"),
        },
    )
    l_reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("lid", body_names="PCO28_1810_PET_bottle_cap_ind"),
        },
    )

@configclass
class RewardsCfg:
    b_reaching_object = RewTerm(func=mdp.object_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("bottle"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_bottle"),
        },
        weight=10.0, 
    )

    l_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("lid"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_lid"),
        },
        weight=15.0,  
    )

    b_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.15, "object_cfg": SceneEntityCfg("bottle")},
        weight=20.0, 
    )

    l_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.015, "object_cfg": SceneEntityCfg("lid")},
        weight=40.0,
    )

    b_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.2,
            "minimal_height": 0.05, 
            "command_name": "rbottle_pose",
            "robot_cfg": SceneEntityCfg("robot_bottle"),
            "object_cfg": SceneEntityCfg("bottle"),
        },
        weight=10.0,
    )

    l_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.2,
            "minimal_height": 0.05, 
            "command_name": "rlid_pose",
            "robot_cfg": SceneEntityCfg("robot_lid"),
            "object_cfg": SceneEntityCfg("lid"),
        },
        weight=10.0,
    )

    b_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.05,
            "command_name": "rbottle_pose",
            "robot_cfg": SceneEntityCfg("robot_bottle"),
            "object_cfg": SceneEntityCfg("bottle"),
        },
        weight=5.0,
    )

    l_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.05, 
            "command_name": "rlid_pose",
            "robot_cfg": SceneEntityCfg("robot_lid"),
            "object_cfg": SceneEntityCfg("lid"),
        },
        weight=5.0,
    )

    b_to_meeting = RewTerm(
        func=mdp.ee_goal_distance,
        params={
            "command_name": "meeting_pose",
            "robot_cfg": SceneEntityCfg("robot_bottle"),
        },
        weight=1.0,
    )

    l_to_meeting = RewTerm(
        func=mdp.ee_goal_distance,
        params={
            "command_name": "meeting_pose",
            "robot_cfg": SceneEntityCfg("robot_lid"),
        },
        weight=1.0,
    )

    alignment_pos = RewTerm(
        func=mdp.lid_bottle_position_error,
        params={
            "lid_cfg": SceneEntityCfg("lid"),
            "bottle_cfg": SceneEntityCfg("bottle"),
            "std": 0.05,
        },
        weight=1.50,
    )
    
    alignment_rot = RewTerm(
        func=mdp.lid_bottle_orientation_error,
        params={
            "lid_cfg": SceneEntityCfg("lid"),
            "bottle_cfg": SceneEntityCfg("bottle"),
            "std": 0.2,
        },
        weight=2.0,
    )

    success = RewTerm(
        func=mdp.is_aligned_and_close,
        params={
            "lid_cfg": SceneEntityCfg("lid"),
            "bottle_cfg": SceneEntityCfg("bottle"),
            "pos_threshold": 0.01,
            "rot_threshold": 0.1,
        },
        weight=1.0,
    )


    b_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot_bottle")}
    )
    l_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot_lid")}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    b_object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("bottle")})
    l_object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("lid")})


@configclass
class CurriculumCfg:
    action_rate_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 200000}
    )
    b_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "b_joint_vel", "weight": -1e-2, "num_steps": 200000}
    )
    l_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "l_joint_vel", "weight": -1e-2, "num_steps": 200000}
    )

    b_to_meeting = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "b_to_meeting", "weight": 10, "num_steps": 5_000_000}
    )
    l_to_meeting = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "l_to_meeting", "weight": 10, "num_steps": 5_000_000}
    )
    alignment_pos = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "alignment_pos", "weight": 15, "num_steps": 15_000_000}
    )
    alignment_rot = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "alignment_rot", "weight": 20, "num_steps": 15_000_000}
    )
    success = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "success", "weight": 100, "num_steps": 25_000_000}
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
        self.scene.robot_bottle = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_Bottle"
        )
        self.scene.robot_lid = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_Lid"
        )
        self.scene.robot_bottle.init_state.pos = [0.0, 0.0, 0.0]
        self.scene.robot_lid.init_state.pos    = [0.0, 1.5, 0.0]

        self.scene.robot_bottle.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.3,      
            "panda_joint3": 0.0,
            "panda_joint4": -2.5,      
            "panda_joint5": 0.0,
            "panda_joint6": 2.8,       
            "panda_joint7": 0.785,     
            "panda_finger_joint.*": 0.04, 
        }
        self.scene.robot_lid.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.3,
            "panda_joint3": 0.0,
            "panda_joint4": -2.5,
            "panda_joint5": 0.0,
            "panda_joint6": 2.8,
            "panda_joint7": 0.785,
            "panda_finger_joint.*": 0.04,  
        }

        self.scene.bottle = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/bottle",
            spawn=sim_utils.UsdFileCfg(
                usd_path='/home/dgargan2/twist_lid/assets/bottle.usdc',
                activate_contact_sensors=True,
                scale= (1.2, 1.2, 1.2), 
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.4, 0.0, 0.01), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )

        self.scene.lid = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/lid",
            spawn=sim_utils.UsdFileCfg(
                usd_path='/home/dgargan2/twist_lid/assets/cap_nomaterial.usdc',
                scale= (1.2, 1.2, 1.2), 
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.4, 1.4, 0.01), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )

        marker_cfg_bottle = FRAME_MARKER_CFG.copy()
        marker_cfg_bottle.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_bottle.prim_path = "{ENV_REGEX_NS}/Visuals/FrameBottle"

        marker_cfg_lid = FRAME_MARKER_CFG.copy()
        marker_cfg_lid.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_lid.prim_path = "{ENV_REGEX_NS}/Visuals/FrameLid"

        self.scene.ee_frame_bottle = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_Bottle/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_bottle,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_Bottle/panda_hand",
                    name="ee_frame_bottle",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )
        self.scene.ee_frame_lid = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_Lid/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_lid,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_Lid/panda_hand",
                    name="ee_frame_lid",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        self.decimation = 2
        self.episode_length_s = 8.0   
        self.sim.dt = 0.01          
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
