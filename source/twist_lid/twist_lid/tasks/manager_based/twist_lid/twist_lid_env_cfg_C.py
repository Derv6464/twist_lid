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


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class TwistLidSceneCfg(InteractiveSceneCfg):
    """Scene with two Frankas: one holds the bottle, one screws the lid."""

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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@configclass
class CommandsCfg:
    """
    Bottle robot: carry bottle to the middle table handoff point.
    Lid robot:    no pose command — it operates purely from alignment rewards.
    """
    rbottle_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_bottle",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Command target for carrying bottle to middle table
            # Keep closer and lower for easier reaching
            pos_x=(0.3, 0.5), pos_y=(-0.2, 0.2), pos_z=(0.15, 0.35),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    arm_bottle = mdp.JointPositionActionCfg(
        asset_name="robot_bottle", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    arm_lid = mdp.JointPositionActionCfg(
        asset_name="robot_lid", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )

    # Continuous gripper control for better grasping
    gripper_bottle = mdp.JointPositionActionCfg(
        asset_name="robot_bottle",
        joint_names=["panda_finger.*"],
        scale=0.04,  # Max 4cm per finger
        use_default_offset=False,
    )
    gripper_lid = mdp.JointPositionActionCfg(
        asset_name="robot_lid",
        joint_names=["panda_finger.*"],
        scale=0.04,  # Max 4cm per finger (lid is smaller, needs finer control)
        use_default_offset=False,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """
    Shared policy: both robots see their own proprioception + all object state.
    The 4-dim relative pose obs (lid_bottle_relative_pose) and screw progress
    give both robots the coordination signal they need.

    Total dims (approx):
        b_joint_pos          9   (7 arm + 2 gripper)
        b_joint_vel          9   (7 arm + 2 gripper)
        l_joint_pos          9   (7 arm + 2 gripper)
        l_joint_vel          9   (7 arm + 2 gripper)
        b_object_position    3   (bottle in bottle-robot frame)
        l_object_position    3   (lid in lid-robot frame)
        lid_bottle_rel_pose  4   (dx,dy,dz, d_yaw)
        screw_progress       1
        b_target_position    7   (pose command wxyz+xyz)
        actions             18   (9 per robot: 7 arm + 2 gripper)
        ─────────────────────
        Total               72
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # --- Bottle robot proprioception ---
        b_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_bottle")}
        )
        b_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_bottle")}
        )

        # --- Lid robot proprioception ---
        l_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_lid")}
        )
        l_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_lid")}
        )

        # --- Object positions in their respective robot frames ---
        b_object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot_bottle"),
                "object_cfg": SceneEntityCfg("bottle"),
            }
        )
        l_object_position = ObsTerm(
            # lid_position_in_robot_root_frame lives in mdp_additions.py
            func=mdp.lid_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot_lid"),
                "lid_cfg": SceneEntityCfg("lid"),
            }
        )

        # --- Coordination: lid-bottle relative pose + screw progress ---
        lid_bottle_rel_pose = ObsTerm(
            func=mdp.lid_bottle_relative_pose,
            params={
                "lid_cfg": SceneEntityCfg("lid"),
                "bottle_cfg": SceneEntityCfg("bottle"),
            }
        )
        screw_progress = ObsTerm(func=mdp.screw_progress_obs)

        # --- Bottle carry command ---
        b_target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "rbottle_pose"}
        )

        # --- Last actions (full set, both robots) ---
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    b_reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Reduce randomization range: keep objects closer for easier learning
            # X: -0.03 to +0.03 from initial position (0.3) = [0.27, 0.33]
            # Y: -0.15 to +0.15 (narrower than before)
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle", body_names="bottle"),
        },
    )
    l_reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Lid: even tighter randomization since it's smaller
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("lid", body_names="lid"),
        },
    )

    # Zero out screw-angle accumulator on reset
    reset_screw = EventTerm(func=mdp.reset_screw_state, mode="reset")


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # ── Bottle robot: reach → grasp → lift → carry ──────────────────────────────────

    b_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("bottle"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_bottle"),
        },
        weight=10.0,  # Strong reaching signal
    )

    # NEW: Reward for gripper being centered on object (encourages approach alignment)
    b_gripper_centering = RewTerm(
        func=mdp.object_gripper_center_distance,
        params={
            "object_cfg": SceneEntityCfg("bottle"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_bottle"),
            "std": 0.05,  # Tight tolerance for good centering
        },
        weight=5.0,
    )

    b_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        # Bottle is at Z=0.0 (table surface), lifting means moving it UP
        # Start with small threshold for initial learning
        params={"minimal_height": 0.10, "object_cfg": SceneEntityCfg("bottle")},
        weight=20.0,  # Strong reward for lifting
    )

    bottle_to_middle_table = RewTerm(
        # Encourage bottle robot to carry bottle towards the middle handoff point
        # This is important so the lid robot can reach the bottle
        func=mdp.object_distance_to_middle_table,
        params={
            "object_cfg": SceneEntityCfg("bottle"),
            "table_offset": (0.0, 0.75, 0.0),  # Middle table at Y=0.75
            "height_offset": 0.15,  # Keep at reasonable height
            "std": 0.20,  # Moderate tolerance
        },
        weight=8.0,  # Higher weight - important for coordination
    )

    # NEW: Reward bottle robot for keeping bottle still during screwing phase
    # This helps the lid robot complete the screwing task
    bottle_stability = RewTerm(
        func=mdp.object_upright_reward,
        params={
            "object_cfg": SceneEntityCfg("bottle"),
            "std": 0.15,  # Allow some tilt
        },
        weight=3.0,  # Keep bottle upright and stable
    )

    b_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.2,
            "minimal_height": 0.05,  # Low threshold - just need object off table
            "command_name": "rbottle_pose",
            "robot_cfg": SceneEntityCfg("robot_bottle"),
            "object_cfg": SceneEntityCfg("bottle"),
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

    # ── Lid robot: reach → grasp → lift → align → screw ─────────────────────────────────────

    l_reaching_object = RewTerm(
        func=mdp.lid_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("lid"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_lid"),
        },
        weight=10.0,  # Strong reaching signal
    )

    # NEW: Reward for lid gripper centering (critical for small lid)
    l_gripper_centering = RewTerm(
        func=mdp.object_gripper_center_distance,
        params={
            "object_cfg": SceneEntityCfg("lid"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_lid"),
            "std": 0.03,  # Very tight for small lid
        },
        weight=8.0,  # Higher weight due to difficulty
    )

    l_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        # Lid only needs to lift slightly
        params={"minimal_height": 0.02, "object_cfg": SceneEntityCfg("lid")},
        weight=20.0,
    )

    # ── Step 3: alignment rewards (lid → bottle) ────────────────────────────────────────────

    lid_position_alignment = RewTerm(
        # Lid mouth sits directly above bottle top
        # This is CRITICAL: lid must be positioned correctly to screw on
        func=mdp.lid_bottle_position_alignment,
        params={
            "lid_cfg": SceneEntityCfg("lid"),
            "bottle_cfg": SceneEntityCfg("bottle"),
            "std": 0.05,   # Tighter tolerance - lid must be well-aligned
        },
        weight=15.0,  # High weight - alignment is essential for screwing
    )

    lid_orientation_alignment = RewTerm(
        # Lid Z-axis matches bottle Z-axis (both upright)
        func=mdp.lid_bottle_orientation_alignment,
        params={
            "lid_cfg": SceneEntityCfg("lid"),
            "bottle_cfg": SceneEntityCfg("bottle"),
            "std": 0.3,    # ~17 degrees tolerance
        },
        weight=10.0,  # High weight - orientation must match for screwing
    )

    # ── Step 4: screwing rewards (make lid actually twist!) ─────────────────────────────────────────────

    lid_screw_progress = RewTerm(
        # Per-step delta-yaw reward: rewards ROTATION of lid relative to bottle
        # This is gated by alignment - only active when lid is positioned correctly
        func=mdp.lid_screw_progress,
        params={
            "lid_cfg": SceneEntityCfg("lid"),
            "bottle_cfg": SceneEntityCfg("bottle"),
            "alignment_pos_threshold": 0.04,   # 4 cm - slightly relaxed
            "alignment_ori_threshold": 0.4,    # ~23 deg - slightly relaxed
        },
        weight=100.0,  # VERY strong signal - this is the main task!
    )

    lid_screw_total = RewTerm(
        # Cumulative bonus: rewards total amount twisted (approaches 2π rad = full turn)
        func=mdp.lid_screw_total,
        params={
            "lid_cfg": SceneEntityCfg("lid"),
            "bottle_cfg": SceneEntityCfg("bottle"),
            "target_angle": 6.28,  # 2π radians = one full clockwise turn
            "std": 1.0,
        },
        weight=200.0,  # HUGE terminal bonus for completing the screw
    )

    # ── Regularisation (both robots) ─────────────────────────────────────────

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


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Only terminate if objects fall significantly below table (prevents early termination)
    b_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.15, "asset_cfg": SceneEntityCfg("bottle")}  # 15cm below table
    )
    l_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.15, "asset_cfg": SceneEntityCfg("lid")}
    )


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@configclass
class CurriculumCfg:
    """
    Phased curriculum for dual-arm bottle screwing:
      Phase 1 (0-8k):     Learn reaching, grasping, and lifting both objects
      Phase 2 (8k-12k):   Bottle robot carries bottle to handoff; lid robot lifts lid
      Phase 3 (12k-16k):  Lid robot learns to align lid with bottle
      Phase 4 (16k+):     Lid robot learns to screw lid onto bottle (twisting motion)
    """

    # Phase 1: Start with strong grasping signals, then ramp up other rewards

    # Phase 2: Ramp up lifting and carrying after grasping is learned
    b_lifting_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "b_lifting_object", "weight": 30.0, "num_steps": 8000}
    )
    l_lifting_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "l_lifting_object", "weight": 30.0, "num_steps": 8000}
    )

    # Phase 3: Ramp up alignment rewards after lifting is learned
    lid_position_alignment_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lid_position_alignment", "weight": 30.0, "num_steps": 12000}
    )
    lid_orientation_alignment_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lid_orientation_alignment", "weight": 20.0, "num_steps": 12000}
    )

    # Phase 4: Screw reward starts low → full weight after alignment is learned
    lid_screw_progress_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lid_screw_progress", "weight": 150.0, "num_steps": 16000}
    )
    lid_screw_total_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lid_screw_total", "weight": 300.0, "num_steps": 16000}
    )

    # Gradually increase penalties to encourage smooth motions
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 8000}
    )
    b_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "b_joint_vel", "weight": -1e-3, "num_steps": 8000}
    )
    l_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "l_joint_vel", "weight": -1e-3, "num_steps": 8000}
    )


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------

@configclass
class TwistLidEnvCfg(ManagerBasedRLEnvCfg):
    scene: TwistLidSceneCfg = TwistLidSceneCfg(num_envs=512, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # ── Robots ───────────────────────────────────────────────────────────
        self.scene.robot_bottle = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_Bottle"
        )
        self.scene.robot_lid = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_Lid"
        )
        self.scene.robot_bottle.init_state.pos = [0.0, 0.0, 0.0]
        self.scene.robot_lid.init_state.pos    = [0.0, 1.5, 0.0]

        # Configure robot to START with EE close to table level
        # Objects are at Z=0.0 (table surface), so EE should start ~5-15cm above
        #
        # Key insight: joint2 (shoulder) controls vertical reach significantly
        # Positive joint2 = arm goes UP, negative = arm goes DOWN
        #
        # This pose puts EE at ~10-15cm above table, forward ~30-35cm
        self.scene.robot_bottle.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.3,      # Negative = lower the arm towards table
            "panda_joint3": 0.0,
            "panda_joint4": -2.5,      # More bent = shorter reach, but lower
            "panda_joint5": 0.0,
            "panda_joint6": 2.8,       # Compensate for joint4 to keep EE horizontal
            "panda_joint7": 0.785,     # 45 deg rotation for gripper alignment
            "panda_finger_joint.*": 0.04,  # OPEN gripper (0.04 = 4cm = max open per finger)
        }
        self.scene.robot_lid.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.3,
            "panda_joint3": 0.0,
            "panda_joint4": -2.5,
            "panda_joint5": 0.0,
            "panda_joint6": 2.8,
            "panda_joint7": 0.785,
            "panda_finger_joint.*": 0.04,  # OPEN gripper (0.04 = 4cm = max open per finger)
        }

        # ── Objects ──────────────────────────────────────────────────────────
        self.scene.bottle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bottle",
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
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), roughness=1.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                # Place closer to robot base for easier reaching: X=0.3 instead of 0.4
                # Z=0.0 is correct (on table surface)
                pos=[0.3, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]
            ),
        )

        self.scene.lid = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/lid",
            spawn=sim_utils.CylinderCfg(
                radius=0.032,   # Slightly larger than bottle radius so it can sit on top
                height=0.04,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.4, 1.0), roughness=1.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                # Place closer to robot base: X=0.3 instead of 0.4
                pos=[0.3, 1.5, 0.0], rot=[1.0, 0.0, 0.0, 0.0]
            ),
        )

        # ── End-effector frames ──────────────────────────────────────────────
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

        # ── Sim settings ─────────────────────────────────────────────────────
        self.decimation = 2
        self.episode_length_s = 8.0   # Increased to give time for full task: grasp → lift → align → screw
        self.sim.dt = 0.01            # 100 Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
