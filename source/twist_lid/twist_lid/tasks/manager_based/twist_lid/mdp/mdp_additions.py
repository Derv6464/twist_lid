# SPDX-License-Identifier: BSD-3-Clause
"""
MDP functions for the TwistLid task.
Add these to your existing mdp/ module (e.g. mdp/rewards.py and mdp/observations.py).

Covers:
  - Lid-bottle position alignment  (Step 3)
  - Lid-bottle orientation alignment  (Step 3)
  - Screwing progress (angular twist around Z)  (Step 4)
  - Grasp quality for the small lid  (Step 3)
  - Shared-policy observation helpers  (Step 4)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_error_magnitude,
    quat_mul,
    quat_inv,
    euler_xyz_from_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_object_pos_w(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> torch.Tensor:
    """Return world-frame position (N, 3) for a RigidObject or Articulation root."""
    obj = env.scene[cfg.name]
    if isinstance(obj, RigidObject):
        return obj.data.root_pos_w  # (N, 3)
    else:  # Articulation – use body_pos_w for the root body
        body_idx = obj.find_bodies(cfg.body_names)[0][0] if cfg.body_names else 0
        return obj.data.body_pos_w[:, body_idx, :]  # (N, 3)


def _get_object_quat_w(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> torch.Tensor:
    """Return world-frame quaternion (N, 4) wxyz for a RigidObject or Articulation root."""
    obj = env.scene[cfg.name]
    if isinstance(obj, RigidObject):
        return obj.data.root_quat_w
    else:
        body_idx = obj.find_bodies(cfg.body_names)[0][0] if cfg.body_names else 0
        return obj.data.body_quat_w[:, body_idx, :]


# ---------------------------------------------------------------------------
# Step 3 — Lid grasping (tight tolerance for small lid)
# ---------------------------------------------------------------------------

def lid_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Gaussian reward for the lid-arm end-effector approaching the lid.
    Uses a tighter std than the bottle (lid radius = 0.02 m).
    Identical signature to object_ee_distance so it drops in as a RewTerm.
    """
    lid_pos = _get_object_pos_w(env, object_cfg)  # (N, 3)
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (N, 3)

    dist = torch.norm(lid_pos - ee_pos, dim=-1)  # (N,)
    return torch.exp(-(dist**2) / (2 * std**2))


# ---------------------------------------------------------------------------
# Step 3 — Alignment rewards
# ---------------------------------------------------------------------------

def lid_bottle_position_alignment(
    env: ManagerBasedRLEnv,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
    std: float = 0.02,
) -> torch.Tensor:
    """
    Reward for placing the lid directly above the bottle mouth.

    The bottle has height=0.2 m (half=0.1) and the lid height=0.04 m (half=0.02).
    Target relative position is (0, 0, +0.12) in the bottle's local XY = world XY
    (assuming bottle stays upright).

    Returns a Gaussian reward in [0, 1].
    """
    lid_pos    = _get_object_pos_w(env, lid_cfg)     # (N, 3)
    bottle_pos = _get_object_pos_w(env, bottle_cfg)  # (N, 3)

    # Bottle mouth in world frame (top center)
    bottle_half_h = 0.1   # half of 0.2 m
    lid_half_h    = 0.02  # half of 0.04 m
    target_offset = torch.tensor(
        [0.0, 0.0, bottle_half_h + lid_half_h], device=env.device
    )
    target_pos = bottle_pos + target_offset  # (N, 3)

    dist = torch.norm(lid_pos - target_pos, dim=-1)  # (N,)
    return torch.exp(-(dist**2) / (2 * std**2))


def lid_bottle_orientation_alignment(
    env: ManagerBasedRLEnv,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
    std: float = 0.3,
) -> torch.Tensor:
    """
    Reward for the lid sharing the same Z-axis orientation as the bottle.
    Measures quaternion angular error (rad) and returns a Gaussian reward.

    std = 0.3 rad ≈ 17 deg tolerance at 0.6 reward value.
    """
    lid_quat    = _get_object_quat_w(env, lid_cfg)     # (N, 4) wxyz
    bottle_quat = _get_object_quat_w(env, bottle_cfg)  # (N, 4) wxyz

    # Relative rotation: q_rel = q_lid * inv(q_bottle)
    q_rel = quat_mul(lid_quat, quat_inv(bottle_quat))  # (N, 4)
    # Identity quaternion in wxyz format for comparison
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(q_rel.shape[0], -1)
    angle_err = quat_error_magnitude(q_rel, identity_quat)  # (N,) in rad

    return torch.exp(-(angle_err**2) / (2 * std**2))


# ---------------------------------------------------------------------------
# Step 4 — Screwing progress
# ---------------------------------------------------------------------------

def lid_screw_progress(
    env: ManagerBasedRLEnv,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
    alignment_pos_threshold: float = 0.03,
    alignment_ori_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Reward for screwing the lid onto the bottle.

    Only active when the lid is already aligned (position + orientation).
    Measures cumulative relative yaw twist between lid and bottle across steps
    by tracking delta-yaw each step and summing (stored in env extras).

    Returns delta-yaw (rad) this step, gated by alignment quality.
    Positive reward = lid rotating relative to bottle in the tightening direction.

    NOTE: You must initialise env.extras["lid_screw_angle"] = zeros(N)
          in your env reset — see reset helper below.
    """
    lid_pos    = _get_object_pos_w(env, lid_cfg)
    bottle_pos = _get_object_pos_w(env, bottle_cfg)

    # --- Gate 1: position alignment ---
    bottle_half_h = 0.1
    lid_half_h    = 0.02
    target_pos = bottle_pos + torch.tensor(
        [0.0, 0.0, bottle_half_h + lid_half_h], device=env.device
    )
    pos_err = torch.norm(lid_pos - target_pos, dim=-1)          # (N,)
    pos_ok  = (pos_err < alignment_pos_threshold).float()

    # --- Gate 2: orientation alignment (XY axes match) ---
    lid_quat    = _get_object_quat_w(env, lid_cfg)
    bottle_quat = _get_object_quat_w(env, bottle_cfg)
    q_rel       = quat_mul(lid_quat, quat_inv(bottle_quat))
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(q_rel.shape[0], -1)
    angle_err   = quat_error_magnitude(q_rel, identity_quat)
    ori_ok      = (angle_err < alignment_ori_threshold).float()

    gate = pos_ok * ori_ok  # (N,)  1 = both aligned

    # --- Delta yaw this step ---
    # Relative yaw = yaw_lid - yaw_bottle
    _, _, yaw_lid    = euler_xyz_from_quat(lid_quat)     # (N,)
    _, _, yaw_bottle = euler_xyz_from_quat(bottle_quat)  # (N,)
    yaw_rel_now = yaw_lid - yaw_bottle                   # (N,)

    key = "lid_screw_yaw_prev"
    if key not in env.extras:
        env.extras[key] = yaw_rel_now.clone()

    delta_yaw = yaw_rel_now - env.extras[key]            # (N,)

    # Wrap to [-pi, pi] to handle discontinuities
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi

    env.extras[key] = yaw_rel_now.clone()

    # Accumulate total twist angle
    acc_key = "lid_screw_angle"
    if acc_key not in env.extras:
        env.extras[acc_key] = torch.zeros(env.num_envs, device=env.device)
    env.extras[acc_key] += delta_yaw * gate

    # Reward = delta_yaw this step, gated, no negative reward for unwinding
    # (clamp so the robot isn't punished for minor backward drift)
    reward = torch.clamp(delta_yaw * gate, min=0.0)
    return reward


def lid_screw_total(
    env: ManagerBasedRLEnv,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
    target_angle: float = 6.28,  # 2 * pi = one full turn
    std: float = 1.0,
) -> torch.Tensor:
    """
    Bonus reward when cumulative screw angle approaches target_angle.
    Gaussian shaped so reward climbs as the lid gets closer to fully tightened.
    """
    acc_key = "lid_screw_angle"
    if acc_key not in env.extras:
        return torch.zeros(env.num_envs, device=env.device)

    progress = env.extras[acc_key]  # (N,)
    remaining = target_angle - torch.clamp(progress, max=target_angle)
    return torch.exp(-(remaining**2) / (2 * std**2))


# ---------------------------------------------------------------------------
# Step 4 — Shared-policy observations
# ---------------------------------------------------------------------------

def lid_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    lid_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Position of the lid in the lid-robot's root frame.
    Mirrors object_position_in_robot_root_frame for the lid robot.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    lid_pos_w  = _get_object_pos_w(env, lid_cfg)          # (N, 3)
    robot_pos_w  = robot.data.root_pos_w                   # (N, 3)
    robot_quat_w = robot.data.root_quat_w                  # (N, 4)

    lid_pos_b, _ = subtract_frame_transforms(
        robot_pos_w, robot_quat_w, lid_pos_w
    )
    return lid_pos_b  # (N, 3)


def lid_bottle_relative_pose(
    env: ManagerBasedRLEnv,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Observation: [relative_position (3), relative_yaw (1)] = (N, 4).
    Gives both robots direct signal about lid-bottle alignment state.
    """
    lid_pos    = _get_object_pos_w(env, lid_cfg)
    bottle_pos = _get_object_pos_w(env, bottle_cfg)
    rel_pos    = lid_pos - bottle_pos  # (N, 3)

    lid_quat    = _get_object_quat_w(env, lid_cfg)
    bottle_quat = _get_object_quat_w(env, bottle_cfg)
    q_rel       = quat_mul(lid_quat, quat_inv(bottle_quat))
    _, _, yaw   = euler_xyz_from_quat(q_rel)  # (N,)

    return torch.cat([rel_pos, yaw.unsqueeze(-1)], dim=-1)  # (N, 4)


def screw_progress_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Observation: accumulated screw angle so far (N, 1).
    Lets the policy know how much twisting has been done.
    """
    acc_key = "lid_screw_angle"
    if acc_key not in env.extras:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.extras[acc_key].unsqueeze(-1)  # (N, 1)


# ---------------------------------------------------------------------------
# Reset helper — call inside your EventCfg reset or env __post_init__
# ---------------------------------------------------------------------------

def reset_screw_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """
    Zero out screw tracking for reset environments.
    Register this as an EventTerm with mode="reset".

    Example in EventCfg:
        reset_screw = EventTerm(func=mdp.reset_screw_state, mode="reset")
    """
    if "lid_screw_angle" in env.extras:
        env.extras["lid_screw_angle"][env_ids] = 0.0
    if "lid_screw_yaw_prev" in env.extras:
        lid_cfg = SceneEntityCfg("lid")
        lid_quat = _get_object_quat_w(env, lid_cfg)
        bottle_cfg = SceneEntityCfg("bottle")
        bottle_quat = _get_object_quat_w(env, bottle_cfg)
        _, _, yaw_lid    = euler_xyz_from_quat(lid_quat)
        _, _, yaw_bottle = euler_xyz_from_quat(bottle_quat)
        env.extras["lid_screw_yaw_prev"][env_ids] = (yaw_lid - yaw_bottle)[env_ids]
