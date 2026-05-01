# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject, Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("bottle")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height.

    Returns a smooth reward that increases as the object gets closer to the target height,
    with a bonus when it exceeds the minimal height.
    """
    object: RigidObject = env.scene[object_cfg.name]
    current_height = object.data.root_pos_w[:, 2]

    # Smooth reward that increases with height (gives gradient even below threshold)
    # Normalize so that reaching minimal_height gives ~0.5, and going higher gives more
    height_progress = torch.tanh(torch.clamp(current_height, min=0.0) / (minimal_height * 0.5))

    # Binary bonus when above threshold
    above_threshold = (current_height > minimal_height).float()

    # Combined: smooth progress + bonus
    return height_progress + above_threshold


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_bottle"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_bottle"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def lid_bottle_position_error(
    env,
    *,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
    std: float = 0.05,
):
    """Reward for lid position aligning with bottle."""

    lid_pos = env.scene[lid_cfg.name].data.root_pos_w
    bottle_pos = env.scene[bottle_cfg.name].data.root_pos_w

    dist = torch.norm(lid_pos - bottle_pos, dim=-1)

    return torch.exp(-(dist**2) / (2 * std**2))

def lid_bottle_orientation_error(
    env,
    *,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
    std: float = 0.2,
):
    """Reward for aligning lid orientation with bottle."""

    from isaaclab.utils.math import quat_inv, quat_mul

    q1 = env.scene[lid_cfg.name].data.root_quat_w
    q2 = env.scene[bottle_cfg.name].data.root_quat_w

    q_diff = quat_mul(quat_inv(q1), q2)

    angle = 2 * torch.acos(torch.clamp(q_diff[:, 0], -1.0, 1.0))

    return torch.exp(-(angle**2) / (2 * std**2))

def is_aligned_and_close(
    env,
    *,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
    pos_threshold: float,
    rot_threshold: float,
):
    """Binary success signal."""

    from isaaclab.utils.math import quat_inv, quat_mul

    lid = env.scene[lid_cfg.name]
    bottle = env.scene[bottle_cfg.name]

    pos_error = torch.norm(lid.data.root_pos_w - bottle.data.root_pos_w, dim=-1)

    q_diff = quat_mul(quat_inv(lid.data.root_quat_w), bottle.data.root_quat_w)
    rot_error = 2 * torch.acos(torch.clamp(q_diff[:, 0], -1.0, 1.0))

    success = (pos_error < pos_threshold) & (rot_error < rot_threshold)

    return success.float()

def lid_bottle_relative_pose(
    env,
    *,
    lid_cfg: SceneEntityCfg,
    bottle_cfg: SceneEntityCfg,
):
    """Relative pose of lid in bottle frame."""

    from isaaclab.utils.math import quat_inv, quat_mul

    lid = env.scene[lid_cfg.name]
    bottle = env.scene[bottle_cfg.name]

    pos_rel = lid.data.root_pos_w - bottle.data.root_pos_w
    quat_rel = quat_mul(quat_inv(bottle.data.root_quat_w), lid.data.root_quat_w)

    return torch.cat([pos_rel, quat_rel], dim=-1)
