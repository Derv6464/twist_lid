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
    height_progress = torch.tanh((current_height - 0.0) / (minimal_height * 0.5))

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

def object_gripper_center_distance(
    env,
    *,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    std: float = 0.1,
):
    """Reward based on distance between object center and gripper midpoint."""

    # Object position in world frame
    object_pos = env.scene[object_cfg.name].data.root_pos_w  # (num_envs, 3)

    # End-effector (gripper midpoint) position in world frame
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]  
    # target_pos_w shape: (num_envs, num_frames, 3)

    # Euclidean distance
    dist = torch.norm(object_pos - ee_pos, dim=-1)

    # Gaussian-shaped reward
    reward = torch.exp(-(dist**2) / (2 * std**2))

    return reward


def object_distance_to_middle_table(
    env,
    *,
    object_cfg,
    table_offset=(0.0, 0.75, 0.0),
    height_offset: float = 0.1,
    std: float = 0.25,
):
    import torch

    # Bottle world positions (num_envs, 3)
    obj_pos = env.scene[object_cfg.name].data.root_pos_w

    env_origins = env.scene.env_origins

    table_offset = torch.tensor(
        table_offset,
        device=obj_pos.device,
        dtype=obj_pos.dtype,
    )

    target_pos = env_origins + table_offset
    target_pos[:, 2] += height_offset

    # Distance
    dist = torch.norm(obj_pos - target_pos, dim=-1)

    # Gaussian reward (nearer = more reward)
    reward = torch.exp(-(dist ** 2) / (2 * std ** 2))

    return reward


def object_upright_reward(
    env,
    object_cfg: SceneEntityCfg,
    std: float = 0.2,
):
    """Reward for keeping an object upright (aligned with world Z axis)."""

    # Object orientation as quaternion (w, x, y, z)
    quat = env.scene[object_cfg.name].data.root_quat_w  # (num_envs, 4)

    # World Z axis
    world_z = torch.tensor(
        [0.0, 0.0, 1.0],
        device=quat.device,
        dtype=quat.dtype,
    )

    # Compute object's local Z axis expressed in world frame
    # Using quaternion rotation: z_obj = q * [0,0,1] * q_conj
    z_obj = torch.stack(
        [
            2 * (quat[:, 1] * quat[:, 3] + quat[:, 0] * quat[:, 2]),
            2 * (quat[:, 2] * quat[:, 3] - quat[:, 0] * quat[:, 1]),
            1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
        ],
        dim=-1,
    )

    # Cosine of tilt angle
    alignment = torch.clamp(
        (z_obj * world_z).sum(dim=-1), -1.0, 1.0
    )

    
    reward = torch.exp(-((1.0 - alignment) ** 2) / (2 * std**2))

    return reward


