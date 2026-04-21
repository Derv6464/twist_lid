# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_bottle"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b



def object_uprightness(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
) -> torch.Tensor:
    """Penalty for how far the object is from upright.

    0.0 = perfectly upright (z-axis aligned with world up)
    1.0 = completely sideways
    """

    obj = env.scene[object_cfg.name]

    quat = obj.data.root_quat_w  # (N, 4)
    norm = torch.linalg.norm(quat, dim=1, keepdim=True)
    quat = quat / (norm + 1e-8)

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    obj_z = torch.stack(
        [
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    )

    world_up = torch.tensor([0.0, 0.0, 1.0], device=quat.device, dtype=quat.dtype)

    alignment = (obj_z * world_up).sum(dim=1)
    alignment = alignment.clamp(min=-1.0, max=1.0)

    penalty = 1.0 - alignment.clamp(min=0.0)

    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=1.0, neginf=1.0)

    return penalty

def root_lin_vel_l2(env, object_cfg: SceneEntityCfg = SceneEntityCfg("bottle")) -> torch.Tensor:
    """L2 norm of the object's linear velocity in world frame."""
    obj = env.scene[object_cfg.name]
    vel = obj.data.root_vel_w[:, :3]  # linear velocity
    return torch.norm(vel, dim=1)

def root_ang_vel_l2(env, object_cfg: SceneEntityCfg = SceneEntityCfg("bottle")) -> torch.Tensor:
    """L2 norm of the object's angular velocity in world frame."""
    obj = env.scene[object_cfg.name]
    ang_vel = obj.data.root_ang_vel_w[:, :3]  # angular velocity
    return torch.norm(ang_vel, dim=1)

def reset_object_to_ee(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ee_frame_name: str,       
    asset_cfg: SceneEntityCfg,
    offset: list = [0.0, 0.0, 0.0],  # offset from EE tip in EE frame
):
    """Reset object pose to match the EE position at reset."""
    asset = env.scene[asset_cfg.name]
    ee_frame = env.scene[ee_frame_name]

    # Get EE world pose — shape (num_envs, 3) and (num_envs, 4)
    ee_pos = ee_frame.data.target_pos_w[env_ids, 0, :]   # (N, 3)
    ee_quat = ee_frame.data.target_quat_w[env_ids, 0, :] # (N, 4)

    # Apply offset in world frame (simple case — offset is small)
    offset_t = torch.tensor(offset, device=env.device).unsqueeze(0).expand(len(env_ids), -1)
    object_pos = ee_pos + offset_t

    # Zero velocity
    object_vel = torch.zeros(len(env_ids), 6, device=env.device)

    # Write to sim
    root_pose = torch.cat([object_pos, ee_quat], dim=-1)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(object_vel, env_ids=env_ids)