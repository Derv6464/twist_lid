#!/usr/bin/env python3
"""Debug script to check initial environment state and rewards."""

import torch
import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
from twist_lid.tasks.manager_based.twist_lid.twist_lid_env_cfg_C import TwistLidEnvCfg

def main():
    # Create environment
    from isaaclab.envs import ManagerBasedRLEnv

    cfg = TwistLidEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=cfg)

    print("\n" + "="*80)
    print("ENVIRONMENT INITIALIZATION CHECK")
    print("="*80)

    # Reset environment
    obs, _ = env.reset()

    # Check positions
    print("\n### Initial Positions ###")
    robot_bottle = env.scene["robot_bottle"]
    robot_lid = env.scene["robot_lid"]
    bottle = env.scene["bottle"]
    lid = env.scene["lid"]
    ee_frame_bottle = env.scene["ee_frame_bottle"]

    print(f"Robot bottle base: {robot_bottle.data.root_pos_w[0].cpu().numpy()}")
    print(f"Robot lid base: {robot_lid.data.root_pos_w[0].cpu().numpy()}")
    print(f"Bottle position: {bottle.data.root_pos_w[0].cpu().numpy()}")
    print(f"Lid position: {lid.data.root_pos_w[0].cpu().numpy()}")
    print(f"Bottle EE position: {ee_frame_bottle.data.target_pos_w[0, 0].cpu().numpy()}")

    # Check distances
    bottle_pos = bottle.data.root_pos_w[0]
    ee_pos = ee_frame_bottle.data.target_pos_w[0, 0]
    distance = torch.norm(bottle_pos - ee_pos).item()

    print(f"\n### Distances ###")
    print(f"EE to bottle distance: {distance:.3f} m")

    # Check rewards at initial state
    print(f"\n### Initial Rewards (without any action) ###")

    # Take a zero action and step
    action = torch.zeros(env.action_manager.total_action_dim, device=env.device)
    action = action.unsqueeze(0).expand(env.num_envs, -1)

    obs, rewards, dones, truncated, info = env.step(action)

    # Print individual rewards
    if hasattr(info, 'episode'):
        print("\nReward breakdown (first env):")
        for key in sorted(info.keys()):
            if 'reward' in key.lower():
                val = info[key][0].item() if torch.is_tensor(info[key]) else info[key]
                print(f"  {key}: {val:.6f}")

    # Print total
    print(f"\nTotal reward: {rewards[0].item():.6f}")

    # Check action space
    print(f"\n### Action Space ###")
    print(f"Action dimension: {env.action_manager.total_action_dim}")

    # Check observation space
    print(f"\n### Observation Space ###")
    print(f"Observation shape: {obs['policy'].shape}")
    print(f"Observation (first env, first 20 dims): {obs['policy'][0, :20].cpu().numpy()}")

    # Try moving towards bottle
    print(f"\n### Testing with random actions ###")
    for step in range(5):
        action = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 0.1
        obs, rewards, dones, truncated, info = env.step(action)

        bottle_pos = bottle.data.root_pos_w[0]
        ee_pos = ee_frame_bottle.data.target_pos_w[0, 0]
        distance = torch.norm(bottle_pos - ee_pos).item()

        print(f"Step {step+1}: distance={distance:.3f}m, reward={rewards[0].item():.6f}")

    print("\n" + "="*80)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
