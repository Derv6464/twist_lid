from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # Increased from 16 for more data per update
    max_iterations = 50000
    save_interval = 4000  # Save more frequently to monitor progress
    experiment_name = "twist_lid"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,  # Enable normalization for better learning
        critic_obs_normalization=True,  # Enable normalization for better learning
        actor_hidden_dims=[128, 128, 64],  # Larger network for complex task
        critic_hidden_dims=[128, 128, 64],  # Larger network for complex task
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,  # Increased from 1.0 for better value learning
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Increased from 0.005 for more exploration
        num_learning_epochs=8,  # Increased from 5 for more updates per batch
        num_mini_batches=8,  # Increased from 4 for better gradient estimates
        learning_rate=3.0e-4,  # Reduced from 1e-3 for more stable learning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,  # Increased from 0.01 for larger updates
        max_grad_norm=1.0,
    )