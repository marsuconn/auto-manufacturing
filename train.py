"""Train a PPO agent on the CNC Pocket Machining environment."""

import argparse
import sys
import os

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Register the custom environment
register(
    id="PocketMachining-v0",
    entry_point="envs.pocket_machining_env:PocketMachiningEnv",
)


def main():
    parser = argparse.ArgumentParser(description="Train PPO on PocketMachining-v0")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000,
                        help="Save a checkpoint every N steps")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = gym.make("PocketMachining-v0")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=args.log_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="ppo_pocket",
    )

    print(f"Training PPO for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_path = os.path.join(args.save_dir, "ppo_pocket_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
