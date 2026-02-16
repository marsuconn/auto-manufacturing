"""Evaluate a trained PPO agent and compare against a greedy baseline."""

import argparse
import os

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO

from sim.tool_library import ToolLibrary

# Register the custom environment
register(
    id="PocketMachining-v0",
    entry_point="envs.pocket_machining_env:PocketMachiningEnv",
)


def run_greedy_episode(env: gym.Env, library: ToolLibrary) -> dict:
    """Greedy baseline: always pick the toolpath with the highest volume removal rate,
    switching to finishing once roughing is nearly done."""
    obs, info = env.reset()

    # Sort toolpaths by volume removal rate (descending)
    roughing_tps = sorted(
        [tp for tp in library.toolpaths
         if library.get_tool(tp.tool_id).tool_type == "roughing"],
        key=lambda tp: tp.volume_removal_rate,
        reverse=True,
    )
    finishing_tps = sorted(
        [tp for tp in library.toolpaths
         if library.get_tool(tp.tool_id).tool_type == "finishing"],
        key=lambda tp: tp.volume_removal_rate,
        reverse=True,
    )

    total_reward = 0.0
    tool_changes = 0
    prev_tool_id = -1

    while True:
        remaining = obs[0]
        quality = obs[1]

        # Use finishing if remaining volume is low enough
        if remaining <= 0.15 and quality < 0.7:
            action = finishing_tps[0].id
        else:
            action = roughing_tps[0].id

        tp = library.get_toolpath(action)
        if tp.tool_id != prev_tool_id and prev_tool_id >= 0:
            tool_changes += 1
        prev_tool_id = tp.tool_id

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "elapsed_time": info["elapsed_time"],
        "total_energy": info["total_energy"],
        "tool_changes": tool_changes,
        "remaining_fraction": info["remaining_fraction"],
        "surface_quality": info["surface_quality"],
        "completed": terminated,
    }


def run_agent_episode(env: gym.Env, model: PPO, library: ToolLibrary,
                      render: bool = False) -> dict:
    """Run one episode with the trained agent."""
    obs, info = env.reset()
    total_reward = 0.0
    tool_changes = 0
    prev_tool_id = -1
    steps = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        tp = library.get_toolpath(action)
        tool = library.get_tool(tp.tool_id)
        if tp.tool_id != prev_tool_id and prev_tool_id >= 0:
            tool_changes += 1
        prev_tool_id = tp.tool_id

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        steps.append({
            "action": action,
            "toolpath": tp.name,
            "tool": tool.name,
            "remaining": info["remaining_fraction"],
            "quality": info["surface_quality"],
        })

        if terminated or truncated:
            break

    if render:
        print("\n--- Agent Step-by-Step ---")
        for i, s in enumerate(steps):
            print(
                f"  Step {i+1:2d}: Action {s['action']} | "
                f"{s['toolpath']:<30s} | Tool: {s['tool']:<25s} | "
                f"Remaining: {s['remaining']:.1%} | Quality: {s['quality']:.2f}"
            )

    return {
        "total_reward": total_reward,
        "elapsed_time": info["elapsed_time"],
        "total_energy": info["total_energy"],
        "tool_changes": tool_changes,
        "remaining_fraction": info["remaining_fraction"],
        "surface_quality": info["surface_quality"],
        "completed": terminated,
    }


def print_results(label: str, results: dict):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Completed:          {results['completed']}")
    print(f"  Total reward:       {results['total_reward']:.2f}")
    print(f"  Machining time:     {results['elapsed_time']:.2f} min")
    print(f"  Energy consumed:    {results['total_energy']:.0f} W·min")
    print(f"  Tool changes:       {results['tool_changes']}")
    print(f"  Remaining fraction: {results['remaining_fraction']:.2%}")
    print(f"  Surface quality:    {results['surface_quality']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("--model", type=str, default="models/ppo_pocket_final",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    library = ToolLibrary()
    env = gym.make("PocketMachining-v0")

    # --- Greedy baseline ---
    print("\nRunning greedy baseline...")
    greedy_results = []
    for _ in range(args.episodes):
        greedy_results.append(run_greedy_episode(env, library))

    avg_greedy = {
        k: np.mean([r[k] for r in greedy_results])
        for k in ["total_reward", "elapsed_time", "total_energy", "tool_changes"]
    }
    avg_greedy["completed"] = all(r["completed"] for r in greedy_results)
    avg_greedy["remaining_fraction"] = np.mean([r["remaining_fraction"] for r in greedy_results])
    avg_greedy["surface_quality"] = np.mean([r["surface_quality"] for r in greedy_results])
    print_results("Greedy Baseline (avg)", avg_greedy)

    # --- Trained agent ---
    model_path = args.model
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}. Train first with: python train.py")
        env.close()
        return

    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)

    agent_results = []
    for i in range(args.episodes):
        render = (i == 0)  # show step-by-step for first episode
        agent_results.append(run_agent_episode(env, model, library, render=render))

    avg_agent = {
        k: np.mean([r[k] for r in agent_results])
        for k in ["total_reward", "elapsed_time", "total_energy", "tool_changes"]
    }
    avg_agent["completed"] = all(r["completed"] for r in agent_results)
    avg_agent["remaining_fraction"] = np.mean([r["remaining_fraction"] for r in agent_results])
    avg_agent["surface_quality"] = np.mean([r["surface_quality"] for r in agent_results])
    print_results("Trained Agent (avg)", avg_agent)

    # --- Comparison ---
    print(f"\n{'='*50}")
    print("  Comparison")
    print(f"{'='*50}")
    time_diff = avg_greedy["elapsed_time"] - avg_agent["elapsed_time"]
    print(f"  Time saved by agent: {time_diff:.2f} min "
          f"({time_diff / avg_greedy['elapsed_time'] * 100:.1f}%)" if avg_greedy["elapsed_time"] > 0 else "")

    env.close()


if __name__ == "__main__":
    main()
