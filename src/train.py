"""
train.py: Complete Training Pipeline for Reinforcement Learning on Adaptive Surfaces.

Supports:
- Single-run training with Soft Actor-Critic (SAC)
- Action space engineering benchmarking (Crystallised+Residual vs End-to-End vs Baseline)
- Multi-subject training cohorts
- Automatic checkpointing and scientific plot generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from .action_space import ActionSpaceMode
from .agent import SACAgent
from .body_model import BodyParameters, PhysiologicalMode
from .environment import AdaptiveSurfaceEnv
from .visualise import (
    plot_action_space_comparison,
    plot_surface_and_pressure_state,
)


def train_agent(
    env: AdaptiveSurfaceEnv,
    agent: SACAgent,
    num_episodes: int = 150,
    batch_size: int = 128,
    eval_freq: int = 10,
    eval_episodes: int = 3,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "sac_agent",
) -> Dict[str, List[float]]:
    """Execute complete SAC training loop."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "rewards": [],
        "comfort_rewards": [],
        "alignment_rewards": [],
        "energy_costs": [],
        "arousals": [],
        "peak_pressures": [],
        "pressure_variances": [],
        "eval_episodes": [],
        "eval_returns": [],
    }

    best_eval_return = -float("inf")
    pbar = tqdm(range(1, num_episodes + 1), desc=f"Training {model_name}")

    for episode in pbar:
        obs, info = env.reset(seed=episode * 17)
        done = False
        ep_reward = 0.0
        ep_comfort = 0.0
        ep_align = 0.0
        ep_energy = 0.0
        ep_arousals = 0
        last_step_info = None

        while not done:
            # Select action with exploration noise
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            # Update parameters
            if len(agent.replay_buffer) >= batch_size:
                agent.update_parameters(batch_size=batch_size)

            ep_reward += reward
            rc = step_info["reward_components"]
            ep_comfort += rc.comfort_reward
            ep_align += rc.alignment_reward
            ep_energy += abs(rc.energy_penalty)
            if step_info["is_aroused"]:
                ep_arousals += 1
            last_step_info = step_info

        # Record metrics
        history["rewards"].append(float(ep_reward))
        history["comfort_rewards"].append(float(ep_comfort))
        history["alignment_rewards"].append(float(ep_align))
        history["energy_costs"].append(float(ep_energy))
        history["arousals"].append(int(ep_arousals))
        history["peak_pressures"].append(float(last_step_info["peak_pressure_mmhg"]))
        history["pressure_variances"].append(float(last_step_info["pressure_variance"]))

        pbar.set_postfix({
            "R": f"{ep_reward:.1f}",
            "P_peak": f"{last_step_info['peak_pressure_mmhg']:.1f}",
            "Arousals": ep_arousals,
            "Alpha": f"{agent.alpha:.3f}",
        })

        # Periodic evaluation
        if episode % eval_freq == 0:
            eval_returns = []
            for ev_i in range(eval_episodes):
                ev_obs, _ = env.reset(seed=1000 + ev_i)
                ev_done = False
                ev_ret = 0.0
                while not ev_done:
                    ev_act = agent.select_action(ev_obs, evaluate=True)
                    ev_obs, ev_r, ev_term, ev_trunc, _ = env.step(ev_act)
                    ev_done = ev_term or ev_trunc
                    ev_ret += ev_r
                eval_returns.append(ev_ret)

            mean_eval_ret = float(np.mean(eval_returns))
            history["eval_episodes"].append(episode)
            history["eval_returns"].append(mean_eval_ret)

            # Checkpoint best model
            if mean_eval_ret > best_eval_return:
                best_eval_return = mean_eval_ret
                best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
                agent.save_checkpoint(best_ckpt)

    # Save final model
    final_ckpt = os.path.join(checkpoint_dir, f"{model_name}_final.pt")
    agent.save_checkpoint(final_ckpt)
    print(f"Training complete. Best eval return: {best_eval_return:.2f}. Checkpoint saved to {best_ckpt}")

    return history


def run_action_space_benchmark(
    config: Dict[str, Any],
    num_episodes: int = 100,
    figures_dir: str = "figures",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Dict[str, List[float]]]:
    """Compare Action Space Architectures:
    1. Crystallised + Learned Residual (WATER Architecture)
    2. Naive End-to-End Continuous RL
    3. Crystallised Baseline Heuristic
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    benchmark_results = {}

    # 1. Crystallised + Learned Residual (WATER)
    print("\n[Benchmark 1/3] Training Crystallised + Learned Residual (WATER Architecture)...")
    env_crys = AdaptiveSurfaceEnv(
        max_steps_per_episode=config["training"]["max_steps_per_episode"],
        action_mode=ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL,
    )
    agent_crys = SACAgent(
        obs_dim=env_crys.obs_dim,
        act_dim=env_crys.num_blocks,
        batch_size=config["agent"]["batch_size"],
        init_temperature=config["agent"]["init_temperature"],
    )
    hist_crys = train_agent(
        env=env_crys,
        agent=agent_crys,
        num_episodes=num_episodes,
        batch_size=config["agent"]["batch_size"],
        eval_freq=config["training"]["eval_frequency"],
        checkpoint_dir=checkpoint_dir,
        model_name="sac_crystallised_residual",
    )
    benchmark_results["Crystallised + Residual (WATER)"] = hist_crys

    # Visualize and save final state from Crystallised + Residual
    obs, info = env_crys.reset(seed=42)
    for _ in range(env_crys.max_steps_per_episode):
        act = agent_crys.select_action(obs, evaluate=True)
        obs, _, _, _, info = env_crys.step(act)

    plot_surface_and_pressure_state(
        block_heights_mm=env_crys.current_blocks_mm,
        pressure_sensors_mmhg=env_crys.sensor_pressures_mmhg,
        crystallised_blocks_mm=env_crys.crystallised_target_mm,
        subject_label=f"{env_crys.body.params.subject_id} ({env_crys.body.params.weight_kg:.0f}kg, {env_crys.body.params.height_m:.2f}m)",
        posture=env_crys.body.posture,
        mode=env_crys.body.mode.value,
        save_path=os.path.join(figures_dir, "surface_pressure_state.png"),
    )

    # 2. Naive End-to-End RL
    print("\n[Benchmark 2/3] Training Naive End-to-End RL...")
    env_e2e = AdaptiveSurfaceEnv(
        max_steps_per_episode=config["training"]["max_steps_per_episode"],
        action_mode=ActionSpaceMode.END_TO_END,
    )
    agent_e2e = SACAgent(
        obs_dim=env_e2e.obs_dim,
        act_dim=env_e2e.num_blocks,
        batch_size=config["agent"]["batch_size"],
        init_temperature=config["agent"]["init_temperature"],
    )
    hist_e2e = train_agent(
        env=env_e2e,
        agent=agent_e2e,
        num_episodes=num_episodes,
        batch_size=config["agent"]["batch_size"],
        eval_freq=config["training"]["eval_frequency"],
        checkpoint_dir=checkpoint_dir,
        model_name="sac_end_to_end",
    )
    benchmark_results["End-to-End RL"] = hist_e2e

    # 3. Crystallised Baseline Heuristic (Zero-Learning)
    print("\n[Benchmark 3/3] Evaluating Crystallised Baseline Heuristic...")
    env_heur = AdaptiveSurfaceEnv(
        max_steps_per_episode=config["training"]["max_steps_per_episode"],
        action_mode=ActionSpaceMode.CRYSTALLISED_ONLY,
    )
    hist_heur = {
        "rewards": [],
        "comfort_rewards": [],
        "alignment_rewards": [],
        "energy_costs": [],
        "arousals": [],
        "peak_pressures": [],
        "pressure_variances": [],
        "eval_episodes": list(range(1, num_episodes + 1)),
        "eval_returns": [],
    }
    dummy_act = np.zeros(env_heur.num_blocks, dtype=np.float32)
    for ep in range(1, num_episodes + 1):
        obs, _ = env_heur.reset(seed=ep * 17)
        ep_r, ep_e, ep_a = 0.0, 0.0, 0
        last_info = None
        for _ in range(env_heur.max_steps_per_episode):
            obs, r, _, _, last_info = env_heur.step(dummy_act)
            ep_r += r
            rc = last_info["reward_components"]
            ep_e += abs(rc.energy_penalty)
            if last_info["is_aroused"]:
                ep_a += 1

        hist_heur["rewards"].append(float(ep_r))
        hist_heur["comfort_rewards"].append(float(rc.comfort_reward))
        hist_heur["alignment_rewards"].append(float(rc.alignment_reward))
        hist_heur["energy_costs"].append(float(ep_e))
        hist_heur["arousals"].append(int(ep_a))
        hist_heur["peak_pressures"].append(float(last_info["peak_pressure_mmhg"]))
        hist_heur["pressure_variances"].append(float(last_info["pressure_variance"]))
        hist_heur["eval_returns"].append(float(ep_r))

    benchmark_results["Crystallised Baseline (Heuristic)"] = hist_heur

    # Generate and save benchmark comparison plots
    plot_action_space_comparison(
        curves=benchmark_results,
        save_path=os.path.join(figures_dir, "action_space_comparison.png"),
    )

    # Save benchmark history to JSON
    out_json = os.path.join(figures_dir, "benchmark_curves.json")
    with open(out_json, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nBenchmark results saved to: {out_json}")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Adaptive Surface RL Training Pipeline")
    parser.add_argument("--config", type=str, default="experiments/config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes (overrides config)")
    parser.add_argument("--action_mode", type=str, default=None, help="Action mode: crystallised_plus_residual, end_to_end, crystallised_only")
    parser.add_argument("--compare_action_spaces", action="store_true", help="Run systematic comparison of action space designs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    num_episodes = args.episodes if args.episodes is not None else config["training"]["num_episodes"]

    if args.compare_action_spaces:
        run_action_space_benchmark(
            config=config,
            num_episodes=num_episodes,
            figures_dir=args.figures_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        mode = args.action_mode or config["action_space"]["mode"]
        env = AdaptiveSurfaceEnv(
            max_steps_per_episode=config["training"]["max_steps_per_episode"],
            action_mode=mode,
            seed=args.seed,
        )
        agent = SACAgent(
            obs_dim=env.obs_dim,
            act_dim=env.num_blocks,
            batch_size=config["agent"]["batch_size"],
            init_temperature=config["agent"]["init_temperature"],
        )
        train_agent(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            batch_size=config["agent"]["batch_size"],
            eval_freq=config["training"]["eval_frequency"],
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"sac_{mode}",
        )


if __name__ == "__main__":
    main()
