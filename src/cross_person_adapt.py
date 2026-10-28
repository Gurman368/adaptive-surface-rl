"""
cross_person_adapt.py: Few-Shot Adaptation across Human Anthropometries.

Evaluates how the Action Space Decomposition (Crystallised Mechanism + Learned Residual)
generalizes across diverse human bodies:
- Pre-trains on N-1 subjects (diverse BMI, heights, and mass distributions).
- Evaluates rapid adaptation on an unseen N-th test subject.
- Compares four foundational control strategies:
    1. Zero-shot Crystallised Baseline: Pure analytical physics (0 episodes, verifiable safety).
    2. Crystallised + Residual: Adapts in 5-10 episodes.
    3. Fine-tuned End-to-End RL: Pre-trained without crystallised prior, fine-tuned.
    4. Scratch End-to-End RL: Trained entirely from scratch on the target subject.

Demonstrates that the crystallised component transfers instantly because gravity and skeletal
kinematics are invariant physics, while the residual converges in minimal sample transitions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from .action_space import ActionSpaceMode
from .agent import SACAgent
from .body_model import BodyParameters, PhysiologicalMode
from .environment import AdaptiveSurfaceEnv


def evaluate_policy_on_subject(
    env: AdaptiveSurfaceEnv,
    agent: Optional[SACAgent],
    subject: BodyParameters,
    n_episodes: int = 5,
    action_mode: ActionSpaceMode = ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL,
) -> Dict[str, float]:
    """Evaluate a policy on a specific subject over multiple episodes."""
    env.set_subject(subject)
    env.set_action_mode(action_mode)

    total_rewards = []
    comfort_rewards = []
    peak_pressures = []
    pressure_vars = []
    arousals = []
    energies = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 100 + 7, options={"mode": "asleep", "start_at_crystallised": True})
        done = False
        ep_reward = 0.0
        ep_arousals = 0
        ep_energy = 0.0

        while not done:
            if agent is not None and action_mode != ActionSpaceMode.CRYSTALLISED_ONLY:
                action = agent.select_action(obs, evaluate=True)
            else:
                action = np.zeros(env.num_blocks, dtype=np.float32)

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            rc = step_info["reward_components"]
            ep_energy += abs(rc.energy_penalty)
            if step_info["is_aroused"]:
                ep_arousals += 1

        total_rewards.append(ep_reward)
        comfort_rewards.append(rc.comfort_reward)
        peak_pressures.append(step_info["peak_pressure_mmhg"])
        pressure_vars.append(rc.pressure_variance)
        arousals.append(ep_arousals)
        energies.append(ep_energy)

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_comfort": float(np.mean(comfort_rewards)),
        "mean_peak_pressure_mmhg": float(np.mean(peak_pressures)),
        "mean_pressure_var": float(np.mean(pressure_vars)),
        "mean_arousals": float(np.mean(arousals)),
        "mean_energy_cost": float(np.mean(energies)),
    }


def adapt_agent_to_subject(
    env: AdaptiveSurfaceEnv,
    agent: SACAgent,
    subject: BodyParameters,
    n_episodes: int = 15,
    batch_size: int = 64,
    action_mode: ActionSpaceMode = ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL,
) -> List[Dict[str, float]]:
    """Adapt agent to target subject episode-by-episode, recording adaptation trajectory."""
    env.set_subject(subject)
    env.set_action_mode(action_mode)

    adaptation_history = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 42, options={"mode": "asleep", "start_at_crystallised": True})
        done = False
        ep_reward = 0.0
        ep_energy = 0.0

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

            # Continuous SAC parameter update
            if len(agent.replay_buffer) >= batch_size:
                agent.update_parameters(batch_size=batch_size)

        # Post-episode evaluation in deterministic mode
        eval_metrics = evaluate_policy_on_subject(
            env=env,
            agent=agent,
            subject=subject,
            n_episodes=2,
            action_mode=action_mode,
        )
        eval_metrics["episode"] = ep + 1
        adaptation_history.append(eval_metrics)

    return adaptation_history


def run_cross_person_experiment(
    config_path: str = "experiments/config.yaml",
    n_train_subjects: int = 4,
    n_adapt_episodes: int = 15,
    output_dir: str = "figures",
) -> Dict[str, Any]:
    """Execute complete cross-person adaptation experiment."""
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Generate diverse subject population
    population = BodyParameters.sample_population(n_train_subjects + 1)
    train_subjects = population[:n_train_subjects]
    target_subject = population[-1]

    print(f"\n================ CROSS-PERSON ADAPTATION EXPERIMENT ================")
    print(f"Training Population ({len(train_subjects)} subjects):")
    for s in train_subjects:
        print(f"  * {s.subject_id}: weight={s.weight_kg:.1f}kg, height={s.height_m:.2f}m, BMI={s.bmi:.1f}")
    print(f"Target Held-Out Subject:")
    print(f"  * {target_subject.subject_id}: weight={target_subject.weight_kg:.1f}kg, height={target_subject.height_m:.2f}m, BMI={target_subject.bmi:.1f}\n")

    # 2. Setup environments
    env_crys_res = AdaptiveSurfaceEnv(
        max_steps_per_episode=config["training"]["max_steps_per_episode"],
        action_mode=ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL,
    )
    env_e2e = AdaptiveSurfaceEnv(
        max_steps_per_episode=config["training"]["max_steps_per_episode"],
        action_mode=ActionSpaceMode.END_TO_END,
    )

    # 3. Create Agents
    agent_crys_res = SACAgent(
        obs_dim=env_crys_res.obs_dim,
        act_dim=env_crys_res.num_blocks,
        batch_size=64,
        init_temperature=0.2,
    )
    agent_e2e_pretrained = SACAgent(
        obs_dim=env_e2e.obs_dim,
        act_dim=env_e2e.num_blocks,
        batch_size=64,
        init_temperature=0.2,
    )
    agent_e2e_scratch = SACAgent(
        obs_dim=env_e2e.obs_dim,
        act_dim=env_e2e.num_blocks,
        batch_size=64,
        init_temperature=0.2,
    )

    # 4. Multi-subject pre-training (quick warm-up across N-1 subjects)
    print("Pre-training agents across training cohort (N-1 subjects)...")
    pretrain_episodes_per_subject = 8
    for subj in train_subjects:
        # Pre-train Crystallised + Residual Agent
        env_crys_res.set_subject(subj)
        for ep in range(pretrain_episodes_per_subject):
            obs, _ = env_crys_res.reset()
            done = False
            while not done:
                act = agent_crys_res.select_action(obs)
                next_obs, rew, term, trunc, _ = env_crys_res.step(act)
                done = term or trunc
                agent_crys_res.replay_buffer.add(obs, act, rew, next_obs, done)
                obs = next_obs
                if len(agent_crys_res.replay_buffer) >= 64:
                    agent_crys_res.update_parameters(64)

        # Pre-train End-to-End Agent
        env_e2e.set_subject(subj)
        for ep in range(pretrain_episodes_per_subject):
            obs, _ = env_e2e.reset()
            done = False
            while not done:
                act = agent_e2e_pretrained.select_action(obs)
                next_obs, rew, term, trunc, _ = env_e2e.step(act)
                done = term or trunc
                agent_e2e_pretrained.replay_buffer.add(obs, act, rew, next_obs, done)
                obs = next_obs
                if len(agent_e2e_pretrained.replay_buffer) >= 64:
                    agent_e2e_pretrained.update_parameters(64)

    # 5. Zero-Shot Baseline Evaluation (Crystallised only)
    zero_shot_crystallised = evaluate_policy_on_subject(
        env=env_crys_res,
        agent=None,
        subject=target_subject,
        n_episodes=5,
        action_mode=ActionSpaceMode.CRYSTALLISED_ONLY,
    )
    print(f"Zero-Shot Crystallised Baseline Reward on Target: {zero_shot_crystallised['mean_reward']:.2f}")

    # 6. Adaptation Trajectory: Crystallised + Residual Architecture
    print(f"Adapting Crystallised + Residual Agent on target ({n_adapt_episodes} episodes)...")
    history_crys_res = adapt_agent_to_subject(
        env=env_crys_res,
        agent=agent_crys_res,
        subject=target_subject,
        n_episodes=n_adapt_episodes,
        action_mode=ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL,
    )

    # 7. Adaptation Trajectory: Pre-trained End-to-End Fine-Tuning
    print(f"Adapting Pre-trained End-to-End Agent on target ({n_adapt_episodes} episodes)...")
    history_e2e_pretrained = adapt_agent_to_subject(
        env=env_e2e,
        agent=agent_e2e_pretrained,
        subject=target_subject,
        n_episodes=n_adapt_episodes,
        action_mode=ActionSpaceMode.END_TO_END,
    )

    # 8. Adaptation Trajectory: End-to-End Trained from Scratch
    print(f"Training Scratch End-to-End Agent on target ({n_adapt_episodes} episodes)...")
    history_e2e_scratch = adapt_agent_to_subject(
        env=env_e2e,
        agent=agent_e2e_scratch,
        subject=target_subject,
        n_episodes=n_adapt_episodes,
        action_mode=ActionSpaceMode.END_TO_END,
    )

    results = {
        "target_subject": {
            "id": target_subject.subject_id,
            "weight_kg": target_subject.weight_kg,
            "height_m": target_subject.height_m,
            "bmi": target_subject.bmi,
        },
        "zero_shot_crystallised": zero_shot_crystallised,
        "history_crystallised_plus_residual": history_crys_res,
        "history_e2e_pretrained": history_e2e_pretrained,
        "history_e2e_scratch": history_e2e_scratch,
    }

    out_json = os.path.join(output_dir, "cross_person_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Cross-person adaptation experiment results saved to: {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--n_adapt_episodes", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    run_cross_person_experiment(
        config_path=args.config,
        n_adapt_episodes=args.n_adapt_episodes,
        output_dir=args.output_dir,
    )
