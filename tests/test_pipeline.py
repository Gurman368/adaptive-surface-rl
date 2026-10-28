"""
test_pipeline.py: Unit and Integration Tests for adaptive-surface-rl.
"""

import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.body_model import BodyParameters, SimulatedHumanBody, PhysiologicalMode
from src.action_space import ActionSpaceDecomposer, ActionSpaceMode
from src.reward import MultiRateRewardCalculator
from src.environment import AdaptiveSurfaceEnv
from src.agent import SACAgent


def test_body_model():
    print("Testing body_model...")
    params = BodyParameters(weight_kg=80.0, height_m=1.80)
    body = SimulatedHumanBody(params=params)
    assert body.params.bmi == 80.0 / (1.80 ** 2)

    blocks = np.full((7, 6), 125.0, dtype=np.float32)
    sensors, full_map, peak_p, z_com = body.compute_pressure_distribution(blocks)
    assert sensors.shape == (14, 12), f"Expected (14, 12), got {sensors.shape}"
    assert peak_p > 0.0, "Peak pressure should be positive"
    assert not np.isnan(peak_p), "Peak pressure contains NaN"
    print(f"  Passed! Peak pressure={peak_p:.2f} mmHg, z_com={z_com:.2f} mm")


def test_action_space():
    print("Testing action_space...")
    params = BodyParameters()
    decomposer = ActionSpaceDecomposer(mode=ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL)
    raw_action = np.zeros(42, dtype=np.float32)
    target, crys, res = decomposer.decode_action(raw_action, params, "supine")
    assert target.shape == (42,)
    assert np.allclose(target, crys), "With zero residual, target should equal crystallised"
    assert np.all(target >= 0.0) and np.all(target <= 250.0)
    print("  Passed! Crystallised action decomposition verified.")


def test_environment_step():
    print("Testing environment...")
    env = AdaptiveSurfaceEnv()
    obs, info = env.reset()
    assert obs.shape == (222,), f"Expected obs shape (222,), got {obs.shape}"

    action = np.zeros(42, dtype=np.float32)
    next_obs, reward, term, trunc, step_info = env.step(action)
    assert next_obs.shape == (222,)
    assert isinstance(reward, float)
    assert "reward_components" in step_info
    print(f"  Passed! Reward={reward:.3f}, Terminated={term}, Truncated={trunc}")


def test_agent_training_step():
    print("Testing SAC agent...")
    agent = SACAgent(obs_dim=222, act_dim=42, batch_size=4)
    obs = np.random.randn(222).astype(np.float32)
    action = agent.select_action(obs)
    assert action.shape == (42,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    for _ in range(10):
        s = np.random.randn(222).astype(np.float32)
        a = np.random.uniform(-1, 1, size=42).astype(np.float32)
        r = float(np.random.randn())
        s2 = np.random.randn(222).astype(np.float32)
        d = False
        agent.replay_buffer.add(s, a, r, s2, d)

    loss_dict = agent.update_parameters(batch_size=4)
    assert "critic_loss" in loss_dict
    assert "actor_loss" in loss_dict
    print(f"  Passed! Critic loss={loss_dict['critic_loss']:.4f}, Actor loss={loss_dict['actor_loss']:.4f}")


if __name__ == "__main__":
    test_body_model()
    test_action_space()
    test_environment_step()
    test_agent_training_step()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
