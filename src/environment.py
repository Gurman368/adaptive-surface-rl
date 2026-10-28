"""
environment.py: Gymnasium-Compatible Adaptive Surface Environment.

Models the 42-block robotic surface (7x6 grid) with Human-in-the-Loop:
- State Space: 42 block heights + 168 pressure sensors + 4 body features + 6 posture one-hot + 2 time features = 222-D
- Action Space: 42 continuous block target commands in [-1, +1]
- Biomechanical Plant: Nonlinear human contact mechanics & static gravity equilibrium
- Transmission: Non-backdrivable self-locking actuators (zero static holding power)
- Multi-rate feedback: 1 Hz hardware tick, 4e-4 Hz sleep micro-arousal return path
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Gymnasium compatibility with fallback
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    # Standalone lightweight Gymnasium drop-in if gymnasium is not installed
    HAS_GYMNASIUM = False
    
    class DummySpace:
        def __init__(self, shape: Tuple[int, ...], low: float, high: float, dtype=np.float32):
            self.shape = shape
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.dtype = dtype

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class spaces:
        Box = DummySpace
        Dict = dict

    class gym:
        class Env:
            pass

from .action_space import ActionSpaceDecomposer, ActionSpaceMode
from .body_model import BodyParameters, PhysiologicalMode, SimulatedHumanBody, POSTURE_NAMES
from .reward import MultiRateRewardCalculator, RewardComponents


class AdaptiveSurfaceEnv(gym.Env):
    """Reinforcement learning environment for adaptive robotic surface control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_rows: int = 7,
        grid_cols: int = 6,
        max_travel_mm: float = 250.0,
        min_travel_mm: float = 0.0,
        max_velocity_mm_s: float = 25.0,
        dt_seconds: float = 1.0,
        sensor_rows: int = 14,
        sensor_cols: int = 12,
        max_steps_per_episode: int = 50,
        action_mode: Union[ActionSpaceMode, str] = ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL,
        body_params: Optional[BodyParameters] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__()

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_blocks = grid_rows * grid_cols
        self.max_travel_mm = max_travel_mm
        self.min_travel_mm = min_travel_mm
        self.max_velocity_mm_s = max_velocity_mm_s
        self.dt_seconds = dt_seconds
        self.sensor_rows = sensor_rows
        self.sensor_cols = sensor_cols
        self.num_sensors = sensor_rows * sensor_cols
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0

        # Action space decomposer
        if isinstance(action_mode, str):
            action_mode = ActionSpaceMode(action_mode)
        self.decomposer = ActionSpaceDecomposer(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            min_travel_mm=min_travel_mm,
            max_travel_mm=max_travel_mm,
            nominal_travel_mm=max_travel_mm * 0.5,
            max_residual_mm=35.0,
            mode=action_mode,
        )

        # Reward calculator
        rw = reward_weights or {}
        self.reward_calculator = MultiRateRewardCalculator(
            weight_comfort=rw.get("weight_comfort", 1.2),
            weight_alignment=rw.get("weight_alignment", 0.8),
            weight_energy=rw.get("weight_energy", 0.04),
            weight_arousal=rw.get("weight_arousal", 3.0),
            weight_smoothness=rw.get("weight_smoothness", 0.15),
            capillary_threshold_mmhg=32.0,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )

        # RNG
        self._rng = np.random.RandomState(seed)

        # Body model
        self.body_params = body_params or BodyParameters()
        self.body = SimulatedHumanBody(
            params=self.body_params,
            surface_length_mm=2000.0,
            surface_width_mm=900.0,
            sensor_grid_shape=(sensor_rows, sensor_cols),
        )

        # State tracking:
        # Blocks: (42,) in [0, 250] mm
        self.current_blocks_mm = np.full(self.num_blocks, max_travel_mm * 0.5, dtype=np.float32)
        self.target_blocks_mm = np.copy(self.current_blocks_mm)
        self.prev_blocks_mm = np.copy(self.current_blocks_mm)
        self.crystallised_target_mm = np.copy(self.current_blocks_mm)
        self.learned_residual_mm = np.zeros(self.num_blocks, dtype=np.float32)

        # Sensors: (14, 12)
        self.sensor_pressures_mmhg = np.zeros((sensor_rows, sensor_cols), dtype=np.float32)
        self.peak_pressure_mmhg = 0.0
        self.equilibrium_z_com = 0.0

        # Sleep cycle timing: 8-hour sleep period (28,800 seconds)
        self.sleep_duration_s = 28800.0
        self.current_time_s = 0.0
        self.is_rem_phase = False
        self.total_episode_arousals = 0

        # Define Observation Space: 42 + 168 + 4 + 6 + 2 = 222 continuous dimensions
        self.obs_dim = self.num_blocks + self.num_sensors + 4 + len(POSTURE_NAMES) + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Define Action Space: 42 continuous actions in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_blocks,),
            dtype=np.float32,
        )

    def set_subject(self, body_params: BodyParameters) -> None:
        """Switch to a new human subject."""
        self.body_params = body_params
        self.body = SimulatedHumanBody(
            params=self.body_params,
            surface_length_mm=2000.0,
            surface_width_mm=900.0,
            sensor_grid_shape=(self.sensor_rows, self.sensor_cols),
        )

    def set_action_mode(self, mode: Union[ActionSpaceMode, str]) -> None:
        """Change action space decomposition mode."""
        if isinstance(mode, str):
            mode = ActionSpaceMode(mode)
        self.decomposer.mode = mode

    def _get_obs(self) -> np.ndarray:
        """Assemble the 222-dimensional observation vector."""
        # 1. Normalized block heights in [0, 1]
        blocks_norm = self.current_blocks_mm / self.max_travel_mm

        # 2. Normalized pressure readings (0 to 100 mmHg normalized to [0, 1])
        pressures_norm = np.clip(self.sensor_pressures_mmhg.flatten() / 100.0, 0.0, 2.0)

        # 3. Normalized body anthropometrics (4 features)
        body_features = self.body.params.to_feature_vector()

        # 4. Posture one-hot (6 features)
        posture_one_hot = self.body.get_posture_one_hot()

        # 5. Time and sleep phase (2 features)
        time_norm = np.clip(self.current_time_s / self.sleep_duration_s, 0.0, 1.0)
        time_features = np.array([time_norm, 1.0 if self.is_rem_phase else 0.0], dtype=np.float32)

        obs = np.concatenate([
            blocks_norm,
            pressures_norm,
            body_features,
            posture_one_hot,
            time_features,
        ]).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start of an episode."""
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self.current_step = 0
        self.total_episode_arousals = 0

        # Randomize initial sleep time or start at 0
        options = options or {}
        randomize_start = options.get("randomize_time", True)
        self.current_time_s = float(self._rng.uniform(0.0, 7200.0)) if randomize_start else 0.0

        # Select initial posture
        initial_posture = options.get("posture", None)
        if initial_posture is None:
            initial_posture = self._rng.choice(["supine", "left_lateral", "right_lateral"])
        self.body.set_posture(initial_posture)

        # Select physiological mode
        mode_str = options.get("mode", "asleep")
        if mode_str == "awake":
            self.body.set_mode(PhysiologicalMode.AWAKE)
        elif mode_str == "rem":
            self.body.set_mode(PhysiologicalMode.REM)
        else:
            self.body.set_mode(PhysiologicalMode.ASLEEP)

        # Compute crystallised baseline
        self.crystallised_target_mm = self.decomposer.compute_crystallised_action(
            self.body.params,
            self.body.posture,
        )

        # Initialize blocks: start either at nominal flat or crystallised baseline
        start_at_baseline = options.get("start_at_crystallised", True)
        if start_at_baseline:
            self.current_blocks_mm = np.copy(self.crystallised_target_mm)
        else:
            self.current_blocks_mm = np.full(self.num_blocks, self.max_travel_mm * 0.5, dtype=np.float32)

        self.prev_blocks_mm = np.copy(self.current_blocks_mm)
        self.target_blocks_mm = np.copy(self.current_blocks_mm)
        self.learned_residual_mm = np.zeros(self.num_blocks, dtype=np.float32)

        # Compute initial pressure equilibrium
        block_grid = self.current_blocks_mm.reshape((self.grid_rows, self.grid_cols))
        sensors, _, peak_p, z_com = self.body.compute_pressure_distribution(block_grid)
        self.sensor_pressures_mmhg = sensors
        self.peak_pressure_mmhg = peak_p
        self.equilibrium_z_com = z_com

        obs = self._get_obs()
        info = {
            "posture": self.body.posture,
            "mode": self.body.mode.value,
            "peak_pressure_mmhg": self.peak_pressure_mmhg,
            "subject_id": self.body.params.subject_id,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one control step (hardware tick dt = 1.0s)."""
        self.current_step += 1
        self.current_time_s += self.dt_seconds

        # Sleep cycle ultradian rhythm (90-minute REM cycles)
        cycle_minute = (self.current_time_s / 60.0) % 90.0
        self.is_rem_phase = 70.0 <= cycle_minute <= 90.0
        if self.body.mode != PhysiologicalMode.AWAKE:
            if self.is_rem_phase:
                self.body.set_mode(PhysiologicalMode.REM)
            else:
                self.body.set_mode(PhysiologicalMode.ASLEEP)

        # Decode action via Action Space Decomposer
        target_blocks, crystallised, residual = self.decomposer.decode_action(
            action,
            self.body.params,
            self.body.posture,
        )
        self.target_blocks_mm = target_blocks
        self.crystallised_target_mm = crystallised
        self.learned_residual_mm = residual

        # Actuator physical velocity limit
        # Max velocity: max_velocity_mm_s * dt_seconds per step
        max_delta = self.max_velocity_mm_s * self.dt_seconds
        displacement = np.clip(target_blocks - self.current_blocks_mm, -max_delta, max_delta)

        self.prev_blocks_mm = np.copy(self.current_blocks_mm)
        self.current_blocks_mm += displacement

        # Velocity and acceleration estimates for arousal detection
        block_velocities = np.abs(displacement) / self.dt_seconds
        max_velocity = float(np.max(block_velocities))
        # Approximate acceleration from velocity step
        max_accel = max_velocity / self.dt_seconds

        # Human-in-the-loop physiological response
        adjusted_posture, is_aroused = self.body.step_physiological_dynamics(
            dt_seconds=self.dt_seconds,
            surface_velocity_max=max_velocity,
            surface_accel_max=max_accel,
            current_time_s=self.current_time_s,
            rng=self._rng,
        )
        if is_aroused:
            self.total_episode_arousals += 1

        # Recompute physical contact pressure equilibrium
        block_grid = self.current_blocks_mm.reshape((self.grid_rows, self.grid_cols))
        sensors, _, peak_p, z_com = self.body.compute_pressure_distribution(block_grid)
        self.sensor_pressures_mmhg = sensors
        self.peak_pressure_mmhg = peak_p
        self.equilibrium_z_com = z_com

        # Evaluate multi-objective reward
        reward_decomp = self.reward_calculator.compute_reward(
            current_blocks_mm=self.current_blocks_mm,
            prev_blocks_mm=self.prev_blocks_mm,
            crystallised_target_mm=self.crystallised_target_mm,
            sensor_pressures_mmhg=self.sensor_pressures_mmhg,
            is_aroused=is_aroused,
        )

        terminated = False
        truncated = self.current_step >= self.max_steps_per_episode

        obs = self._get_obs()
        info = {
            "reward_components": reward_decomp,
            "posture": self.body.posture,
            "mode": self.body.mode.value,
            "is_aroused": is_aroused,
            "total_arousals": self.total_episode_arousals,
            "peak_pressure_mmhg": self.peak_pressure_mmhg,
            "pressure_variance": reward_decomp.pressure_variance,
            "crystallised_blocks": self.crystallised_target_mm,
            "learned_residual": self.learned_residual_mm,
            "target_blocks": self.target_blocks_mm,
        }

        return obs, reward_decomp.total_reward, terminated, truncated, info
