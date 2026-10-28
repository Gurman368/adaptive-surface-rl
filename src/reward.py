"""
reward.py: Multi-Rate Multi-Objective Reward Function for Adaptive Surfaces.

Encapsulates WATER's core physical control philosophy:
1. Comfort: Minimizing pressure variance across cutaneous contact areas and penalizing
   peak pressures exceeding the capillary closing threshold (32 mmHg / 4.27 kPa).
2. Alignment: Penalizing deviation from neutral spinal biomechanics.
3. Energy: Work done during mechanical transition. Holding position costs ZERO energy
   due to non-backdrivable, self-locking lead-screw transmissions.
4. Arousal Penalty: Steep negative reward when rapid surface actuation induces tactile/vestibular
   micro-arousals during sleep, disrupting sleep architecture.
5. Smoothness: Penalizing spatial shear gradients across neighboring actuating blocks.
6. Multi-Rate Structure: Fast sensory feedback (1 Hz) coupled with slow sleep return paths (4e-4 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RewardComponents:
    """Individual breakdown of multi-objective reward terms."""
    total_reward: float
    comfort_reward: float
    alignment_reward: float
    energy_penalty: float
    arousal_penalty: float
    smoothness_penalty: float
    peak_pressure_mmhg: float
    pressure_variance: float
    mean_displacement_mm: float
    is_aroused: bool


class MultiRateRewardCalculator:
    """Computes and tracks multi-objective rewards across fast and slow return paths."""

    def __init__(
        self,
        weight_comfort: float = 1.2,
        weight_alignment: float = 0.8,
        weight_energy: float = 0.04,
        weight_arousal: float = 3.0,
        weight_smoothness: float = 0.15,
        capillary_threshold_mmhg: float = 32.0,
        grid_rows: int = 7,
        grid_cols: int = 6,
    ):
        self.w_comfort = weight_comfort
        self.w_alignment = weight_alignment
        self.w_energy = weight_energy
        self.w_arousal = weight_arousal
        self.w_smoothness = weight_smoothness

        self.capillary_threshold_mmhg = capillary_threshold_mmhg
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_blocks = grid_rows * grid_cols

        # Precompute block adjacency graph for smoothness regularization
        self._adj_pairs = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if r + 1 < grid_rows:
                    self._adj_pairs.append((idx, (r + 1) * grid_cols + c))
                if c + 1 < grid_cols:
                    self._adj_pairs.append((idx, r * grid_cols + (c + 1)))

    def compute_reward(
        self,
        current_blocks_mm: np.ndarray,
        prev_blocks_mm: np.ndarray,
        crystallised_target_mm: np.ndarray,
        sensor_pressures_mmhg: np.ndarray,
        is_aroused: bool,
    ) -> RewardComponents:
        """Compute the composite reward and decomposed components.
        
        Args:
            current_blocks_mm: (42,) array of current block heights (0 - 250 mm).
            prev_blocks_mm: (42,) array of previous block heights (0 - 250 mm).
            crystallised_target_mm: (42,) precomputed analytical gravity baseline.
            sensor_pressures_mmhg: (14, 12) tactile pressure sensor readings.
            is_aroused: Boolean flag if micro-arousal occurred.
            
        Returns:
            RewardComponents with all terms decomposed.
        """
        # 1. Comfort: pressure uniformity on contact cells
        contact_mask = sensor_pressures_mmhg > 1.0  # Threshold to isolate active contact
        if np.any(contact_mask):
            contact_pressures = sensor_pressures_mmhg[contact_mask]
            pressure_var = float(np.var(contact_pressures))
            peak_pressure = float(np.max(contact_pressures))
            
            # Variance penalty normalized (target variance < 25 mmHg^2)
            var_term = -min(pressure_var / 50.0, 5.0)

            # Capillary closure penalty: severe non-linear ischemia penalty above 32 mmHg
            excess_peak = max(0.0, peak_pressure - self.capillary_threshold_mmhg)
            ischemia_term = - (excess_peak / 10.0) ** 1.5
            
            # Mean contact support bonus (reward sustaining distributed body weight)
            support_term = min(float(np.mean(contact_pressures)) / 20.0, 1.0)
            comfort_reward = var_term + ischemia_term + support_term
        else:
            # Body not contacting surface properly
            pressure_var = 100.0
            peak_pressure = 0.0
            comfort_reward = -5.0

        # 2. Alignment: deviation from neutral biomechanical contour
        # The crystallised baseline represents the neutral sagittal curve
        delta_align = (current_blocks_mm - crystallised_target_mm) / 50.0  # Normalized scale
        alignment_penalty = -float(np.mean(delta_align ** 2))

        # 3. Energy: Mechanical transition work
        # Self-locking transmission: Holding position costs EXACTLY ZERO energy!
        # Only changing height costs energy (motor torque overcoming gravity and friction)
        delta_disp = np.abs(current_blocks_mm - prev_blocks_mm)
        mean_disp = float(np.mean(delta_disp))
        energy_penalty = -float(np.mean((delta_disp / 20.0) ** 2))

        # 4. Arousal Penalty (Sleep mode)
        # Steep negative reward if actuator movement startled vestibular/mechanoreceptors
        arousal_term = -5.0 if is_aroused else 0.0

        # 5. Smoothness / Shear penalty between adjacent actuated blocks
        # Large vertical steps between neighboring blocks create shear strain in human tissue
        diffs = []
        for i1, i2 in self._adj_pairs:
            diffs.append((current_blocks_mm[i1] - current_blocks_mm[i2]) / 25.0)
        smoothness_penalty = -float(np.mean(np.array(diffs) ** 2)) if diffs else 0.0

        # Total multi-objective weighted sum
        total = (
            self.w_comfort * comfort_reward
            + self.w_alignment * alignment_penalty
            + self.w_energy * energy_penalty
            + self.w_arousal * arousal_term
            + self.w_smoothness * smoothness_penalty
        )

        return RewardComponents(
            total_reward=float(total),
            comfort_reward=float(comfort_reward),
            alignment_reward=float(alignment_penalty),
            energy_penalty=float(energy_penalty),
            arousal_penalty=float(arousal_term),
            smoothness_penalty=float(smoothness_penalty),
            peak_pressure_mmhg=peak_pressure,
            pressure_variance=pressure_var,
            mean_displacement_mm=mean_disp,
            is_aroused=is_aroused,
        )
