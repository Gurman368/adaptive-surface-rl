"""
action_space.py: Action Space Engineering for Intelligent Adaptive Surfaces.

Implements WATER's core design doctrine:
"Action space engineering: the seam between mechanism and model is something to DESIGN."

Decomposes the 42-dimensional actuation into:
1. CRYSTALLISED COMPONENT (Mechanism):
   - Precomputed, analytical gravity compensation profile derived from body anthropometry
     and posture class.
   - Zero compute at runtime, provably stable, inherently satisfies kinematic safety bounds.
   - Holds position under zero electric power due to self-locking transmissions.
   
2. LEARNED COMPONENT (Model):
   - Bounded residual delta predicted by the RL policy: Delta a in [-Delta_max, +Delta_max].
   - Compensates for individual soft-tissue compliance, local mattress hysteresis,
     and fine-grained interface pressure hotspots.
   - Costs energy during mechanical transition.

3. FINAL ACTION:
   a_final = clip(a_crystallised + a_residual, z_min, z_max)
"""

from __future__ import annotations

import enum
from typing import Optional, Tuple, Union

import numpy as np

from .body_model import BodyParameters, POSTURE_NAMES


class ActionSpaceMode(str, enum.Enum):
    """Action space representation mode."""
    CRYSTALLISED_PLUS_RESIDUAL = "crystallised_plus_residual"  # WATER's hybrid architecture
    END_TO_END = "end_to_end"                                  # Naive RL learning full 42-D profile
    CRYSTALLISED_ONLY = "crystallised_only"                    # Pure analytical heuristic (no RL)


class ActionSpaceDecomposer:
    """Manages mapping between policy action outputs and physical block travel heights."""

    def __init__(
        self,
        grid_rows: int = 7,
        grid_cols: int = 6,
        min_travel_mm: float = 0.0,
        max_travel_mm: float = 250.0,
        nominal_travel_mm: float = 125.0,
        max_residual_mm: float = 35.0,
        mode: ActionSpaceMode = ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_blocks = grid_rows * grid_cols
        self.min_travel_mm = min_travel_mm
        self.max_travel_mm = max_travel_mm
        self.nominal_travel_mm = nominal_travel_mm
        self.max_residual_mm = max_residual_mm
        self.mode = ActionSpaceMode(mode)

    def compute_crystallised_action(
        self,
        body_params: BodyParameters,
        posture: str,
    ) -> np.ndarray:
        """Compute analytical, physics-grounded baseline profile (Mechanism).
        
        This profile encodes:
        - Natural spinal lordosis (elevated lumbar) and kyphosis (lowered thoracic/sacral)
        - Lateral posture trochanter and acromion bone sinking with lateral waist support
        - Heel pressure relief (lowering foot blocks to eliminate decubitus ulcer risk)
        - Weight-dependent sinkage compensation (crystallising gravity)
        
        Returns:
            (42,) array of block heights in mm.
        """
        profile = np.full((self.grid_rows, self.grid_cols), self.nominal_travel_mm, dtype=np.float32)

        # Sinkage offset due to subject weight: heavier subjects sink deeper into soft topper
        # Gravity compensation: heavier body needs higher baseline support
        weight_offset = (body_params.weight_kg - 75.0) * 0.45  # mm

        if posture == "supine":
            # Row 0: Head / pillow support
            profile[0, :] = 135.0
            # Row 1: Cervical spine lordosis arch (requires upward support)
            profile[1, :] = 145.0
            # Row 2: Thoracic kyphosis (upper back needs to sink down)
            profile[2, :] = 115.0
            # Row 3: Lumbar lordosis (lower back arch must be supported upwards)
            profile[3, :] = 155.0
            # Row 4: Sacrum / Pelvis (major mass concentration, needs deeper recess)
            profile[4, :] = 100.0
            # Row 5: Thighs and popliteal fossa (knees slightly elevated for circulation)
            profile[5, :] = 130.0
            # Row 6: Calves and heels (heels offloaded to prevent calcaneal pressure ulcers)
            profile[6, :] = 110.0

            # Lateral cradle: slightly elevate outer columns (cols 0 and 5) by 10mm for lateral stability
            profile[:, 0] += 12.0
            profile[:, -1] += 12.0

        elif posture == "prone":
            # On stomach: abdominal support, thoracic elevation, feet relaxed
            profile[0, :] = 110.0  # Head turned flat
            profile[1, :] = 120.0
            profile[2, :] = 130.0
            profile[3, :] = 125.0
            profile[4, :] = 115.0
            profile[5, :] = 125.0
            profile[6, :] = 120.0

        elif "lateral" in posture:
            # Side sleeping: wide shoulder and hip bony prominences!
            # Head needs high pillow support to keep cervical spine horizontal
            profile[0, :] = 160.0
            profile[1, :] = 155.0
            # Thorax / shoulder (acromion sinks deep)
            profile[2, :] = 95.0
            # Waist / lumbar indentation (deep upward support required to prevent spinal sagging)
            profile[3, :] = 165.0
            # Pelvis / greater trochanter (bone sinks deep)
            profile[4, :] = 95.0
            # Thighs
            profile[5, :] = 125.0
            # Lower legs / feet
            profile[6, :] = 120.0

        elif posture == "reclined_supine":
            # Fowler position: elevated head and torso
            profile[0, :] = 180.0
            profile[1, :] = 170.0
            profile[2, :] = 145.0
            profile[3, :] = 130.0
            profile[4, :] = 105.0
            profile[5, :] = 135.0
            profile[6, :] = 115.0

        elif posture == "fetal":
            # Curled lateral position
            profile[0, :] = 155.0
            profile[1, :] = 145.0
            profile[2, :] = 100.0
            profile[3, :] = 155.0
            profile[4, :] = 105.0
            profile[5, :] = 135.0
            profile[6, :] = 130.0

        # Apply gravity compensation shift
        profile += weight_offset

        # Clip within travel envelope [0, 250] mm
        return np.clip(profile.flatten(), self.min_travel_mm, self.max_travel_mm).astype(np.float32)

    def decode_action(
        self,
        raw_agent_action: np.ndarray,
        body_params: BodyParameters,
        posture: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw normalized agent action in [-1, 1]^42 into physical block target heights.
        
        Args:
            raw_agent_action: (42,) policy action output in [-1, 1].
            body_params: Anthropometrics of current subject.
            posture: Posture string name.
            
        Returns:
            target_profile_mm: (42,) final target block positions in [0, 250] mm.
            crystallised_mm: (42,) analytical baseline.
            learned_residual_mm: (42,) learned residual component.
        """
        raw_agent_action = np.clip(raw_agent_action, -1.0, 1.0).astype(np.float32)
        crystallised_mm = self.compute_crystallised_action(body_params, posture)

        if self.mode == ActionSpaceMode.CRYSTALLISED_PLUS_RESIDUAL:
            # Learned component is a bounded residual: Delta a in [-max_residual, +max_residual]
            learned_residual_mm = raw_agent_action * self.max_residual_mm
            target_profile_mm = np.clip(
                crystallised_mm + learned_residual_mm,
                self.min_travel_mm,
                self.max_travel_mm,
            )

        elif self.mode == ActionSpaceMode.END_TO_END:
            # Naive RL: agent directly maps [-1, 1] to the full travel envelope [0, 250] mm
            # No crystallised mechanism prior!
            target_profile_mm = (
                (raw_agent_action + 1.0) * 0.5 * (self.max_travel_mm - self.min_travel_mm)
                + self.min_travel_mm
            )
            learned_residual_mm = target_profile_mm - crystallised_mm

        elif self.mode == ActionSpaceMode.CRYSTALLISED_ONLY:
            # Zero-shot physics heuristic: residual is forced to zero
            learned_residual_mm = np.zeros_like(crystallised_mm)
            target_profile_mm = crystallised_mm

        else:
            raise ValueError(f"Unknown action space mode: {self.mode}")

        return target_profile_mm, crystallised_mm, learned_residual_mm
