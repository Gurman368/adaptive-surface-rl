"""
body_model.py: Biomechanical Human Body Simulation on Adaptive Robotic Surfaces.

Models a human subject resting on an actuated multi-block surface:
- Anthropometric segment decomposition (Winter & Dempster mass distributions)
- Nonlinear viscoelastic contact mechanics (Hertzian tissue indentation)
- Static gravitational equilibrium solver
- Spinal curvature and neutral postural alignment models
- Three physiological regimes:
    * AWAKE: Active proprioceptive feedback loop, high-gain compensator
    * ASLEEP: Passive mode, low adjustment frequency (~4e-4 Hz), micro-arousal cost
    * REM: Motor stage inhibition (atonia), zero active compensation
- Capillary ischemia detection (32 mmHg threshold)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


class PhysiologicalMode(enum.Enum):
    """Physiological state of the human in the control loop."""
    AWAKE = "awake"
    ASLEEP = "asleep"
    REM = "rem"


POSTURE_NAMES = [
    "supine",
    "prone",
    "left_lateral",
    "right_lateral",
    "reclined_supine",
    "fetal",
]


@dataclass
class BodyParameters:
    """Anthropometric parameters defining a specific human subject.
    
    Attributes:
        weight_kg: Total body mass in kilograms.
        height_m: Stature in meters.
        shoulder_width_m: Biacromial breadth in meters.
        hip_width_m: Bi-iliocristal breadth in meters.
        tissue_stiffness_kpa_mm: Soft tissue compressive modulus (kPa / mm^1.5).
        subject_id: Optional string identifier.
    """
    weight_kg: float = 75.0
    height_m: float = 1.75
    shoulder_width_m: float = 0.44
    hip_width_m: float = 0.36
    tissue_stiffness_kpa_mm: float = 0.35
    subject_id: str = "nominal_subject"

    @property
    def bmi(self) -> float:
        """Body Mass Index (kg / m^2)."""
        return self.weight_kg / (self.height_m ** 2)

    def to_feature_vector(self) -> np.ndarray:
        """Normalized 4D anthropometric feature vector for neural network conditioning."""
        # Normalize against typical adult ranges:
        # Weight: [45, 120] kg -> [0, 1]
        # Height: [1.45, 2.00] m -> [0, 1]
        # Shoulder: [0.30, 0.55] m -> [0, 1]
        # Hip: [0.25, 0.48] m -> [0, 1]
        w_norm = (self.weight_kg - 45.0) / 75.0
        h_norm = (self.height_m - 1.45) / 0.55
        s_norm = (self.shoulder_width_m - 0.30) / 0.25
        hip_norm = (self.hip_width_m - 0.25) / 0.23
        return np.array([w_norm, h_norm, s_norm, hip_norm], dtype=np.float32)

    @staticmethod
    def sample_population(n_subjects: int = 5, rng: Optional[np.random.RandomState] = None) -> List[BodyParameters]:
        """Generate diverse anthropometric population spanning 5th to 95th percentiles."""
        if rng is None:
            rng = np.random.RandomState(42)

        subjects = []
        # Predefined archetypes: slender, average, heavy, tall-athletic, petite
        archetypes = [
            ("ectomorph_slender", 58.0, 1.78, 0.39, 0.31, 0.42),
            ("mesomorph_nominal", 74.0, 1.75, 0.44, 0.36, 0.35),
            ("endomorph_heavy", 98.0, 1.72, 0.48, 0.43, 0.28),
            ("tall_broad", 86.0, 1.90, 0.50, 0.38, 0.38),
            ("petite_light", 50.0, 1.56, 0.36, 0.33, 0.40),
            ("athletic_dense", 82.0, 1.82, 0.47, 0.35, 0.36),
            ("bariatric", 112.0, 1.70, 0.51, 0.46, 0.25),
        ]

        for i in range(n_subjects):
            name, w, h, sw, hw, stiff = archetypes[i % len(archetypes)]
            # Add small jitter for variety if generating more than base archetypes
            if i >= len(archetypes):
                w += rng.uniform(-4.0, 4.0)
                h += rng.uniform(-0.03, 0.03)
                sw += rng.uniform(-0.02, 0.02)
                hw += rng.uniform(-0.02, 0.02)
                stiff += rng.uniform(-0.03, 0.03)
                name = f"subject_{i}_{name}"
            subjects.append(
                BodyParameters(
                    weight_kg=float(np.clip(w, 45.0, 125.0)),
                    height_m=float(np.clip(h, 1.45, 2.05)),
                    shoulder_width_m=float(np.clip(sw, 0.30, 0.55)),
                    hip_width_m=float(np.clip(hw, 0.25, 0.50)),
                    tissue_stiffness_kpa_mm=float(np.clip(stiff, 0.20, 0.55)),
                    subject_id=name,
                )
            )
        return subjects


@dataclass
class BodySegment:
    """Kinematic segment of the human body."""
    name: str
    mass_fraction: float  # Fraction of total body mass (Dempster / Winter)
    length_fraction: float  # Fraction of total body stature
    width_fraction: float  # Relative lateral half-width (m)
    x_rel_center: float  # Longitudinal center along body axis [0, 1]
    y_rel_center: float  # Lateral center along body axis [-0.5, 0.5]
    thickness_mm: float  # Nominal sagittal thickness (mm)
    lordosis_offset_mm: float  # Natural spinal/anatomical offset from neutral (mm)


def get_standard_segments() -> List[BodySegment]:
    """Anthropometric segment decomposition following Winter (2009) & Dempster (1955)."""
    return [
        BodySegment("head_neck", 0.075, 0.14, 0.18, 0.08, 0.0, 140.0, -25.0),
        BodySegment("upper_thorax", 0.185, 0.18, 0.42, 0.24, 0.0, 180.0, 10.0),
        BodySegment("lumbar_abdomen", 0.125, 0.14, 0.34, 0.40, 0.0, 160.0, -35.0),
        BodySegment("pelvis_buttocks", 0.160, 0.16, 0.38, 0.55, 0.0, 190.0, 20.0),
        BodySegment("thighs", 0.210, 0.22, 0.35, 0.74, 0.0, 170.0, -10.0),
        BodySegment("shins_feet", 0.145, 0.24, 0.28, 0.92, 0.0, 130.0, 15.0),
        BodySegment("arms_left", 0.050, 0.35, 0.12, 0.38, -0.32, 110.0, -5.0),
        BodySegment("arms_right", 0.050, 0.35, 0.12, 0.38, 0.32, 110.0, -5.0),
    ]


class SimulatedHumanBody:
    """Continuous biomechanical human body model on a 7x6 actuated block surface.
    
    Coordinate System:
        x: Longitudinal axis along the bed length (0 to 2000 mm, head to feet)
        y: Lateral axis along the bed width (0 to 900 mm, left to right)
        z: Vertical displacement (0 to 250 mm travel)
    """

    def __init__(
        self,
        params: Optional[BodyParameters] = None,
        surface_length_mm: float = 2000.0,
        surface_width_mm: float = 900.0,
        sensor_grid_shape: Tuple[int, int] = (14, 12),
        sim_res: Tuple[int, int] = (56, 48),
    ):
        self.params = params or BodyParameters()
        self.surface_length_mm = surface_length_mm
        self.surface_width_mm = surface_width_mm
        self.sensor_rows, self.sensor_cols = sensor_grid_shape
        self.sim_rows, self.sim_cols = sim_res

        self.segments = get_standard_segments()
        self.posture = "supine"
        self.mode = PhysiologicalMode.AWAKE

        # Postural correction dynamics:
        # Awake: correction rate ~0.1 Hz, Asleep: ~0.0004 Hz, REM: 0.0 Hz
        self.last_correction_time_s = 0.0
        self.arousal_count = 0
        self.total_micro_arousals = 0

        # Precompute simulation coordinate grid
        self.x_grid = np.linspace(0, self.surface_length_mm, self.sim_rows)
        self.y_grid = np.linspace(0, self.surface_width_mm, self.sim_cols)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing="ij")

        # Cell area in mm^2 (for pressure/force integration)
        dx = self.surface_length_mm / (self.sim_rows - 1)
        dy = self.surface_width_mm / (self.sim_cols - 1)
        self.cell_area_mm2 = dx * dy

    def set_posture(self, posture: str) -> None:
        """Set the current posture from available postures."""
        if posture not in POSTURE_NAMES:
            raise ValueError(f"Unknown posture: {posture}. Must be one of {POSTURE_NAMES}")
        self.posture = posture

    def set_mode(self, mode: PhysiologicalMode) -> None:
        """Set the physiological mode (AWAKE, ASLEEP, REM)."""
        self.mode = mode

    def get_posture_one_hot(self) -> np.ndarray:
        """One-hot representation of current posture."""
        idx = POSTURE_NAMES.index(self.posture)
        one_hot = np.zeros(len(POSTURE_NAMES), dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def compute_body_profile(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the intrinsic 3D body resting contour, contact mask, and mass density.
        
        Returns:
            z_body_local: 2D array (sim_rows, sim_cols) of segment bottom height offsets (mm).
            contact_mask: 2D boolean array indicating locations where body tissue exists.
            mass_density: 2D array of mass distribution (kg / mm^2).
        """
        z_body_local = np.full((self.sim_rows, self.sim_cols), np.nan, dtype=np.float32)
        contact_mask = np.zeros((self.sim_rows, self.sim_cols), dtype=bool)
        mass_density = np.zeros((self.sim_rows, self.sim_cols), dtype=np.float32)

        # Scale segment dimensions to subject stature and widths
        body_len_mm = self.params.height_m * 1000.0
        shoulder_w_mm = self.params.shoulder_width_m * 1000.0
        hip_w_mm = self.params.hip_width_m * 1000.0

        # Posture adaptations
        effective_len = body_len_mm
        y_offset_sign = 1.0 if self.posture == "left_lateral" else (-1.0 if self.posture == "right_lateral" else 0.0)

        if self.posture == "fetal":
            effective_len *= 0.78  # Curled posture shortens longitudinal footprint
        elif self.posture == "reclined_supine":
            effective_len *= 0.94

        # Center longitudinal placement on the bed (margin at head)
        x_head_margin = max(50.0, (self.surface_length_mm - effective_len) * 0.35)
        y_center = self.surface_width_mm / 2.0

        for seg in self.segments:
            seg_mass = self.params.weight_kg * seg.mass_fraction
            seg_len = seg.length_fraction * effective_len

            # Lateral width scaled by shoulder or hip width
            if "thorax" in seg.name or "head" in seg.name or "neck" in seg.name:
                seg_half_w = (seg.width_fraction * shoulder_w_mm) / 2.0
            elif "pelvis" in seg.name or "lumbar" in seg.name:
                seg_half_w = (seg.width_fraction * hip_w_mm) / 2.0
            else:
                seg_half_w = (seg.width_fraction * hip_w_mm * 0.8) / 2.0

            # Lateral posture changes: contact is narrower, trochanter / acromion prominence
            if "lateral" in self.posture or self.posture == "fetal":
                seg_half_w *= 0.65  # Narrower contact strip when lying on side

            # Longitudinal center of segment
            x_seg_center = x_head_margin + seg.x_rel_center * effective_len
            y_seg_center = y_center + (seg.y_rel_center * shoulder_w_mm)

            # Lateral spine shift if lateral posture
            if "lateral" in self.posture:
                y_seg_center += y_offset_sign * 30.0

            # Elliptical contact footprint
            dist_sq = ((self.X - x_seg_center) / (seg_len / 2.0)) ** 2 + ((self.Y - y_seg_center) / seg_half_w) ** 2
            in_seg = dist_sq <= 1.0

            if not np.any(in_seg):
                continue

            # Sagittal profile offset:
            # Neutral curvature: Thoracic & sacral kyphosis are lower/prominent,
            # Lumbar lordosis and cervical recess curve upward (require support block elevation).
            offset_mm = seg.lordosis_offset_mm

            if "lateral" in self.posture or self.posture == "fetal":
                # On side, hip and shoulder bones project outward; waist indents strongly
                if "thorax" in seg.name or "pelvis" in seg.name:
                    offset_mm = 25.0  # High lateral projection
                elif "lumbar" in seg.name:
                    offset_mm = -40.0  # Deep lateral waist gap
                elif "head_neck" in seg.name:
                    offset_mm = -50.0  # Head needs thick pillow support on side

            elif self.posture == "prone":
                # Prone: invert spinal lordosis curves
                offset_mm = -offset_mm * 0.6

            elif self.posture == "reclined_supine":
                # Elevate head & upper thorax
                if "head" in seg.name:
                    offset_mm -= 60.0
                elif "upper_thorax" in seg.name:
                    offset_mm -= 30.0

            # Ellipsoidal bottom contour
            # The bottom curvature curves up towards the edges of the segment
            bottom_sagitta = (seg.thickness_mm * 0.35) * (1.0 - np.sqrt(np.clip(1.0 - dist_sq, 0.0, 1.0)))
            z_local = offset_mm + bottom_sagitta

            # Update master arrays
            mask_update = in_seg & (np.isnan(z_body_local) | (z_local < z_body_local))
            z_body_local[mask_update] = z_local[mask_update]
            contact_mask |= in_seg

            # Distribute mass uniformly across segment footprint
            n_cells = np.sum(in_seg)
            if n_cells > 0:
                mass_density[in_seg] += float(seg_mass / (n_cells * self.cell_area_mm2))

        # Default nan values to high clearance
        z_body_local[~contact_mask] = 1000.0
        return z_body_local, contact_mask, mass_density

    def compute_pressure_distribution(
        self,
        surface_profile_mm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Compute contact pressure distribution using nonlinear viscoelastic contact mechanics.
        
        Solves for gravitational equilibrium:
            F_ext = M_body * g
            F_contact(z_com) = sum_{cells} P(x, y) * dA = F_ext
        where:
            P(x, y) = k_tissue * max(0, delta(x, y))^(1.5)  [Hertzian contact]
            delta(x, y) = surface_profile(x, y) - (z_body_local(x, y) + z_com)
        
        Args:
            surface_profile_mm: 2D array (7, 6) or (sim_rows, sim_cols) of block heights.
            
        Returns:
            pressure_sensors_mmhg: (14, 12) array of tactile sensor pressures in mmHg.
            full_pressure_map_mmhg: (sim_rows, sim_cols) high-res pressure field in mmHg.
            peak_pressure_mmhg: Peak contact pressure across the surface.
            equilibrium_height_mm: Resolved body center-of-mass height.
        """
        # Interpolate surface profile from 7x6 block grid to continuous simulation grid
        if surface_profile_mm.shape != (self.sim_rows, self.sim_cols):
            from scipy.ndimage import zoom
            zoom_x = self.sim_rows / surface_profile_mm.shape[0]
            zoom_y = self.sim_cols / surface_profile_mm.shape[1]
            surface_fine = zoom(surface_profile_mm, (zoom_x, zoom_y), order=1)
        else:
            surface_fine = surface_profile_mm

        z_body_local, contact_mask, _ = self.compute_body_profile()

        # Gravitational force in Newtons
        g = 9.80665
        total_gravity_n = self.params.weight_kg * g

        # Soft tissue stiffness in Pa / m^1.5 converted to N / (mm^2 * mm^1.5)
        # 1 kPa = 1000 N / m^2 = 0.001 N / mm^2
        k_stiffness = self.params.tissue_stiffness_kpa_mm * 0.001

        # Binary search / root finding for vertical equilibrium height z_com
        # Higher z_com means body is higher -> less penetration -> less upward reaction force
        z_min = -150.0
        z_max = 350.0

        for _ in range(24):
            z_com = 0.5 * (z_min + z_max)
            # Indentation penetration: delta = z_surf - (z_body_local + z_com)
            indentation = np.maximum(0.0, surface_fine - (z_body_local + z_com))
            indentation[~contact_mask] = 0.0

            # Hertzian pressure: P = k * delta^(1.5) in N / mm^2
            p_n_mm2 = k_stiffness * (indentation ** 1.5)
            f_contact = np.sum(p_n_mm2) * self.cell_area_mm2

            if f_contact < total_gravity_n:
                # Need more penetration -> lower the body
                z_max = z_com
            else:
                z_min = z_com

        # Final contact pressure distribution
        z_com_equil = 0.5 * (z_min + z_max)
        indentation = np.maximum(0.0, surface_fine - (z_body_local + z_com_equil))
        indentation[~contact_mask] = 0.0

        # Convert N/mm^2 (MPa) to mmHg:
        # 1 N/mm^2 = 1,000,000 Pa = 1,000 kPa
        # 1 kPa = 7.50062 mmHg => 1 N/mm^2 = 7500.62 mmHg
        p_n_mm2 = k_stiffness * (indentation ** 1.5)
        pressure_map_mmhg = p_n_mm2 * 7500.62

        # Smooth high-res pressure with gaussian filter (models subcutaneous tissue stress dispersion)
        pressure_map_mmhg = gaussian_filter(pressure_map_mmhg, sigma=1.0)
        pressure_map_mmhg[~contact_mask] = 0.0

        # Subsample to (14, 12) sensor array (tactile sensor mat matching bed dimension)
        from scipy.ndimage import zoom
        zx = self.sensor_rows / self.sim_rows
        zy = self.sensor_cols / self.sim_cols
        sensor_array = zoom(pressure_map_mmhg, (zx, zy), order=1)
        sensor_array = np.clip(sensor_array, 0.0, 150.0).astype(np.float32)

        peak_pressure = float(np.max(pressure_map_mmhg))
        return sensor_array, pressure_map_mmhg, peak_pressure, z_com_equil

    def step_physiological_dynamics(
        self,
        dt_seconds: float,
        surface_velocity_max: float,
        surface_accel_max: float,
        current_time_s: float,
        rng: np.random.RandomState,
    ) -> Tuple[bool, bool]:
        """Simulate human-in-the-loop proprioception and sleep state dynamics.
        
        A person is simultaneously:
            1. The PLANT (their biomechanical mass deforms the mattress)
            2. The SENSOR (cutaneous mechanoreceptors & vestibular system detect motion & ischemia)
            3. The COMPENSATOR (proprioceptive muscle tone & postural shifting)
            
        Returns:
            has_adjusted_posture: Whether the human actively shifted posture.
            has_aroused: Whether a sleep micro-arousal event was triggered.
        """
        has_adjusted_posture = False
        has_aroused = False

        if self.mode == PhysiologicalMode.AWAKE:
            # Active conscious compensation: high gain (~0.1 Hz adjustment rate)
            prob_posture_shift = 1.0 - np.exp(-0.015 * dt_seconds)
            if rng.uniform(0.0, 1.0) < prob_posture_shift:
                # Stochastic posture adjustment
                choices = [p for p in POSTURE_NAMES if p != self.posture]
                self.posture = rng.choice(choices)
                has_adjusted_posture = True

        elif self.mode == PhysiologicalMode.ASLEEP:
            # Sleeping subject: low natural adjustment frequency (4e-4 Hz, ~once per 2500 s)
            natural_shift_prob = 1.0 - np.exp(-0.0004 * dt_seconds)
            if rng.uniform(0.0, 1.0) < natural_shift_prob:
                choices = [p for p in POSTURE_NAMES if p != self.posture]
                self.posture = rng.choice(choices)
                has_adjusted_posture = True
                has_aroused = True  # Natural shift accompanied by micro-arousal

            # Mechanoreceptor / vestibular arousal trigger:
            # If surface acceleration or velocity exceeds perceptual comfort thresholds,
            # an involuntary micro-arousal is induced!
            if surface_accel_max > 30.0 or surface_velocity_max > 18.0:
                has_aroused = True
                self.total_micro_arousals += 1
                # If disturbance is severe, provoke an involuntary posture correction
                if rng.uniform(0.0, 1.0) < 0.25:
                    choices = [p for p in POSTURE_NAMES if p != self.posture]
                    self.posture = rng.choice(choices)
                    has_adjusted_posture = True

        elif self.mode == PhysiologicalMode.REM:
            # REM sleep: motor stage inhibition (muscular atonia)
            # Volitional adjustment rate drops to ZERO.
            # Strong acceleration still causes micro-arousal (waking up or dropping out of REM)
            if surface_accel_max > 45.0 or surface_velocity_max > 22.0:
                has_aroused = True
                self.total_micro_arousals += 1

        return has_adjusted_posture, has_aroused
