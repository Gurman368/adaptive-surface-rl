"""
visualise.py: Publication-Quality Visualization Suite for Adaptive Surfaces.

Generates:
1. 3D Robotic Surface Profile (7x6 actuated grid) with elevation contours.
2. Tactile Interface Pressure Heatmap (14x12 sensor grid) with capillary ischemia threshold.
3. Sagittal Spine Alignment Cross-Section (block profile vs neutral curvature).
4. Multi-rate Reward Temporal Breakdown (Fast tick vs slow arousal return path).
5. Cross-Person Few-Shot Adaptation Learning Curves (WATER architecture vs baselines).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

# Set scientific plotting aesthetics
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
})


def plot_surface_and_pressure_state(
    block_heights_mm: np.ndarray,
    pressure_sensors_mmhg: np.ndarray,
    crystallised_blocks_mm: Optional[np.ndarray] = None,
    subject_label: str = "Subject (75kg, 1.75m)",
    posture: str = "supine",
    mode: str = "asleep",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate comprehensive 3-panel visualization of physical state.
    
    Panel 1: 3D perspective surface topography of the 42 actuated blocks.
    Panel 2: Cutaneous interface pressure distribution (14x12 sensor mat).
    Panel 3: Midline sagittal profile comparing current blocks vs crystallised baseline.
    """
    fig = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.0, 1.0])

    # ---------------- Panel 1: 3D Actuated Block Profile ----------------
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    grid_rows, grid_cols = 7, 6
    blocks_2d = block_heights_mm.reshape((grid_rows, grid_cols))

    X_b = np.linspace(0, 2000, grid_rows)
    Y_b = np.linspace(0, 900, grid_cols)
    X_mesh, Y_mesh = np.meshgrid(X_b, Y_b, indexing="ij")

    surf = ax1.plot_surface(
        X_mesh,
        Y_mesh,
        blocks_2d,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.5,
        alpha=0.88,
    )
    ax1.set_title(f"42-Block Actuated Surface Topography\n({posture.capitalize()} | {mode.upper()})", pad=12)
    ax1.set_xlabel("Longitudinal X (mm)", labelpad=8)
    ax1.set_ylabel("Lateral Y (mm)", labelpad=8)
    ax1.set_zlabel("Travel Z (mm)", labelpad=8)
    ax1.set_zlim(0, 250)
    ax1.view_init(elev=28, azim=-55)

    # ---------------- Panel 2: Tactile Interface Pressure Heatmap ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(
        pressure_sensors_mmhg,
        cmap="magma",
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=max(45.0, float(np.max(pressure_sensors_mmhg))),
    )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Contact Pressure (mmHg)", rotation=270, labelpad=15)
    
    # Capillary ischemia threshold marker
    cbar.ax.axhline(32.0, color="cyan", linestyle="--", linewidth=2.0)
    ax2.text(
        0.5, 0.04,
        "Cyan dashed: 32 mmHg Ischemia Threshold",
        transform=ax2.transAxes,
        color="cyan",
        fontsize=9,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
    )

    ax2.set_title(f"Tactile Interface Pressure (14×12 Grid)\n{subject_label}")
    ax2.set_xlabel("Lateral Sensor Column")
    ax2.set_ylabel("Longitudinal Sensor Row (Head -> Feet)")

    # ---------------- Panel 3: Midline Sagittal Profile ----------------
    ax3 = fig.add_subplot(gs[0, 2])
    # Midline column average (cols 2 and 3)
    midline_curr = np.mean(blocks_2d[:, 2:4], axis=1)
    x_positions = np.linspace(150, 1850, grid_rows)

    ax3.plot(x_positions, midline_curr, "o-", color="#1f77b4", linewidth=2.5, label="Active Surface Profile")
    if crystallised_blocks_mm is not None:
        crys_2d = crystallised_blocks_mm.reshape((grid_rows, grid_cols))
        midline_crys = np.mean(crys_2d[:, 2:4], axis=1)
        ax3.plot(x_positions, midline_crys, "s--", color="#2ca02c", linewidth=2.0, label="Crystallised Mechanism Baseline")

    ax3.axhline(125.0, color="gray", linestyle=":", label="Nominal Mid-Stroke (125mm)")
    ax3.set_title("Midline Sagittal Contouring")
    ax3.set_xlabel("Longitudinal Position (mm)")
    ax3.set_ylabel("Vertical Height (mm)")
    ax3.set_ylim(0, 250)
    ax3.legend(loc="lower right")

    # Annotate anatomical zones
    zones = ["Head", "Cervical", "Thorax", "Lumbar", "Pelvis", "Thigh", "Calf/Heel"]
    for i, z in enumerate(zones):
        ax3.text(x_positions[i], 15.0, z, rotation=90, fontsize=8, ha="center", color="#555555")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_path}")

    return fig


def plot_action_space_comparison(
    curves: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot learning and safety comparison across action space engineering paradigms:
    
    1. Crystallised + Learned Residual (WATER Architecture)
    2. Naive End-to-End RL
    3. Crystallised Baseline Heuristic
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    colors = {
        "Crystallised + Residual (WATER)": "#1f77b4",
        "End-to-End RL": "#d62728",
        "Crystallised Baseline (Heuristic)": "#2ca02c",
    }

    # 1. Total Reward Learning Curves
    ax1 = axes[0, 0]
    for label, metrics in curves.items():
        c = colors.get(label, "black")
        rewards = metrics["rewards"]
        ax1.plot(rewards, label=label, color=c, linewidth=2.2)
    ax1.set_title("Sample Efficiency: Episode Reward")
    ax1.set_xlabel("Training Episodes")
    ax1.set_ylabel("Composite Return")
    ax1.legend(loc="lower right")

    # 2. Peak Contact Pressure (Safety / Ischemia Risk)
    ax2 = axes[0, 1]
    for label, metrics in curves.items():
        c = colors.get(label, "black")
        p_peaks = metrics["peak_pressures"]
        ax2.plot(p_peaks, label=label, color=c, linewidth=2.0)
    ax2.axhline(32.0, color="crimson", linestyle="--", linewidth=1.8, label="Capillary Ischemia Threshold (32 mmHg)")
    ax2.set_title("Tissue Safety: Peak Contact Pressure")
    ax2.set_xlabel("Training Episodes")
    ax2.set_ylabel("Peak Pressure (mmHg)")
    ax2.legend(loc="upper right")

    # 3. Energy Consumption (Self-Locking Advantage)
    ax3 = axes[1, 0]
    for label, metrics in curves.items():
        c = colors.get(label, "black")
        energies = metrics["energy_costs"]
        ax3.plot(energies, label=label, color=c, linewidth=2.0)
    ax3.set_title("Mechanical Transition Energy Cost")
    ax3.set_xlabel("Training Episodes")
    ax3.set_ylabel("Actuator Transition Cost (Work)")
    ax3.legend(loc="upper right")

    # 4. Sleep Micro-Arousals
    ax4 = axes[1, 1]
    for label, metrics in curves.items():
        c = colors.get(label, "black")
        arousals = metrics["arousals"]
        ax4.plot(arousals, label=label, color=c, linewidth=2.0)
    ax4.set_title("Sleep Quality: Induced Micro-Arousals")
    ax4.set_xlabel("Training Episodes")
    ax4.set_ylabel("Arousals per Episode")
    ax4.legend(loc="upper right")

    fig.suptitle("Action Space Engineering Comparison in Adaptive Surface Control", fontsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_path}")

    return fig


def plot_cross_person_adaptation_curves(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot few-shot adaptation trajectories on an unseen held-out subject."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    target_info = results["target_subject"]
    h_crys_res = results["history_crystallised_plus_residual"]
    h_e2e_pre = results["history_e2e_pretrained"]
    h_e2e_scr = results["history_e2e_scratch"]
    zero_shot = results["zero_shot_crystallised"]

    episodes = [x["episode"] for x in h_crys_res]

    # Panel 1: Return Adaptation
    ax1.plot(
        episodes,
        [x["mean_reward"] for x in h_crys_res],
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        label="Crystallised + Residual (WATER)",
    )
    ax1.plot(
        episodes,
        [x["mean_reward"] for x in h_e2e_pre],
        "^-",
        color="#ff7f0e",
        linewidth=2.0,
        label="Fine-Tuned End-to-End",
    )
    ax1.plot(
        episodes,
        [x["mean_reward"] for x in h_e2e_scr],
        "s--",
        color="#d62728",
        linewidth=2.0,
        label="Scratch End-to-End",
    )
    ax1.axhline(
        zero_shot["mean_reward"],
        color="#2ca02c",
        linestyle=":",
        linewidth=2.2,
        label=f"Zero-Shot Crystallised ({zero_shot['mean_reward']:.1f})",
    )

    # Highlight 5-10 episode rapid convergence window
    ax1.axvspan(5, 10, color="#1f77b4", alpha=0.1, label="Rapid Adaptation Window (5-10 ep)")

    ax1.set_title(f"Few-Shot Adaptation to Unseen Subject\n({target_info['id']}, BMI={target_info['bmi']:.1f})")
    ax1.set_xlabel("Adaptation Episodes on Target")
    ax1.set_ylabel("Evaluation Return")
    ax1.legend(loc="lower right")

    # Panel 2: Peak Interface Pressure Adaptation
    ax2.plot(
        episodes,
        [x["mean_peak_pressure_mmhg"] for x in h_crys_res],
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        label="Crystallised + Residual (WATER)",
    )
    ax2.plot(
        episodes,
        [x["mean_peak_pressure_mmhg"] for x in h_e2e_pre],
        "^-",
        color="#ff7f0e",
        linewidth=2.0,
        label="Fine-Tuned End-to-End",
    )
    ax2.plot(
        episodes,
        [x["mean_peak_pressure_mmhg"] for x in h_e2e_scr],
        "s--",
        color="#d62728",
        linewidth=2.0,
        label="Scratch End-to-End",
    )
    ax2.axhline(32.0, color="crimson", linestyle="--", linewidth=1.8, label="Ischemia Limit (32 mmHg)")
    ax2.set_title("Peak Cutaneous Pressure Convergence")
    ax2.set_xlabel("Adaptation Episodes on Target")
    ax2.set_ylabel("Peak Pressure (mmHg)")
    ax2.legend(loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_path}")

    return fig
