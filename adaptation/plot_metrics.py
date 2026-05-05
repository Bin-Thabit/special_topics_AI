"""
adaptation/plot_metrics.py
--------------------------
Generates the prequential accuracy chart required by D1.

PURPOSE
-------
Takes the accuracy history recorded by QueryTopicLearner and produces
a PNG chart showing:
  1. Raw prequential accuracy over time (faint gray line)
  2. Smoothed rolling mean (blue line) — shows the learning trend clearly
  3. Vertical red dashed lines where ADWIN detected drift

This chart is the main visual evidence in the D1 report that:
  - The model is learning (accuracy rises over time)
  - ADWIN correctly detects when the distribution shifts
  - The model recovers after a drift-triggered reset

INPUT
-----
history : list of dicts, each with keys:
    "step"           : how many samples have been seen
    "accuracy"       : prequential accuracy at that step
    "drift_detected" : was drift detected at this step?
    "resets"         : total resets so far

This list comes directly from QueryTopicLearner.history
(saved to JSON via learner.save() and loaded in the notebook).

OUTPUT
------
A PNG file saved to docs/prequential_accuracy.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_prequential(
    history: list[dict],
    drift_steps: Optional[list[int]] = None,
    output_path: str | Path = "docs/prequential_accuracy.png",
    window: int = 20,
) -> None:
    """
    Generate and save the prequential accuracy chart.

    Parameters
    ----------
    history : list of state dicts from QueryTopicLearner.history
              each dict must have 'step' and 'accuracy' keys

    drift_steps : list of step numbers where ADWIN fired
                  each becomes a vertical red dashed line on the chart
                  if None or empty, no drift lines are drawn

    output_path : where to save the PNG
                  parent directory is created automatically if it doesn't exist

    window : rolling mean window size
             20 means each smoothed point = average of last 20 raw points
             larger window = smoother line but slower to show changes
             smaller window = noisier but reacts faster to accuracy changes
    """
    if not history:
        print("No history to plot — run the learner first.")
        return

    # ── Extract values from history ───────────────────────────────────────────
    steps      = [h["step"] for h in history]
    accuracies = [h["accuracy"] for h in history]

    # ── Compute rolling mean ──────────────────────────────────────────────────
    # This smooths out the noise in the raw accuracy line
    # so the learning trend is clearly visible in the chart
    rolling = _rolling_mean(accuracies, window)

    # ── Set up the figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    # ── Plot raw accuracy ─────────────────────────────────────────────────────
    # Shown as a faint gray line in the background
    # It shows the real step-by-step values without smoothing
    ax.plot(
        steps, accuracies,
        color="#c0c0c0",
        linewidth=0.8,
        alpha=0.6,
        label="Prequential accuracy (raw)",
    )

    # ── Plot smoothed rolling mean ────────────────────────────────────────────
    # The main line the reader focuses on
    # Shows the overall learning trend clearly
    ax.plot(
        steps, rolling,
        color="#4361ee",
        linewidth=2.2,
        label=f"Rolling mean (window = {window})",
    )

    # ── Draw drift detection lines ────────────────────────────────────────────
    # Each vertical line marks where ADWIN fired and the classifier was reset
    # The reader can see how accuracy drops before drift and recovers after
    if drift_steps:
        for step in drift_steps:
            ax.axvline(
                x=step,
                color="#e63946",
                linewidth=1.3,
                linestyle="--",
                alpha=0.85,
            )
        # Add a legend entry for the drift lines
        drift_patch = mpatches.Patch(
            color="#e63946",
            alpha=0.85,
            label="Drift detected — ADWIN reset",
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(drift_patch)
        ax.legend(handles=handles, fontsize=9, loc="lower right")
    else:
        ax.legend(fontsize=9, loc="lower right")

    # ── Labels and formatting ─────────────────────────────────────────────────
    ax.set_xlabel("Samples seen", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        "Prequential accuracy — Query-to-topic classifier  (River · TF-IDF · ADWIN)",
        fontsize=12,
        pad=12,
    )

    # Y axis from 0 to 1.05 so the 1.0 line is clearly visible
    ax.set_ylim(0, 1.05)

    # Horizontal grid lines only — easier to read accuracy values
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Remove top and right borders for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)  # close to free memory — important in notebooks

    print(f"Chart saved → {output_path}")


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """
    Computes a simple rolling average over a list of floats.

    For the first few points where we don't have a full window yet,
    we average whatever points are available so far.

    Example with window=3:
        input:  [0.1, 0.3, 0.2, 0.4, 0.5]
        output: [0.1, 0.2, 0.2, 0.3, 0.37]
                 ^^^  ^^^  ^^^  ^^^  ^^^^
                 1pt  2pt  3pt  3pt  3pt averaged
    """
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start: i + 1])))
    return result