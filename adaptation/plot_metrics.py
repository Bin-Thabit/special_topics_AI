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
  3. Rolling accuracy over last 50 queries (green line) — shows recent performance
  4. Vertical red dashed lines where ADWIN detected drift

This chart is the main visual evidence in the D1 report that:
  - The model is learning (accuracy rises over time)
  - ADWIN correctly detects when the distribution shifts
  - The model recovers after a drift-triggered reset

INPUT
-----
history : list of dicts, each with keys:
    "step"             : how many samples have been seen
    "accuracy"         : cumulative prequential accuracy at that step
    "rolling_accuracy" : accuracy over last 50 queries
    "drift_detected"   : was drift detected at this step?
    "resets"           : total resets so far

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
    show_rolling: bool = True,
) -> None:
    """
    Generate and save the prequential accuracy chart.

    Parameters
    ----------
    history      : list of state dicts from QueryTopicLearner.history
                   each dict must have 'step' and 'accuracy' keys
                   optionally 'rolling_accuracy' for the green line

    drift_steps  : list of step numbers where ADWIN fired
                   each becomes a vertical red dashed line
                   if None or empty, no drift lines are drawn

    output_path  : where to save the PNG
                   parent directory is created automatically

    window       : rolling mean window size for the smoothed blue line
                   20 means each point = average of last 20 snapshots

    show_rolling : whether to show the rolling_accuracy green line
                   only shown if 'rolling_accuracy' key exists in history
    """
    if not history:
        print("No history to plot — run the learner first.")
        return

    # ── Extract values ────────────────────────────────────────────────────────
    steps      = [h["step"] for h in history]
    accuracies = [h["accuracy"] for h in history]

    # Rolling accuracy (last 50 queries) — only if available
    has_rolling = "rolling_accuracy" in history[0]
    rolling_accs = [h["rolling_accuracy"] for h in history] if has_rolling else []

    # Compute smoothed line for cumulative accuracy
    smoothed = _rolling_mean(accuracies, window)

    # ── Set up figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    # ── Raw cumulative accuracy (faint gray) ──────────────────────────────────
    ax.plot(
        steps, accuracies,
        color="#c0c0c0",
        linewidth=0.8,
        alpha=0.5,
        label="Cumulative accuracy (raw)",
    )

    # ── Smoothed cumulative accuracy (blue) ───────────────────────────────────
    ax.plot(
        steps, smoothed,
        color="#4361ee",
        linewidth=2.2,
        label=f"Cumulative accuracy — rolling mean (w={window})",
    )

    # ── Rolling accuracy last 50 queries (green) ──────────────────────────────
    # This shows recent performance — more honest for an online learner
    if show_rolling and has_rolling:
        ax.plot(
            steps, rolling_accs,
            color="#0f6e56",
            linewidth=1.8,
            linestyle="-.",
            alpha=0.85,
            label="Rolling accuracy (last 50 queries)",
        )

    # ── Drift detection lines (red dashed) ───────────────────────────────────
    if drift_steps:
        for step in drift_steps:
            ax.axvline(
                x=step,
                color="#e63946",
                linewidth=1.3,
                linestyle="--",
                alpha=0.85,
            )
        drift_patch = mpatches.Patch(
            color="#e63946",
            alpha=0.85,
            label=f"ADWIN drift reset (steps: {drift_steps})",
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
        "Prequential accuracy — Query-to-topic classifier  "
        "(River · BagOfWords · MultinomialNB · ADWIN)",
        fontsize=12,
        pad=12,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {output_path}")


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """
    Computes a simple rolling average.
    For early points without a full window, averages available points.
    """
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start: i + 1])))
    return result