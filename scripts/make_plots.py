"""
make_plots.py
Generates illustrative (conceptual) plots for the nn-evolution-doc page.

Usage:
    python scripts/make_plots.py

Output files:
    assets/plots/loss_curve.png
    assets/plots/tradeoff_scatter.png
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── stylistic defaults ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BLUE   = "#0078D4"
PURPLE = "#7C3AED"
GREEN  = "#16A34A"
AMBER  = "#D97706"
GRAY   = "#6B7280"


# ── Plot A: Illustrative loss curve ─────────────────────────────────────────
def plot_loss_curve():
    steps = np.arange(0, 200)
    # Smooth exponential decay + small noise
    rng = np.random.default_rng(42)
    loss = 2.5 * np.exp(-steps / 60) + 0.25 + rng.normal(0, 0.04, len(steps))
    loss = np.clip(loss, 0.2, None)

    val_loss = 2.5 * np.exp(-steps / 70) + 0.35 + rng.normal(0, 0.06, len(steps))
    val_loss = np.clip(val_loss, 0.3, None)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(steps, loss, color=BLUE, lw=2, label="Train loss")
    ax.plot(steps, val_loss, color=PURPLE, lw=2, linestyle="--", label="Val loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Illustrative loss curve (conceptual — not real measurements)", fontsize=9, color=GRAY)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "loss_curve.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot B: Ergonomics vs Performance scatter ────────────────────────────────
def plot_tradeoff_scatter():
    # (ergonomics score 0-10, performance/opt potential 0-10, label, color)
    points = [
        (2,  3,  "NumPy / Manual",  GRAY),
        (3,  6,  "Graph-first\n(TF1/Theano)", AMBER),
        (8,  5,  "Keras / High-level", GREEN),
        (6,  7,  "Eager / Define-by-run", BLUE),
        (7,  8,  "PyTorch\nMainstream", BLUE),
        (7,  9.5,"Compile era\n(torch.compile)", PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for x, y, label, color in points:
        ax.scatter(x, y, s=140, color=color, zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 3),
                    fontsize=8, color=color)

    ax.set_xlabel("Developer ergonomics →", fontsize=10)
    ax.set_ylabel("Performance / optimization potential →", fontsize=10)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_title("Illustrative: ergonomics vs. performance (conceptual — not real benchmarks)",
                 fontsize=9, color=GRAY)
    ax.axhline(5, color="#E5E7EB", lw=1, zorder=0)
    ax.axvline(5, color="#E5E7EB", lw=1, zorder=0)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "tradeoff_scatter.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot C (optional): Cognitive load bars ──────────────────────────────────
def plot_cognitive_load():
    categories = ["Manual\ngradients", "Autograd\n(eager)", "High-level\nfit()"]
    cognitive  = [9, 5, 2]
    control    = [10, 9, 4]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars1 = ax.bar(x - width/2, cognitive, width, label="Cognitive load", color=AMBER, alpha=0.85)
    bars2 = ax.bar(x + width/2, control,   width, label="Explicit control", color=BLUE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Relative score (conceptual)")
    ax.set_ylim(0, 12)
    ax.set_title("Illustrative: cognitive load vs explicit control (conceptual)", fontsize=9, color=GRAY)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "cognitive_load.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_loss_curve()
    plot_tradeoff_scatter()
    plot_cognitive_load()
    print("All plots generated.")
