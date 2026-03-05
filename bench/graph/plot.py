#!/usr/bin/env python3
"""
bench/graph/plot.py — Generate comparison graphs from numc vs numpy CSV benchmarks.

Reads bench/numc/results.csv and bench/numpy/results.csv, produces PNG charts
in bench/graph/output/.

Usage:
    bench/graph/.venv/bin/python3 bench/graph/plot.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMC_CSV = os.path.join(ROOT, "numc", "results.csv")
NUMPY_CSV = os.path.join(ROOT, "numpy", "results.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Colors ────────────────────────────────────────────────────────────

C_NUMC = "#2563eb"
C_NUMPY = "#dc2626"
C_SPEEDUP = "#16a34a"
BG = "#fafafa"

# ── Load data ─────────────────────────────────────────────────────────

def load():
    if not os.path.exists(NUMC_CSV):
        print(f"ERROR: {NUMC_CSV} not found. Run ./run.sh bench first.")
        sys.exit(1)
    if not os.path.exists(NUMPY_CSV):
        print(f"ERROR: {NUMPY_CSV} not found. Run ./run.sh bench first.")
        sys.exit(1)

    numc = pd.read_csv(NUMC_CSV)
    numpy = pd.read_csv(NUMPY_CSV)
    return numc, numpy


def merge(numc, numpy, keys=None):
    """Merge numc and numpy on shared columns, suffix _numc/_numpy."""
    if keys is None:
        keys = ["category", "operation", "dtype", "size", "shape"]
    merged = pd.merge(
        numc, numpy,
        on=keys,
        suffixes=("_numc", "_numpy"),
    )
    merged["speedup"] = merged["time_us_numpy"] / merged["time_us_numc"]
    return merged


# ── Plot helpers ──────────────────────────────────────────────────────

def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


def grouped_bar(ax, labels, numc_vals, numpy_vals, ylabel="Time (us)"):
    """Side-by-side bar chart with log scale when range exceeds 10x."""
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, numc_vals, w, label="numc", color=C_NUMC, zorder=3)
    ax.bar(x + w/2, numpy_vals, w, label="numpy", color=C_NUMPY, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_facecolor(BG)
    all_vals = np.concatenate([numc_vals, numpy_vals])
    if all_vals.max() / max(all_vals[all_vals > 0].min(), 1e-9) > 10:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + " (log scale)", fontsize=9)


def speedup_bar(ax, labels, speedups):
    """Horizontal bar chart of speedup factors."""
    y = np.arange(len(labels))
    colors = [C_SPEEDUP if s >= 1 else C_NUMPY for s in speedups]
    bars = ax.barh(y, speedups, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=1.0, color="black", linewidth=0.8, linestyle="--", zorder=4)
    ax.set_xlabel("Speedup (numc / numpy)", fontsize=9)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_facecolor(BG)
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}x", va="center", fontsize=7, fontweight="bold")


# ── Chart generators ──────────────────────────────────────────────────

def plot_category_speedup(merged, category, title, filename):
    """Speedup chart for one category, float32 dtype only."""
    sub = merged[(merged["category"] == category) & (merged["dtype"] == "float32")]
    if sub.empty:
        sub = merged[merged["category"] == category]
        if sub.empty:
            return
        # Pick first dtype available per operation
        sub = sub.drop_duplicates(subset=["operation"], keep="first")

    sub = sub.sort_values("speedup", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.4)))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    speedup_bar(ax, sub["operation"].values, sub["speedup"].values)
    save(fig, filename)


def plot_category_time(merged, category, title, filename):
    """Time comparison bars for one category, float32."""
    sub = merged[(merged["category"] == category) & (merged["dtype"] == "float32")]
    if sub.empty:
        sub = merged[merged["category"] == category]
        if sub.empty:
            return
        sub = sub.drop_duplicates(subset=["operation"], keep="first")

    sub = sub.sort_values("operation")
    fig, ax = plt.subplots(figsize=(max(6, len(sub) * 0.8), 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    grouped_bar(ax, sub["operation"].values,
                sub["time_us_numc"].values, sub["time_us_numpy"].values)
    save(fig, filename)


def plot_dtype_heatmap(merged, category, operation, title, filename):
    """Speedup across all dtypes for a single operation."""
    sub = merged[(merged["category"] == category) & (merged["operation"] == operation)]
    if sub.empty:
        return

    sub = sub.sort_values("dtype")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.4)))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    speedup_bar(ax, sub["dtype"].values, sub["speedup"].values)
    save(fig, filename)


def plot_matmul(merged):
    """Matmul comparison across shapes, float32 only."""
    sub = merged[(merged["category"] == "matmul") & (merged["dtype"] == "float32")]
    if sub.empty:
        return

    sub = sub.sort_values("size")
    labels = sub["shape"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrix Multiplication: numc vs numpy", fontsize=12, fontweight="bold")

    grouped_bar(ax1, labels, sub["time_us_numc"].values, sub["time_us_numpy"].values)
    ax1.set_title("Time (us)", fontsize=10)

    speedup_bar(ax2, labels, sub["speedup"].values)
    ax2.set_title("Speedup", fontsize=10)

    fig.tight_layout()
    save(fig, "matmul.png")


def plot_reduction_full(merged):
    """Full reductions (sum, mean, max, min, argmax, argmin) across dtypes."""
    ops = ["sum", "mean", "max", "min", "argmax", "argmin"]
    sub = merged[(merged["category"] == "reduction") & (merged["operation"].isin(ops))]
    if sub.empty:
        return

    # float32 summary
    f32 = sub[sub["dtype"] == "float32"].sort_values("operation")
    if f32.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Reductions (full, float32): numc vs numpy", fontsize=12, fontweight="bold")

    grouped_bar(ax1, f32["operation"].values,
                f32["time_us_numc"].values, f32["time_us_numpy"].values)
    ax1.set_title("Time (us)", fontsize=10)

    speedup_bar(ax2, f32["operation"].values, f32["speedup"].values)
    ax2.set_title("Speedup", fontsize=10)

    fig.tight_layout()
    save(fig, "reduction_full.png")


def plot_overview(merged):
    """One big summary: average speedup per category."""
    summary = merged.groupby("category")["speedup"].median().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(summary) * 0.5)))
    fig.suptitle("numc vs numpy: Median Speedup by Category", fontsize=13, fontweight="bold", y=1.02)

    y = np.arange(len(summary))
    colors = [C_SPEEDUP if s >= 1 else C_NUMPY for s in summary.values]
    bars = ax.barh(y, summary.values, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index, fontsize=9)
    ax.axvline(x=1.0, color="black", linewidth=0.8, linestyle="--", zorder=4)
    ax.set_xlabel("Median Speedup (higher = numc faster)", fontsize=10)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_facecolor(BG)

    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}x", va="center", fontsize=8, fontweight="bold")

    save(fig, "overview.png")


def plot_dtype_comparison(merged):
    """Binary add speedup across all dtypes."""
    plot_dtype_heatmap(merged, "binary", "add",
                       "Binary Add: Speedup by dtype", "binary_add_dtypes.png")


def plot_all_ops_speedup(merged):
    """Big chart: speedup for every float32 operation."""
    sub = merged[merged["dtype"] == "float32"].copy()
    if sub.empty:
        sub = merged.drop_duplicates(subset=["category", "operation"], keep="first")

    sub["label"] = sub["category"] + "/" + sub["operation"]
    sub = sub.sort_values("speedup", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(sub) * 0.25)))
    fig.suptitle("All Operations (float32): numc vs numpy Speedup",
                 fontsize=13, fontweight="bold", y=1.01)
    speedup_bar(ax, sub["label"].values, sub["speedup"].values)
    save(fig, "all_ops_speedup.png")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading benchmark CSVs...")
    numc, numpy = load()
    merged = merge(numc, numpy)
    print(f"Merged: {len(merged)} rows\n")

    print("Generating graphs:")

    # Overview
    plot_overview(merged)

    # Per-category speedup charts
    categories = [
        ("binary", "Binary Element-wise: Speedup (float32)", "binary_speedup.png"),
        ("ternary", "Ternary Ops: Speedup (float32)", "ternary_speedup.png"),
        ("scalar", "Scalar Ops: Speedup (float32)", "scalar_speedup.png"),
        ("scalar_inplace", "Scalar Inplace: Speedup (float32)", "scalar_inplace_speedup.png"),
        ("unary", "Unary Ops: Speedup (float32)", "unary_speedup.png"),
        ("unary_inplace", "Unary Inplace: Speedup (float32)", "unary_inplace_speedup.png"),
        ("comparison", "Comparison Ops: Speedup (float32)", "comparison_speedup.png"),
        ("comparison_scalar", "Comparison Scalar: Speedup (float32)", "comparison_scalar_speedup.png"),
        ("random", "Random Generation: Speedup (float32)", "random_speedup.png"),
    ]
    for cat, title, fname in categories:
        plot_category_speedup(merged, cat, title, fname)

    # Per-category time comparison
    for cat, title, fname in categories:
        time_fname = fname.replace("_speedup.png", "_time.png")
        time_title = title.replace("Speedup", "Time Comparison")
        plot_category_time(merged, cat, time_title, time_fname)

    # Specific charts
    plot_reduction_full(merged)
    plot_matmul(merged)
    plot_dtype_comparison(merged)
    plot_all_ops_speedup(merged)

    print(f"\nDone! Charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
