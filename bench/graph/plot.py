#!/usr/bin/env python3
"""
bench/graph/plot.py — Generate informative comparison graphs from numc vs numpy CSV benchmarks.

Reads bench/numc/results.csv and bench/numpy/results.csv, produces PNG charts
in bench/graph/output/.

Usage:
    bench/graph/.venv/bin/python3 bench/graph/plot.py
"""

import os
import sys
import platform
import subprocess
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

C_NUMC = "#2563eb"    # Blue
C_NUMPY = "#dc2626"   # Red
C_SPEEDUP = "#16a34a" # Green (Fast)
C_SLOWDOWN = "#991b1b" # Dark Red (Slow)
BG = "#ffffff"        # Clean White
GRID = "#e5e7eb"      # Light Gray

# ── System Info ───────────────────────────────────────────────────────

def get_system_info():
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor(),
        "Architecture": platform.machine(),
    }
    if platform.system() == "Linux":
        try:
            # Try lscpu for detailed model name
            cpu_info = subprocess.check_output("lscpu", shell=True).decode()
            for line in cpu_info.split("\n"):
                if "Model name:" in line:
                    info["CPU"] = line.split("Model name:")[1].strip()
                    break
        except Exception:
            pass
    return info

SYS_INFO = get_system_info()
SYS_STR = f"CPU: {SYS_INFO['CPU']} | OS: {SYS_INFO['OS']} | Arch: {SYS_INFO['Architecture']}"

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
    # Add system info as footer
    fig.text(0.5, 0.01, SYS_STR, ha='center', fontsize=8, color='#6b7280', style='italic')
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


def setup_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(True, axis="both", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9ca3af')
    ax.spines['bottom'].set_color('#9ca3af')
    ax.tick_params(colors='#4b5563', labelsize=8)


def grouped_bar(ax, labels, numc_vals, numpy_vals, ylabel="Time (us)"):
    """Side-by-side bar chart with log scale when range exceeds 10x."""
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, numc_vals, w, label="numc", color=C_NUMC, zorder=3, alpha=0.9)
    ax.bar(x + w/2, numpy_vals, w, label="numpy", color=C_NUMPY, zorder=3, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    
    all_vals = np.concatenate([numc_vals, numpy_vals])
    if len(all_vals) > 0 and all_vals.max() / max(all_vals[all_vals > 0].min(), 1e-9) > 10:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + " (log scale)", fontsize=9, fontweight="bold")
    setup_ax(ax)


def speedup_bar(ax, labels, speedups):
    """Horizontal bar chart of speedup factors."""
    y = np.arange(len(labels))
    colors = [C_SPEEDUP if s >= 1 else C_SLOWDOWN for s in speedups]
    bars = ax.barh(y, speedups, color=colors, zorder=3, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(x=1.0, color="#1f2937", linewidth=1.5, linestyle="-", zorder=4, alpha=0.6)
    ax.set_xlabel("Speedup (Higher is Faster)", fontsize=9, fontweight="bold")
    
    # Text labels for speedup
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}x", va="center", fontsize=7, fontweight="bold",
                color=C_SPEEDUP if val >= 1 else C_SLOWDOWN)
    setup_ax(ax)


# ── Chart generators ──────────────────────────────────────────────────

def plot_category_combined(merged, category, title, filename):
    """Combined chart: Time (left) and Speedup (right) for one category."""
    sub = merged[(merged["category"] == category) & (merged["dtype"] == "float32")]
    if sub.empty:
        sub = merged[merged["category"] == category]
        if sub.empty: return
        sub = sub.drop_duplicates(subset=["operation"], keep="first")

    sub_speedup = sub.sort_values("speedup", ascending=True)
    sub_time = sub.sort_values("operation")

    fig, (ax_time, ax_speed) = plt.subplots(1, 2, figsize=(14, max(5, len(sub) * 0.4)))
    fig.suptitle(f"{title} (float32)", fontsize=14, fontweight="bold", y=0.98)

    # Time Comparison
    grouped_bar(ax_time, sub_time["operation"].values,
                sub_time["time_us_numc"].values, sub_time["time_us_numpy"].values)
    ax_time.set_title("Average Execution Time", fontsize=11, pad=10)

    # Speedup
    speedup_bar(ax_speed, sub_speedup["operation"].values, sub_speedup["speedup"].values)
    ax_speed.set_title("numc Speedup Factor", fontsize=11, pad=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save(fig, filename)


def plot_dtype_heatmap(merged, category, operation, title, filename):
    """Heatmap of speedup across all dtypes for a single operation."""
    sub = merged[(merged["category"] == category) & (merged["operation"] == operation)]
    if sub.empty: return

    sub = sub.sort_values("dtype")
    labels = sub["dtype"].values
    speedups = sub["speedup"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    y = np.arange(len(labels))
    colors = [C_SPEEDUP if s >= 1 else C_SLOWDOWN for s in speedups]
    bars = ax.bar(y, speedups, color=colors, zorder=3, alpha=0.8)
    ax.set_xticks(y)
    ax.set_xticklabels(labels, rotation=0)
    ax.axhline(y=1.0, color="#1f2937", linewidth=1, linestyle="--", zorder=4)
    ax.set_ylabel("Speedup", fontsize=10, fontweight="bold")

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

    setup_ax(ax)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save(fig, filename)


def plot_overview(merged):
    """High-level summary: Geometric Mean speedup per category."""
    import math

    def geo_mean(x):
        return math.exp(np.log(x[x > 0]).mean())

    # Calculate overall stats per category
    summary = merged.groupby("category")["speedup"].agg([geo_mean, "max", "min"])
    
    # Calculate float-only stats for a more "fair" comparison
    float_sub = merged[merged["dtype"].isin(["float32", "float64"])]
    float_summary = float_sub.groupby("category")["speedup"].agg(geo_mean)
    
    # Sort by overall geo_mean
    summary = summary.sort_values("geo_mean", ascending=True)
    float_summary = float_summary.reindex(summary.index)

    fig, ax = plt.subplots(figsize=(10, max(5, len(summary) * 0.7)))
    fig.suptitle("numc Performance Overview: Geometric Mean Speedup", fontsize=15, fontweight="bold", y=0.98)

    y = np.arange(len(summary))
    height = 0.4
    
    means = summary["geo_mean"].values
    float_means = float_summary.values
    
    # Plot main bars (Overall)
    colors = [C_SPEEDUP if s >= 1 else C_SLOWDOWN for s in means]
    bars = ax.barh(y + height/2, means, height, color=colors, zorder=3, alpha=0.8, label="Overall (All DTypes)")
    
    # Plot secondary bars (Float-only)
    ax.barh(y - height/2, float_means, height, color="#94a3b8", zorder=3, alpha=0.6, label="Floating Point (fp32/64)")
    
    # Error bars for overall range
    ax.errorbar(means, y + height/2, xerr=[means - summary["min"], summary["max"] - means], 
                fmt='none', ecolor='#9ca3af', elinewidth=1, capsize=3, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ").title() for c in summary.index], fontsize=10, fontweight="bold")
    ax.axvline(x=1.0, color="#1f2937", linewidth=2, linestyle="-", zorder=4, alpha=0.7)
    ax.set_xlabel("Speedup Factor (Higher is Faster)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=BG)

    # Text labels for overall speedup
    for i, val in enumerate(means):
        ax.text(max(val, summary["max"].iloc[i]) + 0.1, i + height/2,
                f"{val:.2f}x", va="center", fontsize=9, fontweight="bold",
                color=C_SPEEDUP if val >= 1 else C_SLOWDOWN)

    # Note about the metrics
    ax.text(0.95, 0.02, "Geometric mean is used to aggregate speedup ratios.\nError bars show Min/Max speedup in category.", 
            transform=ax.transAxes, ha='right', fontsize=8, color='#6b7280', style='italic')

    setup_ax(ax)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save(fig, "overview.png")


def plot_all_ops_detailed(merged):
    """Large detailed chart for all float32 operations."""
    sub = merged[merged["dtype"] == "float32"].copy()
    if sub.empty:
        sub = merged.drop_duplicates(subset=["category", "operation"], keep="first")

    sub["label"] = sub["category"].str.replace("_", " ").str.title() + " / " + sub["operation"]
    sub = sub.sort_values("speedup", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(10, len(sub) * 0.25)))
    fig.suptitle("Detailed Operation Speedup (float32)", fontsize=15, fontweight="bold", y=0.99)
    
    speedup_bar(ax, sub["label"].values, sub["speedup"].values)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save(fig, "all_ops_speedup.png")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    cat_filter = None
    if "--filter" in sys.argv:
        idx = sys.argv.index("--filter")
        if idx + 1 < len(sys.argv):
            cat_filter = sys.argv[idx + 1]

    print("Loading benchmark results...")
    numc, numpy = load()
    merged = merge(numc, numpy)
    print(f"Loaded {len(merged)} shared data points.")

    print("\nGenerating Informative Charts:")

    if cat_filter:
        print(f"  Filtering for category: {cat_filter}")
        plot_category_combined(merged, cat_filter, f"{cat_filter.title()} Performance", f"{cat_filter}_combined.png")
        if cat_filter == "binary":
            plot_dtype_heatmap(merged, "binary", "add", "Binary Add Speedup across Data Types", "binary_add_dtypes_heatmap.png")
    else:
        # 1. Overview
        plot_overview(merged)

        # 2. Combined Category Plots
        categories = [
            ("binary", "Binary Operations"),
            ("ternary", "Ternary Operations"),
            ("scalar", "Scalar Operations"),
            ("scalar_inplace", "Inplace Scalar Operations"),
            ("unary", "Unary Operations"),
            ("unary_inplace", "Inplace Unary Operations"),
            ("comparison", "Comparison Operations"),
            ("comparison_scalar", "Scalar Comparison"),
            ("random", "Random Number Generation"),
            ("reduction", "Reduction Operations"),
            ("matmul", "Matrix Multiplication"),
        ]
        
        for cat, title in categories:
            plot_category_combined(merged, cat, title, f"{cat}_combined.png")

        # 3. Special DType Heatmap
        plot_dtype_heatmap(merged, "binary", "add", "Performance Scalability: Binary Add by Data Type", "binary_add_dtypes_heatmap.png")

        # 4. Detailed Ops
        plot_all_ops_detailed(merged)

    print(f"\nSuccess! Visualizations are available in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
