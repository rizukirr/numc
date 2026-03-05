"""
NumPy random benchmark — mirrors bench_random.c output format.

Usage:
    python bench/numpy_bench_random.py
"""

import numpy as np
import time

WARMUP = 5
ITERS  = 50

# ── Timer helpers ──────────────────────────────────────────────────────

def time_randn(size, dt):
    """Benchmark np.random.standard_normal, return avg microseconds."""
    # Warmup
    for _ in range(WARMUP):
        a = np.random.standard_normal(size).astype(dt)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        a = np.random.standard_normal(size).astype(dt)
    t1 = time.perf_counter()

    return (t1 - t0) / ITERS * 1e6


# ── Helpers ────────────────────────────────────────────────────────────

ALL_DTYPES = [
    ("float32", np.float32), ("float64", np.float64),
]


# ── Benchmark: randn ───────────────────────────────────────────────────

def bench_randn(size):
    print(f"{'━' * 82}")
    print(f"  RANDN  ({size} elements, {ITERS} iters)")
    print(f"\n  {'dtype':<8s} {'time (us)':>10s} {'Mop/s':>10s}")
    print(f"  {'─' * 30}")

    for name, dt in ALL_DTYPES:
        us = time_randn(size, dt)
        print(f"  {name:<8s} {us:10.2f} {size/us:10.1f}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print(f"\n  numpy random benchmark")
    print(f"  numpy {np.__version__}\n")

    bench_randn(1_000_000)

    print()


if __name__ == "__main__":
    main()
