"""
NumPy reduction (sum) benchmark — mirrors bench_reduction.c output format.

Usage:
    python bench/numpy_bench_reduction.py
"""

import numpy as np
import time

WARMUP = 20
ITERS  = 200

# ── Helpers ────────────────────────────────────────────────────────────

ALL_DTYPES = [
    ("int8",    np.int8),    ("uint8",   np.uint8),
    ("int16",   np.int16),   ("uint16",  np.uint16),
    ("int32",   np.int32),   ("uint32",  np.uint32),
    ("int64",   np.int64),   ("uint64",  np.uint64),
    ("float32", np.float32), ("float64", np.float64),
]


def bench(fn, iters=ITERS):
    """Time a zero-arg callable, return avg microseconds."""
    for _ in range(WARMUP):
        fn()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()

    return (t1 - t0) / iters * 1e6


# ── Benchmark: full sum — all dtypes ──────────────────────────────────

def bench_sum_full(size):
    print(f"{'━' * 82}")
    print(f"  SUM (full reduction)  ({size} elements, {ITERS} iters)")
    print(f"\n  {'dtype':<8s} {'time (us)':>10s} {'Mop/s':>10s}")
    print(f"  {'─' * 32}")

    for name, dt in ALL_DTYPES:
        a = np.ones(size, dtype=dt)

        us   = bench(lambda: a.sum())
        mops = size / us

        print(f"  {name:<8s} {us:10.2f} {mops:10.1f}")


# ── Benchmark: sum axis=0 on 2D array ─────────────────────────────────

def bench_sum_axis0(rows, cols):
    total = rows * cols
    print(f"\n{'━' * 82}")
    print(f"  SUM AXIS=0  ({rows}x{cols} = {total} elements, {ITERS} iters)")
    print(f"\n  {'dtype':<8s} {'time (us)':>10s} {'Mop/s':>10s}")
    print(f"  {'─' * 32}")

    for name, dt in ALL_DTYPES:
        a = np.ones((rows, cols), dtype=dt)

        us   = bench(lambda: a.sum(axis=0))
        mops = total / us

        print(f"  {name:<8s} {us:10.2f} {mops:10.1f}")


# ── Benchmark: sum axis=1 on 2D array ─────────────────────────────────

def bench_sum_axis_last(rows, cols):
    total = rows * cols
    print(f"\n{'━' * 82}")
    print(f"  SUM AXIS=1  ({rows}x{cols} = {total} elements, {ITERS} iters)")
    print(f"\n  {'dtype':<8s} {'time (us)':>10s} {'Mop/s':>10s}")
    print(f"  {'─' * 32}")

    for name, dt in ALL_DTYPES:
        a = np.ones((rows, cols), dtype=dt)

        us   = bench(lambda: a.sum(axis=1))
        mops = total / us

        print(f"  {name:<8s} {us:10.2f} {mops:10.1f}")


# ── Benchmark: size scaling (float32 full sum) ────────────────────────

def bench_scaling():
    print(f"\n{'━' * 82}")
    print(f"  SIZE SCALING  (float32 sum, {ITERS} iters)")
    print(f"\n  {'elements':>10s} {'time (us)':>10s} {'Mop/s':>10s} {'GB/s':>10s}")
    print(f"  {'─' * 42}")

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    for n in sizes:
        a = np.ones(n, dtype=np.float32)

        us   = bench(lambda: a.sum())
        mops = n / us
        gbs  = (n * 4) / (us * 1e3)   # read 1 array

        print(f"  {n:10d} {us:10.2f} {mops:10.1f} {gbs:10.2f}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print(f"\n  numpy reduction benchmark")
    print(f"  numpy {np.__version__}\n")

    bench_sum_full(1_000_000)
    bench_sum_axis0(1000, 1000)
    bench_sum_axis_last(1000, 1000)
    bench_scaling()

    print()


if __name__ == "__main__":
    main()
