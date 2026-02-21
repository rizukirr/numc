"""
NumPy binary element-wise benchmark — mirrors bench_elemwise.c output format.

Usage:
    python bench/numpy_bench_elemwise.py
"""

import numpy as np
import time

WARMUP = 20
ITERS  = 200

# ── Timer helpers ──────────────────────────────────────────────────────

def time_binary(op, a, b, out):
    """Benchmark a binary op, return avg microseconds."""
    try:
        for _ in range(WARMUP):
            op(a, b, out=out)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a, b, out=out)
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS * 1e6
    except (TypeError, np.exceptions.DTypePromotionError,
            np._core._exceptions._UFuncOutputCastingError):
        # NumPy upcasts integer division to float64 — benchmark without out=
        for _ in range(WARMUP):
            op(a, b)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a, b)
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS * 1e6


# ── Helpers ────────────────────────────────────────────────────────────

ALL_DTYPES = [
    ("int8",    np.int8),    ("uint8",   np.uint8),
    ("int16",   np.int16),   ("uint16",  np.uint16),
    ("int32",   np.int32),   ("uint32",  np.uint32),
    ("int64",   np.int64),   ("uint64",  np.uint64),
    ("float32", np.float32), ("float64", np.float64),
]

FILL_VALUES = {
    np.int8: 3, np.int16: 7, np.int32: 42, np.int64: 42,
    np.uint8: 3, np.uint16: 7, np.uint32: 42, np.uint64: 42,
    np.float32: 1.5, np.float64: 1.5,
}


def print_header(title):
    print(f"\n  {title:<8s} {'add':>8s} {'sub':>8s} {'mul':>8s} {'div':>8s}   "
          f"{'add':>8s} {'sub':>8s} {'mul':>8s} {'div':>8s}")
    print(f"  {'':<8s} {'(us)':>8s} {'(us)':>8s} {'(us)':>8s} {'(us)':>8s}   "
          f"{'(Mop/s)':>8s} {'(Mop/s)':>8s} {'(Mop/s)':>8s} {'(Mop/s)':>8s}")
    print(f"  {'─' * 78}")


def print_row(name, us, mops):
    print(f"  {name:<8s} {us[0]:8.2f} {us[1]:8.2f} {us[2]:8.2f} {us[3]:8.2f}   "
          f"{mops[0]:8.1f} {mops[1]:8.1f} {mops[2]:8.1f} {mops[3]:8.1f}")


# ── Benchmark: contiguous binary ops ──────────────────────────────────

def bench_contiguous(size):
    print(f"{'━' * 82}")
    print(f"  CONTIGUOUS BINARY  ({size} elements, {ITERS} iters)")
    print_header("dtype")

    ops = [np.add, np.subtract, np.multiply, np.divide]

    for name, dt in ALL_DTYPES:
        v   = FILL_VALUES[dt]
        a   = np.full(size, v, dtype=dt)
        b   = np.full(size, v, dtype=dt)
        out = np.empty(size, dtype=dt)

        us   = [time_binary(op, a, b, out) for op in ops]
        mops = [size / u for u in us]
        print_row(name, us, mops)


# ── Benchmark: strided (transposed view) ──────────────────────────────

def bench_strided(rows, cols):
    total = rows * cols
    print(f"\n{'━' * 82}")
    print(f"  STRIDED  ({rows}x{cols} transposed, {total} elements, {ITERS} iters)")
    print_header("dtype")

    ops = [np.add, np.subtract, np.multiply, np.divide]
    dtypes = [
        ("int32",   np.int32),
        ("float32", np.float32),
        ("float64", np.float64),
    ]

    for name, dt in dtypes:
        v   = FILL_VALUES[dt]
        a   = np.full((rows, cols), v, dtype=dt).T
        b   = np.full((rows, cols), v, dtype=dt).T
        out = np.empty((cols, rows), dtype=dt)

        us   = [time_binary(op, a, b, out) for op in ops]
        mops = [total / u for u in us]
        print_row(name, us, mops)


# ── Benchmark: broadcast patterns ─────────────────────────────────────

def bench_bcast_pattern(a_shape, b_shape, out_shape, total):
    ops = [np.add, np.subtract, np.multiply, np.divide]
    dtypes = [
        ("int32",   np.int32),
        ("float32", np.float32),
        ("float64", np.float64),
    ]

    for name, dt in dtypes:
        v   = FILL_VALUES[dt]
        a   = np.full(a_shape, v, dtype=dt)
        b   = np.full(b_shape, v, dtype=dt)
        out = np.empty(out_shape, dtype=dt)

        us   = [time_binary(op, a, b, out) for op in ops]
        mops = [total / u for u in us]
        print_row(name, us, mops)


def bench_broadcast(M, N):
    total = M * N

    # Row broadcast: (1,N) + (M,N) → (M,N)
    print(f"\n{'━' * 82}")
    print(f"  BROADCAST ROW  (1,{N}) + ({M},{N}) -> ({M},{N}), {ITERS} iters")
    print_header("dtype")
    bench_bcast_pattern((1, N), (M, N), (M, N), total)

    # Outer broadcast: (M,1) + (1,N) → (M,N)
    print(f"\n{'━' * 82}")
    print(f"  BROADCAST OUTER  ({M},1) + (1,{N}) -> ({M},{N}), {ITERS} iters")
    print_header("dtype")
    bench_bcast_pattern((M, 1), (1, N), (M, N), total)

    # Rank broadcast: (N,) + (M,N) → (M,N)
    print(f"\n{'━' * 82}")
    print(f"  BROADCAST RANK  ({N},) + ({M},{N}) -> ({M},{N}), {ITERS} iters")
    print_header("dtype")
    bench_bcast_pattern((N,), (M, N), (M, N), total)


# ── Benchmark: scaling across sizes ───────────────────────────────────

def bench_scaling():
    print(f"\n{'━' * 82}")
    print(f"  SIZE SCALING  (float32 add, {ITERS} iters)")
    print(f"\n  {'elements':>10s} {'time (us)':>10s} {'Mops/s':>10s} {'GB/s':>10s}")
    print(f"  {'─' * 42}")

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    for n in sizes:
        a   = np.full(n, 1.5, dtype=np.float32)
        b   = np.full(n, 2.5, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)

        us   = time_binary(np.add, a, b, out)
        mops = n / us
        gbs  = (3.0 * n * 4) / (us * 1e3)

        print(f"  {n:10d} {us:10.2f} {mops:10.1f} {gbs:10.2f}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print(f"\n  numpy binary element-wise benchmark")
    print(f"  numpy {np.__version__}\n")

    bench_contiguous(1_000_000)
    bench_strided(1000, 1000)
    bench_broadcast(1000, 1000)
    bench_scaling()

    print()


if __name__ == "__main__":
    main()
