"""
NumPy unary element-wise benchmark — mirrors bench_unary.c output format.

Usage:
    python bench/numpy_bench_unary.py
"""

import numpy as np
import time

WARMUP = 20
ITERS  = 200

# ── Timer helpers ──────────────────────────────────────────────────────

def time_unary(op, a, out):
    """Benchmark a unary op with output array, return avg microseconds."""
    try:
        for _ in range(WARMUP):
            op(a, out=out)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a, out=out)
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS * 1e6
    except (TypeError, np.exceptions.DTypePromotionError,
            np._core._exceptions._UFuncOutputCastingError):
        # log/exp on integer arrays returns float64 — out= incompatible
        for _ in range(WARMUP):
            op(a)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a)
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS * 1e6


def time_unary_inplace(op, a):
    """Benchmark a unary inplace op (out=a), return avg microseconds."""
    try:
        for _ in range(WARMUP):
            op(a, out=a)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a, out=a)
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS * 1e6
    except (TypeError, np.exceptions.DTypePromotionError,
            np._core._exceptions._UFuncOutputCastingError):
        # Integer inplace log/exp not supported — measure allocating fallback
        for _ in range(WARMUP):
            op(a)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a)
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

# Small fill for exp — exp(2) ≈ 7.39 → fits int8/uint8.
FILL_VALUES_EXP = {
    np.int8: 2, np.int16: 2, np.int32: 2, np.int64: 2,
    np.uint8: 2, np.uint16: 2, np.uint32: 2, np.uint64: 2,
    np.float32: 1.5, np.float64: 1.5,
}


def print_header(title):
    print(f"\n  {title:<8s} {'log':>8s} {'exp':>8s} {'abs':>8s} {'sqrt':>8s}   "
          f"{'log':>8s} {'exp':>8s} {'abs':>8s} {'sqrt':>8s}")
    print(f"  {'':<8s} {'(us)':>8s} {'(us)':>8s} {'(us)':>8s} {'(us)':>8s}   "
          f"{'(Mop/s)':>8s} {'(Mop/s)':>8s} {'(Mop/s)':>8s} {'(Mop/s)':>8s}")
    print(f"  {'─' * 78}")


def print_row(name, us, mops):
    print(f"  {name:<8s} {us[0]:8.2f} {us[1]:8.2f} {us[2]:8.2f} {us[3]:8.2f}   "
          f"{mops[0]:8.1f} {mops[1]:8.1f} {mops[2]:8.1f} {mops[3]:8.1f}")


# ── Benchmark: unary ops ───────────────────────────────────────────────

def bench_unary_ops(size):
    print(f"{'━' * 82}")
    print(f"  UNARY OPS  ({size} elements, {ITERS} iters)")
    print_header("dtype")

    for name, dt in ALL_DTYPES:
        v     = FILL_VALUES[dt]
        v_exp = FILL_VALUES_EXP[dt]
        a_log  = np.full(size, v, dtype=dt)
        a_exp  = np.full(size, v_exp, dtype=dt)
        a_abs  = np.full(size, v, dtype=dt)
        a_sqrt = np.full(size, v, dtype=dt)
        out    = np.empty(size, dtype=dt)

        us   = [time_unary(np.log,  a_log,  out),
                time_unary(np.exp,  a_exp,  out),
                time_unary(np.abs,  a_abs,  out),
                time_unary(np.sqrt, a_sqrt, out)]
        mops = [size / u for u in us]
        print_row(name, us, mops)


# ── Benchmark: unary inplace ops ──────────────────────────────────────

def bench_unary_inplace_ops(size):
    print(f"\n{'━' * 82}")
    print(f"  UNARY INPLACE  ({size} elements, {ITERS} iters)")
    print_header("dtype")

    for name, dt in ALL_DTYPES:
        v     = FILL_VALUES[dt]
        v_exp = FILL_VALUES_EXP[dt]
        a_log  = np.full(size, v, dtype=dt)
        a_exp  = np.full(size, v_exp, dtype=dt)
        a_abs  = np.full(size, v, dtype=dt)
        a_sqrt = np.full(size, v, dtype=dt)

        us   = [time_unary_inplace(np.log,  a_log),
                time_unary_inplace(np.exp,  a_exp),
                time_unary_inplace(np.abs,  a_abs),
                time_unary_inplace(np.sqrt, a_sqrt)]
        mops = [size / u for u in us]
        print_row(name, us, mops)


# ── Benchmark: size scaling (float32 sqrt) ─────────────────────────────

def bench_unary_scaling():
    print(f"\n{'━' * 82}")
    print(f"  SIZE SCALING  (float32 sqrt, {ITERS} iters)")
    print(f"\n  {'elements':>10s} {'time (us)':>10s} {'Mops/s':>10s} {'GB/s':>10s}")
    print(f"  {'─' * 42}")

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    for n in sizes:
        a   = np.full(n, 1.5, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)

        us   = time_unary(np.sqrt, a, out)
        mops = n / us
        gbs  = (2.0 * n * 4) / (us * 1e3)  # read 1 + write 1

        print(f"  {n:10d} {us:10.2f} {mops:10.1f} {gbs:10.2f}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print(f"\n  numpy unary element-wise benchmark")
    print(f"  numpy {np.__version__}\n")

    bench_unary_ops(1_000_000)
    bench_unary_inplace_ops(1_000_000)
    bench_unary_scaling()

    print()


if __name__ == "__main__":
    main()
