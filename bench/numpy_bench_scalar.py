"""
NumPy scalar element-wise benchmark — mirrors bench_scalar.c output format.

Usage:
    python bench/numpy_bench_scalar.py
"""

import numpy as np
import time

WARMUP = 20
ITERS  = 200

# ── Timer helpers ──────────────────────────────────────────────────────

def time_scalar(op, a, scalar, out):
    """Benchmark a scalar op, return avg microseconds."""
    try:
        for _ in range(WARMUP):
            op(a, scalar, out=out)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a, scalar, out=out)
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS * 1e6
    except (TypeError, np.exceptions.DTypePromotionError,
            np._core._exceptions._UFuncOutputCastingError):
        for _ in range(WARMUP):
            op(a, scalar)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            op(a, scalar)
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS * 1e6


def time_scalar_inplace(op, a, scalar):
    """Benchmark an inplace scalar op (a op= scalar), return avg microseconds."""
    for _ in range(WARMUP):
        op(a, scalar, out=a)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        op(a, scalar, out=a)
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


# ── Benchmark: scalar ops ─────────────────────────────────────────────

def bench_scalar_ops(size):
    print(f"{'━' * 82}")
    print(f"  SCALAR OPS  ({size} elements, {ITERS} iters)")
    print_header("dtype")

    ops = [np.add, np.subtract, np.multiply, np.divide]

    for name, dt in ALL_DTYPES:
        v   = FILL_VALUES[dt]
        a   = np.full(size, v, dtype=dt)
        out = np.empty(size, dtype=dt)
        scalar = dt(2)

        us   = [time_scalar(op, a, scalar, out) for op in ops]
        mops = [size / u for u in us]
        print_row(name, us, mops)


# ── Benchmark: scalar inplace ops ─────────────────────────────────────

def bench_scalar_inplace_ops(size):
    print(f"\n{'━' * 82}")
    print(f"  SCALAR INPLACE  ({size} elements, {ITERS} iters)")
    print_header("dtype")

    ops = [np.add, np.subtract, np.multiply, np.divide]

    for name, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)

        # For integer inplace divide, NumPy rejects out=a (type promotion).
        # Use floor_divide for integers, true_divide for floats.
        inplace_ops = list(ops)
        if not np.issubdtype(dt, np.floating):
            inplace_ops[3] = np.floor_divide

        us = []
        for op in inplace_ops:
            # Reset array before each op to avoid overflow/underflow drift
            a[:] = v
            try:
                us.append(time_scalar_inplace(op, a, dt(1)))
            except (TypeError, np.exceptions.DTypePromotionError,
                    np._core._exceptions._UFuncOutputCastingError):
                # Fallback: measure without out=
                for _ in range(WARMUP):
                    op(a, dt(1))
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    op(a, dt(1))
                t1 = time.perf_counter()
                us.append((t1 - t0) / ITERS * 1e6)

        mops = [size / u for u in us]
        print_row(name, us, mops)


# ── Benchmark: scaling across sizes ───────────────────────────────────

def bench_scalar_scaling():
    print(f"\n{'━' * 82}")
    print(f"  SIZE SCALING  (float32 add_scalar, {ITERS} iters)")
    print(f"\n  {'elements':>10s} {'time (us)':>10s} {'Mops/s':>10s} {'GB/s':>10s}")
    print(f"  {'─' * 42}")

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    for n in sizes:
        a   = np.full(n, 1.5, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)

        us   = time_scalar(np.add, a, np.float32(2.0), out)
        mops = n / us
        gbs  = (2.0 * n * 4) / (us * 1e3)  # read 1 + write 1

        print(f"  {n:10d} {us:10.2f} {mops:10.1f} {gbs:10.2f}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print(f"\n  numpy scalar element-wise benchmark")
    print(f"  numpy {np.__version__}\n")

    bench_scalar_ops(1_000_000)
    bench_scalar_inplace_ops(1_000_000)
    bench_scalar_scaling()

    print()


if __name__ == "__main__":
    main()
