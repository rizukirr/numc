"""
NumPy benchmark — mirrors numc's comprehensive_benchmark.c

Run:  python3 examples/numpy_benchmark.py
Compare with:  ./run.sh benchmark
"""

import numpy as np
import time
import sys


def bench(func, iters=100):
    """Benchmark a function, return elapsed microseconds per call."""
    # warmup
    for _ in range(10):
        func()
    t0 = time.perf_counter()
    for _ in range(iters):
        func()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


def run_benchmarks(dtype_name, dtype, N, iters):
    """Run all benchmarks for a given dtype."""
    is_int = np.issubdtype(dtype, np.integer)

    if is_int:
        a = np.random.randint(1, 100, N, dtype=dtype)
        b = np.random.randint(1, 100, N, dtype=dtype)
    else:
        a = np.random.rand(N).astype(dtype) + 1.0
        b = np.random.rand(N).astype(dtype) + 1.0

    out = np.empty_like(a)
    scalar = dtype(3)

    print(f"\n{dtype_name} Array ({N:,} elements):\n")

    # Binary ops
    print("  Binary Operations:")
    for name, op in [("Add", np.add), ("Subtract", np.subtract),
                     ("Multiply", np.multiply)]:
        t = bench(lambda op=op: op(a, b, out=out), iters)
        print(f"    {name + ':':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    if not is_int:
        t = bench(lambda: np.divide(a, b, out=out), iters)
        print(f"    {'Divide:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    print()

    # Scalar ops
    print("  Scalar Operations:")
    for name, op in [("Add scalar", np.add), ("Sub scalar", np.subtract),
                     ("Mul scalar", np.multiply)]:
        t = bench(lambda op=op: op(a, scalar, out=out), iters)
        print(f"    {name + ':':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    if not is_int:
        t = bench(lambda: np.divide(a, scalar, out=out), iters)
        print(f"    {'Div scalar:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    print()

    # Reductions
    print("  Reduction Operations:")
    t = bench(lambda: np.sum(a), iters)
    print(f"    {'Sum:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.min(a), iters)
    print(f"    {'Min:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.max(a), iters)
    print(f"    {'Max:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.prod(a), iters)
    print(f"    {'Prod:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.dot(a, b), iters)
    print(f"    {'Dot product:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.mean(a), iters)
    print(f"    {'Mean:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.std(a), iters)
    print(f"    {'Std:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    print()

    # Axis reductions (1000x1000)
    side = int(N ** 0.5)
    a2d = a[:side * side].reshape(side, side)
    print(f"  Axis Reduction Operations (axis=0 on 2D {side}x{side}):")
    t = bench(lambda: np.mean(a2d, axis=0), iters)
    print(f"    {'Mean:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.mean(a2d, axis=1), iters)
    print(f"    {'Mean(ax=1):':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.std(a2d, axis=0), iters)
    print(f"    {'Std:':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")
    t = bench(lambda: np.std(a2d, axis=1), iters)
    print(f"    {'Std(ax=1):':<17}{t:8.2f} us  ({N / t:10,.0f} Mops/sec)")


def main():
    N = 1_000_000
    ITERS = 100

    print("=" * 67)
    print(f"  NumPy Benchmark — v{np.__version__}")
    print(f"  Compare with: ./run.sh benchmark")
    print(f"  Python {sys.version.split()[0]} | {N:,} elements | {ITERS} iterations")
    print("=" * 67)

    types = [
        ("BYTE (INT8)", np.int8),
        ("UBYTE (UINT8)", np.uint8),
        ("SHORT (INT16)", np.int16),
        ("USHORT (UINT16)", np.uint16),
        ("INT32", np.int32),
        ("UINT32", np.uint32),
        ("INT64", np.int64),
        ("UINT64", np.uint64),
        ("FLOAT", np.float32),
        ("DOUBLE", np.float64),
    ]

    for dtype_name, dtype in types:
        run_benchmarks(dtype_name, dtype, N, ITERS)

    print("\n" + "=" * 67)
    print("  All operations completed.")
    print("=" * 67)


if __name__ == "__main__":
    main()
