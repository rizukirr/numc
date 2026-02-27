"""
NumPy matmul benchmark — mirrors bench_matmul.c output format.

Sections:
  1. Square size scaling  — float32, sizes 32..512, reports GFLOP/s
  2. Dtype comparison     — all 10 dtypes at fixed 256x256
  3. Shape variants       — tall/square/wide at float32

Usage:
    python bench/numpy_bench_matmul.py
"""

import numpy as np
import time

# ── Timer ───────────────────────────────────────────────────────────────

def bench(a, b, out, warmup, iters):
    """Warmup + timed matmul, return avg microseconds."""
    for _ in range(warmup):
        np.matmul(a, b, out=out)

    t0 = time.perf_counter()
    for _ in range(iters):
        np.matmul(a, b, out=out)
    return (time.perf_counter() - t0) / iters * 1e6


def gflops(M, K, N, us):
    return 2.0 * M * K * N / (us * 1e3)


# ── Helpers ─────────────────────────────────────────────────────────────

ALL_DTYPES = [
    ("int8",    np.int8),    ("int16",   np.int16),
    ("int32",   np.int32),   ("int64",   np.int64),
    ("uint8",   np.uint8),   ("uint16",  np.uint16),
    ("uint32",  np.uint32),  ("uint64",  np.uint64),
    ("float32", np.float32), ("float64", np.float64),
]


# ── Section 1: Square size scaling ──────────────────────────────────────

def bench_square_scaling():
    print()
    print("━" * 60)
    print("  SQUARE SIZE SCALING  (float32, NxN @ NxN -> NxN)")
    print(f"\n  {'N':>6s}  {'iters':>6s}  {'time(us)':>8s}  {'time(ms)':>8s}  {'GFLOP/s':>8s}")
    print("  " + "─" * 53)

    sizes = [
        ( 32, 50, 500),
        ( 64, 20, 200),
        (128, 10,  50),
        (256,  5,  20),
        (512,  2,   5),
    ]

    for n, warmup, iters in sizes:
        a   = np.ones((n, n), dtype=np.float32)
        b   = np.ones((n, n), dtype=np.float32)
        out = np.empty((n, n), dtype=np.float32)

        us = bench(a, b, out, warmup, iters)
        print(f"  {n:6d}  {iters:6d}  {us:8.2f}  {us/1e3:8.3f}  {gflops(n,n,n,us):8.3f}")


# ── Section 2: Dtype comparison ──────────────────────────────────────────

def bench_dtype_comparison():
    N = 256
    print()
    print("━" * 60)
    print(f"  DTYPE COMPARISON  ({N}x{N} @ {N}x{N}, 20 iters)")
    print(f"\n  {'dtype':<8s}  {'time(us)':>8s}  {'GFLOP/s':>8s}")
    print("  " + "─" * 30)

    for name, dt in ALL_DTYPES:
        # np.matmul promotes integer types internally — use float32 out for integers
        # to avoid dtype promotion errors; report as if same-dtype
        a   = np.ones((N, N), dtype=dt)
        b   = np.ones((N, N), dtype=dt)

        # np.matmul on integer types returns the same dtype on recent NumPy
        try:
            out = np.empty((N, N), dtype=dt)
            us  = bench(a, b, out, 5, 20)
        except TypeError:
            # fallback: no out= arg
            for _ in range(5):
                np.matmul(a, b)
            t0 = time.perf_counter()
            for _ in range(20):
                np.matmul(a, b)
            us = (time.perf_counter() - t0) / 20 * 1e6

        print(f"  {name:<8s}  {us:8.2f}  {gflops(N,N,N,us):8.3f}")


# ── Section 3: Shape variants ─────────────────────────────────────────────

def bench_shape_variants():
    print()
    print("━" * 60)
    print("  SHAPE VARIANTS  (float32, 20 iters)")
    print(f"\n  {'shape (M,K)@(K,N)':<24s}  {'time(us)':>8s}  {'GFLOP/s':>8s}  {'flops':>8s}")
    print("  " + "─" * 54)

    shapes = [
        (512,  32, 512, "wide K (512x32@32x512)"),
        (512, 512, 512, "square  (512x512@512x512)"),
        ( 32, 512,  32, "tall K  (32x512@512x32)"),
        (256, 128, 512, "rect    (256x128@128x512)"),
        (  1, 256, 256, "vec-mat (1x256@256x256)"),
        (256, 256,   1, "mat-vec (256x256@256x1)"),
    ]

    for M, K, N, label in shapes:
        a   = np.ones((M, K), dtype=np.float32)
        b   = np.ones((K, N), dtype=np.float32)
        out = np.empty((M, N), dtype=np.float32)

        us = bench(a, b, out, 5, 20)
        total_flops = 2.0 * M * K * N
        print(f"  {label:<24s}  {us:8.2f}  {gflops(M,K,N,us):8.3f}  {total_flops/1e3:8.0f} K")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print()
    print("  numpy matmul benchmark")
    print(f"  numpy {np.__version__}  |  BLAS: {np.show_config.__module__}")

    bench_square_scaling()
    bench_dtype_comparison()
    bench_shape_variants()

    print()


if __name__ == "__main__":
    main()
