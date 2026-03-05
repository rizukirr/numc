"""
NumPy pow benchmark — mirrors bench_pow.c output format.

Usage:
    python bench/numpy_bench_pow.py
"""

import numpy as np
import time

WARMUP = 20
ITERS  = 200

# ── Timer helpers ──────────────────────────────────────────────────────

def time_pow(a, b, out):
    for _ in range(WARMUP):
        np.power(a, b, out=out)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        np.power(a, b, out=out)
    t1 = time.perf_counter()

    return (t1 - t0) / ITERS * 1e6


def time_pow_inplace(a, b):
    for _ in range(WARMUP):
        np.power(a, b, out=a)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        np.power(a, b, out=a)
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

BASE_VALUES = {
    np.int8: 3, np.int16: 3, np.int32: 3, np.int64: 3,
    np.uint8: 3, np.uint16: 3, np.uint32: 3, np.uint64: 3,
    np.float32: 2.0, np.float64: 2.0,
}

EXP_VALUES = {
    np.int8: 3, np.int16: 3, np.int32: 3, np.int64: 3,
    np.uint8: 3, np.uint16: 3, np.uint32: 3, np.uint64: 3,
    np.float32: 3.0, np.float64: 3.0,
}


# ── Benchmark: contiguous pow ──────────────────────────────────────────

def bench_contiguous(size):
    print(f"{'━' * 82}")
    print(f"  POW CONTIGUOUS  ({size} elements, {ITERS} iters)")
    print(f"\n  {'dtype':<8s} {'pow':>10s} {'inplace':>10s}   {'pow':>10s} {'inplace':>10s}")
    print(f"  {'':<8s} {'(us)':>10s} {'(us)':>10s}   {'(Mop/s)':>10s} {'(Mop/s)':>10s}")
    print(f"  {'─' * 58}")

    for name, dt in ALL_DTYPES:
        vb = BASE_VALUES[dt]
        ve = EXP_VALUES[dt]
        a   = np.full(size, vb, dtype=dt)
        b   = np.full(size, ve, dtype=dt)
        out = np.empty(size, dtype=dt)
        a_ip = np.full(size, vb, dtype=dt)

        us_pow = time_pow(a, b, out)
        us_ip  = time_pow_inplace(a_ip, b)

        print(f"  {name:<8s} {us_pow:10.2f} {us_ip:10.2f}   "
              f"{size/us_pow:10.1f} {size/us_ip:10.1f}")


# ── Benchmark: size scaling ────────────────────────────────────────────

def bench_scaling():
    print(f"\n{'━' * 82}")
    print(f"  SIZE SCALING  (float32 pow, {ITERS} iters)")
    print(f"\n  {'elements':>10s} {'time (us)':>10s} {'Mops/s':>10s} {'GB/s':>10s}")
    print(f"  {'─' * 42}")

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    for n in sizes:
        a   = np.full(n, 2.0, dtype=np.float32)
        b   = np.full(n, 3.0, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)

        us   = time_pow(a, b, out)
        mops = n / us
        gbs  = (3.0 * n * 4) / (us * 1e3)

        print(f"  {n:10d} {us:10.2f} {mops:10.1f} {gbs:10.2f}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print(f"\n  numpy pow benchmark")
    print(f"  numpy {np.__version__}\n")

    bench_contiguous(1_000_000)
    bench_scaling()

    print()


if __name__ == "__main__":
    main()
