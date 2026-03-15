"""
bench/numpy/bench.py — Comprehensive NumPy benchmark (CSV output)

Mirrors bench/numc/bench.c exactly: same operations, dtypes, sizes, CSV format.

CSV columns: library,category,operation,dtype,size,shape,time_us,throughput_mops

Usage:
    python bench/numpy/bench.py > results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
import sys

# ── Config ────────────────────────────────────────────────────────────

WARMUP = 20
ITERS = 200
SIZE = 1_000_000

ALL_DTYPES = [
    ("int8", np.int8), ("uint8", np.uint8),
    ("int16", np.int16), ("uint16", np.uint16),
    ("int32", np.int32), ("uint32", np.uint32),
    ("int64", np.int64), ("uint64", np.uint64),
    ("float32", np.float32), ("float64", np.float64),
]

FLOAT_DTYPES = [
    ("float32", np.float32), ("float64", np.float64),
]

FILL_VALUES = {
    np.int8: 3, np.int16: 7, np.int32: 42, np.int64: 42,
    np.uint8: 3, np.uint16: 7, np.uint32: 42, np.uint64: 42,
    np.float32: 1.5, np.float64: 1.5,
}

FILL_VALUES_EXP = {
    np.int8: 2, np.int16: 2, np.int32: 2, np.int64: 2,
    np.uint8: 2, np.uint16: 2, np.uint32: 2, np.uint64: 2,
    np.float32: 1.5, np.float64: 1.5,
}

FILL_VALUES_POW_EXP = {
    np.int8: 3, np.int16: 3, np.int32: 3, np.int64: 3,
    np.uint8: 3, np.uint16: 3, np.uint32: 3, np.uint64: 3,
    np.float32: 3.0, np.float64: 3.0,
}

UNSIGNED = {np.uint8, np.uint16, np.uint32, np.uint64}


# ── Helpers ───────────────────────────────────────────────────────────

def csv(cat, op, dt, size, shape, us):
    mops = size / us
    print(f"numpy,{cat},{op},{dt},{size},{shape},{us:.4f},{mops:.4f}")


def bench_fn(fn, iters=ITERS):
    """Time a zero-arg callable, return minimum microseconds (most stable)."""
    for _ in range(WARMUP):
        fn()
    min_us = float('inf')
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1e6
        if elapsed < min_us:
            min_us = elapsed
    return min_us


# ── Binary element-wise ──────────────────────────────────────────────

def bench_binary(name, np_op, size):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)
        b = np.full(size, v, dtype=dt)
        out = np.empty(size, dtype=dt)

        try:
            us = bench_fn(lambda: np_op(a, b, out=out))
        except (TypeError, np.exceptions.DTypePromotionError,
                np._core._exceptions._UFuncOutputCastingError):
            us = bench_fn(lambda: np_op(a, b))

        csv("binary", name, dname, size, f"({size})", us)


# ── Pow ───────────────────────────────────────────────────────────────

def bench_pow(size):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        e = FILL_VALUES_POW_EXP[dt]
        a = np.full(size, v, dtype=dt)
        b = np.full(size, e, dtype=dt)

        try:
            out = np.empty(size, dtype=dt)
            us = bench_fn(lambda: np.power(a, b, out=out))
        except (TypeError, np.exceptions.DTypePromotionError,
                np._core._exceptions._UFuncOutputCastingError):
            us = bench_fn(lambda: np.power(a, b))

        csv("binary", "pow", dname, size, f"({size})", us)


# ── FMA (a*b + c) ────────────────────────────────────────────────────

def bench_fma(size):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)
        b = np.full(size, v, dtype=dt)
        c = np.full(size, v, dtype=dt)

        # NumPy has no fused FMA, so emulate with mul + add
        try:
            tmp = np.empty(size, dtype=dt)
            out = np.empty(size, dtype=dt)
            us = bench_fn(lambda: np.add(np.multiply(a, b, out=tmp), c, out=out))
        except (TypeError, np.exceptions.DTypePromotionError,
                np._core._exceptions._UFuncOutputCastingError):
            us = bench_fn(lambda: a * b + c)

        csv("ternary", "fma", dname, size, f"({size})", us)


# ── Where ─────────────────────────────────────────────────────────────

def bench_where(size):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        cond = np.ones(size, dtype=np.bool_)
        a = np.full(size, v, dtype=dt)
        b = np.full(size, v, dtype=dt)

        us = bench_fn(lambda: np.where(cond, a, b))
        csv("ternary", "where", dname, size, f"({size})", us)


# ── Scalar ops ────────────────────────────────────────────────────────

def bench_scalar_op(name, np_op, size, scalar=2.0):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)
        out = np.empty(size, dtype=dt)

        try:
            us = bench_fn(lambda: np_op(a, scalar, out=out))
        except (TypeError, np.exceptions.DTypePromotionError,
                np._core._exceptions._UFuncOutputCastingError):
            us = bench_fn(lambda: np_op(a, scalar))

        csv("scalar", name, dname, size, f"({size})", us)


# ── Scalar inplace ops ───────────────────────────────────────────────

def bench_scalar_inplace(name, size, scalar=1.01):
    """NumPy inplace scalar: a += scalar, a -= scalar, etc."""
    op_map = {
        "add_scalar_inplace": lambda a: np.add(a, scalar, out=a),
        "sub_scalar_inplace": lambda a: np.subtract(a, scalar, out=a),
        "mul_scalar_inplace": lambda a: np.multiply(a, scalar, out=a),
        "div_scalar_inplace": lambda a: np.divide(a, scalar, out=a),
    }
    op = op_map[name]

    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)

        try:
            us = bench_fn(lambda: op(a))
        except (TypeError, np.exceptions.DTypePromotionError,
                np._core._exceptions._UFuncOutputCastingError):
            # Fallback: cast scalar to match dtype so out= works
            s = dt(scalar)
            fallback_map = {
                "add_scalar_inplace": lambda a: np.add(a, s, out=a),
                "sub_scalar_inplace": lambda a: np.subtract(a, s, out=a),
                "mul_scalar_inplace": lambda a: np.multiply(a, s, out=a),
                "div_scalar_inplace": lambda a: np.divide(a, s, out=a),
            }
            fb = fallback_map[name]
            try:
                us = bench_fn(lambda: fb(a))
            except (TypeError, np.exceptions.DTypePromotionError,
                    np._core._exceptions._UFuncOutputCastingError):
                # Last resort: no out=
                np_op_map = {
                    "add_scalar_inplace": np.add,
                    "sub_scalar_inplace": np.subtract,
                    "mul_scalar_inplace": np.multiply,
                    "div_scalar_inplace": np.divide,
                }
                np_fn = np_op_map[name]
                us = bench_fn(lambda: np_fn(a, s))

        csv("scalar_inplace", name, dname, size, f"({size})", us)


# ── Unary ops ─────────────────────────────────────────────────────────

def bench_unary(name, np_op, size, skip_unsigned=False, use_exp_fill=False):
    for dname, dt in ALL_DTYPES:
        if skip_unsigned and dt in UNSIGNED:
            continue

        v = FILL_VALUES_EXP[dt] if use_exp_fill else FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)

        try:
            out = np.empty(size, dtype=dt)
            us = bench_fn(lambda: np_op(a, out=out))
        except (TypeError, np.exceptions.DTypePromotionError,
                np._core._exceptions._UFuncOutputCastingError):
            us = bench_fn(lambda: np_op(a))

        csv("unary", name, dname, size, f"({size})", us)


# ── Unary inplace ops ────────────────────────────────────────────────

def bench_unary_inplace(name, np_op, size, skip_unsigned=False, use_exp_fill=False):
    for dname, dt in ALL_DTYPES:
        if skip_unsigned and dt in UNSIGNED:
            continue

        v = FILL_VALUES_EXP[dt] if use_exp_fill else FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)

        try:
            us = bench_fn(lambda: np_op(a, out=a))
        except (TypeError, np.exceptions.DTypePromotionError,
                np._core._exceptions._UFuncOutputCastingError):
            us = bench_fn(lambda: np_op(a))

        csv("unary_inplace", name, dname, size, f"({size})", us)


# ── Clip ──────────────────────────────────────────────────────────────

def bench_clip(size):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)

        us = bench_fn(lambda: np.clip(a, 1.0, 5.0))
        csv("unary", "clip", dname, size, f"({size})", us)


# ── Comparison ops ────────────────────────────────────────────────────

def bench_comparison(name, np_op, size):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)
        b = np.full(size, v, dtype=dt)

        us = bench_fn(lambda: np_op(a, b))
        csv("comparison", name, dname, size, f"({size})", us)


# ── Comparison scalar ops ────────────────────────────────────────────

def bench_comparison_scalar(name, np_op, size, scalar=2.0):
    for dname, dt in ALL_DTYPES:
        v = FILL_VALUES[dt]
        a = np.full(size, v, dtype=dt)

        us = bench_fn(lambda: np_op(a, scalar))
        csv("comparison_scalar", name, dname, size, f"({size})", us)


# ── Full reductions ──────────────────────────────────────────────────

def bench_reduce_full(name, size):
    for dname, dt in ALL_DTYPES:
        a = np.ones(size, dtype=dt)
        method = getattr(a, name)

        us = bench_fn(method)
        csv("reduction", name, dname, size, f"({size})", us)


# ── Axis reductions ──────────────────────────────────────────────────

def bench_reduce_axis(name, axis, rows, cols):
    total = rows * cols
    op_name = f"{name}_axis{axis}"
    for dname, dt in ALL_DTYPES:
        a = np.ones((rows, cols), dtype=dt)
        method = getattr(a, name)

        us = bench_fn(lambda: method(axis=axis))
        csv("reduction", op_name, dname, total, f"({rows}x{cols})", us)


# ── Matmul ────────────────────────────────────────────────────────────

def bench_matmul(M, K, N, warmup, iters):
    """Matmul benchmark: per-iteration timing, reports minimum (most stable)."""
    np.random.seed(42)  # Deterministic random data

    dtypes = [
        ("float32", np.float32), ("float64", np.float64),
        ("int8", np.int8), ("int16", np.int16),
        ("int32", np.int32), ("int64", np.int64),
        ("uint8", np.uint8), ("uint16", np.uint16),
        ("uint32", np.uint32), ("uint64", np.uint64),
    ]

    for dname, dt in dtypes:
        if dt in (np.float32, np.float64):
            # Random data for float types (avoids special-case optimizations)
            a = np.random.randn(M, K).astype(dt)
            b = np.random.randn(K, N).astype(dt)
            w, it = warmup, iters
        else:
            # Integer types: use fill with 2 (random integers can overflow)
            v = dt(2)
            a = np.full((M, K), v, dtype=dt)
            b = np.full((K, N), v, dtype=dt)
            # NumPy integer matmul has no BLAS — use reduced iters
            w = min(warmup, 5)
            it = min(iters, 10)

        # NumPy matmul doesn't support int8/uint8 etc natively via @
        # Use np.matmul which works for all types
        try:
            for _ in range(w):
                np.matmul(a, b)

            # Per-iteration timing: report minimum
            min_us = float('inf')
            for _ in range(it):
                t0 = time.perf_counter()
                np.matmul(a, b)
                elapsed = (time.perf_counter() - t0) * 1e6
                if elapsed < min_us:
                    min_us = elapsed
        except TypeError:
            # Some dtypes may not be supported, use float64 cast
            af = a.astype(np.float64)
            bf = b.astype(np.float64)
            for _ in range(w):
                np.matmul(af, bf)
            min_us = float('inf')
            for _ in range(it):
                t0 = time.perf_counter()
                np.matmul(af, bf)
                elapsed = (time.perf_counter() - t0) * 1e6
                if elapsed < min_us:
                    min_us = elapsed

        total = M * N
        # GFLOPS = 2*M*K*N / time_us / 1e3
        flops = 2.0 * M * K * N
        gflops = flops / min_us / 1e3
        csv("matmul", "matmul", dname, total,
            f"({M}x{K})@({K}x{N})", min_us)


# ── Dot product ───────────────────────────────────────────────────────

def bench_dot(size):
    for dname, dt in FLOAT_DTYPES:
        a = np.random.randn(size).astype(dt)
        b = np.random.randn(size).astype(dt)

        iters = 1000
        for _ in range(WARMUP):
            np.dot(a, b)

        t0 = time.perf_counter()
        for _ in range(iters):
            np.dot(a, b)
        t1 = time.perf_counter()
        us = (t1 - t0) / iters * 1e6

        csv("linalg", "dot", dname, size, f"({size})", us)


# ── Random ────────────────────────────────────────────────────────────

def bench_random(name, size):
    iters = 50
    gen_map = {
        "rand": lambda dt: np.random.rand(size).astype(dt),
        "randn": lambda dt: np.random.randn(size).astype(dt),
    }
    gen = gen_map[name]

    for dname, dt in FLOAT_DTYPES:
        # Warmup
        for _ in range(5):
            gen(dt)

        t0 = time.perf_counter()
        for _ in range(iters):
            gen(dt)
        t1 = time.perf_counter()
        us = (t1 - t0) / iters * 1e6

        csv("random", name, dname, size, f"({size})", us)


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("library,category,operation,dtype,size,shape,time_us,throughput_mops")

    # Optional --matmul flag for matmul-only benchmark
    if "--matmul" in sys.argv:
        bench_matmul(64, 64, 64, 200, 2000)
        bench_matmul(128, 128, 128, 100, 500)
        bench_matmul(256, 256, 256, 50, 100)
        bench_matmul(512, 512, 512, 20, 50)
        bench_matmul(1024, 1024, 1024, 10, 20)
        return

    # Binary element-wise
    bench_binary("add", np.add, SIZE)
    bench_binary("sub", np.subtract, SIZE)
    bench_binary("mul", np.multiply, SIZE)
    bench_binary("div", np.divide, SIZE)
    bench_binary("maximum", np.maximum, SIZE)
    bench_binary("minimum", np.minimum, SIZE)
    bench_pow(SIZE)

    # FMA and Where
    bench_fma(SIZE)
    bench_where(SIZE)

    # Scalar ops
    bench_scalar_op("add_scalar", np.add, SIZE)
    bench_scalar_op("sub_scalar", np.subtract, SIZE)
    bench_scalar_op("mul_scalar", np.multiply, SIZE)
    bench_scalar_op("div_scalar", np.divide, SIZE)

    # Scalar inplace
    bench_scalar_inplace("add_scalar_inplace", SIZE)
    bench_scalar_inplace("sub_scalar_inplace", SIZE)
    bench_scalar_inplace("mul_scalar_inplace", SIZE)
    bench_scalar_inplace("div_scalar_inplace", SIZE)

    # Unary ops
    bench_unary("neg", np.negative, SIZE)
    bench_unary("abs", np.abs, SIZE, skip_unsigned=True)
    bench_unary("log", np.log, SIZE)
    bench_unary("exp", np.exp, SIZE, use_exp_fill=True)
    bench_unary("sqrt", np.sqrt, SIZE)
    bench_clip(SIZE)

    # Unary inplace
    bench_unary_inplace("neg_inplace", np.negative, SIZE)
    bench_unary_inplace("abs_inplace", np.abs, SIZE, skip_unsigned=True)
    bench_unary_inplace("log_inplace", np.log, SIZE)
    bench_unary_inplace("exp_inplace", np.exp, SIZE, use_exp_fill=True)
    bench_unary_inplace("sqrt_inplace", np.sqrt, SIZE)

    # Comparison ops
    bench_comparison("eq", np.equal, SIZE)
    bench_comparison("gt", np.greater, SIZE)
    bench_comparison("lt", np.less, SIZE)
    bench_comparison("ge", np.greater_equal, SIZE)
    bench_comparison("le", np.less_equal, SIZE)

    # Comparison scalar ops
    bench_comparison_scalar("eq_scalar", np.equal, SIZE)
    bench_comparison_scalar("gt_scalar", np.greater, SIZE)
    bench_comparison_scalar("lt_scalar", np.less, SIZE)
    bench_comparison_scalar("ge_scalar", np.greater_equal, SIZE)
    bench_comparison_scalar("le_scalar", np.less_equal, SIZE)

    # Full reductions
    for op in ("sum", "mean", "max", "min", "argmax", "argmin"):
        bench_reduce_full(op, SIZE)

    # Axis reductions (1000x1000)
    for op in ("sum", "mean", "max", "min", "argmax", "argmin"):
        bench_reduce_axis(op, 0, 1000, 1000)
        bench_reduce_axis(op, 1, 1000, 1000)

    # Matmul — various sizes
    bench_matmul(64, 64, 64, 200, 2000)
    bench_matmul(128, 128, 128, 100, 500)
    bench_matmul(256, 256, 256, 50, 100)
    bench_matmul(512, 512, 512, 20, 50)
    bench_matmul(1024, 1024, 1024, 10, 20)

    # Dot product
    bench_dot(SIZE)

    # Random
    bench_random("rand", SIZE)
    bench_random("randn", SIZE)


if __name__ == "__main__":
    main()
