# Benchmark Results

Automated performance comparison of numc against NumPy across all supported
operations, data types, and array configurations.

## Test Environment

| Component    | Specification                                    |
|--------------|--------------------------------------------------|
| CPU          | 13th Gen Intel Core i7-13620H (10C/16T, 4.9 GHz)|
| Architecture | x86_64 (AVX2, FMA)                               |
| OS           | Linux 6.12.74-1-lts (Arch)                        |
| Compiler     | Clang 21.1.8 + OpenMP                             |
| NumPy        | 2.4.2 (OpenBLAS backend)                          |
| Array Size   | 1,000,000 elements (contiguous)                   |
| Iterations   | 200 timed / 20 warmup                             |

## Overview

Median speedup across all operations and data types, grouped by category.
Values above 1.0 indicate numc is faster than NumPy.

![Overview](output/overview.png)

## Element-wise Operations

### Binary (add, sub, mul, div, pow, maximum, minimum)

![Binary Speedup](output/binary_speedup.png)

![Binary Time](output/binary_time.png)

### Binary Add Across Data Types

Speedup for binary `add` across all 10 supported data types (int8 through
float64), demonstrating performance characteristics at different element widths.

![Binary Add Dtypes](output/binary_add_dtypes.png)

### Ternary (fma, where)

![Ternary Speedup](output/ternary_speedup.png)

![Ternary Time](output/ternary_time.png)

## Scalar Operations

### Scalar (a + scalar, a * scalar, ...)

![Scalar Speedup](output/scalar_speedup.png)

![Scalar Time](output/scalar_time.png)

### Scalar Inplace (a += scalar, a *= scalar, ...)

![Scalar Inplace Speedup](output/scalar_inplace_speedup.png)

![Scalar Inplace Time](output/scalar_inplace_time.png)

## Unary Operations

### Unary (log, exp, sqrt, abs, neg, clip)

![Unary Speedup](output/unary_speedup.png)

![Unary Time](output/unary_time.png)

### Unary Inplace

![Unary Inplace Speedup](output/unary_inplace_speedup.png)

![Unary Inplace Time](output/unary_inplace_time.png)

## Comparison Operations

### Array vs Array (eq, gt, lt, ge, le)

![Comparison Speedup](output/comparison_speedup.png)

![Comparison Time](output/comparison_time.png)

### Array vs Scalar

![Comparison Scalar Speedup](output/comparison_scalar_speedup.png)

![Comparison Scalar Time](output/comparison_scalar_time.png)

## Reductions

Full reduction (1M elements to scalar) for sum, mean, max, min, argmax,
argmin on float32.

![Reduction Full](output/reduction_full.png)

## Matrix Multiplication

Square matrix multiplication at 64x64, 128x128, 256x256, and 512x512
(float32). NumPy delegates to optimized BLAS (OpenBLAS/MKL); numc uses
a vendored BLIS backend.

![Matmul](output/matmul.png)

## Random Number Generation

![Random Speedup](output/random_speedup.png)

![Random Time](output/random_time.png)

## All Operations (float32)

Complete speedup chart for every benchmarked operation at float32 precision.

![All Ops Speedup](output/all_ops_speedup.png)

## Reproducing These Results

```bash
# Build release and run both benchmark suites
./run.sh bench

# Generate charts
bench/graph/.venv/bin/python3 bench/graph/plot.py
```

Charts are written to `bench/graph/output/`. See the
[bench README](../README.md) for detailed instructions on environment setup,
CSV format, and interpretation guidelines.

Results are hardware-dependent. Always run both numc and NumPy benchmarks on
the same machine in the same session for valid comparisons.
