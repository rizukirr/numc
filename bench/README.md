# Benchmarks

This directory contains the benchmark suite for comparing numc against NumPy.
Both libraries are tested with identical operations, data types, array sizes,
and iteration counts to ensure a fair comparison.

## Directory Structure

```
bench/
  numc/
    bench.c              # C benchmark — all numc operations (CSV output)
    results.csv          # Generated output from bench.c
  numpy/
    bench.py             # Python benchmark — mirrors bench.c exactly
    results.csv          # Generated output from bench.py
    requirements.txt     # Python dependencies (numpy)
  graph/
    plot.py              # Generates comparison charts from both CSVs
    requirements.txt     # Python dependencies (pandas, matplotlib, numpy)
    output/              # Generated PNG charts
  bench_*.c              # Standalone benchmarks (elemwise, matmul, dot, etc.)
  numpy_bench_*.py       # Standalone NumPy counterparts
```

## Quick Start

### Run all benchmarks and generate graphs

```bash
# 1. Build and run both numc and numpy benchmarks
./run.sh bench

# 2. Generate comparison charts
bench/graph/.venv/bin/python3 bench/graph/plot.py
```

Charts are saved as PNG files in `bench/graph/output/`.

### Run benchmarks separately

```bash
# numc only (requires a release build)
./run.sh release
./build/bin/bench_numc_csv > bench/numc/results.csv

# numpy only
python3 bench/numpy/bench.py > bench/numpy/results.csv
```

### Standalone benchmarks

Individual benchmarks are useful for profiling or testing a single category:

```bash
./build/bin/bench_elemwise
./build/bin/bench_matmul
./build/bin/bench_dot
./build/bin/bench_reduction
./build/bin/bench_scalar
./build/bin/bench_unary
./build/bin/bench_pow
./build/bin/bench_fma
./build/bin/bench_random
```

## Setting Up Python Environments

The numpy benchmark and graph scripts require Python with specific packages.

```bash
# For numpy benchmarks
python3 -m venv bench/numpy/.venv
bench/numpy/.venv/bin/pip install -r bench/numpy/requirements.txt

# For graph generation
python3 -m venv bench/graph/.venv
bench/graph/.venv/bin/pip install -r bench/graph/requirements.txt
```

## CSV Format

Both `bench/numc/results.csv` and `bench/numpy/results.csv` share the same schema:

```
library,category,operation,dtype,size,shape,time_us,throughput_mops
```

| Column         | Description                                         |
|----------------|-----------------------------------------------------|
| library        | `numc` or `numpy`                                   |
| category       | Operation group (binary, unary, scalar, reduction, matmul, etc.) |
| operation      | Specific operation (add, sub, mul, div, log, exp, sum, etc.)     |
| dtype          | Data type (int8, uint8, int16, ..., float32, float64)            |
| size           | Total number of elements                             |
| shape          | Array shape as a string, e.g. `(1000000)` or `(512x512)@(512x512)` |
| time_us        | Average time per iteration in microseconds           |
| throughput_mops| Throughput in millions of operations per second       |

## Reading the Results

**time_us** — Lower is better. This is the average wall-clock time for one
operation, measured after a warmup phase (20 iterations by default) over 200
timed iterations.

**throughput_mops** — Higher is better. Computed as `size / time_us`. This
normalizes across different array sizes and makes it easier to compare
operations that run on different element counts.

**speedup** — Shown in the generated charts. Computed as
`numpy_time / numc_time`. Values above 1.0 mean numc is faster; below 1.0
means numpy is faster.

## Benchmark Configuration

Both the C and Python benchmarks use the same defaults:

| Parameter | Value     | Purpose                              |
|-----------|-----------|--------------------------------------|
| WARMUP    | 20        | Iterations before timing starts      |
| ITERS     | 200       | Timed iterations (average reported)  |
| SIZE      | 1,000,000 | Default array size for element-wise  |

Matmul benchmarks use separate configs (smaller sizes, fewer iterations)
because matrix multiplication is significantly more expensive per call.

## Notes for Contributors

- Always run both numc and numpy benchmarks on the same machine in the same
  session. Do not compare results across different machines or runs with
  different system load.
- Close other CPU-intensive programs before benchmarking.
- The CSV files are gitignored. Results are machine-specific and should not
  be committed.
- When adding a new operation to numc, add the corresponding benchmark to
  both `bench/numc/bench.c` and `bench/numpy/bench.py` to keep them in sync.
- The `plot.py` script automatically picks up new categories from the CSV
  files, so new operations will appear in charts without modifying the
  plotting code (as long as they use existing category names).
