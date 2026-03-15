# Benchmarks

Performance is a core pillar of `numc`. This directory contains a comprehensive
benchmark suite designed to compare `numc` against NumPy under identical
conditions.

## Quick Start

### 1. Build numc in Release Mode
Benchmarks must be run on optimized binaries to be meaningful.

```bash
./run.sh release
```

### 2. Run All Benchmarks
This command runs both the `numc` and `numpy` benchmark suites, compares the
results, and generates visualization charts.

```bash
./run.sh bench
```

Charts are saved as PNG files in `bench/graph/output/`.

---

## How to Interpret the Results

We use three primary metrics to evaluate performance:

### 1. time_us (Lower is Better)
The average wall-clock time for a single execution of the operation, measured
in microseconds. We use a **Warmup-then-Timed** approach:
- **Warmup:** 20 iterations to prime the instruction cache and branch predictor.
- **Timed:** 200 iterations to calculate a stable average.

### 2. throughput_mops (Higher is Better)
**Millions of Operations Per Second.** This normalizes performance across
different array sizes.
- Formula: `(Total Elements) / (time_us)`
- Useful for comparing efficiency across different data types or hardware.

### 3. Speedup (Higher is Faster for numc)
The relative performance factor compared to NumPy.
- Formula: `numpy_time / numc_time`
- **> 1.0x:** `numc` is faster than NumPy.
- **< 1.0x:** NumPy is faster than `numc`.

---

## Benchmarking Checklist for Reproducibility

To get consistent and high-fidelity results, follow these guidelines:

1.  **Set CPU Governor to Performance:** On Linux, modern CPUs throttle
    frequencies aggressively.
    ```bash
    sudo cpupower frequency-set -g performance
    ```
2.  **Close Background Applications:** Heavy IDEs, browsers, or other background
    tasks can cause context-switching noise.
3.  **Use the Same Machine:** Never compare `numc` results from one machine to
    NumPy results from another.
4.  **Check OpenMP Bindings:** `run.sh` sets `OMP_PROC_BIND=close` and
    `OMP_PLACES=cores` by default to minimize thread migration.

---

## CSV Format

The benchmark results are stored in `bench/numc/results.csv` and
`bench/numpy/results.csv` with the following schema:

| Column           | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `library`        | `numc` or `numpy`                                          |
| `category`       | Operation group (binary, unary, reduction, matmul, etc.)   |
| `operation`      | Specific function name (add, sub, log, sum)                |
| `dtype`          | Data type (float32, int64, etc.)                           |
| `size`           | Total number of elements processed                         |
| `shape`          | String representation of geometry, e.g. `(1024, 1024)`      |
| `time_us`        | Average execution time in microseconds                     |
| `throughput_mops`| Calculated throughput (Millions of Ops/sec)                |

---

## Directory Structure

-   `numc/`: C-based benchmark implementations and results.
-   `numpy/`: Python-based NumPy counterparts and results.
-   `graph/`: Visualization logic and generated comparison charts.
-   `compare.py`: CLI tool for quick text-based comparison of CSV files.

## Adding New Benchmarks

When adding a new feature to `numc`, ensure it is benchmarked:
1.  Add the operation to `bench/numc/bench.c`.
2.  Add the equivalent NumPy call to `bench/numpy/bench.py`.
3.  Run `./run.sh bench` to verify the performance delta.
