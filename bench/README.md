# Benchmarks

Performance is a core pillar of `numc`. This directory contains a comprehensive
benchmark suite that measures `numc` across all operation categories under
controlled conditions.

## Quick Start

### 1. Build numc in Release Mode
Benchmarks must be run on optimized binaries to be meaningful.

```bash
./run.sh release
```

### 2. Run All Benchmarks
This command runs the `numc` benchmark suite and writes the results to
`bench/numc/results.csv`.

```bash
./run.sh bench
```

You can also benchmark a single category, e.g. `./run.sh bench matmul`.

---

## How to Interpret the Results

We use two primary metrics to evaluate performance:

### 1. time_us (Lower is Better)
The average wall-clock time for a single execution of the operation, measured
in microseconds. We use a **Warmup-then-Timed** approach:
- **Warmup:** 20 iterations to prime the instruction cache and branch predictor.
- **Timed:** 200 iterations, reporting the minimum per-iteration time (most
  stable, least affected by OS scheduling noise).

### 2. throughput_mops (Higher is Better)
**Millions of Operations Per Second.** This normalizes performance across
different array sizes.
- Formula: `(Total Elements) / (time_us)`
- Useful for comparing efficiency across different data types or hardware.

> **Cache caveat:** most categories run at a fixed `1M`-element size, whose
> working set is largely **L3-resident** — so those throughput numbers reflect
> in-cache speed, not stream-from-DRAM speed. The `cache` category runs `add`
> across sizes that grow from L1 into DRAM to make this explicit; expect
> throughput to fall several-fold once the working set exceeds L3.

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
3.  **Use the Same Machine:** Never compare results gathered on different
    machines.
4.  **Check OpenMP Bindings:** `run.sh` sets `OMP_PROC_BIND=close` and
    `OMP_PLACES=cores` by default to minimize thread migration.

---

## CSV Format

The benchmark results are stored in `bench/numc/results.csv` with the following
schema:

| Column           | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `library`        | `numc`                                                     |
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

## Adding New Benchmarks

When adding a new feature to `numc`, ensure it is benchmarked:
1.  Add the operation to `bench/numc/bench.c`.
2.  Run `./run.sh bench` to record its performance.
