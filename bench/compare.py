#!/usr/bin/env python3
"""Print a comparison table from numc and numpy CSV benchmark results."""

import csv
import sys
from pathlib import Path

NUMC_CSV = Path(__file__).parent / "numc" / "results.csv"
NUMPY_CSV = Path(__file__).parent / "numpy" / "results.csv"

# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def load_csv(path):
    """Load CSV into dict keyed by (category, operation, dtype, shape)."""
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["category"], row["operation"], row["dtype"], row["shape"])
            results[key] = float(row["time_us"])
    return results


def colorize_speedup(speedup):
    if speedup >= 2.0:
        return f"{GREEN}{BOLD}{speedup:6.2f}x{RESET}"
    elif speedup >= 1.0:
        return f"{GREEN}{speedup:6.2f}x{RESET}"
    elif speedup >= 0.8:
        return f"{YELLOW}{speedup:6.2f}x{RESET}"
    else:
        return f"{RED}{speedup:6.2f}x{RESET}"


def main():
    if not NUMC_CSV.exists() or not NUMPY_CSV.exists():
        print("Missing CSV files. Run ./run.sh bench first.", file=sys.stderr)
        sys.exit(1)

    numc = load_csv(NUMC_CSV)
    numpy = load_csv(NUMPY_CSV)

    # Match keys present in both
    common = sorted(set(numc) & set(numpy))
    if not common:
        print("No matching benchmarks found.", file=sys.stderr)
        sys.exit(1)

    # Build rows
    rows = []
    slower = []
    for key in common:
        cat, op, dtype, shape = key
        t_numc = numc[key]
        t_numpy = numpy[key]
        speedup = t_numpy / t_numc if t_numc > 0 else float("inf")
        rows.append((cat, op, dtype, shape, t_numc, t_numpy, speedup))
        if speedup < 1.0:
            slower.append((cat, op, dtype, shape, t_numc, t_numpy, speedup))

    # Print full table
    hdr = f"{'Category':<12} {'Operation':<24} {'Dtype':<7} {'Shape':<16} {'numc(us)':>10} {'numpy(us)':>10} {'Speedup':>10}"
    sep = "-" * len(hdr)
    print(f"\n{BOLD}numc vs NumPy Benchmark Comparison{RESET}")
    print(f"{DIM}{len(rows)} benchmarks matched{RESET}\n")
    print(hdr)
    print(sep)

    prev_cat = None
    for cat, op, dtype, shape, t_numc, t_numpy, speedup in rows:
        if prev_cat and cat != prev_cat:
            print(sep)
        prev_cat = cat
        sp_str = colorize_speedup(speedup)
        print(
            f"{cat:<12} {op:<24} {dtype:<7} {shape:<16} {t_numc:>10.2f} {t_numpy:>10.2f} {sp_str}"
        )

    # Summary
    import math

    speedups = [r[6] for r in rows]
    finite = [s for s in speedups if math.isfinite(s) and s > 0]
    geo_mean = math.exp(sum(math.log(s) for s in finite) / len(finite)) if finite else 0

    faster = sum(1 for s in speedups if s >= 1.0)
    at_parity = sum(1 for s in speedups if 0.95 <= s < 1.0)
    n_slower = sum(1 for s in speedups if s < 0.95)

    print(f"\n{BOLD}Summary{RESET}")
    print(f"  Total:     {len(rows)}")
    print(f"  {GREEN}Faster:    {faster}{RESET}")
    if at_parity:
        print(f"  {YELLOW}Parity:    {at_parity}{RESET}  (0.95-1.0x)")
    print(f"  {RED}Slower:    {n_slower}{RESET}  (<0.95x)")
    print(f"  Geo mean:  {colorize_speedup(geo_mean)}")

    # Slower-than-numpy table
    if slower:
        slower.sort(key=lambda r: r[6])
        print(f"\n{BOLD}{RED}Operations slower than NumPy ({len(slower)}):{RESET}\n")
        print(f"{'Category':<12} {'Operation':<24} {'Dtype':<7} {'Shape':<16} {'numc(us)':>10} {'numpy(us)':>10} {'Speedup':>10}")
        print(sep)
        for cat, op, dtype, shape, t_numc, t_numpy, speedup in slower:
            sp_str = colorize_speedup(speedup)
            print(
                f"{cat:<12} {op:<24} {dtype:<7} {shape:<16} {t_numc:>10.2f} {t_numpy:>10.2f} {sp_str}"
            )


if __name__ == "__main__":
    main()
