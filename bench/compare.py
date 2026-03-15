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
            if not row or row.get("category") is None or row.get("time_us") is None:
                continue
            key = (row["category"], row["operation"], row["dtype"], row["shape"])
            try:
                results[key] = float(row["time_us"])
            except (TypeError, ValueError):
                continue
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

    # Optional category filter (e.g., --filter matmul)
    cat_filter = None
    if "--filter" in sys.argv:
        idx = sys.argv.index("--filter")
        if idx + 1 < len(sys.argv):
            cat_filter = sys.argv[idx + 1]

    numc = load_csv(NUMC_CSV)
    numpy = load_csv(NUMPY_CSV)

    # Match keys present in both
    common = sorted(set(numc) & set(numpy))
    if cat_filter:
        common = [k for k in common if k[0] == cat_filter]
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

        # Compute GFLOPS for matmul rows: parse shape "(MxK)@(KxN)"
        gflops_numc = None
        gflops_numpy = None
        if cat == "matmul":
            try:
                # shape format: "(MxK)@(KxN)"
                parts = shape.replace("(", "").replace(")", "").split("@")
                mk = parts[0].split("x")
                kn = parts[1].split("x")
                M, K, N = int(mk[0]), int(mk[1]), int(kn[1])
                flops = 2.0 * M * K * N
                gflops_numc = flops / t_numc / 1e3  # us->s=1e6, /1e9=G => /1e3
                gflops_numpy = flops / t_numpy / 1e3
            except (IndexError, ValueError):
                pass

        rows.append((cat, op, dtype, shape, t_numc, t_numpy, speedup, gflops_numc, gflops_numpy))
        if speedup < 1.0:
            slower.append((cat, op, dtype, shape, t_numc, t_numpy, speedup, gflops_numc, gflops_numpy))

    # Check if any matmul rows exist (for GFLOPS columns)
    has_matmul = any(r[0] == "matmul" for r in rows)

    # Print full table
    if has_matmul and cat_filter == "matmul":
        hdr = f"{'Category':<12} {'Operation':<24} {'Dtype':<7} {'Shape':<16} {'numc(us)':>10} {'numpy(us)':>10} {'Speedup':>10} {'numc GF':>9} {'numpy GF':>9}"
    else:
        hdr = f"{'Category':<12} {'Operation':<24} {'Dtype':<7} {'Shape':<16} {'numc(us)':>10} {'numpy(us)':>10} {'Speedup':>10}"
    sep = "-" * len(hdr)
    print(f"\n{BOLD}numc vs NumPy Benchmark Comparison{RESET}")
    print(f"{DIM}{len(rows)} benchmarks matched | timing: minimum (per-iteration){RESET}\n")
    print(hdr)
    print(sep)

    prev_cat = None
    for cat, op, dtype, shape, t_numc, t_numpy, speedup, gf_numc, gf_numpy in rows:
        if prev_cat and cat != prev_cat:
            print(sep)
        prev_cat = cat
        sp_str = colorize_speedup(speedup)
        if has_matmul and cat_filter == "matmul" and gf_numc is not None:
            print(
                f"{cat:<12} {op:<24} {dtype:<7} {shape:<16} {t_numc:>10.2f} {t_numpy:>10.2f} {sp_str} {gf_numc:>9.1f} {gf_numpy:>9.1f}"
            )
        else:
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

    # Peak GFLOPS summary for matmul
    matmul_rows = [r for r in rows if r[0] == "matmul" and r[7] is not None]
    if matmul_rows:
        # Group by dtype for peak reporting
        by_dtype = {}
        for r in matmul_rows:
            dtype = r[2]
            if dtype not in by_dtype or r[7] > by_dtype[dtype][7]:
                by_dtype[dtype] = r
        print(f"\n{BOLD}Peak GFLOPS (numc){RESET}")
        for dtype in sorted(by_dtype.keys()):
            r = by_dtype[dtype]
            print(f"  {dtype:<8}  {r[7]:>7.1f} GF  @ {r[3]}")

    # Slower-than-numpy table
    if slower:
        slower.sort(key=lambda r: r[6])
        print(f"\n{BOLD}{RED}Operations slower than NumPy ({len(slower)}):{RESET}\n")
        if has_matmul and cat_filter == "matmul":
            print(f"{'Category':<12} {'Operation':<24} {'Dtype':<7} {'Shape':<16} {'numc(us)':>10} {'numpy(us)':>10} {'Speedup':>10} {'numc GF':>9} {'numpy GF':>9}")
        else:
            print(f"{'Category':<12} {'Operation':<24} {'Dtype':<7} {'Shape':<16} {'numc(us)':>10} {'numpy(us)':>10} {'Speedup':>10}")
        print(sep)
        for cat, op, dtype, shape, t_numc, t_numpy, speedup, gf_numc, gf_numpy in slower:
            sp_str = colorize_speedup(speedup)
            if has_matmul and cat_filter == "matmul" and gf_numc is not None:
                print(
                    f"{cat:<12} {op:<24} {dtype:<7} {shape:<16} {t_numc:>10.2f} {t_numpy:>10.2f} {sp_str} {gf_numc:>9.1f} {gf_numpy:>9.1f}"
                )
            else:
                print(
                    f"{cat:<12} {op:<24} {dtype:<7} {shape:<16} {t_numc:>10.2f} {t_numpy:>10.2f} {sp_str}"
                )


if __name__ == "__main__":
    main()
