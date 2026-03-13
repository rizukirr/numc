# numc

A high-performance N-dimensional tensor library in pure C23. No external dependencies.

Packed SIMD GEMM, vectorized elementwise ops, and OpenMP parallelization across **AVX2**, **AVX-512**, **NEON**, **SVE**, and **RVV** — all with a scalar fallback so it builds anywhere.

## Why numc

- **Zero dependencies** — no BLAS, no Fortran runtime, no Python. Just C and a compiler.
- **Arena allocator** — all tensors owned by a context. Free the context, free everything. O(1).
- **10 data types** — int8/16/32/64, uint8/16/32/64, float32, float64. Every op supports all of them.
- **NumPy-compatible broadcasting** — same semantics, faster execution.

## Performance

Benchmarked against NumPy (OpenBLAS) on i7-13620H. numc wins across **all 13 operation categories** (1.7x–11.4x median speedup).

![Overview](bench/graph/output/overview.png)

Full breakdown with per-operation charts: [benchmark results](bench/graph/README.md).

> GEMM micro-kernels for AVX-512, NEON, SVE, and RVV pass correctness tests via QEMU but lack native hardware benchmarks. Contributions welcome — run `./run.sh bench` and open a PR.

## Quick Start

```bash
git clone https://github.com/rizukirr/numc.git
cd numc
./run.sh release    # optimized build
./run.sh test       # test suite (ASan enabled)
./run.sh bench      # benchmark vs NumPy
```

### Cross-Compilation

Requires cross-compilers and QEMU user-mode.

```bash
./run.sh neon test    # AArch64 NEON
./run.sh sve test     # AArch64 SVE
./run.sh rvv test     # RISC-V RVV
```

### CMake Options

| Option | Default | |
|---|---|---|
| `NUMC_ENABLE_ASAN` | `OFF` | AddressSanitizer |
| `NUMC_WERROR` | `OFF` | Warnings as errors |
| `NUMC_OPTIMIZE_NATIVE` | `ON` | `-march=native` |

## Architecture

```
Public API → Dispatcher → Kernels
```

1. **API** (`include/numc/`) — validates inputs, allocates output via arena
2. **Dispatcher** (`src/*/dispatch.h`) — contiguous fast-path (flat SIMD loop) vs strided N-D walk
3. **Kernels** (`src/*/kernel.h`) — per-type code via X-macros, `__builtin_assume_aligned`, OpenMP

Matmul dispatch: **GEMMSUP** (small matrices) → **Packed GEMM** (5-loop Goto's algorithm) → **Naive fallback**.

## Documentation

[Project Wiki](https://github.com/rizukirr/numc/wiki) — API reference, architecture guide, usage examples.

## Known Slow Paths — Help Wanted

These are specific operations where NumPy (OpenBLAS) still beats numc. Contributions and ideas welcome.

| Area | What's slow | vs NumPy | Why | Where to look |
|---|---|---|---|---|
| Matmul | f64 128×128 | 0.41x | OpenBLAS hand-tuned asm micro-kernels | `src/intrinsics/gemm_avx2.h` |
| Matmul | f32 128×128 | 0.49x | Same — small GEMM dispatch overhead | `src/intrinsics/gemm_avx2.h` |
| Reduction | int64/uint64 min/max axis-1 | 0.53x–0.57x | No AVX2 min/max for 64-bit integers | `src/reduction/min.c`, `max.c` |
| Comparison | float64 scalar comparisons | 0.66x–0.77x | Scalar kernel not SIMD-optimized | `src/elemwise/compare.c` |
| Matmul | f64 256×256, 1024×1024 | 0.74x–0.77x | K-loop unrolling, prefetching gaps | `src/intrinsics/gemm_avx2.h` |
| Elemwise | int8/uint8 binary min/max/cmp | 0.78x–0.89x | Byte-width elemwise loops not SIMD | `src/elemwise/compare.c`, `src/elemwise/arithmetic.c` |

Reference implementations for GEMM optimization are available in `external/blis/` and `external/openblas/` (gitignored, clone separately).

## Contributing

See the [contributing guide](https://github.com/rizukirr/numc/blob/main/CONTRIBUTING.md). For benchmarks, see [bench/README.md](bench/README.md).

## Support

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/rizukirr)

## License

Copyright (c) 2026 Rizki Rakasiwi

numc is free software: you can redistribute it and/or modify it under the terms of the **GNU Lesser General Public License v3.0** as published by the Free Software Foundation.

See [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING) for details.
