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

## Hardware Testing — Help Wanted

All SIMD kernels (AVX-512, NEON, SVE, RVV) pass correctness tests via QEMU cross-compilation, but **performance has only been benchmarked on AVX2** (Intel i7-13620H). The library needs testing and benchmarking on native hardware for other architectures:

- **AVX-512** — Intel Xeon / Ice Lake+ / AMD Zen 4+
- **NEON** — Apple Silicon, Ampere Altra, AWS Graviton
- **SVE/SVE2** — AWS Graviton 3/4, Fujitsu A64FX, NVIDIA Grace
- **RVV** — SiFive P670, SpacemiT K1, Milk-V Pioneer

If you have access to any of these, run `./run.sh bench` and open a PR with results. GEMM micro-kernel tuning parameters (cache blocking sizes, K-loop unroll factors) may need per-architecture adjustments.

## Contributing

See the [contributing guide](https://github.com/rizukirr/numc/blob/main/CONTRIBUTING.md). For benchmarks, see [bench/README.md](bench/README.md).

## Support

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/rizukirr)

## License

Copyright (c) 2026 Rizki Rakasiwi

numc is free software: you can redistribute it and/or modify it under the terms of the **GNU Lesser General Public License v3.0** as published by the Free Software Foundation.

See [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING) for details.
