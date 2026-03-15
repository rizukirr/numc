# numc

A high-performance N-dimensional tensor library in pure C. Zero dependencies — no BLAS, no Fortran, no Python runtime.

Hand-written SIMD kernels (AVX2, AVX-512, NEON, SVE, RVV), packed GEMM with Goto's algorithm, arena-based memory, and OpenMP parallelization. Scalar fallback ensures it builds on any platform.

## Example

```c
#include <numc/numc.h>

int main(void) {
  NumcCtx *ctx = numc_ctx_create();

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_fill(ctx, shape, 2, NUMC_DTYPE_FLOAT32, &(float){1.0f});
  NumcArray *b = numc_array_fill(ctx, shape, 2, NUMC_DTYPE_FLOAT32, &(float){2.0f});
  NumcArray *c = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  numc_add(a, b, c);       // c = a + b
  numc_array_print(c);     // [[3, 3, 3], [3, 3, 3]]

  numc_ctx_free(ctx);      // frees a, b, c in O(1)
}
```

## Features

- **Arena allocator** — all arrays owned by a `NumcCtx`. Free the context, free everything in O(1). Checkpoint/restore for scoped temporaries.
- **10 dtypes** — `int8/16/32/64`, `uint8/16/32/64`, `float32`, `float64`. Every operation supports all of them.
- **NumPy-compatible broadcasting** — same shape semantics, same edge cases.
- **Zero-copy views** — slice, transpose, reshape without copying data.
- **GEMM** — 3-tier dispatch: GEMMSUP (small), packed 5-loop Goto's (large), naive fallback. ISA-specific micro-kernels (e.g., 6x16 f32 on AVX2).
- **Elementwise ops** — add, sub, mul, div, pow, abs, neg, exp, log, sqrt, clip, fma, where, comparisons. All vectorized with direct SIMD fast-paths.
- **Reductions** — sum, mean, max, min, argmax, argmin. Per-axis with keepdim support.
- **OpenMP** — auto-parallelization above 512 KB threshold. 2D IC x JR parallelism for GEMM.

## Performance

Benchmarked against NumPy (OpenBLAS backend) on i7-13620H. numc wins across all 13 operation categories (1.7x-11.4x median speedup).

![Overview](bench/graph/output/overview.png)

Per-operation breakdown: [benchmark results](bench/graph/README.md).

## Build

```bash
git clone https://github.com/rizukirr/numc.git
cd numc
./run.sh release        # optimized build
./run.sh test           # tests with AddressSanitizer
./run.sh bench          # benchmark vs NumPy
./run.sh check          # clang-format + clang-tidy + test+ASan
```

### Cross-compilation

Requires cross-compilers and QEMU user-mode emulation.

```bash
./run.sh neon test      # AArch64 NEON
./run.sh sve test       # AArch64 SVE
./run.sh rvv test       # RISC-V RVV
./run.sh avx512 test    # x86 AVX-512
```

### CMake options

| Option | Default | Purpose |
|---|---|---|
| `NUMC_ENABLE_ASAN` | `OFF` | AddressSanitizer |
| `NUMC_WERROR` | `OFF` | `-Werror` |
| `NUMC_OPTIMIZE_NATIVE` | `ON` | `-march=native` |

## Architecture

```
Public API (include/numc/) → Dispatcher (src/*/dispatch.h) → Kernels (src/*/kernel.h)
```

- **API layer** validates inputs, allocates output via the arena
- **Dispatcher** selects contiguous fast-path (flat SIMD loop) or strided N-D walker
- **Kernels** are X-macro generated per-dtype, with `__builtin_assume_aligned` and OpenMP

GEMM dispatch: GEMMSUP → packed GEMM (5-loop, cache-blocked MC/KC/NC panels) → naive fallback.

## Documentation

[Project Wiki](https://github.com/rizukirr/numc/wiki) — API reference, architecture details, usage examples.

## Hardware testing wanted

SIMD kernels pass correctness tests via QEMU but only AVX2 has native benchmarks. If you have access to AVX-512, NEON (Apple Silicon / Graviton), SVE, or RVV hardware, run `./run.sh bench` and open a PR.

## Contributing

See [CONTRIBUTING.md](https://github.com/rizukirr/numc/blob/main/CONTRIBUTING.md).

## Support

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/rizukirr)

## License

Copyright (c) 2026 Rizki Rakasiwi. Licensed under [LGPL-3.0](COPYING.LESSER).
