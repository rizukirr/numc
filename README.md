# numc

A fast, lightweight NumPy-like N-dimensional tensor library in C, built for high-performance scientific computing and AI workloads. Hand-written SIMD kernels, packed GEMM, arena memory, and OpenMP parallelization.

> Warning, this is not production ready yet

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

## Why numc

- **Fast** — hand-written AVX2, AVX-512, NEON, SVE/SVE2, and RVV kernels. Packed GEMM with Goto's algorithm. Beats NumPy (OpenBLAS) across all operation categories.
- **Complete** — 10 dtypes (`int8`–`int64`, `uint8`–`uint64`, `float32`, `float64`), elementwise ops, reductions, matmul, broadcasting, slicing, random. Every operation supports every dtype.
- **Simple** — arena allocator owns all memory. Create a context, use it, free it. No manual `malloc`/`free` per array.
- **Portable** — pure C23 with scalar fallback. Builds on any platform. SIMD is a bonus, not a requirement.

## Performance

Benchmarked against NumPy (OpenBLAS backend) on i7-13620H:

![Overview](bench/graph/output/overview.png)

Per-operation breakdown: [benchmark results](bench/graph/README.md).

## Build

```bash
git clone https://github.com/rizukirr/numc.git
cd numc
./run.sh release        # optimized build
./run.sh test           # tests with AddressSanitizer
./run.sh bench          # benchmark vs NumPy
```

Cross-compilation (requires cross-compilers + QEMU):

```bash
./run.sh neon test      # AArch64 NEON
./run.sh sve test       # AArch64 SVE
./run.sh rvv test       # RISC-V RVV
./run.sh avx512 test    # x86 AVX-512
```

CMake options: `NUMC_ENABLE_ASAN` (AddressSanitizer), `NUMC_WERROR` (`-Werror`), `NUMC_OPTIMIZE_NATIVE` (`-march=native`, default ON).

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
