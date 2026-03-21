# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

numc is a high-performance N-dimensional tensor library in pure C23. Zero dependencies — no BLAS, no Fortran, no Python runtime. Hand-written SIMD kernels (AVX2, AVX-512, NEON, SVE/SVE2, RVV), packed GEMM with Goto's algorithm, arena-based memory, and OpenMP parallelization. Licensed under LGPL-3.0.

## Build & Test Commands

```bash
./run.sh release          # Optimized build (CC=clang by default)
./run.sh debug            # Debug build with ASan
./run.sh test             # Debug build + run all tests with ASan
./run.sh test gcc         # Tests with a specific compiler
./run.sh check            # CI simulation: clang-format + clang-tidy + test+ASan
./run.sh bench            # Full benchmark suite vs NumPy
./run.sh bench matmul     # Benchmark a specific category
./run.sh clean            # Remove build directories
./run.sh rebuild          # Clean + fresh debug build
```

Cross-compilation (requires cross-compilers + QEMU):
```bash
./run.sh neon test        # AArch64 NEON
./run.sh sve test         # AArch64 SVE
./run.sh rvv test         # RISC-V RVV
./run.sh avx512 test      # x86 AVX-512 via QEMU
```

Manual CMake:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DNUMC_ENABLE_ASAN=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Run a single test: `./build/bin/test_add` (binary name matches test name in `tests/CMakeLists.txt`).

## External Reference Repositories

The `externals/` folder contains three reference codebases. **Study these before modifying numc kernel or SIMD code**:

- **`externals/blis/`** — BLIS framework. Reference for Goto's GEMM algorithm, micro-kernel design (MR×NR tiles), packing strategies (MC/KC/NC panels), and multi-level cache blocking. See `kernels/` for ISA-specific micro-kernels and `docs/KernelsHowTo.md` for the kernel API.
- **`externals/numpy/`** — NumPy source. Reference for ufunc dispatch patterns, broadcasting rules, and NumPy's SIMD abstraction layer (`numpy/_core/src/common/simd/`). Useful when ensuring numc matches NumPy semantics.
- **`externals/OpenBLAS/`** — OpenBLAS source. Reference for optimized BLAS kernels, especially `kernel/x86_64/` for AVX2/AVX-512 assembly patterns and thread-level parallelism in GEMM.

## Architecture

```
Public API (include/numc/) → Dispatcher (src/*/dispatch.h) → Kernels (src/*/kernel.h)
                                                            → SIMD intrinsics (src/intrinsics/)
```

### Key layers

1. **Public API** (`include/numc/`): Opaque types (`NumcCtx`, `NumcArray`), operations declared in `math.h`, `array.h`, `random.h`. The umbrella header is `numc/numc.h`.

2. **Internal structs** (`src/internal.h`): `NumcArray` has inline shape/stride buffers (`NUMC_MAX_INLINE_DIMS=8`), contiguity flag, arena pointer. `NumcCtx` wraps an `Arena*`. OMP threshold macros (`NUMC_OMP_FOR`, `NUMC_OMP_REDUCE_FOR`) live here.

3. **Arena allocator** (`src/arena.h`): Header-only with `#ifdef ARENA_IMPLEMENTATION`. Bump-pointer allocation, checkpoint/restore for scoped temporaries. Blocks grow by doubling.

4. **Dispatchers** (`src/elemwise/dispatch.h`, `src/reduction/dispatch.h`, `src/matmul/dispatch.h`): Validate inputs, then select contiguous fast-path (flat SIMD loop) or recursive N-D stride walker.

5. **Kernels** (`src/*/kernel.h`): X-macro generated per-dtype via `GENERATE_NUMC_TYPES`. Binary kernel macros (`DEFINE_BINARY_KERNEL`) expand to stride-aware loops with `__builtin_assume_aligned` and `NUMC_OMP_FOR`.

6. **SIMD intrinsics** (`src/intrinsics/`): ISA-specific implementations. Files named `{op}_{isa}.h` (e.g., `elemwise_avx2.h`, `reduce_neon.h`, `gemm_avx512.h`). Hand-written `.S` assembly for GEMM micro-kernels (e.g., `gemm_ukernel_f32_6x16_avx2.S`).

7. **Arch dispatch** (`src/arch_dispatch.h`): Compile-time `NUMC_HAVE_AVX2`, `NUMC_HAVE_AVX512`, `NUMC_HAVE_NEON`, `NUMC_HAVE_SVE`, `NUMC_HAVE_RVV` flags driven by compiler-defined macros.

### GEMM dispatch hierarchy

`numc_matmul()` dispatches through three tiers:
1. **GEMMSUP** (small matrix) — SIMD kernels for small M/K/N, no packing overhead
2. **Packed GEMM** — 5-loop Goto's algorithm with MC/KC/NC cache-blocked panels, ISA-specific MR×NR micro-kernels (e.g., 6×16 f32 on AVX2), 2D IC×JR OpenMP parallelism
3. **Naive fallback** — O(M·K·N) triple loop with OMP on outer loop

### X-macro dtype system

All operations support 10 dtypes. `GENERATE_NUMC_TYPES(MACRO)` expands `MACRO(NUMC_DTYPE_INT8, NUMC_INT8)` through all 10 types. Specialized generators exist: `GENERATE_FLOAT_NUMC_TYPES`, `GENERATE_INT_NUMC_TYPES`, `GENERATE_32BIT_NUMC_TYPES`, etc. Each operation `.c` file stamps kernels per-dtype, then uses a dispatch table indexed by `NumcDType`.

### Elemwise fast-paths

Binary kernels have three stride specializations in priority order:
1. `sa == es && sb == es && so == es` — fully contiguous (SIMD + OMP)
2. `sb == 0` — scalar broadcast (one operand is scalar)
3. General strided — per-element stride arithmetic

## Code Style

- LLVM-based clang-format (2-space indent, 80-col limit, pointer right-aligned)
- Format check: `find src include tests examples bench -type f \( -name '*.c' -o -name '*.h' \) -not -path '*/.venv/*' | xargs clang-format -i`
- Static analysis: clang-tidy with bugprone, performance, clang-analyzer checks
- C23 standard (`alignas` keyword, `nullptr`)

## Performance Conventions

- Use multi-accumulation in loops to break dependency chains
- OpenMP parallelization kicks in above byte thresholds (1MB for elemwise, 256KB for reductions) — see `NUMC_OMP_FOR` macros in `internal.h`
- All data is 32-byte aligned (`NUMC_SIMD_ALIGN=32`) for AVX2 compatibility
- `KMP_BLOCKTIME=200` is set via constructor to avoid libomp thread sleep latency
- OpenMP thread pool is pre-warmed during `numc_ctx_create()`

## Adding a New Operation

1. Add kernel in `src/{category}/kernel.h` using the appropriate `DEFINE_*_KERNEL` macro
2. Add SIMD intrinsics in `src/intrinsics/{op}_{isa}.h` if needed
3. Add dispatch logic in the operation's `.c` file with dtype table
4. Declare public API in `include/numc/math.h` (or appropriate header)
5. Add tests in `tests/{category}/test_{op}.c` and register in `tests/CMakeLists.txt`
6. Add benchmarks in `bench/numc/bench.c` and `bench/numpy/bench.py`

## Optimization Rules

When adding or modifying any performance-critical code (kernels, SIMD intrinsics, GEMM, reductions, elementwise ops):

1. **All 10 dtypes** — every operation must support int8/16/32/64, uint8/16/32/64, float32, float64. No dtype left behind. Use X-macro generators to ensure coverage.
2. **All supported ISAs** — provide optimized implementations for AVX2, AVX-512, NEON, SVE/SVE2, and RVV. Scalar fallback must always exist. When adding a new SIMD kernel for one ISA, create at least a C intrinsics version for all other ISAs in the same PR.
3. **Cross-platform correctness** — validate all ISA variants via QEMU cross-compilation (`./run.sh {neon,sve,rvv,avx512} test`) before merging. Native benchmarking is required for the primary development ISA (currently AVX2).
4. **Benchmark before and after** — run `./run.sh bench` (or the relevant category) and record the speedup. Regressions in any dtype or size are blockers.

## Benchmark Categories

matmul, binary, scalar, unary, comparison, comparison_scalar, ternary, reduction, linalg, random
