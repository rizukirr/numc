# numc: High-Performance N-Dimensional Tensor Library

`numc` is a blazingly fast C library engineered for high-performance N-dimensional array manipulation and tensor computation. Architected in pure **C23**, it provides a robust mathematical foundation for scientific computing, machine learning, and high-frequency data processing.

By synthesizing modern language features with hardware-aware architectural patterns, `numc` delivers deterministic performance that consistently outperforms industry-standard libraries in both raw throughput and cache efficiency.

## Key Architectural Advantages

- **Hardware-Informed Performance:** Leverages `__builtin_assume_aligned` for SIMD alignment and `__builtin_prefetch` to minimize cache miss penalties during strided N-D iteration.
- **Three-Tier Dispatch System:**
    1. **Accelerated Backend:** Packed GEMM micro-kernels for all 10 data types across **AVX2**, **AVX-512**, **NEON**, **SVE**, and **RVV**, with optional **BLIS**/**OpenBLAS** offloading for floating-point BLAS Level 1/2/3.
    2. **Parallelized Engine:** Multi-accumulator **OpenMP** kernels for high-bandwidth integer and strided-floating operations.
    3. **Native Fallback:** Highly optimized, C23-native kernels for edge cases and small tensors.
- **Arena-Based Memory Management:** Utilizes a high-concurrency arena allocator (`NumcCtx`) for O(1) tensor lifecycle management, ensuring zero memory fragmentation and deterministic cleanup.
- **Zero-Copy Geometry:** Perform reshapes, transposes, and complex slicing via metadata manipulation, eliminating redundant data movement.
- **Modern C23 Foundation:** Built on the latest C standard for enhanced type safety and rigorous static analysis compatibility.

## Technical Capabilities

`numc` implements a comprehensive suite of operations for production-ready engineering:

- **Tensor Math:** Full support for binary arithmetic, unary functions, and scalar operations with broadcasting.
- **Transcendental Ops:** Hardware-accelerated `exp`, `log`, `sqrt`, and `pow`.
- **Advanced Reductions:** Multi-dimensional `sum`, `mean`, `max/min`, and `argmax/argmin` with axis-specific reduction logic.
- **Universal Broadcasting:** Strict adherence to NumPy-style broadcasting semantics across all dimension-mismatched operations.
- **N-Dimensional Support:** Every mathematical operation (excluding strictly 2D `matmul`) supports arbitrary N-D tensor shapes and strided memory layouts.

## Benchmarks

numc is rigorously benchmarked against NumPy across all supported operations, data types, and array configurations. The benchmark suite runs identical workloads on both libraries to ensure a fair comparison.

> **Note:** Current benchmarks were collected on an x86-64 AVX2 machine (i7-13620H). GEMM micro-kernels for **AVX-512**, **NEON**, **SVE**, and **RVV** pass correctness tests (via QEMU) but have not been performance-benchmarked on native hardware. If you have access to any of these platforms, we would love benchmark contributions — just run `./run.sh bench` and open a PR with your results!

![Overview](bench/graph/output/overview.png)

For the complete set of benchmark charts, device specifications, and per-operation breakdowns, see the [benchmark results](bench/graph/README.md).

To reproduce benchmarks on your own hardware:

```bash
./run.sh bench                                        # Run numc + numpy benchmarks
bench/graph/.venv/bin/python3 bench/graph/plot.py     # Generate comparison charts
```

Detailed benchmark methodology, CSV format documentation, and environment setup instructions are available in the [bench README](bench/README.md).

## TODO

### Performance — Help Wanted

These are known slow paths where numc loses to NumPy. Community contributions and ideas are welcome.

#### Matmul (SIMD packed GEMM)

The packed SIMD GEMM is the primary matmul path (BLIS/OpenBLAS are optional). It wins at 512×512 but loses at small/medium sizes where packing overhead dominates.

| Size | numc vs NumPy | Bottleneck |
|------|---------------|------------|
| 64×64 | 1.03x (on par) | — |
| 128×128 | 0.47x (2.1x slower) | Packing overhead |
| 256×256 | 0.66x (1.5x slower) | Compiler scheduling |
| 512×512 | 1.25x (faster) | — |
| 1024×1024 | 0.78x (1.3x slower) | 1D threading |

- [x] **Assembly micro-kernels** — Inline asm f32 6×16 and f64 6×8 with pinned YMM registers, 4× K-loop unroll (GCC/Clang; C intrinsics fallback for MSVC)
- [x] **Small-matrix unpacked kernels (gemmsup)** — AVX2 unpacked f32 6×16 and f64 6×8 micro-kernels, skip packing for M×K×N < 128³
- [x] **B-panel prefetching** — Integrated into inline asm micro-kernels

#### Other Slow Paths

- [x] **Per-op-class OMP threshold for reductions** — Separate 256KB threshold for reductions vs 1MB for elemwise
- [x] **SIMD uint8 comparison kernels** — AVX2 XOR-0x80 trick for unsigned `eq/gt/lt/ge/le`
- [x] **SIMD int8/uint8 min/max kernels** — Multi-accumulator vectorized helpers for all integer types
- [x] **SIMD log/exp intrinsics** (AVX2) — Vectorized fdlibm/Cephes log/exp for f32 (8-wide) and f64 (4-wide)
- [x] **SIMD pow intrinsics** (AVX2) — Vectorized `exp(b*log(a))` for contiguous float data
- [x] **SIMD randn (Box-Muller)** — SIMD Box-Muller with vectorized log/sqrt/sincos for f32

### Features

- [x] **SIMD gemm for all architectures** — Packed GEMM micro-kernels for AVX2, AVX-512, NEON, SVE, and RVV (all 10 types)
- [x] **AVX-512 integer GEMM micro-kernels** — `vpmulld`/`vpmullw`/`vpmullq`-based kernels for all 8 integer types (i8/u8 with i32 accumulators)
- [ ] **Intel hybrid CPU P-core detection** — Runtime sysfs-based detection removed for portability; consider optional opt-in

### ARM & RISC-V SIMD Parity

The following optimizations are currently AVX2-only and need NEON/SVE/RVV equivalents:

- [ ] **NEON/SVE log/exp intrinsics** — Port `math_avx2.h` vectorized log/exp to `math_neon.h` and `math_sve.h`
- [ ] **RVV log/exp intrinsics** — Port vectorized log/exp to `math_rvv.h` using RVV intrinsics
- [ ] **NEON/SVE sin/cos intrinsics** — Port Cephes sin/cos from `math_avx2.h` for SIMD randn on ARM
- [ ] **RVV sin/cos intrinsics** — Port Cephes sin/cos for SIMD randn on RISC-V
- [ ] **NEON/SVE SIMD pow** — Port vectorized `exp(b*log(a))` pow to ARM
- [ ] **RVV SIMD pow** — Port vectorized pow to RISC-V
- [ ] **NEON/SVE SIMD randn (Box-Muller)** — SIMD Box-Muller using ARM vectorized log/sincos
- [ ] **RVV SIMD randn (Box-Muller)** — SIMD Box-Muller using RVV vectorized log/sincos
- [ ] **NEON/SVE uint8 comparison kernels** — Port XOR-0x80 unsigned comparison trick to ARM
- [ ] **RVV uint8 comparison kernels** — Port unsigned comparison kernels to RISC-V
- [ ] **NEON/SVE int8/uint8 min/max reduction** — Port multi-accumulator min/max helpers to ARM
- [ ] **RVV int8/uint8 min/max reduction** — Port multi-accumulator min/max helpers to RISC-V
- [ ] **ARM gemmsup (small-matrix unpacked GEMM)** — Port unpacked f32/f64 micro-kernels to NEON/SVE
- [ ] **RVV gemmsup (small-matrix unpacked GEMM)** — Port unpacked f32/f64 micro-kernels to RVV
- [ ] **AArch64 assembly GEMM micro-kernels** — Hand-tuned asm for NEON/SVE f32/f64 GEMM (matching AVX2 inline asm)

## Documentation

For detailed API specifications, architecture deep-dives, and comprehensive usage guides, visit the [**numc Project Wiki**](https://github.com/rizukirr/numc/wiki).

## Build and Development

`numc` utilizes a modern CMake build system and provides a unified developer-experience script.

### Installation

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/rizukirr/numc.git
cd numc

# Build optimized Release binaries
./run.sh release

# Validate with the comprehensive test suite (ASan/LSan included)
./run.sh test

# Benchmark against local Python/NumPy installation
./run.sh bench
```

### Build Configuration

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `NUMC_BLAS_BACKEND` | `BLIS` | BLAS backend: `BLIS` or `OPENBLAS` |
| `NUMC_VENDOR_BLIS` | `ON` | Build BLIS from internal submodule |
| `NUMC_VENDOR_OPENBLAS` | `OFF` | Build OpenBLAS from internal submodule |
| `NUMC_USE_BLAS` | `ON` | Enable BLAS acceleration |
| `BLIS_CONFIG` | `auto` | BLIS target (e.g., `haswell`, `zen`, `skx`) |

To switch BLAS backend:

```bash
# Default (BLIS — lightweight, ~5MB)
./run.sh release

# OpenBLAS (heavier, ~200MB, but faster float matmul)
NUMC_BLAS_BACKEND=OPENBLAS NUMC_VENDOR_OPENBLAS=ON ./run.sh release

# A/B benchmark both backends
./run.sh bench-blas
```

### Cross-Compilation (ARM / RISC-V)

Requires cross-compilers and QEMU user-mode. Tests run automatically via QEMU emulation.

```bash
./run.sh neon test    # AArch64 NEON (armv8-a baseline)
./run.sh sve test     # AArch64 SVE  (armv8-a+sve)
./run.sh sve2 test    # AArch64 SVE2 (armv9-a)
./run.sh rvv test     # RISC-V RVV   (rv64gcv)
```

## Contributing

Contributions are welcome. Whether it is a bug fix, new operation, performance optimization, or documentation improvement, all contributions help strengthen the library.

Please refer to the [contributing guide](https://github.com/rizukirr/numc/blob/main/CONTRIBUTING.md) for coding standards, commit conventions, and pull request guidelines. For benchmark-related contributions, see the [bench README](bench/README.md).

## Support

If you find this library useful, consider supporting its development:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/rizukirr)

## License

`numc` is released under the **MIT License**.

This project includes optional vendored dependencies under their own licenses:
- **BLIS** — 3-Clause BSD License (Copyright The University of Texas at Austin et al.)
- **OpenBLAS** — 3-Clause BSD License (Copyright The OpenBLAS Project)

See `external/blis/LICENSE` and `external/openblas/LICENSE` for full terms.
