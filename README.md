# numc: High-Performance N-Dimensional Tensor Library

`numc` is a blazingly fast C library engineered for high-performance N-dimensional array manipulation and tensor computation. Architected in pure **C23**, it provides a robust mathematical foundation for scientific computing, machine learning, and high-frequency data processing.

By synthesizing modern language features with hardware-aware architectural patterns, `numc` delivers deterministic performance that consistently outperforms industry-standard libraries in both raw throughput and cache efficiency.

## Key Architectural Advantages

- **Hardware-Informed Performance:** Leverages `__builtin_assume_aligned` for SIMD alignment and `__builtin_prefetch` to minimize cache miss penalties during strided N-D iteration.
- **Three-Tier Dispatch System:**
    1. **Accelerated Backend:** Automatic offloading of floating-point operations (`dot`, `matmul`, `sum`, `add`, `sub`) to **BLIS** (Level 1/2/3 BLAS) with runtime CPU feature detection (AVX2, AVX-512).
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

![Overview](bench/graph/output/overview.png)

For the complete set of benchmark charts, device specifications, and per-operation breakdowns, see the [benchmark results](bench/graph/README.md).

To reproduce benchmarks on your own hardware:

```bash
./run.sh bench                                        # Run numc + numpy benchmarks
bench/graph/.venv/bin/python3 bench/graph/plot.py     # Generate comparison charts
```

Detailed benchmark methodology, CSV format documentation, and environment setup instructions are available in the [bench README](bench/README.md).

## TODO

### Performance

- [ ] **Per-op-class OMP threshold for reductions** — Global 256KB threshold regresses element-wise ops; reductions need a separate, lower threshold to enable OMP on 1-byte types at 1M elements
- [ ] **AVX-512 dot product intrinsics** (`src/reduction/intrinsics/dot_avx512.h`) — Close the MKL gap for float32/float64 dot (currently 0.34x–0.76x vs numpy)
- [ ] **NEON dot product intrinsics** (`src/reduction/intrinsics/dot_neon.h`)
- [ ] **RVV dot product intrinsics** (`src/reduction/intrinsics/dot_rvv.h`)
- [ ] **SIMD uint8 comparison kernels** — XOR-0x80 trick for unsigned `eq/gt/lt/ge/le` (5 ops slower than numpy)
- [ ] **SIMD int8/uint8 min/max kernels** — Direct `vpminub`/`vpmaxub`/`vpminsb`/`vpmaxsb` (4 ops slower than numpy)
- [ ] **Raise BLIS small-matrix threshold** — 64x64 matmul dispatches to BLIS where overhead > compute
- [ ] **SIMD log/exp intrinsics** (AVX2/NEON/RVV) — Minimax polynomial; currently scalar-only (`NOSIMD`)
- [ ] **SIMD pow intrinsics** (AVX2/NEON/RVV) — Vectorized exp-by-squaring or `exp(b*log(a))`
- [ ] **SIMD randn (Box-Muller)** — Batch PRNG + SIMD log/sin/cos for `numc_array_randn`

### Features

- [ ] **Integer SIMD gemm for NEON/RVV** — AVX2 gemm is complete, port to ARM and RISC-V
- [ ] **Intel hybrid CPU P-core detection** — Runtime sysfs-based detection removed for portability; consider optional opt-in

See [`notes/PERF_REPORT.md`](notes/PERF_REPORT.md) for full analysis and benchmark data.

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
| `NUMC_VENDOR_BLIS` | `ON` | Build BLIS from internal submodule |
| `NUMC_USE_BLAS` | `ON` | Enable BLAS/BLIS acceleration |
| `BLIS_CONFIG` | `auto` | BLIS target (e.g., `haswell`, `zen`, `skx`) |

## Contributing

Contributions are welcome. Whether it is a bug fix, new operation, performance optimization, or documentation improvement, all contributions help strengthen the library.

Please refer to the [contributing guide](https://github.com/rizukirr/numc/blob/main/CONTRIBUTING.md) for coding standards, commit conventions, and pull request guidelines. For benchmark-related contributions, see the [bench README](bench/README.md).

## Support

If you find this library useful, consider supporting its development:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/rizukirr)

## License

`numc` is released under the **MIT License**. Engineered for reliability and performance.
