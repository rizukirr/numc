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

## Documentation

For detailed API specifications, architecture deep-dives, and comprehensive usage guides, please visit the [**numc Project Wiki**](https://github.com/rizukirr/numc/wiki).

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

## License

`numc` is released under the **MIT License**. Engineered for reliability and performance.
