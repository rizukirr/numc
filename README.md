# numc

`numc` is a high-performance C library for N-dimensional array manipulation and tensor computation. Engineered in pure C23, it provides a robust mathematical foundation for scientific computing, machine learning, and data processing.

By combining modern C language features with hardware-aware architectural patterns, `numc` delivers a professional-grade toolset that often outperforms industry-standard libraries in both raw throughput and memory efficiency.

## Key Features

- **C23 Native Implementation:** Leverages the latest C standard for improved type safety, performance, and modern syntax.
- **Hardware-Aware Performance:** Utilizes `__builtin_assume_aligned` for optimal SIMD vectorization and `__builtin_prefetch` to minimize CPU cache stalls during non-contiguous operations.
- **Arena-Based Memory Management:** Utilizes a highly efficient arena allocator (`NumcCtx`) to manage tensor lifecycles, ensuring zero fragmentation and O(1) cleanup.
- **Optimized Compute Kernels:** Features hand-tuned kernels with reciprocal multiplication for division and power-of-two bit-shift optimizations.
- **Zero-Copy Strided Views:** Perform reshapes, transposes, and complex slicing operations without redundant data movement or allocation.
- **Vendored BLIS Integration:** Ships with [BLIS](https://github.com/flame/blis) for optimized `sgemm`/`dgemm` with runtime CPU detection (AVX2, AVX-512) and automatic threading.

## Technical Capabilities

`numc` provides a comprehensive suite of operations for production-ready environments:

- **Tensor Lifecycle:** Flexible creation, initialization (zeros, ones, fill), and deep-copy utilities.
- **Element-wise Mathematics:** Full support for binary, unary, and scalar operations, including transcendental functions (`exp`, `log`, `sqrt`, `pow`).
- **Standard Broadcasting:** Transparent handling of mismatched dimensions across all binary operations with optimized semi-contiguous fast paths.
- **Reductions:** Multi-dimensional sum, mean, max, min, and argmax/argmin with support for axis-specific reduction.
- **Matrix Multiplication:** Three-tier dispatch: BLIS for large float ops, system BLAS fallback, and optimized naive C23 kernels that outperform NumPy by 4-5x on small matrices.

## Documentation

Comprehensive API documentation and usage guides are available in the [Project Wiki](https://github.com/rizukirr/numc/wiki).

## Building and Development

`numc` uses CMake and includes a developer-focused helper script for common workflows.

### Quick Start

```bash
./run.sh release        # Build optimized Release mode
./run.sh test           # Execute the comprehensive test suite (with ASan/LSan)
./run.sh bench          # Run performance benchmarks against NumPy
```

### Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `NUMC_VENDOR_BLIS` | `ON` | Build BLIS from vendored submodule |
| `NUMC_USE_BLAS` | `ON` | Enable BLAS integration (system or vendored) |
| `BLIS_CONFIG` | `x86_64` | BLIS target (`x86_64`, `auto`, `haswell`, etc.) |

## License

`numc` is released under the MIT License. See the [LICENSE](LICENSE) file for details.
