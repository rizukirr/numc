# numc

`numc` is a high-performance C library for N-dimensional array manipulation and tensor computation. Engineered in pure C23, it provides a robust mathematical foundation for scientific computing, machine learning, and data processing.

By combining modern C language features with efficient architectural patterns, `numc` delivers a professional-grade toolset for developers who require predictable performance and low-level control.

## Key Features

- **C23 Native Implementation:** Leverages the latest C standard for improved type safety, performance, and modern syntax.
- **Arena-Based Memory Management:** Utilizes a highly efficient arena allocator (`NumcCtx`) to manage tensor lifecycles, minimizing fragmentation and simplifying cleanup.
- **Optimized Compute Kernels:** Features auto-vectorized kernels and OpenMP parallelism, ensuring maximum hardware utilization across multi-core systems.
- **Zero-Copy Strided Views:** Perform reshapes, transposes, and complex slicing operations without redundant data movement or allocation.
- **Type-Generic Architecture:** Supports 10 numeric data types from `int8` to `float64` through a unified, optimized dispatch system.
- **Vendored BLIS Integration:** Ships with [BLIS](https://github.com/flame/blis) as an optional vendored submodule, delivering optimized `sgemm`/`dgemm` with runtime CPU detection (AVX2, AVX-512, etc.) and OpenMP threading out of the box.

## Technical Capabilities

`numc` provides a comprehensive suite of operations for production-ready environments:

- **Tensor Lifecycle:** Flexible creation, initialization (zeros, ones, fill), and deep-copy utilities.
- **Element-wise Mathematics:** Full support for binary, unary, and scalar operations, including transcendental functions (`exp`, `log`, `sqrt`, `pow`).
- **Standard Broadcasting:** Transparent handling of mismatched dimensions across all binary operations.
- **Reductions:** Multi-dimensional sum, mean, max, min, and argmax/argmin with support for axis-specific reduction and dimension preservation.
- **Matrix Multiplication:** High-performance matmul with a three-tier dispatch: vendored BLIS for large float ops, system BLAS fallback, and optimized naive C23 kernels for integer types and small matrices.
- **Initialization & Randomness:** Built-in xoshiro256** PRNG with standard normal, uniform, and specialized weight initialization (He, Xavier/Glorot).

## Performance

Benchmark on Intel i7-13620H (10C/16T), clang 21, vendored BLIS `x86_64`:

**Square matmul scaling (float32):**

| N | numc (GFLOP/s) | NumPy (GFLOP/s) |
|---|---------------|-----------------|
| 128 | 214 | 200 |
| 256 | 349 | 367 |
| 512 | 486 | 469 |

**Integer dtype matmul (256x256):**

| dtype | numc (GFLOP/s) | NumPy (GFLOP/s) | speedup |
|-------|---------------|-----------------|---------|
| int16 | 66 | 3.3 | 20x |
| int32 | 30 | 3.2 | 9x |
| uint16 | 70 | 3.2 | 22x |

numc matches NumPy on float operations and is **9-22x faster** on integer matmul thanks to specialized C23 kernels.

## Documentation

Comprehensive API documentation and usage guides are available in the [Project Wiki](https://github.com/rizukirr/numc/wiki).

## Building and Development

`numc` uses CMake and includes a developer-focused helper script for common workflows.

### Quick Start

```bash
./run.sh release        # Build optimized Release mode
./run.sh test           # Execute the comprehensive test suite
./run.sh bench          # Run performance benchmarks against NumPy
./run.sh bench-matmul   # Run matmul-specific benchmark
```

### BLIS Configuration

By default, `numc` vendors BLIS from a git submodule with runtime CPU dispatch:

```bash
# Default: vendored BLIS with x86_64 fat binary (all Intel + AMD kernels)
./run.sh release

# Faster compile during development (only your CPU's kernels)
BLIS_CONFIG=auto ./run.sh release

# Disable BLIS entirely (naive kernels only)
NUMC_VENDOR_BLIS=OFF NUMC_USE_BLAS=OFF ./run.sh release

# Use system-installed BLIS instead of vendored
NUMC_VENDOR_BLIS=OFF NUMC_USE_BLAS=ON ./run.sh release
```

### Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `NUMC_VENDOR_BLIS` | `ON` | Build BLIS from vendored submodule |
| `NUMC_USE_BLAS` | `ON` | Enable BLAS integration (system or vendored) |
| `BLIS_CONFIG` | `x86_64` | BLIS configuration target (`x86_64`, `auto`, `amd64`, `intel64`, ...) |
| `NUMC_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer for debug builds |

### OpenMP Tuning (Hybrid CPUs)

On Intel hybrid CPUs (P-core + E-core), pin threads to performance cores for consistent results:

```bash
OMP_PLACES=cores BLIS_NUM_THREADS=<P-core count> ./run.sh bench-matmul
```

## License

`numc` is released under the MIT License. See the [LICENSE](LICENSE) file for details.
