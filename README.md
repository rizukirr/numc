# numc

`numc` is a high-performance, standalone C library for N-dimensional array manipulation and tensor computation. Engineered in pure C23, it provides a robust mathematical foundation for scientific computing, machine learning, and data processing with zero external dependencies.

By combining modern C language features with efficient architectural patterns, `numc` delivers a professional-grade toolset for developers who require predictable performance and low-level control.

## Key Features

- **C23 Native implementation:** Leverages the latest C standard for improved type safety, performance, and modern syntax.
- **Arena-based memory management:** Utilizes a highly efficient arena allocator (`NumcCtx`) to manage tensor lifecycles, minimizing fragmentation and simplifying cleanup.
- **Optimized Compute Kernels:** Features auto-vectorized kernels and OpenMP parallelism, ensuring maximum hardware utilization across multi-core systems.
- **Zero-Copy Strided Views:** Perform reshapes, transposes, and complex slicing operations without redundant data movement or allocation.
- **Type-Generic Architecture:** Supports 10 numeric data types from `int8` to `float64` through a unified, optimized dispatch system.
- **Advanced BLAS Integration:** Seamlessly integrates with BLIS for high-performance floating-point matrix operations on modern CPU architectures.

## Technical Capabilities

`numc` provides a comprehensive suite of operations for production-ready environments:

- **Tensor Lifecycle:** Flexible creation, initialization (zeros, ones, fill), and deep-copy utilities.
- **Element-wise Mathematics:** Full support for binary, unary, and scalar operations, including transcendental functions (`exp`, `log`, `sqrt`, `pow`).
- **Standard Broadcasting:** Transparent handling of mismatched dimensions across all binary operations.
- **Reductions:** Multi-dimensional sum, mean, max, min, and argmax/argmin with support for axis-specific reduction and dimension preservation.
- **Matrix Operations:** High-performance matrix multiplication with optimized paths for both integer and floating-point data.
- **Initialization & Randomness:** Built-in xoshiro256** PRNG with standard normal, uniform, and specialized weight initialization (He, Xavier/Glorot).

## Documentation

Comprehensive API documentation and usage guides are available in the [Project Wiki](https://github.com/rizukirr/numc/wiki).

## Building and Development

`numc` includes a developer-focused build system for various environments and architectures.

### Quick Start

```bash
./run.sh release        # Build optimized Release mode
./run.sh test           # Execute the comprehensive test suite
./run.sh bench          # Run performance benchmarks against industry baselines
```

## License

`numc` is released under the MIT License. See the [LICENSE](LICENSE) file for details.
