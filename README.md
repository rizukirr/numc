# numc

A lightweight, NumPy-inspired N-dimensional array library written in C.

> [!WARNING]
> **This project is under heavy development and the API is not stable.**
> Breaking changes may occur between commits. Not recommended for production use yet.

## Building

### Using run.sh

```bash
./run.sh debug       # Build debug mode (with AddressSanitizer) and run demo
./run.sh release     # Build release mode (with -O3 -march=native) and run demo
./run.sh test        # Build and run all tests
./run.sh benchmark   # Build release and run benchmarks
./run.sh clean       # Remove build directory
./run.sh rebuild     # Clean and rebuild in debug mode
./run.sh help        # Show all commands
```

### Manual

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run demo
./build/bin/numc_demo

# Run tests
ctest --test-dir build

# Build options
cmake -B build -DCMAKE_BUILD_TYPE=Debug      # Debug with AddressSanitizer
cmake -B build -DBUILD_SHARED=ON             # Shared library (.so)
cmake -B build -DBUILD_TESTS=OFF             # Disable tests
cmake -B build -DBUILD_EXAMPLES=OFF          # Disable examples
```

## Benchmark Report

System: Intel Core i7-13620H | AVX2 | Clang -O3 -march=native | 1M elements

### Element-wise Binary Operations (Mops/sec)

| Type   | Add     | Sub     | Mul     | Div    |
|--------|---------|---------|---------|--------|
| INT32  | 11,427  | 11,747  | 10,960  | 4,490  |
| INT64  | 4,615   | 3,415   | 1,981   | 1,331  |
| FLOAT  | 10,673  | 11,787  | 9,097   | 6,484  |
| DOUBLE | 3,930   | 1,492   | 2,323   | 2,883  |

### Scalar Operations (Mops/sec)

| Type   | Add     | Sub     | Mul     | Div     |
|--------|---------|---------|---------|---------|
| INT32  | 17,088  | 16,476  | 17,441  | 15,796  |
| INT64  | 2,856   | 3,742   | 6,063   | 3,387   |
| FLOAT  | 6,250   | 6,162   | 9,318   | 10,263  |
| DOUBLE | 6,330   | 7,178   | 6,355   | 6,851   |

### Reduction Operations (Mops/sec)

| Type   | Sum     | Min     | Max     | Prod    | Dot     | Mean    | Std    |
|--------|---------|---------|---------|---------|---------|---------|--------|
| INT32  | 15,481  | 12,887  | 12,902  | 12,033  | 6,812   | 13,915  | 6,568  |
| INT64  | 7,460   | 6,173   | 6,230   | 4,888   | 2,828   | 4,976   | 2,397  |
| FLOAT  | 14,678  | 14,209  | 14,205  | 16,412  | 7,578   | 13,774  | 6,631  |
| DOUBLE | 6,389   | 6,352   | 6,757   | 7,479   | 2,999   | 7,452   | 3,713  |

### Axis Reduction Operations (Mops/sec, 1000x1000 2D)

| Type   | Mean ax=0 | Mean ax=1 | Std ax=0 | Std ax=1 |
|--------|-----------|-----------|----------|----------|
| INT32  | 8,595     | 13,540    | 3,801    | 6,244    |
| INT64  | 4,405     | 4,783     | 2,033    | 2,300    |
| FLOAT  | 8,396     | 13,185    | 3,821    | 6,437    |
| DOUBLE | 5,957     | 7,932     | 3,204    | 4,814    |

## Performance Characteristics

### Memory Hierarchy Impact

NUMC performance is highly dependent on whether data fits in CPU cache:

| Array Size | Memory   | Cache Level   | Add Performance | Performance Drop |
|------------|----------|---------------|-----------------|------------------|
| 1K elems   | 0.004 MB | L1 Cache      | ~8,300 Mops/s   | baseline         |
| 65K elems  | 0.256 MB | L2 Cache      | ~10,400 Mops/s  | ✅ +25%          |
| 524K elems | 2 MB     | L3 Cache      | ~10,100 Mops/s  | ✅ similar       |
| 4M elems   | 16 MB    | **Main Memory** | ~1,600 Mops/s | **-84% cliff** |

**Why this happens:**
- **L3 Cache** (18 MB): ~40 cycle latency, 200 GB/s bandwidth → Fast ✅
- **Main Memory** (DDR4): ~300 cycle latency, 50 GB/s bandwidth → Slow ⚠️

When working set exceeds cache size (~18 MB on i7-13620H), the system becomes **memory-bound** rather than compute-bound. This is fundamental to all high-performance libraries (NumPy, BLAS, TensorFlow) and is not a bug.

### 32-bit vs 64-bit Performance

**INT32/FLOAT (4 bytes):**
- 1M elements × 3 arrays = **12 MB total** → Fits in L3 cache ✅
- Achieves ~3,900 Mops/s (excellent performance)
- 8 elements per 256-bit AVX2 vector
- Cache-resident, compute-bound

**INT64/DOUBLE (8 bytes):**
- 1M elements × 3 arrays = **24 MB total** → Exceeds L3 cache ⚠️
- Achieves ~1,300 Mops/s (memory-bound)
- 4 elements per 256-bit AVX2 vector
- Main memory access required

This explains why 32-bit types are ~3× faster than 64-bit types for large arrays.

### Optimization Approach

**Auto-vectorization over hand-written intrinsics:** Explicit AVX2 intrinsics were tested in a separate translation unit but performed worse than auto-vectorization (`-O3 -march=native`) for most types due to cross-TU call overhead preventing inlining. LTO didn't fully recover the loss. The current approach uses X-macro generated type-specific kernels that the compiler auto-vectorizes effectively.

**Implemented:**
- ✅ SIMD auto-vectorization (AVX2/SSE2) via `-O3 -march=native`
- ✅ OpenMP parallelization for arrays > 100K elements
- ✅ 32-byte aligned allocation for SIMD compatibility
- ✅ Typed accumulation kernels for balanced axis reduction performance
- ✅ All 10 precision levels (INT8-64, FLOAT, DOUBLE)

**Planned:**
- Cache blocking (tiling) for memory-bound workloads
- Operation fusion to reduce memory traffic

## Progress

### Completed
- [x] `array_create` - Create array from specification (zeroed when no data)
- [x] `array_empty` - Uninitialized array (fast, no zeroing)
- [x] `array_zeros` - Zero-filled array
- [x] `array_ones` - Ones-filled array
- [x] `array_full` - Fill with value
- [x] `array_free` - Free array memory
- [x] `array_get` / type-safe accessors - Get element pointer
- [x] `array_offset` - Compute byte offset
- [x] `array_bounds_check` - Bounds checking
- [x] `array_reshape` - Reshape in-place (contiguous only)
- [x] `array_slice` - Zero-copy view
- [x] `array_index_axis` - Index along axis (returns ndim-1 view)
- [x] `array_copy` - Contiguous copy
- [x] `array_ascontiguousarray` - Convert to contiguous in-place
- [x] `array_concat` - Concatenate along axis
- [x] `array_transpose` - Transpose
- [x] `array_is_contiguous` - Contiguity check
- [x] `array_add` / `sub` / `mul` / `div` - Element-wise binary ops
- [x] `array_add_scalar` / `sub` / `mul` / `div` - Scalar ops
- [x] `array_sum` / `array_min` / `array_max` / `array_prod` / `array_dot` - Full reductions
- [x] `array_mean` / `array_std` - Statistical reductions (double output)
- [x] `array_sum_axis` / `array_prod_axis` / `array_min_axis` / `array_max_axis` - Axis reductions
- [x] `array_mean_axis` / `array_std_axis` - Axis statistical reductions (double output)
- [x] `array_print` - Print to stdout
- [x] `array_arange` - Range of values
- [x] `array_linspace` - Linearly spaced values
- [x] `array_flatten` / `array_ravel` - Flatten to 1D
- [x] `array_astype` - Type conversion
- [x] `array_equal` / `array_allclose` - Comparison
- [x] Auto-vectorization (AVX2/SSE2) via X-macro kernels
- [x] Separate 32-bit / 64-bit optimization strategies
- [x] Cache-friendly memory access patterns
- [x] 32-byte aligned allocation for SIMD
- [x] Stack allocation for small arrays (ndim <= 8)
- [x] OpenMP parallelization (binary ops, scalar ops, reductions, fill)
- [x] Comprehensive benchmark suite

### Roadmap — Neural Network from Scratch

Goal: build a working MLP (multi-layer perceptron) using only numc.
Tiers are ordered by what unblocks the most NN functionality.

**Tier 1 — Cannot train any NN without these**
- [ ] `array_matmul` - Matrix multiplication (forward pass, backward pass, gradients)
- [ ] Broadcasting for binary ops - Bias addition `z = X @ W + b` where b is [1, N]
- [ ] `array_exp` / `array_log` / `array_negate` - Softmax, cross-entropy loss
- [ ] `array_random` / `array_randn` - Weight initialization (Xavier/He)

```
# Forward:  z = X @ W + b;  a = relu(z);  out = softmax(z2)
# Backward: dW = a.T @ dz / batch;  db = mean(dz, axis=0)  ← already have!
# Update:   W -= lr * dW  ← already have scalar ops!
```

**Tier 2 — Practical training loop**
- [ ] `array_clip` - Gradient clipping, numerical stability
- [ ] `array_where` - ReLU backward `dz = da * (z > 0)`, conditional ops
- [ ] `array_argmax` - Prediction labels, accuracy computation
- [ ] `array_sqrt` / `array_abs` / `array_power` - Adam optimizer, L2 regularization
- [ ] `array_squeeze` / `array_expand_dims` - Dimension manipulation for broadcasting

**Tier 3 — Convenience and I/O**
- [ ] `array_eye` - Identity matrix
- [ ] Binary file I/O (`.npy` / `.npz`) - Save/load weights
- [ ] `array_vstack` / `array_hstack` - Batch assembly
- [ ] `array_shuffle` / `array_permutation` - Mini-batch SGD
- [ ] `array_flip` - Data augmentation

**Tier 4 — Advanced**
- [ ] `array_sort` / `array_argsort`
- [ ] `array_unique`
- [ ] `array_nonzero`
- [ ] Boolean/conditional indexing
- [ ] Text file I/O (CSV/TXT)

### Performance
- [ ] Cache blocking (tiling) for memory-bound workloads
- [ ] Operation fusion to reduce memory traffic
- [ ] Arena allocator for temporary arrays

## License

MIT License - See [LICENSE](LICENSE) for details.
