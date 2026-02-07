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

## Benchmark Report (v0.0.0-RC01)

System: Intel Core i7-13620H | AVX2 | Clang -O3 -march=native | 1M elements

### Element-wise Binary Operations (Mops/sec)

| Type   | Add    | Sub    | Mul    | Div    | Avg Improvement |
|--------|--------|--------|--------|--------|-----------------|
| INT32  | 3,888  | 3,956  | 3,832  | 626    | 32% faster      |
| INT64  | 1,355  | 1,311  | 1,205  | 588    | 14% faster      |
| FLOAT  | 3,831  | 3,811  | 3,943  | 3,562  | 36% faster      |
| DOUBLE | 1,255  | 1,130  | 1,109  | 1,109  | 1% faster       |

### Scalar Operations (Mops/sec)

| Type   | Add    | Sub    | Mul    | Div    |
|--------|--------|--------|--------|--------|
| INT32  | 5,100  | 5,200  | 5,500  | 6,000  |
| INT64  | 2,600  | 2,700  | 2,800  | 2,900  |
| FLOAT  | 5,500  | 5,700  | 5,900  | 6,200  |
| DOUBLE | 1,600  | 1,800  | 2,200  | 2,700  |

### Reduction Operations (Mops/sec)

| Type   | Sum    | Min    | Max    | Dot    |
|--------|--------|--------|--------|--------|
| INT32  | 14,400 | 7,800  | 8,200  | 9,100  |
| INT64  | 7,000  | 2,900  | 3,100  | 4,200  |
| FLOAT  | 1,900  | 900    | 1,000  | 1,500  |
| DOUBLE | 1,750  | 850    | 900    | 1,200  |

### Theoretical Efficiency

| Type  | Achieved   | Peak       | Efficiency | Bottleneck     |
|-------|------------|------------|------------|----------------|
| INT32 | 3.9 Gops/s | 8.0 Gops/s | 49%        | Memory/cache   |
| FLOAT | 3.9 Gops/s | 4.0 Gops/s | 97%        | Near-optimal   |
| INT64 | 1.4 Gops/s | 8.0 Gops/s | 17%        | DRAM bandwidth |
| DOUBLE| 1.3 Gops/s | 4.0 Gops/s | 32%        | DRAM bandwidth |

## Performance Characteristics

### Memory Hierarchy Impact

NUMC performance is highly dependent on whether data fits in CPU cache:

| Array Size | Memory   | Cache Level   | Add Performance | Performance Drop |
|------------|----------|---------------|-----------------|------------------|
| 1K elems   | 0.004 MB | L1 Cache      | ~8,300 Mops/s   | baseline         |
| 65K elems  | 0.256 MB | L2 Cache      | ~10,400 Mops/s  | ✅ +25%          |
| 524K elems | 2 MB     | L3 Cache      | ~10,100 Mops/s  | ✅ similar       |
| 4M elems   | 16 MB    | **Main Memory** | ~1,600 Mops/s | ⚠️ **-84% cliff** |

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

### Planned Optimizations for Memory-Bound Workloads

The following optimizations are planned to improve performance when data exceeds cache:

#### 1. **Cache Blocking (Tiling)**
Process large arrays in cache-sized chunks to maintain L3 residency:
```c
// Future API
array_add_blocked(a, b, out, block_size);  // Process in cache-friendly chunks
```

#### 2. **Explicit SIMD Intrinsics**
Use AVX2/AVX-512 intrinsics for better control over memory access patterns:
- Software prefetching (`_mm_prefetch`)
- Non-temporal stores for write-only data (`_mm_stream_si256`)
- Aligned vs unaligned load optimization

#### ~~3. **Multithreading**~~ ✅ Done
OpenMP parallelization implemented across all math operations (binary ops, scalar ops, reductions) and fill functions. Operations automatically parallelize when array size exceeds 100K elements.

#### 4. **Algorithm Fusion**
Combine multiple operations to reduce memory traffic:
```c
// Instead of: temp = a + b; result = temp * c;  (2 memory passes)
// Do: result = (a + b) * c;  (1 memory pass)
```

#### ~~5. **Compressed Data Formats**~~ ✅ Already Supported
The type system supports all precision levels (INT8-64, FLOAT, DOUBLE). Users can choose lower-precision types for better performance.

### Learning Resources

**Academic Papers:**
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) (Ulrich Drepper, 2007)
  - Comprehensive guide to memory hierarchy, cache behavior, and optimization strategies

- [Cache-Oblivious Algorithms](https://erikdemaine.org/papers/BRICS2002/paper.pdf) (Frigo et al., 1999)
  - Optimal cache-aware algorithms without knowing cache sizes

- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf) (Goto & van de Geijn, 2008)
  - Industry-standard techniques for blocking and tiling

- [Software Prefetching](https://dl.acm.org/doi/10.1145/384286.264207) (Mowry et al., 1992)
  - Cache miss reduction through prefetch instructions

**Books:**
- *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson)
- *Optimizing Software in C++* (Agner Fog) - [Free PDF](https://www.agner.org/optimize/)
- *Systems Performance* (Brendan Gregg)

**Online Resources:**
- [Intel Optimization Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Agner Fog's Optimization Guides](https://www.agner.org/optimize/)
- [Gallery of Processor Cache Effects](http://igoro.com/archive/gallery-of-processor-cache-effects/)

**Current Status:**
- ✅ SIMD auto-vectorization working (AVX2)
- ✅ Cache-friendly for typical workloads (<10 MB)
- ✅ Competitive with NumPy/BLAS for cache-resident data
- ✅ OpenMP multithreading for large arrays (>100K elements)
- 🔄 Cache blocking and explicit SIMD intrinsics planned (see roadmap above)

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
- [x] `array_copy` - Contiguous copy
- [x] `array_concat` - Concatenate along axis
- [x] `array_transpose` - Transpose
- [x] `array_is_contiguous` - Contiguity check
- [x] `array_add` / `sub` / `mul` / `div` - Element-wise binary ops
- [x] `array_add_scalar` / `sub` / `mul` / `div` - Scalar ops
- [x] `array_sum` / `array_min` / `array_max` / `array_dot` - Reductions
- [x] `array_print` - Print to stdout
- [x] Type-specific SIMD-ready kernels
- [x] Auto-vectorization (AVX2/SSE)
- [x] Separate 32-bit / 64-bit optimization strategies
- [x] Cache-friendly memory access patterns
- [x] Contiguous fast paths
- [x] 16-byte aligned allocation
- [x] Stack allocation for small arrays (ndim <= 8)
- [x] Comprehensive benchmark suite
- [x] Multithreading (OpenMP) - binary ops, scalar ops, reductions, fill

### Tier 1 — Foundation (implement next, other features depend on these)
- [ ] `array_arange` - Range of values
- [ ] `array_linspace` - Linearly spaced values
- [ ] `array_flatten` / `array_ravel` - Flatten to 1D
- [ ] `array_astype` - Type conversion (prerequisite for mixed-type math)
- [ ] `array_equal` / `array_allclose` - Comparison (needed for reliable testing)

### Tier 2 — Core Numeric (makes the library useful for real workloads)
- [ ] `array_mean` / `array_prod` / `array_std` - Statistical reductions
- [ ] Axis-based reductions (`sum_axis`, `mean_axis`)
- [ ] `array_argmin` / `array_argmax`
- [ ] `array_power` - Element-wise power
- [ ] `array_matmul` - Matrix multiplication

### Tier 3 — Shape Manipulation
- [ ] `array_squeeze` / `array_expand_dims` - Dimension manipulation
- [ ] `array_flip` - Reverse along axis
- [ ] `array_vstack` / `array_hstack` / `array_hsplit` - Stack/split
- [ ] `array_broadcast_to` - Broadcasting
- [ ] Broadcasting support for math ops

### Tier 4 — Nice to Have
- [ ] `array_eye` - Identity matrix
- [ ] `array_get_1d` / `_2d` / `_3d` - Fast dimensional access
- [ ] Boolean/conditional indexing
- [ ] `array_sort` / `array_argsort`
- [ ] `array_unique`
- [ ] `array_where` / `array_nonzero`
- [ ] `array_random` / `array_randint`
- [ ] Binary file I/O (`.npy` / `.npz`)
- [ ] Text file I/O (CSV/TXT)

### Performance (after API is stable)
- [ ] Explicit SIMD intrinsics (SSE2/AVX2/NEON)
- [ ] Runtime SIMD detection and dispatch
- [ ] Arena allocator for temporary arrays

## License

MIT License - See [LICENSE](LICENSE) for details.
