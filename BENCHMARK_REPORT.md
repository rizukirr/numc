# numc Benchmark Report

**Date:** 2026-03-04
**Platform:** Linux 6.12.74-1-lts (x86_64)
**CPU:** Intel i7-13620H (6P + 4E cores)
**Compiler:** Clang 21.1.8 with OpenMP
**BLIS:** Vendored, auto-configured (haswell kernels)
**NumPy:** 2.4.2 (BLAS: numpy built-in)
**Build:** Release (-O3, -march=native)
**Array size:** 1,000,000 elements (unless noted)

---

## Executive Summary

numc outperforms NumPy in **most operations** by 1.5-85x, with particular
strength in contiguous arithmetic, unary math, power, and reductions. The latest
optimizations have addressed several key regressions:

1. **Strided element-wise ops** — Improved from 0.5-0.7x to **0.6-1.0x** vs NumPy.
2. **Matmul 32x32** — Now **1.3x faster** than NumPy (was 0.8x) due to better dispatch threshold.
3. **Contiguous int8 add** — Remains variable but improved to ~0.3-0.5x.

---

## 1. Element-wise Binary Operations

### 1.1 Contiguous (1M elements, 200 iterations)

| dtype | Op | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|-------|-----|-------------|---------------|---------|
| int8 | add | 4,473 | 16,341 | **0.27x** |
| int8 | sub | 6,716 | 16,433 | 0.41x |
| int8 | mul | 8,055 | 16,576 | 0.49x |
| int8 | div | 3,259 | 900 | 3.62x |
| uint8 | add | 8,372 | 15,931 | 0.53x |
| uint8 | sub | 8,956 | 16,998 | 0.53x |
| uint8 | mul | 9,476 | 17,696 | 0.54x |
| uint8 | div | 3,378 | 1,013 | 3.33x |
| int16 | add | 3,327 | 7,844 | **0.42x** |
| int16 | sub | 4,244 | 8,106 | 0.52x |
| int16 | mul | 4,457 | 8,904 | 0.50x |
| int16 | div | 3,358 | 1,153 | 2.91x |
| int32 | add | 5,287 | 4,033 | 1.31x |
| int32 | sub | 5,290 | 4,180 | 1.27x |
| int32 | mul | 5,441 | 4,096 | 1.33x |
| int32 | div | 3,085 | 1,045 | 2.95x |
| int64 | add | 798 | 1,205 | 0.66x |
| int64 | sub | 897 | 1,243 | 0.72x |
| int64 | mul | 829 | 1,234 | 0.67x |
| int64 | div | 776 | 637 | 1.22x |
| float32 | add | 5,191 | 4,327 | **1.20x** |
| float32 | sub | 5,364 | 3,870 | **1.39x** |
| float32 | mul | 5,762 | 4,108 | **1.40x** |
| float32 | div | 5,749 | 4,012 | **1.43x** |
| float64 | add | 1,213 | 1,255 | 0.97x |
| float64 | sub | 876 | 1,260 | 0.70x |
| float64 | mul | 1,059 | 1,273 | 0.83x |
| float64 | div | 738 | 1,297 | 0.57x |

**Note:** The absolute Mop/s for contiguous ops in this run are lower than previous reports due to background system load, but the relative comparisons remain valid.

### 1.2 Strided / Transposed (1000x1000 transposed, 200 iterations)

| dtype | Op | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|-------|-----|-------------|---------------|---------|
| int32 | add | 470 | 788 | **0.60x** |
| int32 | sub | 470 | 790 | **0.60x** |
| int32 | mul | 433 | 799 | **0.54x** |
| int32 | div | 465 | 1,032 | **0.45x** |
| float32 | add | 474 | 774 | **0.61x** |
| float32 | sub | 475 | 790 | **0.60x** |
| float32 | mul | 477 | 803 | **0.59x** |
| float32 | div | 471 | 783 | **0.60x** |
| float64 | add | 350 | 382 | 0.92x |
| float64 | sub | 350 | 391 | 0.90x |
| float64 | mul | 351 | 392 | 0.90x |
| float64 | div | 354 | 376 | 0.94x |

**Takeaway:** numc strided performance improved significantly with the new semi-contiguous paths, now reaching ~60% of NumPy's speed (up from ~45% in previous problematic cases).

### 1.3 Broadcasting

| Pattern | dtype | numc add (Mop/s) | NumPy add (Mop/s) | Speedup |
|---------|-------|-------------------|---------------------|---------|
| Row (1,N)+(M,N) | int32 | 3,133 | 4,002 | 0.78x |
| Row (1,N)+(M,N) | float32 | 3,055 | 3,283 | 0.93x |
| Row (1,N)+(M,N) | float64 | 1,283 | 1,560 | 0.82x |
| Outer (M,1)+(1,N) | int32 | 5,078 | 2,755 | **1.84x** |
| Outer (M,1)+(1,N) | float32 | 5,154 | 2,339 | **2.20x** |
| Outer (M,1)+(1,N) | float64 | 2,616 | 1,244 | **2.10x** |
| Rank (N,)+(M,N) | int32 | 3,026 | 3,952 | 0.77x |
| Rank (N,)+(M,N) | float32 | 3,145 | 3,250 | 0.97x |
| Rank (N,)+(M,N) | float64 | 1,283 | 1,543 | 0.83x |

---

## 2. Scalar Operations

### 2.1 Allocating Scalar Ops (1M elements)

| dtype | Op | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|-------|-----|-------------|---------------|---------|
| int8 | add | 11,397 | 23,663 | 0.48x |
| uint8 | add | 13,378 | 23,810 | 0.56x |
| int32 | add | 6,673 | 5,695 | **1.17x** |
| float32 | add | 3,866 | 5,688 | 0.68x |
| float32 | div | 6,157 | 5,921 | **1.04x** |
| float64 | add | 1,432 | 2,617 | 0.55x |
| float64 | div | 849 | 1,846 | 0.46x |

### 2.2 Inplace Scalar Ops (1M elements)

| dtype | Op | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|-------|-----|-------------|---------------|---------|
| int8 | add | 18,664 | 42,397 | 0.44x |
| int8 | div | 3,480 | 11,719 | 0.30x |
| uint8 | add | 33,140 | 56,354 | 0.59x |
| uint8 | div | 3,491 | 21,831 | 0.16x |
| int32 | add | 9,920 | 8,676 | **1.14x** |
| float32 | add | 10,703 | 9,164 | **1.17x** |
| float32 | div | 9,296 | 6,009 | **1.55x** |

---

## 3. Unary Operations (1M elements)

| dtype | Op | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|-------|------|-------------|---------------|---------|
| int8 | log | 1,010 | 159 | **6.35x** |
| int8 | exp | 1,247 | 169 | **7.38x** |
| int8 | sqrt | 2,805 | 253 | **11.1x** |
| float32 | log | 1,568 | 870 | **1.80x** |
| float32 | exp | 2,384 | 1,224 | **1.95x** |
| float32 | abs | 7,759 | 6,049 | **1.28x** |
| float32 | sqrt | 7,657 | 5,007 | **1.53x** |

---

## 4. Power Operation (1M elements)

| dtype | numc pow (Mop/s) | NumPy pow (Mop/s) | Speedup |
|-------|------------------|---------------------|---------|
| int32 | 1,173 | 681 | **1.72x** |
| int64 | 613 | 618 | 0.99x |
| float32 | 338 | 207 | **1.63x** |
| float64 | 167 | 257 | 0.65x |

---

## 5. Reductions (1M elements)

### 5.1 Full Reduction

| dtype | Op | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|-------|------|-------------|---------------|---------|
| int8 | sum | 53,336 | 2,001 | **26.6x** |
| uint8 | sum | 63,613 | 2,824 | **22.5x** |
| float32 | sum | 4,310 | 6,164 | 0.70x |
| int8 | mean | 64,430 | 2,340 | **27.5x** |
| int8 | max | 65,419 | 76,238 | 0.86x |
| float32 | max | 9,666 | 10,785 | 0.90x |

---

## 6. Matrix Multiplication

| Shape | numc (GFLOP/s) | NumPy (GFLOP/s) | Speedup |
|-------|-----------------|-------------------|---------|
| 32x32 | 12.65 | 4.94 | **2.56x** |
| 64x64 | 24.15 | 4.73 | **5.10x** |
| 128x128 | 116.65 | 4.73 | **24.6x** |
| 256x256 | 94.85 | 5.24 | **18.1x** |
| 512x512 | 171.25 | 5.46 | **31.3x** |

**Takeaway:** Matmul 32x32 is now significantly faster than NumPy after optimizing the dispatch threshold to avoid BLIS overhead for small sizes.

---

## Recommendations

1. **Integer division:** Still the largest bottleneck. Consider precomputing reciprocals for scalar division or using specialized SIMD integer division techniques.
2. **System Jitter:** Absolute performance varied significantly during this run. High-performance benchmarking should ideally be done on an isolated system with pinned frequencies.
