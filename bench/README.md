# numc vs NumPy Benchmark

All benchmarks run on contiguous 1M-element arrays, 200 iterations, best-of-warmup.

**System:** Intel i7-13620H (16 threads) | Clang 21.1.8 + OpenMP | NumPy 2.4.2 | Linux 6.12.73-1-lts

---

## Binary Element-wise (add, sub, mul, div)

1M contiguous elements. Throughput in Mop/s (higher is better).

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | add | 17,493 | 13,362 | 1.3x |
| int8 | sub | 15,664 | 13,248 | 1.2x |
| int8 | mul | 10,110 | 13,210 | 0.8x |
| int8 | div | 5,915 | 1,118 | **5.3x** |
| uint8 | add | 14,712 | 12,843 | 1.1x |
| uint8 | sub | 14,799 | 13,405 | 1.1x |
| uint8 | mul | 14,598 | 13,164 | 1.1x |
| uint8 | div | 6,039 | 1,073 | **5.6x** |
| int16 | add | 7,618 | 6,551 | 1.2x |
| int16 | sub | 8,147 | 6,833 | 1.2x |
| int16 | mul | 8,645 | 6,801 | 1.3x |
| int16 | div | 6,023 | 1,151 | **5.2x** |
| uint16 | add | 7,943 | 6,879 | 1.2x |
| uint16 | sub | 8,595 | 6,844 | 1.3x |
| uint16 | mul | 8,745 | 6,897 | 1.3x |
| uint16 | div | 5,981 | 1,168 | **5.1x** |
| int32 | add | 4,524 | 3,312 | 1.4x |
| int32 | sub | 7,672 | 3,237 | **2.4x** |
| int32 | mul | 9,090 | 3,396 | **2.7x** |
| int32 | div | 5,280 | 993 | **5.3x** |
| uint32 | add | 7,680 | 3,339 | **2.3x** |
| uint32 | sub | 10,531 | 3,354 | **3.1x** |
| uint32 | mul | 10,390 | 3,342 | **3.1x** |
| uint32 | div | 5,445 | 832 | **6.5x** |
| int64 | add | 4,336 | 1,110 | **3.9x** |
| int64 | sub | 2,223 | 1,171 | **1.9x** |
| int64 | mul | 2,448 | 1,330 | **1.8x** |
| int64 | div | 1,974 | 573 | **3.4x** |
| uint64 | add | 5,930 | 942 | **6.3x** |
| uint64 | sub | 5,620 | 1,196 | **4.7x** |
| uint64 | mul | 5,454 | 1,346 | **4.1x** |
| uint64 | div | 2,299 | 548 | **4.2x** |
| float32 | add | 9,780 | 3,374 | **2.9x** |
| float32 | sub | 9,743 | 3,151 | **3.1x** |
| float32 | mul | 10,442 | 3,552 | **2.9x** |
| float32 | div | 10,459 | 3,514 | **3.0x** |
| float64 | add | 1,619 | 911 | **1.8x** |
| float64 | sub | 4,319 | 980 | **4.4x** |
| float64 | mul | 2,557 | 884 | **2.9x** |
| float64 | div | 3,349 | 1,306 | **2.6x** |

**Summary:** numc matches or beats NumPy on 8-bit/16-bit types (both are memory-bandwidth bound). numc is 2-6x faster on 32-bit and 64-bit types due to zero Python/ufunc dispatch overhead. Integer division is 3-6x faster across the board.

---

## Strided (transposed)

1M elements (1000x1000, one operand transposed). Tests tiled gather/compute/scatter PATH 3.

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int32 | add | 579 | 788 | 0.7x |
| int32 | sub | 560 | 771 | 0.7x |
| int32 | mul | 576 | 785 | 0.7x |
| int32 | div | 482 | 1,025 | 0.5x |
| float32 | add | 465 | 728 | 0.6x |
| float32 | sub | 585 | 769 | 0.8x |
| float32 | mul | 521 | 755 | 0.7x |
| float32 | div | 559 | 682 | 0.8x |
| float64 | add | 355 | 341 | 1.0x |
| float64 | sub | 352 | 316 | 1.1x |
| float64 | mul | 315 | 291 | 1.1x |
| float64 | div | 326 | 340 | 1.0x |

**Summary:** Strided ops use tiled gather/compute/scatter (tile size 256). For cheap ops (add/mul), the gather/scatter overhead exceeds the vectorization gain — NumPy's nditer is faster on int32/float32. For float64 (larger elements, fewer tiles), numc is roughly even. The tiling wins on compute-heavy ops (pow, log, exp) where vectorized tiles amortize the gather/scatter cost.

---

## Broadcasting

1M output elements (1000x1000). Three broadcast patterns tested: row `(1,N)+(M,N)`, outer `(M,1)+(1,N)`, rank `(N,)+(M,N)`.

### Row Broadcast: (1,1000) + (1000,1000) → (1000,1000)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int32 | add | 5,298 | 3,996 | 1.3x |
| int32 | sub | 5,295 | 3,783 | 1.4x |
| int32 | mul | 5,284 | 3,774 | 1.4x |
| int32 | div | 1,886 | 1,060 | **1.8x** |
| float32 | add | 5,365 | 3,034 | **1.8x** |
| float32 | sub | 5,548 | 3,099 | **1.8x** |
| float32 | mul | 5,900 | 3,191 | **1.8x** |
| float32 | div | 5,839 | 3,142 | **1.9x** |
| float64 | add | 2,754 | 1,452 | **1.9x** |
| float64 | sub | 2,514 | 1,322 | **1.9x** |
| float64 | mul | 2,513 | 1,552 | **1.6x** |
| float64 | div | 1,822 | 1,485 | 1.2x |

### Outer Broadcast: (1000,1) + (1,1000) → (1000,1000)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int32 | add | 8,871 | 2,756 | **3.2x** |
| int32 | sub | 8,494 | 2,718 | **3.1x** |
| int32 | mul | 9,118 | 2,706 | **3.4x** |
| int32 | div | 1,888 | 937 | **2.0x** |
| float32 | add | 9,067 | 2,249 | **4.0x** |
| float32 | sub | 8,962 | 2,307 | **3.9x** |
| float32 | mul | 9,183 | 2,332 | **3.9x** |
| float32 | div | 6,001 | 2,294 | **2.6x** |
| float64 | add | 4,647 | 1,235 | **3.8x** |
| float64 | sub | 4,662 | 1,220 | **3.8x** |
| float64 | mul | 4,651 | 1,242 | **3.7x** |
| float64 | div | 1,888 | 1,123 | **1.7x** |

### Rank Broadcast: (1000,) + (1000,1000) → (1000,1000)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int32 | add | 5,293 | 3,726 | 1.4x |
| int32 | sub | 5,319 | 4,014 | 1.3x |
| int32 | mul | 5,324 | 3,750 | 1.4x |
| int32 | div | 1,890 | 1,124 | **1.7x** |
| float32 | add | 5,264 | 3,122 | **1.7x** |
| float32 | sub | 5,265 | 3,143 | **1.7x** |
| float32 | mul | 5,263 | 3,159 | **1.7x** |
| float32 | div | 5,224 | 3,119 | **1.7x** |
| float64 | add | 2,497 | 1,546 | **1.6x** |
| float64 | sub | 2,319 | 1,500 | **1.5x** |
| float64 | mul | 2,429 | 1,405 | **1.7x** |
| float64 | div | 1,834 | 1,451 | 1.3x |

**Summary:** Row and rank broadcast are 1.3-1.9x faster than NumPy — one input reads linearly while the other repeats via stride=0, giving good cache locality. **Outer broadcast is 2.6-4.0x faster** thanks to PATH 2.5 (left scalar broadcast), which vectorizes + OMP the previously-scalar inner loop. Prior to PATH 2.5, outer broadcast was 0.5-1.1x NumPy.

---

## Scalar Element-wise (a + scalar, a * scalar, ...)

1M contiguous elements.

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | add | 23,254 | 23,700 | 1.0x |
| int8 | mul | 23,679 | 22,304 | 1.1x |
| uint8 | add | 28,010 | 22,085 | 1.3x |
| uint8 | mul | 27,194 | 21,501 | 1.3x |
| int32 | add | 11,740 | 5,520 | **2.1x** |
| int32 | mul | 14,593 | 5,613 | **2.6x** |
| int32 | div | 5,456 | 1,352 | **4.0x** |
| int64 | add | 6,190 | 2,510 | **2.5x** |
| int64 | mul | 5,276 | 2,291 | **2.3x** |
| float32 | add | 13,663 | 5,392 | **2.5x** |
| float32 | mul | 14,343 | 5,562 | **2.6x** |
| float32 | div | 10,311 | 5,593 | **1.8x** |
| float64 | add | 7,238 | 2,502 | **2.9x** |
| float64 | mul | 9,291 | 2,345 | **4.0x** |

**Inplace scalar** (a += scalar):

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | add | 50,819 | 54,919 | 0.9x |
| int8 | mul | 49,242 | 46,352 | 1.1x |
| int32 | add | 24,916 | 9,092 | **2.7x** |
| int32 | mul | 21,837 | 9,149 | **2.4x** |
| float32 | add | 28,235 | 9,142 | **3.1x** |
| float32 | mul | 28,036 | 9,238 | **3.0x** |
| float64 | add | 10,674 | 4,515 | **2.4x** |

---

## Unary Operations (log, exp, abs, sqrt)

1M contiguous elements.

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | log | 1,746 | 159 | **11.0x** |
| int8 | exp | 2,152 | 169 | **12.7x** |
| int8 | abs | 24,633 | 23,384 | 1.1x |
| int8 | sqrt | 4,867 | 254 | **19.2x** |
| int16 | log | 1,753 | 807 | **2.2x** |
| int16 | exp | 2,273 | 1,133 | **2.0x** |
| int16 | sqrt | 4,936 | 3,503 | 1.4x |
| int32 | log | 2,122 | 311 | **6.8x** |
| int32 | exp | 1,798 | 280 | **6.4x** |
| int32 | abs | 14,372 | 6,044 | **2.4x** |
| int32 | sqrt | 3,639 | 983 | **3.7x** |
| int64 | log | 709 | 287 | **2.5x** |
| int64 | exp | 618 | 261 | **2.4x** |
| int64 | abs | 6,382 | 2,513 | **2.5x** |
| int64 | sqrt | 1,777 | 800 | **2.2x** |
| float32 | log | 5,859 | 841 | **7.0x** |
| float32 | exp | 7,003 | 1,259 | **5.6x** |
| float32 | abs | 13,853 | 5,793 | **2.4x** |
| float32 | sqrt | 13,443 | 4,985 | **2.7x** |
| float64 | log | 1,614 | 333 | **4.8x** |
| float64 | exp | 1,348 | 299 | **4.5x** |
| float64 | abs | 7,771 | 2,489 | **3.1x** |
| float64 | sqrt | 2,446 | 1,226 | **2.0x** |

**Summary:** numc custom log/exp helpers are 2-13x faster than NumPy across all types. abs is 2-3x faster for 32-bit+. sqrt is now 2-3x faster on float types thanks to `-fno-math-errno` (allowing the compiler to inline `vsqrtps`/`vsqrtpd` without an errno-setting call).

---

## Pow

1M contiguous elements.

| dtype | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---:|---:|---:|
| int8 | 1,871 | 657 | **2.8x** |
| uint8 | 1,645 | 704 | **2.3x** |
| int16 | 1,029 | 687 | **1.5x** |
| uint16 | 1,013 | 747 | 1.4x |
| int32 | 2,091 | 658 | **3.2x** |
| uint32 | 2,085 | 745 | **2.8x** |
| int64 | 2,001 | 609 | **3.3x** |
| uint64 | 2,041 | 633 | **3.2x** |
| float32 | 2,364 | 207 | **11.4x** |
| float64 | 447 | 101 | **4.4x** |

**Summary:** numc uses branchless fixed-iteration for 8/16-bit, exponentiation-by-squaring for 32/64-bit integers, and fused `exp(b * log(a))` for floats. 2-11x faster across the board.

---

## Reductions

### Full Reduction (1M elements -> scalar)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 55,185 | 2,341 | **23.6x** |
| int8 | mean | 104,142 | 2,324 | **44.8x** |
| int8 | max | 88,185 | 67,526 | 1.3x |
| int8 | min | 107,298 | 70,253 | **1.5x** |
| int8 | argmax | 111,202 | 37,621 | **3.0x** |
| int8 | argmin | 110,145 | 39,555 | **2.8x** |
| int16 | sum | 19,795 | 2,174 | **9.1x** |
| int16 | max | 32,171 | 29,427 | 1.1x |
| int16 | argmax | 23,313 | 20,655 | 1.1x |
| int32 | sum | 23,234 | 3,256 | **7.1x** |
| int32 | mean | 30,409 | 2,744 | **11.1x** |
| int32 | max | 27,375 | 12,861 | **2.1x** |
| int32 | min | 28,676 | 13,285 | **2.2x** |
| int32 | argmax | 10,990 | 10,622 | 1.0x |
| int32 | argmin | 11,075 | 10,910 | 1.0x |
| int64 | sum | 8,935 | 5,761 | **1.6x** |
| int64 | max | 9,348 | 5,904 | **1.6x** |
| int64 | argmax | 5,456 | 4,109 | 1.3x |
| float32 | sum | 11,694 | 6,110 | **1.9x** |
| float32 | mean | 9,255 | 6,048 | **1.5x** |
| float32 | max | 19,067 | 10,897 | **1.8x** |
| float32 | min | 19,739 | 11,070 | **1.8x** |
| float32 | argmax | 11,610 | 8,788 | 1.3x |
| float32 | argmin | 10,725 | 8,710 | 1.2x |
| float64 | sum | 7,109 | 5,036 | 1.4x |
| float64 | mean | 8,430 | 4,866 | **1.7x** |
| float64 | max | 13,958 | 5,061 | **2.8x** |
| float64 | min | 13,260 | 5,154 | **2.6x** |
| float64 | argmax | 5,525 | 4,216 | 1.3x |
| float64 | argmin | 5,597 | 4,168 | 1.3x |

**Note:** float32 argmax/argmin use a chunked single-pass algorithm (chunk size 1024): each chunk finds the extreme via vectorized multi-accumulator, then only scans for the matching index when a chunk beats the global best. This reads data once (vs. two-pass before), yielding 1.2-1.3x over NumPy (up from 0.9x with the old two-pass algorithm).

### Axis=0 Reduction (1000x1000 -> 1000 cols)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 34,544 | 2,275 | **15.2x** |
| int8 | mean | 49,327 | 2,597 | **19.0x** |
| int8 | max | 44,140 | 24,597 | **1.8x** |
| int8 | min | 44,297 | 25,060 | **1.8x** |
| int8 | argmax | 3,013 | 1,836 | **1.6x** |
| int8 | argmin | 3,028 | 1,816 | **1.7x** |
| int32 | sum | 11,134 | 2,621 | **4.2x** |
| int32 | max | 12,406 | 7,363 | **1.7x** |
| int32 | argmax | 6,082 | 1,512 | **4.0x** |
| int32 | argmin | 6,458 | 1,604 | **4.0x** |
| float32 | sum | 10,111 | 8,704 | 1.2x |
| float32 | mean | 8,786 | 8,106 | 1.1x |
| float32 | max | 10,738 | 7,313 | **1.5x** |
| float32 | min | 13,865 | 7,341 | **1.9x** |
| float32 | argmax | 6,613 | 1,466 | **4.5x** |
| float32 | argmin | 6,631 | 1,540 | **4.3x** |
| float64 | sum | 5,424 | 4,912 | 1.1x |
| float64 | max | 5,636 | 3,880 | **1.5x** |
| float64 | argmax | 5,257 | 903 | **5.8x** |
| float64 | argmin | 5,324 | 930 | **5.7x** |

**Note:** Axis argmax/argmin now use fused row-reduce kernels that process all rows in a single call, tracking both values (VLA scratch) and indices (output). This eliminates per-row ND iterator overhead. Axis=0 argmax went from ~920 Mop/s (generic ND path) to 4,000-6,600 Mop/s — a **4-7x improvement** over the old code, and **4-6x faster than NumPy**.

### Axis=1 Reduction (1000x1000 -> 1000 rows)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 48,096 | 2,644 | **18.2x** |
| int8 | mean | 46,322 | 2,545 | **18.2x** |
| int8 | max | 47,481 | 22,111 | **2.1x** |
| int8 | argmax | 52,635 | 8,141 | **6.5x** |
| int8 | argmin | 52,592 | 8,027 | **6.6x** |
| int32 | sum | 10,232 | 3,664 | **2.8x** |
| int32 | max | 10,139 | 10,303 | 1.0x |
| int32 | argmax | 10,344 | 9,955 | 1.0x |
| int32 | argmin | 11,063 | 9,918 | 1.1x |
| float32 | sum | 6,716 | 5,781 | 1.2x |
| float32 | max | 10,665 | 6,476 | **1.6x** |
| float32 | argmax | 10,668 | 7,890 | 1.4x |
| float32 | argmin | 10,565 | 8,025 | 1.3x |
| float64 | sum | 4,949 | 5,049 | 1.0x |
| float64 | max | 8,028 | 4,185 | **1.9x** |
| float64 | argmax | 5,479 | 4,089 | 1.3x |
| float64 | argmin | 5,391 | 4,084 | 1.3x |

---

## Size Scaling (float32 sum)

| elements | numc (us) | numpy (us) | numc GB/s | numpy GB/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.02 | 0.74 | 26.02 | 0.54 | **37.0x** |
| 1,000 | 0.14 | 0.96 | 28.44 | 4.17 | **6.9x** |
| 10,000 | 1.46 | 2.53 | 27.44 | 15.80 | **1.7x** |
| 100,000 | 14.13 | 18.10 | 28.32 | 22.10 | 1.3x |
| 1,000,000 | 59.02 | 163.89 | 67.77 | 24.41 | **2.8x** |

**Summary:** numc has near-zero dispatch overhead. At 100 elements, numc is 37x faster due to NumPy's Python/ufunc dispatch cost (~0.74us baseline). At 1M elements numc is 2.8x faster, reaching 67.77 GB/s bandwidth.

---

## How to reproduce

```bash
# numc benchmarks
./run.sh bench              # all benchmarks
./run.sh bench-elemwise     # binary element-wise only
./run.sh bench-scalar       # scalar ops only
./run.sh bench-unary        # unary ops only
./run.sh bench-pow          # pow only
./run.sh bench-reduction    # reductions only

# numpy benchmarks
python bench/numpy_bench_elemwise.py
python bench/numpy_bench_scalar.py
python bench/numpy_bench_unary.py
python bench/numpy_bench_pow.py
python bench/numpy_bench_reduction.py
```
