# numc vs NumPy Benchmark

All benchmarks run on contiguous 1M-element arrays, 200 iterations, best-of-warmup.

**System:** Intel i7-13620H (16 threads) | Clang 21.1.8 + OpenMP | NumPy 2.4.2 | Linux 6.12.73-1-lts

---

## Binary Element-wise (add, sub, mul, div)

1M contiguous elements. Throughput in Mop/s (higher is better).

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | add | 12,655 | 14,711 | 0.9x |
| int8 | sub | 15,794 | 14,524 | 1.1x |
| int8 | mul | 16,632 | 16,937 | 1.0x |
| int8 | div | 6,023 | 1,007 | **6.0x** |
| uint8 | add | 16,252 | 16,249 | 1.0x |
| uint8 | sub | 17,100 | 15,776 | 1.1x |
| uint8 | mul | 18,202 | 17,549 | 1.0x |
| uint8 | div | 6,044 | 1,050 | **5.8x** |
| int16 | add | 8,814 | 8,292 | 1.1x |
| int16 | sub | 8,737 | 8,349 | 1.0x |
| int16 | mul | 8,265 | 7,814 | 1.1x |
| int16 | div | 5,972 | 1,142 | **5.2x** |
| uint16 | add | 8,862 | 8,367 | 1.1x |
| uint16 | sub | 9,104 | 7,878 | 1.2x |
| uint16 | mul | 8,658 | 8,266 | 1.0x |
| uint16 | div | 6,032 | 1,153 | **5.2x** |
| int32 | add | 9,516 | 3,875 | **2.5x** |
| int32 | sub | 10,916 | 3,806 | **2.9x** |
| int32 | mul | 10,734 | 3,757 | **2.9x** |
| int32 | div | 4,642 | 1,029 | **4.5x** |
| uint32 | add | 9,535 | 4,217 | **2.3x** |
| uint32 | sub | 10,013 | 4,331 | **2.3x** |
| uint32 | mul | 9,540 | 1,950 | **4.9x** |
| uint32 | div | 5,455 | 439 | **12.4x** |
| int64 | add | 5,909 | 1,100 | **5.4x** |
| int64 | sub | 6,320 | 1,202 | **5.3x** |
| int64 | mul | 4,934 | 1,230 | **4.0x** |
| int64 | div | 2,286 | 618 | **3.7x** |
| uint64 | add | 4,913 | 1,100 | **4.5x** |
| uint64 | sub | 5,389 | 1,088 | **5.0x** |
| uint64 | mul | 5,740 | 1,033 | **5.6x** |
| uint64 | div | 2,865 | 584 | **4.9x** |
| float32 | add | 9,914 | 3,730 | **2.7x** |
| float32 | sub | 10,282 | 3,787 | **2.7x** |
| float32 | mul | 10,372 | 3,586 | **2.9x** |
| float32 | div | 9,577 | 3,833 | **2.5x** |
| float64 | add | 5,688 | 1,087 | **5.2x** |
| float64 | sub | 4,635 | 1,120 | **4.1x** |
| float64 | mul | 3,181 | 1,124 | **2.8x** |
| float64 | div | 2,324 | 1,147 | **2.0x** |

**Summary:** numc matches NumPy on 8-bit/16-bit add/sub/mul (both are memory-bandwidth bound at ~17 Gop/s). numc is 2-6x faster on 32-bit and 64-bit types due to zero Python/ufunc dispatch overhead. Integer division is 5-12x faster.

---

## Scalar Element-wise (a + scalar, a * scalar, ...)

1M contiguous elements.

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | add | 16,577 | 22,931 | 0.7x |
| int8 | mul | 25,883 | 22,551 | 1.1x |
| uint8 | add | 25,390 | 25,424 | 1.0x |
| uint8 | mul | 27,077 | 24,396 | 1.1x |
| int32 | add | 13,675 | 5,570 | **2.5x** |
| int32 | mul | 14,224 | 5,373 | **2.6x** |
| int32 | div | 5,393 | 1,345 | **4.0x** |
| int64 | add | 7,288 | 2,422 | **3.0x** |
| int64 | mul | 6,961 | 2,370 | **2.9x** |
| float32 | add | 14,115 | 5,599 | **2.5x** |
| float32 | mul | 14,419 | 5,848 | **2.5x** |
| float32 | div | 14,066 | 5,833 | **2.4x** |
| float64 | add | 7,905 | 2,680 | **2.9x** |
| float64 | mul | 8,742 | 2,619 | **3.3x** |

**Inplace scalar** (a += scalar):

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | add | 54,418 | 42,627 | 1.3x |
| int8 | mul | 45,635 | 41,339 | 1.1x |
| int32 | add | 25,673 | 8,711 | **2.9x** |
| int32 | mul | 25,886 | 8,165 | **3.2x** |
| float32 | add | 28,596 | 9,156 | **3.1x** |
| float32 | mul | 28,194 | 9,114 | **3.1x** |
| float64 | add | 12,455 | 4,532 | **2.7x** |

---

## Unary Operations (log, exp, abs, sqrt)

1M contiguous elements.

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | log | 1,729 | 158 | **10.9x** |
| int8 | exp | 2,165 | 169 | **12.8x** |
| int8 | abs | 27,894 | 23,846 | 1.2x |
| int8 | sqrt | 4,876 | 252 | **19.4x** |
| int16 | log | 1,806 | 778 | **2.3x** |
| int16 | exp | 2,289 | 1,098 | **2.1x** |
| int16 | sqrt | 4,950 | 3,516 | 1.4x |
| int32 | log | 2,061 | 311 | **6.6x** |
| int32 | exp | 1,734 | 280 | **6.2x** |
| int32 | abs | 13,575 | 5,532 | **2.5x** |
| int32 | sqrt | 3,656 | 989 | **3.7x** |
| int64 | log | 632 | 288 | **2.2x** |
| int64 | exp | 595 | 261 | **2.3x** |
| int64 | abs | 6,417 | 2,439 | **2.6x** |
| int64 | sqrt | 1,547 | 823 | **1.9x** |
| float32 | log | 5,443 | 868 | **6.3x** |
| float32 | exp | 6,681 | 1,218 | **5.5x** |
| float32 | abs | 13,830 | 5,819 | **2.4x** |
| float32 | sqrt | 3,386 | 4,983 | 0.7x |
| float64 | log | 1,223 | 334 | **3.7x** |
| float64 | exp | 985 | 299 | **3.3x** |
| float64 | abs | 7,364 | 2,690 | **2.7x** |
| float64 | sqrt | 1,472 | 1,229 | 1.2x |

**Summary:** numc custom log/exp helpers are 2-13x faster than NumPy across all types. abs is 2-3x faster for 32-bit+. sqrt is comparable (both use hardware vsqrtps/vsqrtpd).

---

## Pow

1M contiguous elements.

| dtype | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---:|---:|---:|
| int8 | 1,872 | 687 | **2.7x** |
| uint8 | 1,714 | 706 | **2.4x** |
| int16 | 1,030 | 686 | 1.5x |
| uint16 | 1,014 | 753 | 1.3x |
| int32 | 1,969 | 659 | **3.0x** |
| uint32 | 2,128 | 720 | **3.0x** |
| int64 | 2,074 | 613 | **3.4x** |
| uint64 | 1,994 | 671 | **3.0x** |
| float32 | 2,328 | 207 | **11.2x** |
| float64 | 588 | 101 | **5.8x** |

**Summary:** numc uses branchless fixed-iteration for 8/16-bit, exponentiation-by-squaring for 32/64-bit integers, and fused `exp(b * log(a))` for floats. 2-11x faster across the board.

---

## Reductions

### Full Reduction (1M elements -> scalar)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 37,895 | 3,934 | **9.6x** |
| int8 | mean | 96,554 | 2,791 | **34.6x** |
| int8 | max | 97,317 | 71,095 | 1.4x |
| int8 | min | 99,695 | 74,278 | 1.3x |
| int8 | argmax | 98,848 | 42,008 | **2.4x** |
| int8 | argmin | 99,333 | 43,478 | **2.3x** |
| int16 | sum | 11,900 | 4,009 | **3.0x** |
| int16 | max | 33,483 | 30,026 | 1.1x |
| int16 | argmax | 30,164 | 22,722 | 1.3x |
| int32 | sum | 16,801 | 3,897 | **4.3x** |
| int32 | mean | 15,303 | 2,833 | **5.4x** |
| int32 | max | 26,877 | 13,186 | **2.0x** |
| int32 | min | 31,501 | 13,059 | **2.4x** |
| int32 | argmax | 15,103 | 11,065 | 1.4x |
| int32 | argmin | 14,533 | 11,496 | 1.3x |
| int64 | sum | 17,852 | 5,937 | **3.0x** |
| int64 | max | 9,189 | 6,123 | 1.5x |
| int64 | argmax | 6,605 | 3,456 | **1.9x** |
| float32 | sum | 14,096 | 6,147 | **2.3x** |
| float32 | mean | 16,542 | 6,107 | **2.7x** |
| float32 | max | 19,435 | 10,859 | **1.8x** |
| float32 | min | 18,392 | 10,774 | **1.7x** |
| float32 | argmax | 7,528 | 8,548 | 0.9x |
| float32 | argmin | 7,540 | 8,447 | 0.9x |
| float64 | sum | 10,913 | 5,026 | **2.2x** |
| float64 | mean | 13,113 | 5,147 | **2.5x** |
| float64 | max | 11,133 | 5,161 | **2.2x** |
| float64 | min | 13,843 | 5,210 | **2.7x** |
| float64 | argmax | 7,230 | 4,434 | **1.6x** |
| float64 | argmin | 7,028 | 4,316 | **1.6x** |

### Axis=0 Reduction (1000x1000 -> 1000 cols)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 36,851 | 2,772 | **13.3x** |
| int8 | mean | 49,590 | 2,224 | **22.3x** |
| int8 | max | 41,104 | 20,990 | **2.0x** |
| int8 | min | 27,386 | 21,001 | 1.3x |
| int32 | sum | 12,990 | 2,800 | **4.6x** |
| int32 | max | 12,885 | 10,339 | 1.2x |
| float32 | sum | 14,822 | 9,200 | **1.6x** |
| float32 | mean | 14,042 | 9,112 | **1.5x** |
| float32 | max | 14,559 | 7,513 | **1.9x** |
| float32 | min | 10,750 | 7,566 | 1.4x |
| float64 | sum | 7,377 | 4,640 | **1.6x** |
| float64 | max | 7,070 | 3,918 | **1.8x** |

### Axis=1 Reduction (1000x1000 -> 1000 rows)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 46,629 | 3,775 | **12.3x** |
| int8 | mean | 45,350 | 2,355 | **19.3x** |
| int8 | max | 47,152 | 22,971 | **2.1x** |
| int8 | argmax | 50,142 | 8,287 | **6.1x** |
| int32 | sum | 15,187 | 3,513 | **4.3x** |
| int32 | max | 14,494 | 10,831 | 1.3x |
| int32 | argmax | 13,052 | 10,385 | 1.3x |
| float32 | sum | 6,819 | 5,832 | 1.2x |
| float32 | max | 12,318 | 6,465 | **1.9x** |
| float32 | argmax | 12,744 | 6,940 | **1.8x** |
| float64 | sum | 5,369 | 5,109 | 1.1x |
| float64 | max | 7,689 | 4,276 | **1.8x** |
| float64 | argmax | 7,620 | 4,252 | **1.8x** |

---

## Size Scaling (float32 sum)

| elements | numc (us) | numpy (us) | numc GB/s | numpy GB/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.02 | 0.76 | 26.47 | 0.53 | **38.0x** |
| 1,000 | 0.14 | 0.95 | 28.52 | 4.22 | **6.8x** |
| 10,000 | 1.43 | 2.52 | 27.89 | 15.85 | **1.8x** |
| 100,000 | 14.18 | 17.70 | 28.22 | 22.60 | 1.2x |
| 1,000,000 | 103.33 | 159.51 | 38.71 | 25.08 | **1.5x** |

**Summary:** numc has near-zero dispatch overhead. At 100 elements, numc is 38x faster due to NumPy's Python/ufunc dispatch cost (~0.76us baseline). At 1M elements both approach memory bandwidth limits, with numc still 1.5x ahead.

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
