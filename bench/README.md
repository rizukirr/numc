# numc vs NumPy Benchmark

All benchmarks run on contiguous 1M-element arrays, 200 iterations, best-of-warmup.

**System:** Intel i7-13620H (16 threads) | Clang 21.1.8 + OpenMP | NumPy 2.4.2 | Linux 6.12.73-1-lts

---

## Binary Element-wise (add, sub, mul, div)

1M contiguous elements. Throughput in Mop/s (higher is better).

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | add | 15,875 | 14,545 | 1.1x |
| int8 | sub | 17,100 | 13,699 | 1.2x |
| int8 | mul | 18,418 | 13,736 | 1.3x |
| int8 | div | 6,050 | 1,004 | **6.0x** |
| uint8 | add | 18,244 | 16,217 | 1.1x |
| uint8 | sub | 19,041 | 17,424 | 1.1x |
| uint8 | mul | 19,092 | 15,761 | 1.2x |
| uint8 | div | 6,053 | 1,049 | **5.8x** |
| int16 | add | 9,103 | 8,736 | 1.0x |
| int16 | sub | 8,933 | 9,064 | 1.0x |
| int16 | mul | 8,531 | 8,706 | 1.0x |
| int16 | div | 5,970 | 1,146 | **5.2x** |
| uint16 | add | 8,725 | 8,098 | 1.1x |
| uint16 | sub | 8,961 | 8,909 | 1.0x |
| uint16 | mul | 8,838 | 8,589 | 1.0x |
| uint16 | div | 6,006 | 1,161 | **5.2x** |
| int32 | add | 8,771 | 3,863 | **2.3x** |
| int32 | sub | 10,454 | 3,948 | **2.6x** |
| int32 | mul | 10,338 | 3,881 | **2.7x** |
| int32 | div | 5,442 | 1,010 | **5.4x** |
| uint32 | add | 9,734 | 4,304 | **2.3x** |
| uint32 | sub | 10,252 | 4,272 | **2.4x** |
| uint32 | mul | 10,269 | 4,209 | **2.4x** |
| uint32 | div | 5,245 | 867 | **6.0x** |
| int64 | add | 3,497 | 922 | **3.8x** |
| int64 | sub | 4,946 | 1,107 | **4.5x** |
| int64 | mul | 3,802 | 1,266 | **3.0x** |
| int64 | div | 2,793 | 648 | **4.3x** |
| uint64 | add | 4,070 | 1,097 | **3.7x** |
| uint64 | sub | 4,841 | 826 | **5.9x** |
| uint64 | mul | 5,248 | 1,192 | **4.4x** |
| uint64 | div | 2,796 | 585 | **4.8x** |
| float32 | add | 10,160 | 3,703 | **2.7x** |
| float32 | sub | 10,223 | 3,610 | **2.8x** |
| float32 | mul | 9,976 | 3,648 | **2.7x** |
| float32 | div | 10,147 | 3,694 | **2.7x** |
| float64 | add | 4,499 | 1,142 | **3.9x** |
| float64 | sub | 2,398 | 1,109 | **2.2x** |
| float64 | mul | 2,542 | 1,143 | **2.2x** |
| float64 | div | 1,476 | 1,163 | 1.3x |

**Summary:** numc matches NumPy on 8-bit/16-bit add/sub/mul (both are memory-bandwidth bound at ~17 Gop/s). numc is 2-6x faster on 32-bit and 64-bit types due to zero Python/ufunc dispatch overhead. Integer division is 4-6x faster.

---

## Broadcasting

1M output elements (1000x1000). Three broadcast patterns tested: row `(1,N)+(M,N)`, outer `(M,1)+(1,N)`, rank `(N,)+(M,N)`.

### Row Broadcast: (1,1000) + (1000,1000) → (1000,1000)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int32 | add | 5,533 | 3,897 | **1.4x** |
| int32 | sub | 5,625 | 4,039 | **1.4x** |
| int32 | mul | 5,798 | 3,930 | **1.5x** |
| int32 | div | 1,881 | 1,143 | **1.6x** |
| float32 | add | 5,846 | 3,334 | **1.8x** |
| float32 | sub | 5,818 | 3,121 | **1.9x** |
| float32 | mul | 5,524 | 3,154 | **1.8x** |
| float32 | div | 5,676 | 3,175 | **1.8x** |
| float64 | add | 2,586 | 1,404 | **1.8x** |
| float64 | sub | 2,771 | 1,436 | **1.9x** |
| float64 | mul | 2,474 | 1,542 | **1.6x** |
| float64 | div | 1,805 | 1,514 | 1.2x |

### Outer Broadcast: (1000,1) + (1,1000) → (1000,1000)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int32 | add | 2,901 | 2,921 | 1.0x |
| int32 | sub | 2,923 | 2,925 | 1.0x |
| int32 | mul | 2,934 | 2,911 | 1.0x |
| int32 | div | 946 | 942 | 1.0x |
| float32 | add | 2,723 | 2,421 | 1.1x |
| float32 | sub | 2,745 | 2,424 | 1.1x |
| float32 | mul | 2,722 | 2,436 | 1.1x |
| float32 | div | 1,261 | 2,418 | 0.5x |
| float64 | add | 2,693 | 1,200 | **2.2x** |
| float64 | sub | 2,640 | 1,203 | **2.2x** |
| float64 | mul | 2,640 | 1,207 | **2.2x** |
| float64 | div | 945 | 1,122 | 0.8x |

### Rank Broadcast: (1000,) + (1000,1000) → (1000,1000)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int32 | add | 5,846 | 3,909 | **1.5x** |
| int32 | sub | 5,456 | 3,615 | **1.5x** |
| int32 | mul | 5,626 | 3,635 | **1.5x** |
| int32 | div | 1,882 | 1,176 | **1.6x** |
| float32 | add | 5,949 | 3,230 | **1.8x** |
| float32 | sub | 5,799 | 3,098 | **1.9x** |
| float32 | mul | 5,715 | 3,173 | **1.8x** |
| float32 | div | 5,685 | 3,213 | **1.8x** |
| float64 | add | 2,322 | 1,486 | **1.6x** |
| float64 | sub | 2,613 | 1,534 | **1.7x** |
| float64 | mul | 2,399 | 1,636 | **1.5x** |
| float64 | div | 1,774 | 1,494 | 1.2x |

**Summary:** Row and rank broadcast are 1.4-1.9x faster than NumPy — one input reads linearly while the other repeats via stride=0, giving good cache locality. Outer broadcast (both dims broadcast) is roughly tied on 32-bit types since both libraries hit the same cache-unfriendly access pattern. numc pulls ahead on float64 outer broadcast (2.2x) due to lower dispatch overhead.

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
| int8 | sum | 92,320 | 2,341 | **39.4x** |
| int8 | mean | 86,945 | 2,338 | **37.2x** |
| int8 | max | 94,314 | 76,818 | 1.2x |
| int8 | min | 94,303 | 72,631 | 1.3x |
| int8 | argmax | 95,718 | 38,383 | **2.5x** |
| int8 | argmin | 93,372 | 38,738 | **2.4x** |
| int16 | sum | 26,939 | 2,197 | **12.3x** |
| int16 | max | 25,505 | 26,091 | 1.0x |
| int16 | argmax | 29,507 | 21,000 | 1.4x |
| int32 | sum | 18,534 | 3,639 | **5.1x** |
| int32 | mean | 29,820 | 2,763 | **10.8x** |
| int32 | max | 29,754 | 13,407 | **2.2x** |
| int32 | min | 31,016 | 15,056 | **2.1x** |
| int32 | argmax | 14,128 | 10,794 | 1.3x |
| int32 | argmin | 12,384 | 10,871 | 1.1x |
| int64 | sum | 12,876 | 5,832 | **2.2x** |
| int64 | max | 11,025 | 5,852 | **1.9x** |
| int64 | argmax | 6,400 | 4,146 | **1.5x** |
| float32 | sum | 14,022 | 6,189 | **2.3x** |
| float32 | mean | 17,362 | 6,123 | **2.8x** |
| float32 | max | 18,920 | 10,851 | **1.7x** |
| float32 | min | 19,685 | 11,002 | **1.8x** |
| float32 | argmax | 7,551 | 8,760 | 0.9x |
| float32 | argmin | 7,552 | 8,754 | 0.9x |
| float64 | sum | 12,688 | 4,885 | **2.6x** |
| float64 | mean | 13,467 | 4,871 | **2.8x** |
| float64 | max | 14,124 | 5,083 | **2.8x** |
| float64 | min | 14,503 | 5,126 | **2.8x** |
| float64 | argmax | 6,845 | 4,268 | **1.6x** |
| float64 | argmin | 6,651 | 4,207 | **1.6x** |

### Axis=0 Reduction (1000x1000 -> 1000 cols)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 41,731 | 2,262 | **18.4x** |
| int8 | mean | 47,566 | 2,585 | **18.4x** |
| int8 | max | 42,862 | 19,890 | **2.2x** |
| int8 | min | 43,102 | 20,996 | **2.1x** |
| int32 | sum | 12,101 | 2,695 | **4.5x** |
| int32 | max | 13,100 | 8,558 | **1.5x** |
| float32 | sum | 12,554 | 10,498 | 1.2x |
| float32 | mean | 12,939 | 10,040 | 1.3x |
| float32 | max | 12,858 | 7,917 | **1.6x** |
| float32 | min | 13,544 | 7,959 | **1.7x** |
| float64 | sum | 6,605 | 4,845 | 1.4x |
| float64 | max | 6,081 | 3,931 | **1.5x** |

### Axis=1 Reduction (1000x1000 -> 1000 rows)

| dtype | op | numc (Mop/s) | numpy (Mop/s) | speedup |
|---|---|---:|---:|---:|
| int8 | sum | 47,301 | 2,647 | **17.9x** |
| int8 | mean | 44,621 | 2,551 | **17.5x** |
| int8 | max | 46,069 | 21,969 | **2.1x** |
| int8 | argmax | 49,708 | 8,666 | **5.7x** |
| int32 | sum | 13,487 | 3,714 | **3.6x** |
| int32 | max | 13,668 | 10,176 | 1.3x |
| int32 | argmax | 13,054 | 9,617 | 1.4x |
| float32 | sum | 6,839 | 5,784 | 1.2x |
| float32 | max | 12,317 | 6,540 | **1.9x** |
| float32 | argmax | 13,124 | 7,857 | **1.7x** |
| float64 | sum | 5,749 | 4,794 | 1.2x |
| float64 | max | 7,618 | 4,179 | **1.8x** |
| float64 | argmax | 7,522 | 4,107 | **1.8x** |

---

## Size Scaling (float32 sum)

| elements | numc (us) | numpy (us) | numc GB/s | numpy GB/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.01 | 0.76 | 26.85 | 0.53 | **76.0x** |
| 1,000 | 0.14 | 0.91 | 28.60 | 4.39 | **6.5x** |
| 10,000 | 1.43 | 2.47 | 27.89 | 16.23 | **1.7x** |
| 100,000 | 14.12 | 17.84 | 28.34 | 22.43 | 1.3x |
| 1,000,000 | 58.26 | 160.95 | 68.66 | 24.85 | **2.8x** |

**Summary:** numc has near-zero dispatch overhead. At 100 elements, numc is 76x faster due to NumPy's Python/ufunc dispatch cost (~0.76us baseline). At 1M elements numc is 2.8x faster, reaching 68.66 GB/s bandwidth.

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
