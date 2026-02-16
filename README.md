# numc

A fast, NumPy-inspired tensor library written in C23.

numc aims to rewrite the core of NumPy from scratch — same semantics (N-dimensional arrays, broadcasting, type-generic ops), but leaner and faster by cutting through Python overhead and ufunc dispatch. It also serves as the tensor backend for [ctorch](https://github.com/rizukirr/ctorch).

**Highlights:**
- 10 numeric dtypes (`int8` through `float64`) via X-macro code generation
- Arena-allocated memory — create a context, allocate tensors, free everything at once
- Auto-vectorized kernels (`-O3 -march=native`) — no hand-written SIMD intrinsics
- Matches or beats NumPy on element-wise ops (2-3x faster on float32/int32)

---

## Feature Checklist

### 1. Memory

- [x] Arena allocator (bump-pointer, block-based)
- [x] Checkpoint / restore (scratch memory for temporaries)
- [x] Aligned alloc helper (portable `aligned_alloc` wrapper)

### 2. Tensor Core

- [x] `array_create(ctx, shape, dim, dtype)` — allocate tensor
- [x] `array_zeros(ctx, shape, dim, dtype)`
- [x] `array_fill_with(ctx, shape, dim, dtype, value)`
- [x] `array_write_data(arr, data)` — import raw data
- [x] `array_print(arr)`
- [x] `array_free(ctx)` — free context + all tensors
- [x] `array_copy(arr)` — deep copy
- [x] `array_reshape_inplace(arr, new_shape, new_dim)`
- [x] `array_reshape_copy(arr, new_shape, new_dim)`
- [x] `array_transpose_inplace(arr)` — stride swap in-place
- [x] `array_transpose_copy(arr)` — stride swap return new array
- [x] `array_is_contiguous(arr)`
- [x] `array_as_contiguous(arr)` — copy to contiguous layout
- [x] `array_slice(arr, axis, start, end)` — view into sub-range

### 3. Element-wise Math

All support broadcasting. Both allocating and in-place variants.

- [x] `array_add(a, b)` / `array_sub(a, b)` / `array_mul(a, b)` / `array_div(a, b)`
- [x] `array_add_scalar(a, s)` / `array_sub_scalar(a, s)` / `array_mul_scalar(a, s)` / `array_div_scalar(a, s)`
- [x] `array_neg(a)` / `array_neg_inplace(a)`
- [ ] `array_exp(a)` / `array_log(a)` / `array_sqrt(a)` / `array_abs(a)`
- [ ] `array_pow(a, exp)`
- [ ] `array_clamp(a, min, max)`
- [ ] `array_maximum(a, b)` / `array_minimum(a, b)` — element-wise

In-place variants (`_inplace` suffix) for optimizer weight updates:

- [x] `array_add_inplace(a, b)` / `array_sub_inplace(a, b)` / `array_mul_inplace(a, b)` / `array_div_inplace(a, b)`
- [x] `array_mul_scalar_inplace(a, s)` / `array_add_scalar_inplace(a, s)`

### 4. Reductions

- [ ] `array_sum(a)` / `array_sum_axis(a, axis, keepdim)`
- [ ] `array_mean(a)` / `array_mean_axis(a, axis, keepdim)`
- [ ] `array_max(a)` / `array_max_axis(a, axis, keepdim)`
- [ ] `array_min(a)` / `array_min_axis(a, axis, keepdim)`
- [ ] `array_argmax(a, axis)` / `array_argmin(a, axis)`

### 5. Broadcasting

- [ ] Shape compatibility check
- [ ] Broadcast shape computation
- [ ] Virtual expansion (stride=0 for broadcast dims)
- [ ] Works with all binary ops

### 6. Matrix Operations

- [ ] `array_matmul(a, b)` — (M,K) @ (K,N) -> (M,N)

### 7. Comparison / Selection

- [ ] `array_eq(a, b)` / `array_gt(a, b)` / `array_lt(a, b)` / `array_ge(a, b)` / `array_le(a, b)`
- [ ] `array_where(cond, a, b)` — ternary select

### 8. Random

- [ ] Seedable PRNG (xoshiro256**)
- [ ] `numc_manual_seed(seed)`
- [ ] `array_rand(ctx, shape, dim, dtype)` — uniform [0, 1)
- [ ] `array_randn(ctx, shape, dim, dtype)` — normal N(0,1) via Box-Muller

---

## Implementation Phases

### Phase 1 — Tensor core ✓
Create, zeros, fill, clone, free, print, reshape, transpose, slice, contiguous.

### Phase 2 — Element-wise math (in progress)
add, sub, mul, div, neg, exp, log, sqrt, abs, maximum, minimum, clamp.
Scalar variants. In-place variants.

### Phase 3 — Reductions + broadcasting
sum, mean, max, min, argmax (full + axis + keepdim).
Broadcasting for all binary ops.

### Phase 4 — Matrix multiplication
Naive matmul first. BLAS (`cblas_sgemm`) later for performance.

### Phase 5 — Comparison + selection
gt, lt, eq, where. Needed for relu backward.

### Phase 6 — Random
Seedable PRNG, randn (Box-Muller), rand. Needed for weight initialization.

### Phase 7 — Performance
`-O3 -march=native` auto-vectorization. OpenMP for element-wise ops.
Tiled matmul or BLAS backend for large matrices.

---

## Benchmarks

Median of 4 runs, 1M elements, 200 iterations each. Clang 21, `-O3 -march=native`, NumPy 2.4.2.

**Contiguous binary `add`:**

| dtype   | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|---------|-------------|---------------|---------|
| int8    | 11,718      | 15,844        | 0.7x    |
| int32   | 9,766       | 4,117         | 2.4x    |
| int64   | 3,949       | 1,042         | 3.8x    |
| float32 | 10,395      | 3,517         | 3.0x    |
| float64 | 3,975       | 1,234         | 3.2x    |

**Scalar `add`:**

| dtype   | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|---------|-------------|---------------|---------|
| int8    | 25,577      | 23,161        | 1.1x    |
| int32   | 12,523      | 5,787         | 2.2x    |
| int64   | 6,177       | 2,494         | 2.5x    |
| float32 | 12,438      | 5,824         | 2.1x    |
| float64 | 7,950       | 2,637         | 3.0x    |

**Scalar inplace `add`:**

| dtype   | numc (Mop/s) | NumPy (Mop/s) | Speedup |
|---------|-------------|---------------|---------|
| int8    | 56,348      | 54,888        | 1.0x    |
| int32   | 24,450      | 9,111         | 2.7x    |
| float32 | 25,130      | 9,088         | 2.8x    |
| float64 | 12,610      | 4,512         | 2.8x    |

numc is 2-3x faster than NumPy on 32/64-bit types. The gap comes from eliminating Python/ufunc dispatch overhead — both libraries emit near-identical SIMD. int8 is bandwidth-bound, so both saturate DRAM equally.

```bash
./run.sh bench                          # Run all benchmarks
python bench/bench_numpy.py             # NumPy comparison
```

## Build

```bash
./run.sh debug          # Build debug (ASan) + run
./run.sh release        # Build release (-O3 -march=native) + run
./run.sh test           # Build + run tests
./run.sh clean          # Remove build/
CC=gcc ./run.sh release # Use GCC instead of Clang
```

## License

MIT
