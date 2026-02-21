# numc

A fast, NumPy-inspired N-dimensional array library written in C23.

numc provides the same core abstractions as NumPy but more more faster — N-dimensional arrays, type-generic operations, broadcasting, and strided views — implemented in pure C with zero dependencies. It's designed for anyone who needs efficient tensor operations from C: scientific computing, data processing, machine learning backends, game engines, or embedded systems.

**Highlights:**
- 10 numeric dtypes (`int8` through `float64`) via X-macro code generation
- Arena-allocated memory — create a context, allocate tensors, free everything at once
- Auto-vectorized kernels (`-O3 -march=native`) — no hand-written SIMD intrinsics
- Strided views, slices, reshapes, and transposes without copying data
- OpenMP parallelism for large arrays (>1 MB)

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
- [x] `array_exp(a)` / `array_log(a)` / `array_sqrt(a)` / `array_abs(a)`
- [x] `array_exp_inplace(a)` / `array_log_inplace(a)` / `array_sqrt_inplace(a)` / `array_abs_inplace(a)`
- [x] `array_pow(a, exp)`
- [x] `array_clip(a, min, max)`
- [x] `array_maximum(a, b)` / `array_minimum(a, b)` — element-wise

In-place variants (`_inplace` suffix) for optimizer weight updates:

- [x] `array_add_inplace(a, b)` / `array_sub_inplace(a, b)` / `array_mul_inplace(a, b)` / `array_div_inplace(a, b)`
- [x] `array_mul_scalar_inplace(a, s)` / `array_add_scalar_inplace(a, s)`

### 4. Reductions

- [x] `array_sum(a)` / `array_sum_axis(a, axis, keepdim)`
- [x] `array_mean(a)` / `array_mean_axis(a, axis, keepdim)`
- [x] `array_max(a)` / `array_max_axis(a, axis, keepdim)`
- [x] `array_min(a)` / `array_min_axis(a, axis, keepdim)`
- [x] `array_argmax(a, axis)` / `array_argmin(a, axis)`

### 5. Broadcasting

- [x] Shape compatibility check
- [x] Broadcast shape computation
- [x] Virtual expansion (stride=0 for broadcast dims)
- [x] Works with all binary ops (add, sub, mul, div, pow, maximum, minimum)

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

### Phase 2 — Element-wise math ✓
add, sub, mul, div, neg, exp, log, sqrt, abs, pow, maximum, minimum, clamp.
Scalar variants. In-place variants.

### Phase 3 — Reductions + broadcasting ✓
sum, mean, max, min, argmax, argmin (full + axis + keepdim).
NumPy-style broadcasting for all binary ops.

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

## Documentation

Full API reference is available on the [wiki](https://github.com/rizukirr/numc/wiki).

---

## Build

```bash
./run.sh debug          # Build debug (ASan) + run
./run.sh release        # Build release (-O3 -march=native) + run
./run.sh test           # Build + run tests
./run.sh bench          # Build + run all benchmarks
./run.sh clean          # Remove build/
CC=gcc ./run.sh release # Use GCC instead of Clang
```

## License

MIT
