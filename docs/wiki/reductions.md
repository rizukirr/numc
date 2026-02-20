# Reductions

Reduce an array along one or all axes to produce a smaller result.

## `numc_sum` — full reduction

```c
int numc_sum(const NumcArray *a, NumcArray *out);
```

Sums **all elements** of `a` and writes the result to `out`. `out` must be a 1-element array with the same dtype as `a`.

```c
size_t sshape[] = {1};
NumcArray *result = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
numc_sum(a, result);
// result[0] = a[0] + a[1] + ... + a[n-1]
```

Works on non-contiguous arrays (views, transposes).

## `numc_sum_axis` — axis reduction

```c
int numc_sum_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);
```

Reduces `a` along one axis.

| Parameter | Meaning |
|---|---|
| `axis` | Which dimension to reduce (0-indexed) |
| `keepdim` | `0` = remove that axis; `1` = keep it as size 1 |
| `out` | Pre-allocated output with the correct reduced shape |

### Shape rules for a 2×3 input

```
axis=0, keepdim=0  →  out shape: (3,)
axis=0, keepdim=1  →  out shape: (1, 3)
axis=1, keepdim=0  →  out shape: (2,)
axis=1, keepdim=1  →  out shape: (2, 1)
```

### Examples

```c
// a is 2×3: [[1, 2, 3], [4, 5, 6]]
size_t shape[] = {2, 3};
NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
float da[] = {1, 2, 3, 4, 5, 6};
numc_array_write(a, da);

// Sum rows (axis=0) → [5, 7, 9]
size_t oshape0[] = {3};
NumcArray *out0 = numc_array_zeros(ctx, oshape0, 1, NUMC_DTYPE_FLOAT32);
numc_sum_axis(a, 0, 0, out0);

// Sum cols (axis=1) → [6, 15]
size_t oshape1[] = {2};
NumcArray *out1 = numc_array_zeros(ctx, oshape1, 1, NUMC_DTYPE_FLOAT32);
numc_sum_axis(a, 1, 0, out1);

// keepdim=1: [[5, 7, 9]] (shape 1×3, useful for broadcasting)
size_t oshape_kd[] = {1, 3};
NumcArray *out_kd = numc_array_zeros(ctx, oshape_kd, 2, NUMC_DTYPE_FLOAT32);
numc_sum_axis(a, 0, 1, out_kd);
```

### 3D example

```c
// b is 2×3×4
size_t shape3d[] = {2, 3, 4};
// ... write data ...

// sum axis=1: (2, 3, 4) → (2, 4)
size_t oshape3d[] = {2, 4};
NumcArray *out3d = numc_array_zeros(ctx, oshape3d, 2, NUMC_DTYPE_INT32);
numc_sum_axis(b, 1, 0, out3d);
```

## Caller responsibility

You must allocate `out` with the correct shape before calling. numc does not allocate `out` for you.

```c
// Wrong: out has wrong shape
size_t wrong[] = {5};
NumcArray *bad = numc_array_zeros(ctx, wrong, 1, NUMC_DTYPE_FLOAT32);
numc_sum_axis(a, 0, 0, bad);  // will return an error

// Correct: out shape matches the expected reduced shape
size_t correct[] = {3};
NumcArray *good = numc_array_zeros(ctx, correct, 1, NUMC_DTYPE_FLOAT32);
numc_sum_axis(a, 0, 0, good);  // ok
```
