# Element-wise Binary Ops

All binary ops apply an operation **element by element** between two arrays of the same shape and dtype, writing the result to `out`.

## Signature pattern

```c
int numc_xxx(const NumcArray *a, const NumcArray *b, NumcArray *out);
```

- `a`, `b`, `out` must have the **same shape** and **same dtype**.
- `out` may alias `a` or `b` (safe for in-place use).
- Works on contiguous **and** non-contiguous arrays.
- Returns `0` on success, negative error code on failure.

## Arithmetic

### `numc_add`

```c
int numc_add(const NumcArray *a, const NumcArray *b, NumcArray *out);
// out[i] = a[i] + b[i]
```

### `numc_sub`

```c
int numc_sub(const NumcArray *a, const NumcArray *b, NumcArray *out);
// out[i] = a[i] - b[i]
```

### `numc_mul`

```c
int numc_mul(const NumcArray *a, const NumcArray *b, NumcArray *out);
// out[i] = a[i] * b[i]
```

### `numc_div`

```c
int numc_div(const NumcArray *a, const NumcArray *b, NumcArray *out);
// out[i] = a[i] / b[i]
```

### `numc_pow` / `numc_pow_inplace`

```c
int numc_pow(NumcArray *a, NumcArray *b, NumcArray *out);
int numc_pow_inplace(NumcArray *a, NumcArray *b);
// out[i] = a[i] ^ b[i]
```

- **Float types:** implemented as `exp(b * log(a))` using the custom `log`/`exp` kernels.
- **Integer types:** branchless fixed-iteration loop (8/16-bit auto-vectorizes; 32/64-bit uses early-exit).

## Elementwise extrema

### `numc_maximum` / `numc_maximum_inplace`

```c
int numc_maximum(const NumcArray *a, const NumcArray *b, NumcArray *out);
int numc_maximum_inplace(NumcArray *a, const NumcArray *b);
// out[i] = a[i] > b[i] ? a[i] : b[i]
```

### `numc_minimum` / `numc_minimum_inplace`

```c
int numc_minimum(const NumcArray *a, const NumcArray *b, NumcArray *out);
int numc_minimum_inplace(NumcArray *a, const NumcArray *b);
// out[i] = a[i] < b[i] ? a[i] : b[i]
```

## Example

```c
size_t shape[] = {2, 3};
NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
NumcArray *b   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

float da[] = {10, 20, 30, 40, 50, 60};
float db[] = { 1,  2,  3,  4,  5,  6};
numc_array_write(a, da);
numc_array_write(b, db);

numc_add(a, b, out);      // [[11, 22, 33], [44, 55, 66]]
numc_sub(a, b, out);      // [[ 9, 18, 27], [36, 45, 54]]
numc_mul(a, b, out);      // [[10, 40, 90], [160, 250, 360]]
numc_div(a, b, out);      // [[10, 10, 10], [10, 10, 10]]
numc_maximum(a, b, out);  // [[10, 20, 30], [40, 50, 60]]
numc_minimum(a, b, out);  // [[ 1,  2,  3], [ 4,  5,  6]]
```

## In-place aliasing

`out` can be the same pointer as `a` (or `b`) for a safe in-place result:

```c
numc_add(a, b, a);   // a = a + b (valid)
```

## Error conditions

| Error | Cause |
|---|---|
| `NUMC_ERR_NULL` | Any of `a`, `b`, `out` is `NULL` |
| `NUMC_ERR_SHAPE` | Shapes don't match |
| `NUMC_ERR_TYPE` | Dtypes don't match |
