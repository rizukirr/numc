# Array Creation

All creation functions allocate from the context's arena and return `NULL` on failure (e.g., out of arena memory).

## `numc_array_create` — uninitialized

```c
NumcArray *numc_array_create(NumcCtx *ctx,
                              const size_t *shape,
                              size_t ndim,
                              NumcDType dtype);
```

Allocates an array with the given shape. Memory is **not initialized** — write data before reading.

```c
size_t shape[] = {2, 3};
NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
// 2×3 float32, contents undefined
```

Use this when you will immediately overwrite the data (e.g., `numc_array_write`, or as an `out` buffer for a math op).

## `numc_array_zeros` — zero-initialized

```c
NumcArray *numc_array_zeros(NumcCtx *ctx,
                             const size_t *shape,
                             size_t ndim,
                             NumcDType dtype);
```

Same as `numc_array_create` but initializes all bytes to zero.

```c
NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
// safe to use immediately as an accumulator
```

## `numc_array_fill` — constant value

```c
NumcArray *numc_array_fill(NumcCtx *ctx,
                            const size_t *shape,
                            size_t ndim,
                            NumcDType dtype,
                            const void *value);
```

Creates an array where every element equals `*value`. The pointer must point to a scalar whose C type matches `dtype`.

```c
float v = 1.0f;
NumcArray *ones = numc_array_fill(ctx, shape, 2, NUMC_DTYPE_FLOAT32, &v);

double d = 3.14;
NumcArray *pi = numc_array_fill(ctx, shape, 2, NUMC_DTYPE_FLOAT64, &d);

int32_t i = -1;
NumcArray *neg1 = numc_array_fill(ctx, shape, 2, NUMC_DTYPE_INT32, &i);
```

## `numc_array_copy` — deep copy

```c
NumcArray *numc_array_copy(const NumcArray *arr);
```

Deep-copies `arr` into the **same context** as `arr`. The copy is always contiguous, even if the source is not.

```c
NumcArray *copy = numc_array_copy(original);
// copy is independent — modifying copy does not affect original
```

## `numc_array_write` — fill from C array

```c
void numc_array_write(NumcArray *arr, const void *data);
```

Copies raw bytes from `data` into the array's buffer. The source must be a C array of the matching type in **row-major (C) order**, with exactly `numc_array_size(arr)` elements.

```c
// 1D
size_t shape[] = {6};
NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
int32_t data[] = {1, 2, 3, 4, 5, 6};
numc_array_write(a, data);

// 2D — row-major
size_t shape2[] = {2, 3};
NumcArray *b = numc_array_create(ctx, shape2, 2, NUMC_DTYPE_INT32);
int32_t data2[] = {1, 2, 3,   // row 0
                   4, 5, 6};  // row 1
numc_array_write(b, data2);

// 3D — nested C array
size_t shape3[] = {2, 2, 4};
NumcArray *c = numc_array_create(ctx, shape3, 3, NUMC_DTYPE_INT32);
int32_t data3[2][2][4] = {
    {{1, 2, 3, 4}, {5, 6, 7, 8}},
    {{9, 10, 11, 12}, {13, 14, 15, 16}},
};
numc_array_write(c, data3);
```
