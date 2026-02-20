# Array Properties

Functions for inspecting an array without modifying it.

## Dimensions and size

### `numc_array_ndim`

```c
size_t numc_array_ndim(const NumcArray *arr);
```

Number of dimensions (rank). A 2×3 array has ndim `2`.

### `numc_array_size`

```c
size_t numc_array_size(const NumcArray *arr);
```

Total number of elements — product of all dimension sizes.

```c
// 2×3×4 → size = 24
```

### `numc_array_capacity`

```c
size_t numc_array_capacity(const NumcArray *arr);
```

Allocated capacity in elements. For freshly created arrays `capacity == size`. After an in-place reshape the capacity may be larger.

### `numc_array_elem_size`

```c
size_t numc_array_elem_size(const NumcArray *arr);
```

Size in bytes of one element. For `NUMC_DTYPE_FLOAT32` this is `4`.

## Type

### `numc_array_dtype`

```c
NumcDType numc_array_dtype(const NumcArray *arr);
```

Returns the dtype enum, e.g., `NUMC_DTYPE_FLOAT32`. Compare with the `NUMC_DTYPE_*` constants.

## Shape and strides

### `numc_array_shape`

```c
void numc_array_shape(const NumcArray *arr, size_t *shape);
```

Copies the shape into the caller-provided buffer. Buffer must hold at least `numc_array_ndim(arr)` elements.

```c
size_t ndim = numc_array_ndim(arr);
size_t shape[ndim];
numc_array_shape(arr, shape);
// for a 2×3 array: shape = [2, 3]
```

### `numc_array_strides`

```c
void numc_array_strides(const NumcArray *arr, size_t *strides);
```

Copies **byte-strides** into the caller-provided buffer. A stride tells you how many bytes to skip to advance one step along that dimension.

```c
size_t strides[ndim];
numc_array_strides(arr, strides);
```

For a contiguous 2×3 `float32` array:
- `strides[0] = 12` — advancing one row skips 3 floats × 4 bytes
- `strides[1] = 4`  — advancing one column skips 1 float × 4 bytes

After a transpose the strides swap, but the data does not move.

### `numc_array_is_contiguous`

```c
bool numc_array_is_contiguous(NumcArray *arr);
```

Returns `true` if the array is laid out in contiguous C (row-major) order in memory. Freshly created arrays are always contiguous. Arrays become non-contiguous after `numc_array_transpose` or `numc_array_slice`.

## Raw data pointer

### `numc_array_data`

```c
void *numc_array_data(const NumcArray *arr);
```

Returns a raw pointer to the first element of the data buffer. Cast to the appropriate type for direct access.

```c
float *ptr = (float *)numc_array_data(arr);
printf("first element: %f\n", ptr[0]);
```

> Only safe for **contiguous** arrays when iterating with pointer arithmetic. For non-contiguous arrays, use strides to compute element offsets.

## Full inspection example

```c
size_t shape[] = {2, 3, 4};
float one = 1.0f;
NumcArray *a = numc_array_fill(ctx, shape, 3, NUMC_DTYPE_FLOAT32, &one);

size_t ndim = numc_array_ndim(a);      // 3
size_t size = numc_array_size(a);      // 24
size_t esz  = numc_array_elem_size(a); // 4

size_t sh[ndim], st[ndim];
numc_array_shape(a, sh);               // [2, 3, 4]
numc_array_strides(a, st);             // [48, 16, 4]

bool contiguous = numc_array_is_contiguous(a); // true
```
