# Shape Manipulation

Functions for changing the layout or view of an array without (necessarily) moving data.

## Reshape

### `numc_array_reshape` — in-place

```c
int numc_array_reshape(NumcArray *arr, const size_t *new_shape, size_t new_dim);
```

Changes the shape of `arr` in-place. The total element count must stay the same. Returns `0` on success, `-1` on error (size mismatch or non-contiguous source).

```c
// 2×3 → 3×2
size_t new_shape[] = {3, 2};
numc_array_reshape(arr, new_shape, 2);

// 2×3 → 6 (flatten)
size_t flat[] = {6};
numc_array_reshape(arr, flat, 1);
```

### `numc_array_reshape_copy` — returns new array

```c
NumcArray *numc_array_reshape_copy(const NumcArray *arr,
                                   const size_t *new_shape,
                                   size_t new_dim);
```

Returns a new contiguous array with the requested shape. The original is unchanged.

```c
size_t flat[] = {6, 1};
NumcArray *flat_arr = numc_array_reshape_copy(arr, flat, 2);
// arr still 2×3, flat_arr is 6×1
```

## Transpose

### `numc_array_transpose` — in-place

```c
int numc_array_transpose(NumcArray *arr, const size_t *axes);
```

Reorders dimensions by permuting strides and shape. **Does not move data** — the result is a non-contiguous view. `axes` must be a valid permutation of `[0, ndim-1]`.

```c
// 2D: transpose rows ↔ columns
size_t axes[] = {1, 0};
numc_array_transpose(arr, axes);     // 2×3 → 3×2 (non-contiguous)

// 3D: move last axis to front
size_t axes3[] = {2, 0, 1};
numc_array_transpose(vol, axes3);    // (A, B, C) → (C, A, B)
```

After transpose, `numc_array_is_contiguous()` returns `false`.

### `numc_array_transpose_copy` — returns new array

```c
NumcArray *numc_array_transpose_copy(const NumcArray *arr, const size_t *axes);
```

Returns a new **contiguous** array with the reordered layout. The original is unchanged.

```c
NumcArray *t = numc_array_transpose_copy(arr, axes);
// t is contiguous, arr is unchanged
```

## Make contiguous

### `numc_array_contiguous`

```c
int numc_array_contiguous(NumcArray *arr);
```

Re-packs a non-contiguous array's data into a new contiguous buffer within the same context. After this call `numc_array_is_contiguous()` returns `true`.

```c
numc_array_transpose(arr, axes);              // non-contiguous
numc_array_contiguous(arr);                   // contiguous again
```

Use this before passing a view to code that requires contiguous layout.

## Slice

### `numc_array_slice` / `numc_slice` macro

```c
NumcArray *numc_array_slice(const NumcArray *arr, NumcSlice *slice);

// preferred macro:
#define numc_slice(arr, ...)  numc_array_slice((arr), &(NumcSlice){__VA_ARGS__})
```

Returns a **view** — no data is copied. The `NumcSlice` struct has four fields:

| Field | Meaning |
|---|---|
| `axis` | Which dimension to slice |
| `start` | First index (inclusive) |
| `stop` | Last index (exclusive) |
| `step` | Step size (must be ≥ 1) |

```c
// arr is 4×3. Get rows 1 and 2:
NumcArray *view = numc_slice(arr, .axis=0, .start=1, .stop=3, .step=1);
// view is 2×3, shares arr's memory

// Get every other column from a 1×6 array:
NumcArray *evens = numc_slice(arr1d, .axis=0, .start=0, .stop=6, .step=2);
// evens has 3 elements, non-contiguous
```

Math functions work directly on non-contiguous views. Call `numc_array_contiguous()` only when you specifically need a flat buffer.

## Summary: copy vs. view

| Function | Copies data? | Result contiguous? |
|---|---|---|
| `numc_array_reshape` | no | yes (requires source to be contiguous) |
| `numc_array_reshape_copy` | yes | yes |
| `numc_array_transpose` | no | **no** |
| `numc_array_transpose_copy` | yes | yes |
| `numc_array_slice` | no | depends on step/axis |
| `numc_array_contiguous` | yes (in-place repacking) | yes |
| `numc_array_copy` | yes | yes |
