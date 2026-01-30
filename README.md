# numc - N-Dimensional Array Library for C

A lightweight, NumPy-inspired N-dimensional array library written in C. Designed for scientific computing and numerical applications.

## Features

- N-dimensional arrays with flexible stride-based memory layout
- NumPy-like API for familiar usage
- Zero-copy views and slicing
- Contiguous memory optimization for SIMD operations
- Type-agnostic design supporting any data type
- Geometric growth for efficient dynamic arrays

## Building

```bash
# Configure
cmake -B build

# Build
cd build && make

# Run tests
./bin/numc
```

### Build Options

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # Optimized build
cmake -B build -DCMAKE_BUILD_TYPE=Debug     # Debug with AddressSanitizer
cmake -B build -DBUILD_SHARED=ON            # Build shared library
```

## API Reference

### Array Creation

**Implemented:**
- `Array *array_create(size_t ndim, const size_t *shape, size_t elem_size)` - Create uninitialized array
- `Array *array_from_data(size_t ndim, const size_t *shape, size_t elem_size, const void *data)` - Create from existing data
- `void array_free(Array *array)` - Free array memory

**Not Yet Implemented:**
- [ ] `Array *array_zeros(size_t ndim, const size_t *shape, size_t elem_size)` - Create array filled with zeros
- [ ] `Array *array_ones(size_t ndim, const size_t *shape, size_t elem_size)` - Create array filled with ones
- [ ] `Array *array_fill(size_t ndim, const size_t *shape, size_t elem_size, const void *value)` - Create array filled with value
- [ ] `Array *array_arange(double start, double stop, double step)` - Create range of values
- [ ] `Array *array_linspace(double start, double stop, size_t num)` - Create linearly spaced values
- [ ] `Array *array_eye(size_t n, size_t m, size_t elem_size)` - Create identity matrix

### Array Access

**Implemented:**
- `void *array_at(const Array *array, const size_t *indices)` - Get pointer to element (renamed from array_get_ptr)
- `size_t array_offset(const Array *array, const size_t *indices)` - Compute byte offset

**Not Yet Implemented:**
- [ ] `void *array_at_1d(const Array *array, size_t i)` - Fast 1D access
- [ ] `void *array_at_2d(const Array *array, size_t i, size_t j)` - Fast 2D access
- [ ] `void *array_at_3d(const Array *array, size_t i, size_t j, size_t k)` - Fast 3D access

### Array Properties

**Implemented:**
- `size_t array_size(const Array *array)` - Get total number of elements (renamed from array_numof_elem)
- `int array_is_contiguous(const Array *array)` - Check if array is contiguous (renamed from array_contiguous)

**Not Yet Implemented:**
- [ ] `size_t array_ndim(const Array *array)` - Get number of dimensions
- [ ] `const size_t *array_shape(const Array *array)` - Get shape array
- [ ] `size_t array_shape_at(const Array *array, size_t axis)` - Get size of specific dimension
- [ ] `size_t array_itemsize(const Array *array)` - Get element size in bytes
- [ ] `size_t array_nbytes(const Array *array)` - Get total bytes used

### Array Manipulation

**Implemented:**
- `int array_reshape(Array *array, size_t ndim, const size_t *shape)` - Reshape array in-place
- `Array *array_slice(Array *base, const size_t *start, const size_t *stop, const size_t *step)` - Create view slice
- `Array *array_copy(const Array *src)` - Create contiguous copy (renamed from array_copy_contiguous)
- `int array_append(Array *array, const void *elem)` - Append element to 1D array
- `Array *array_concat(const Array *a, const Array *b, size_t axis)` - Concatenate arrays

**Not Yet Implemented:**
- [ ] `Array *array_transpose(const Array *array)` - Transpose array
- [ ] `Array *array_transpose_axes(const Array *array, const size_t *axes)` - Transpose with custom axis order
- [ ] `Array *array_flatten(const Array *array)` - Flatten to 1D array
- [ ] `Array *array_ravel(const Array *array)` - Flatten to 1D view (no copy if possible)
- [ ] `Array *array_squeeze(const Array *array)` - Remove single-dimensional entries
- [ ] `Array *array_expand_dims(const Array *array, size_t axis)` - Add dimension
- [ ] `Array *array_broadcast_to(const Array *array, size_t ndim, const size_t *shape)` - Broadcast to shape
- [ ] `int array_resize(Array *array, size_t ndim, const size_t *shape)` - Resize array (may reallocate)

### Array Operations

**Not Yet Implemented:**
- [ ] `int array_fill_value(Array *array, const void *value)` - Fill array with value
- [ ] `int array_copy_data(Array *dst, const Array *src)` - Copy data between arrays
- [ ] `Array *array_view(const Array *array)` - Create view without copy
- [ ] `Array *array_astype(const Array *array, size_t new_elem_size)` - Convert element type

### Mathematical Operations

**Not Yet Implemented:**
- [ ] `int array_add(const Array *a, const Array *b, Array *result)` - Element-wise addition
- [ ] `int array_subtract(const Array *a, const Array *b, Array *result)` - Element-wise subtraction
- [ ] `int array_multiply(const Array *a, const Array *b, Array *result)` - Element-wise multiplication
- [ ] `int array_divide(const Array *a, const Array *b, Array *result)` - Element-wise division
- [ ] `Array *array_dot(const Array *a, const Array *b)` - Matrix multiplication
- [ ] `Array *array_matmul(const Array *a, const Array *b)` - Matrix multiplication (alias)

### Reduction Operations

**Not Yet Implemented:**
- [ ] `int array_sum(const Array *array, void *result)` - Sum all elements
- [ ] `int array_sum_axis(const Array *array, size_t axis, Array *result)` - Sum along axis
- [ ] `int array_mean(const Array *array, void *result)` - Mean of all elements
- [ ] `int array_mean_axis(const Array *array, size_t axis, Array *result)` - Mean along axis
- [ ] `int array_min(const Array *array, void *result)` - Minimum element
- [ ] `int array_max(const Array *array, void *result)` - Maximum element
- [ ] `int array_argmin(const Array *array, size_t *result)` - Index of minimum
- [ ] `int array_argmax(const Array *array, size_t *result)` - Index of maximum

### Comparison Operations

**Not Yet Implemented:**
- [ ] `int array_equal(const Array *a, const Array *b)` - Check if arrays are equal
- [ ] `int array_allclose(const Array *a, const Array *b, double rtol, double atol)` - Check if approximately equal
- [ ] `Array *array_where(const Array *condition, const Array *x, const Array *y)` - Conditional selection

### I/O Operations

**Implemented:**
- `void array_print(const Array *array)` - Print array to stdout

**Not Yet Implemented:**
- [ ] `int array_save(const Array *array, const char *filename)` - Save array to file
- [ ] `Array *array_load(const char *filename)` - Load array from file
- [ ] `char *array_tostring(const Array *array)` - Convert to string representation

### Utility Functions

**Not Yet Implemented:**
- [ ] `int array_is_c_contiguous(const Array *array)` - Check C-contiguous (row-major)
- [ ] `int array_is_f_contiguous(const Array *array)` - Check Fortran-contiguous (column-major)
- [ ] `int array_is_square(const Array *array)` - Check if matrix is square
- [ ] `int array_can_broadcast(const Array *a, const Array *b)` - Check if arrays can broadcast

## Examples

### Creating Arrays

```c
// Create empty array
size_t shape[] = {2, 3, 4};
Array *arr = array_create(3, shape, sizeof(double));

// Create from static array
int data[2][3] = {{1, 2, 3}, {4, 5, 6}};
Array *arr2 = array_from_data(2, (size_t[]){2, 3}, sizeof(int), data);

// Create zeros (when implemented)
// Array *zeros = array_zeros(2, shape, sizeof(float));
```

### Accessing Elements

```c
// Get element at [1, 2]
float *elem = array_at(arr, (size_t[]){1, 2});
printf("Value: %f\n", *elem);

// Modify element
*elem = 99.0f;
```

### Slicing

```c
// Original array shape: [10, 20]
// Slice: rows 2-8, columns 5-15, step 2
Array *sliced = array_slice(arr,
    (size_t[]){2, 5},     // start
    (size_t[]){8, 15},    // stop (exclusive)
    (size_t[]){2, 1}      // step
);
// Result shape: [3, 10]
```

### Reshaping

```c
// Reshape [2, 6] -> [3, 4]
Array *arr = array_create(2, (size_t[]){2, 6}, sizeof(int));
array_reshape(arr, 2, (size_t[]){3, 4});
```

### Concatenation

```c
Array *a = array_create(2, (size_t[]){3, 4}, sizeof(float));
Array *b = array_create(2, (size_t[]){2, 4}, sizeof(float));

// Concatenate along axis 0 (rows)
Array *result = array_concat(a, b, 0);  // Shape: [5, 4]
```

## Performance Considerations

### Contiguous vs Non-Contiguous Arrays

- **Contiguous arrays**: All elements packed in memory, enables fast operations
- **Non-contiguous arrays**: Created by slicing with step > 1, slower operations

Check contiguity:
```c
if (array_is_contiguous(arr)) {
    // Can use fast SIMD operations
}
```

### Memory Alignment

For SIMD optimizations, ensure arrays are properly aligned:
- SSE/NEON: 16-byte alignment
- AVX2: 32-byte alignment
- AVX-512: 64-byte alignment

See `UNDERSTANDING_SIMD.md` for details.

### Dynamic Arrays

`array_append()` uses geometric growth (2x capacity) for O(n) amortized performance:
```c
Array *arr = array_create(1, (size_t[]){0}, sizeof(int));
for (int i = 0; i < 10000; i++) {
    array_append(arr, &i);  // Efficient: only ~14 reallocations
}
```

## Compatibility

- **C Standard**: C11
- **Compiler**: clang (primary), gcc (compatible)
- **Platforms**: Linux, macOS, Windows (with appropriate compiler)

## Contributing

When adding new functions:
1. Follow NumPy naming conventions when possible
2. Document with Doxygen comments
3. Add examples to `main.c`
4. Update this README
5. Ensure memory safety (no leaks, use AddressSanitizer)

## License

Licensed under the MIT License.
