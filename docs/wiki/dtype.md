# Data Types (NumcDType)

## Enum values

| Constant | C type | Bytes | Notes |
|---|---|---|---|
| `NUMC_DTYPE_INT8` | `int8_t` | 1 | |
| `NUMC_DTYPE_INT16` | `int16_t` | 2 | |
| `NUMC_DTYPE_INT32` | `int32_t` | 4 | |
| `NUMC_DTYPE_INT64` | `int64_t` | 8 | |
| `NUMC_DTYPE_UINT8` | `uint8_t` | 1 | |
| `NUMC_DTYPE_UINT16` | `uint16_t` | 2 | |
| `NUMC_DTYPE_UINT32` | `uint32_t` | 4 | |
| `NUMC_DTYPE_UINT64` | `uint64_t` | 8 | |
| `NUMC_DTYPE_FLOAT32` | `float` | 4 | **Primary dtype for neural networks** |
| `NUMC_DTYPE_FLOAT64` | `double` | 8 | |

## Usage

Pass a `NumcDType` constant wherever a dtype is required:

```c
NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
NumcArray *b = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT8);
```

Mismatched dtypes between operands are caught at runtime and return `NUMC_ERR_TYPE`.

## Dtype-specific behavior

Some operations have dtype-dependent behavior:

- **`numc_abs`** — only defined for signed types (`int8`, `int16`, `int32`, `int64`, `float32`, `float64`). Unsigned types have no negatives.
- **`numc_log` / `numc_exp`** — for integer types, the value is cast through float before computing and truncated back.
- **`numc_pow`** — for floats uses `exp(b * log(a))`; for integers uses a branchless loop.
- **Scalar ops** — the `double` scalar is cast to the array's dtype before the operation.

## Runtime size/alignment lookup

`dtype.h` exposes two compile-time lookup tables:

```c
numc_type_size[NUMC_DTYPE_FLOAT32]   // → 4 (bytes)
numc_type_align[NUMC_DTYPE_INT64]    // → 8 (alignment in bytes)
```
