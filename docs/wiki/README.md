# numc API Wiki

numc is a C tensor library providing N-dimensional arrays with element-wise math, reductions, and shape operations.

## Pages

| Page | What's covered |
|---|---|
| [quickstart.md](quickstart.md) | Include, build, first program |
| [dtype.md](dtype.md) | `NumcDType` enum, all 10 supported types |
| [context.md](context.md) | `numc_ctx_create`, `numc_ctx_free` |
| [creation.md](creation.md) | `numc_array_create`, `zeros`, `fill`, `copy`, `write` |
| [properties.md](properties.md) | `ndim`, `size`, `shape`, `strides`, `dtype`, `data` |
| [shape.md](shape.md) | `reshape`, `transpose`, `slice`, `contiguous` |
| [binary-ops.md](binary-ops.md) | `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum` |
| [scalar-ops.md](scalar-ops.md) | `add_scalar`, `mul_scalar`, inplace variants |
| [unary-ops.md](unary-ops.md) | `neg`, `abs`, `log`, `exp`, `sqrt`, `clip` |
| [reductions.md](reductions.md) | `numc_sum`, `numc_sum_axis` |
| [error.md](error.md) | Error codes, `numc_get_error`, `numc_set_error` |

## Key rules

- **One context, all arrays.** `numc_ctx_free()` frees everything at once — no per-array cleanup.
- **Return value convention.** All math functions return `0` on success and a negative error code on failure.
- **Strides are in bytes**, not elements.
- **`float32` is the primary dtype** for neural network use. All 10 types are supported via X-macros.
