# numc Agent Memory - HPC Matrix Engineer

## Critical Known Bugs (Audit 2026-02-10)

- `numc_type_is_unsigned()` in `dtype.h:135-137`: range check is wrong. UBYTE/USHORT/UINT/ULONG are at odd enum positions (1,3,5,7) -- range check also matches signed types SHORT(2), INT(4), LONG(6). Needs explicit comparison or bitmask.
- `error.c:14`: `strncpy` without null-terminator when msg fills 256-byte buffer. Use `snprintf` or manual terminate.
- `error.c:6-7`: plain `static` globals for error state, NOT `_Thread_local`. Data race with OpenMP.
- `math.c` scalar int division (`divs_INT`/`divs_UINT`): uses reciprocal multiplication which is incorrect for non-power-of-2 divisors (e.g., 3/3 via reciprocal = 0.999 truncates to 0 instead of 1).
- `CMakeLists.txt:84-97`: LTO `set_target_properties` references `numc_demo` and benchmark targets BEFORE they are defined (line 100+). CMake error on Release+LTO builds.

## Key Architecture Notes

- See [audit-details.md](audit-details.md) for full findings
- `numc_type_sizes[]` is `static const` in public header -- duplicated per TU (29+ copies)
- `NUMC_MAX_ELEMENT_LOOP` in `internal.h` is defined but never used (dead code)
- `numc_get_error()` returns `char *` not `const char *` -- callers can corrupt buffer
- No way to clear error state (`set_error(NUMC_OK,...)` returns early without clearing)
- Public headers use quote includes (`"dtype.h"`) instead of angle brackets (`<numc/dtype.h>`)
- `array_arange`/`array_linspace` params are `int`, limiting LONG/DOUBLE range
- `numc_realloc` leaks when `old_size==0 && ptr!=NULL`

## array_core.c / array_shape.c Deep Audit (2026-02-10)

- **astype restrict UB** (core:862-868): In-place conversion with `new_data==array->data` creates two `restrict` pointers to same memory -- C11 UB per 6.7.3.1.
- **array_offset error** (core:646): Returns 0 on NULL (valid offset), not a sentinel.
- **const-cast** (core:904-914): `array_equal`/`array_allclose` cast away `const` via `(Array *)a`.
- **array_create wasted alloc** (core:228-233): `owns_data=false` path does alloc then free.
- **concat wasted zeroing** (shape:334): Uses `array_create` (zeroes) instead of `array_empty`.
- **No ND inner-contiguous copy** (core:635): Only 1D/2D fast paths, 3D+ is element-by-element.
- **sizeof(Array) ~200 bytes** due to embedded `_shape_buff[8]` + `_strides_buff[8]`.
- Transpose is truly O(1) data, O(ndim) metadata. Reshape rejects non-contiguous correctly.
- Flatten delegates to reshape, no unnecessary copies.

## Build System

- CMake sets compiler to clang in CMakeLists.txt with CACHE; switching via CC= requires clean build
- `-ffast-math` + `-fno-math-errno` redundant (fast-math implies it)
- Debug ASan flags set globally but no link flag on library target (works for static, breaks shared)
- No `-j` parallel build in run.sh
- `date +%s%3N` in run.sh is Linux-only (no macOS support)

## Performance Patterns Confirmed

- 32-bit types: `__builtin_assume_aligned` + auto-vectorization is the sweet spot
- 64-bit types: alignment hints cause 6-41% regressions due to cache line conflicts
- Small types (BYTE/SHORT): explicit clang vectorize_width pragmas + higher OMP thresholds (16MB)
- Reductions: no OpenMP (outlining kills vectorization), clang loop pragmas with interleave_count(4)
- Scalar div INT/UINT: reciprocal optimization is INCORRECT (needs actual division)

## Vectorization Width Audit (math.c)

- BYTE/UBYTE `vectorize_width(16)` should be 32 for AVX2 (256-bit / 1-byte = 32 elements)
- SHORT/USHORT `vectorize_width(8)` should be 16 (256-bit / 2-byte = 16 elements)
- All 32-bit reductions use fixed `vectorize_width(8)` -- wrong for BYTE(32) and SHORT(16)
- 64-bit dot product uses `vectorize_width(8)` should be 4
- `sum_to_double` 32-bit uses `vectorize_width(8)` but double accumulator limits width to 4
- `array_binary_op_out` (line 481): does NOT check `out->is_contiguous` -- writing to strided view = UB
- `std_along_axis` (line 1515): no NULL check on `numc_malloc` return
- No OpenMP on axis reduction outer loops (trivially parallelizable for large outer_size)
- ~170 kernel functions, ~30-40KB code in single TU, 21 dispatch tables
