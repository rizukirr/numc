---
title: numc sigmoid
date: 2026-04-24
status: approved
---

# numc sigmoid - Design

## Problem
`numc_sigmoid` is listed as immediate roadmap work and is needed for logistic
regression and neural-network gate-style math. The codebase now has activation
separation work (`unary_activation.c`) and needs sigmoid added with the same API
shape and ISA parity expectations as existing activation ops.

## Goals
- Add `numc_sigmoid(a, out)` and `numc_sigmoid_inplace(a)` to public API.
- Keep dtype behavior consistent with existing unary ops (all 10 dtypes).
- Use numerically stable float formulation for large-magnitude inputs.
- Provide SIMD fast paths for float32/float64 across AVX2/AVX512/SVE/NEON/RVV.
- Add tests covering correctness, bounds/inequality checks, error handling, and
  inplace parity.

## Non-goals
- No new fused ops (e.g. SiLU) in this change.
- No changes to unary dispatch architecture beyond adding sigmoid wiring.
- No behavior changes for existing ops (`tanh`, `exp`, etc.).

## Constraints
- Must follow existing unary dispatch and table patterns in
  `src/elemwise/unary_activation.c`.
- Must preserve scalar fallback for non-contiguous arrays and unsupported dtypes.
- Must keep ISA parity (not single-ISA-only optimization).
- Tests should use epsilon/inequality checks for extreme values, not brittle
  exact-value assertions.

## Approach
Implement sigmoid in activation family (`src/elemwise/unary_activation.c`) using
the stable split formula for floating point:

- `x >= 0`: `1 / (1 + exp(-x))`
- `x < 0`: `exp(x) / (1 + exp(x))`

This is algebraically equivalent to sigmoid and avoids overflow behavior from
always computing `exp(-x)` for large negative values.

### Core activation wiring
- Add typed scalar kernels for all dtypes:
  - small ints/uints cast through float path,
  - 32/64-bit ints cast through double path,
  - float32/float64 use stable split logic.
- Add `sigmoid_table` with full dtype coverage.
- Add `sigmoid_fast_table` for float32/float64 and wire through
  `DEFINE_UNARY_SIMD(sigmoid, sigmoid_fast_table, sigmoid_table)`.
- Keep no-SIMD build using `DEFINE_ELEMWISE_UNARY(sigmoid, sigmoid_table)`.

### Intrinsics
Add sigmoid vector helpers and packed wrappers in:

- `src/intrinsics/math_avx2.h`
- `src/intrinsics/math_neon.h`
- `src/intrinsics/math_sve.h`
- `src/intrinsics/math_rvv.h`

Each ISA implementation computes split-form sigmoid for f32/f64 and exports
`_fast_sigmoid_f32_*` and `_fast_sigmoid_f64_*` consumed by activation dispatch.

### Public API
Add declarations/docs in `include/numc/math.h`:

- `numc_sigmoid(NumcArray *a, NumcArray *out)`
- `numc_sigmoid_inplace(NumcArray *a)`

## Alternatives considered
- Keep sigmoid in `src/elemwise/unary.c`:
  rejected because activation-family split has already started and this would
  weaken module boundaries.
- Scalar-only sigmoid first, SIMD later:
  rejected because ISA parity is an explicit project rule for performance paths.
- Derive sigmoid via tanh identity (`0.5 * (tanh(x/2) + 1)`):
  rejected due to extra ops/coupling and less direct numerical-control path.

## Testing
- Add `tests/elemwise/test_sigmoid.c` and register it in `tests/CMakeLists.txt`.
- Coverage:
  - float32/float64 representative points and tolerances,
  - range checks (`0 <= y <= 1`) and monotonicity,
  - extreme-input near-0/near-1 checks with inequality/epsilon,
  - integer cast/truncation behavior,
  - inplace parity with out-of-place,
  - null/type/shape mismatch behavior.
- Verification command: `./run.sh test`.

## Open questions
N/A - behavior and numeric policy are approved (stable split + epsilon/inequality).
