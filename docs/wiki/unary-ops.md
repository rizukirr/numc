# Unary Ops

Apply an operation to every element of one array. Each op has two forms:

- **Allocating:** `numc_xxx(a, out)` — writes to `out`, leaves `a` unchanged.
- **In-place:** `numc_xxx_inplace(a)` — mutates `a` directly.

Both work on contiguous and non-contiguous arrays.

---

## `numc_neg` — negate

```c
int numc_neg(NumcArray *a, NumcArray *out);
int numc_neg_inplace(NumcArray *a);
// out[i] = -a[i]
```

Works on all 10 dtypes.

```c
float da[] = {1, -2, 3, -4};
numc_array_write(a, da);

numc_neg(a, out);         // [-1, 2, -3, 4]
numc_neg_inplace(a);      // a becomes [-1, 2, -3, 4]
```

---

## `numc_abs` — absolute value

```c
int numc_abs(NumcArray *a, NumcArray *out);
int numc_abs_inplace(NumcArray *a);
// out[i] = |a[i]|
```

Only defined for **signed** types: `int8`, `int16`, `int32`, `int64`, `float32`, `float64`. Unsigned types have no negatives.

```c
int32_t da[] = {-10, -20, 30};
numc_abs(a, out);    // [10, 20, 30]
```

> **Edge case:** `abs(INT8_MIN)` = `abs(-128)` overflows back to `-128`, matching C's standard behavior.

---

## `numc_log` — natural logarithm

```c
int numc_log(NumcArray *a, NumcArray *out);
int numc_log_inplace(NumcArray *a);
// out[i] = ln(a[i])
```

Uses a custom fdlibm-derived implementation (no `<math.h>` dependency).

- **`float32` / `float64`:** bit-manipulation argument reduction + Horner polynomial.
- **Integer types:** cast to float, compute log, truncate back to integer.

```c
float da[] = {1.0f, 2.0f, 4.0f, 8.0f};
numc_log(a, out);   // [0.000, 0.693, 1.386, 2.079]
```

---

## `numc_exp` — natural exponential

```c
int numc_exp(NumcArray *a, NumcArray *out);
int numc_exp_inplace(NumcArray *a);
// out[i] = e^a[i]
```

Uses a Cephes-style polynomial (< 1 ULP error for `float32`).

- **float32 overflow:** `exp(x)` for `x > 88.7` → `+inf`
- **float32 underflow:** `exp(x)` for `x < -103.9` → `0`
- **Integer types:** cast through float, result truncated.

```c
float da[] = {0.0f, 1.0f, 2.0f, 3.0f};
numc_exp(a, out);   // [1.000, 2.718, 7.389, 20.085]
```

---

## `numc_sqrt` — square root

```c
int numc_sqrt(NumcArray *a, NumcArray *out);
int numc_sqrt_inplace(NumcArray *a);
// out[i] = sqrt(a[i])
```

Delegates to hardware `sqrtf` / `sqrt` — compiles to `vsqrtps` / `vsqrtpd` with `-march=native`.

```c
float da[] = {1.0f, 4.0f, 9.0f, 16.0f};
numc_sqrt(a, out);   // [1.0, 2.0, 3.0, 4.0]
```

---

## `numc_clip` — clamp to range

```c
int numc_clip(NumcArray *a, NumcArray *out, double min, double max);
int numc_clip_inplace(NumcArray *a, double min, double max);
// out[i] = clamp(a[i], min, max)
```

`min` and `max` are `double` and are cast to `a`'s dtype before the operation.

```c
float da[] = {-2, 0.5, 3, 10};
numc_clip(a, out, 0.0, 1.0);   // [0, 0.5, 1, 1]

// Gradient clipping
numc_clip_inplace(grad, -1.0, 1.0);

// ReLU-style lower bound
numc_clip_inplace(activations, 0.0, 1e9);
```

---

## Summary table

| Function | Formula | Dtypes |
|---|---|---|
| `numc_neg` | `-a` | all 10 |
| `numc_abs` | `\|a\|` | signed only (int8/16/32/64, float32/64) |
| `numc_log` | `ln(a)` | all 10 (integers cast through float) |
| `numc_exp` | `e^a` | all 10 (integers cast through float) |
| `numc_sqrt` | `√a` | all 10 |
| `numc_clip` | `clamp(a, min, max)` | all 10 |
