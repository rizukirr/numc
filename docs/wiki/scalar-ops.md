# Scalar Ops

Apply a single scalar value to every element of an array. The scalar is passed as `double` and cast to the array's dtype before the operation.

## Allocating variants

Write result to `out`, leave `a` unchanged.

```c
int numc_add_scalar(const NumcArray *a, double scalar, NumcArray *out);
int numc_sub_scalar(const NumcArray *a, double scalar, NumcArray *out);
int numc_mul_scalar(const NumcArray *a, double scalar, NumcArray *out);
int numc_div_scalar(const NumcArray *a, double scalar, NumcArray *out);
// out[i] = a[i] op scalar
```

`a` and `out` must have the **same shape and dtype**. Works on contiguous and non-contiguous arrays.

```c
NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

numc_add_scalar(a, 100.0, out);   // out = a + 100
numc_sub_scalar(a, 5.0,   out);   // out = a - 5
numc_mul_scalar(a, 0.5,   out);   // out = a * 0.5
numc_div_scalar(a, 3.0,   out);   // out = a / 3
```

## In-place variants

Mutate `a` directly — no `out` needed.

```c
int numc_add_scalar_inplace(NumcArray *a, double scalar);
int numc_sub_scalar_inplace(NumcArray *a, double scalar);
int numc_mul_scalar_inplace(NumcArray *a, double scalar);
int numc_div_scalar_inplace(NumcArray *a, double scalar);
// a[i] op= scalar
```

```c
numc_add_scalar_inplace(bias, 1.0);          // bias += 1
numc_mul_scalar_inplace(weights, 0.99);      // weights *= 0.99
numc_sub_scalar_inplace(weights, 1e-4);      // weight decay
numc_div_scalar_inplace(grad, (double)n);    // normalize by batch size
```

## Common neural network uses

```c
// Gradient descent step: w = w - lr * g
numc_mul_scalar(grad, learning_rate, scaled_grad);
numc_sub(weights, scaled_grad, weights);

// Or with inplace:
numc_mul_scalar_inplace(grad, learning_rate);
numc_sub(weights, grad, weights);
```

## Scalar casting

The `double` scalar is **cast to the array's dtype** before the operation:

```c
// INT32 array, scalar 2.9 → cast to int32 → 2
numc_mul_scalar_inplace(int_arr, 2.9);   // multiplies by 2, not 2.9
```

For float arrays (`float32` / `float64`), the cast preserves the value normally.

## Error conditions

| Error | Cause |
|---|---|
| `NUMC_ERR_NULL` | `a` or `out` is `NULL` |
| `NUMC_ERR_SHAPE` | `a` and `out` shapes don't match |
| `NUMC_ERR_TYPE` | `a` and `out` dtypes don't match |
