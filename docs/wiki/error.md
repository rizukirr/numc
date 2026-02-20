# Error Handling

numc uses a simple return-code + thread-local error state model.

## Return values

All math and shape functions return:

- `0` — success
- negative integer — failure (see error codes below)

```c
int rc = numc_add(a, b, out);
if (rc != 0) {
    // something went wrong
}
```

## Error codes

| Constant | Value | Meaning |
|---|---|---|
| `NUMC_ERR_NULL` | `-1` | A pointer argument was `NULL` |
| `NUMC_ERR_SHAPE` | `-3` | Shape mismatch between arrays |
| `NUMC_ERR_TYPE` | `-4` | Dtype mismatch between arrays |

## `numc_get_error`

```c
NumcError numc_get_error(void);
```

Returns the last error as a `NumcError` struct:

```c
typedef struct {
    int   code;   // one of NUMC_ERR_*
    char *msg;    // human-readable description
} NumcError;
```

```c
int rc = numc_add(a, b, out);
if (rc != 0) {
    NumcError e = numc_get_error();
    fprintf(stderr, "numc error %d: %s\n", e.code, e.msg);
}
```

## `numc_set_error`

```c
int numc_set_error(int code, const char *msg);
```

Manually sets the error state. Useful for testing or wrapping numc in a higher-level API.

```c
numc_set_error(NUMC_ERR_NULL, "weights pointer was null");
```

## `numc_log_error`

```c
void numc_log_error(const NumcError *err);
```

Prints the error to `stderr` with a `[ERROR] numc:` prefix. Convenience wrapper for demos and CLIs.

```c
NumcError e = numc_get_error();
numc_log_error(&e);   // prints: [ERROR] numc: shape mismatch ...
```

## Common errors and fixes

**Shape mismatch:**
```c
// a is 2×3, b is 3×2 — different shapes
numc_add(a, b, out);   // NUMC_ERR_SHAPE
// Fix: ensure a and b have the same shape before calling
```

**Dtype mismatch:**
```c
// a is float32, b is int32
numc_mul(a, b, out);   // NUMC_ERR_TYPE
// Fix: use the same dtype for all operands
```

**NULL pointer:**
```c
numc_add(NULL, b, out);   // NUMC_ERR_NULL
// Fix: always check that array creation succeeded before use
NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
if (!a) { /* handle */ }
```

## Checking creation failures

Array creation returns `NULL` on failure (out of arena memory):

```c
NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
if (!a) {
    fprintf(stderr, "allocation failed\n");
    return 1;
}
```
