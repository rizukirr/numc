# Context

The context (`NumcCtx`) is the memory owner for all arrays. It wraps a bump-pointer arena allocator — fast allocation, bulk free.

## `numc_ctx_create`

```c
NumcCtx *numc_ctx_create(void);
```

Creates a new context with an 8 MB arena. Returns `NULL` on allocation failure.

```c
NumcCtx *ctx = numc_ctx_create();
if (!ctx) {
    fprintf(stderr, "out of memory\n");
    return 1;
}
```

## `numc_ctx_free`

```c
void numc_ctx_free(NumcCtx *ctx);
```

Frees the context **and every array** created from it in a single call. NULL-safe.

```c
numc_ctx_free(ctx);   // all arrays are gone
```

## Design notes

- **One context per computation.** Creating multiple contexts gives memory isolation — arrays from different contexts cannot be mixed in math ops.
- **No per-array free.** There is no `numc_array_free()`. Arrays live until their context is freed.
- **8 MB default.** For models with large weight tensors, one context is typically enough. The arena is bump-allocated, so consecutive arrays are laid out contiguously in memory.

## Pattern: one context per model

```c
NumcCtx *ctx = numc_ctx_create();

// allocate all weights, activations, gradients...
NumcArray *w1 = numc_array_fill(ctx, shape1, 2, NUMC_DTYPE_FLOAT32, &init);
NumcArray *w2 = numc_array_fill(ctx, shape2, 2, NUMC_DTYPE_FLOAT32, &init);

// train...

numc_ctx_free(ctx);   // teardown: everything freed at once
```
