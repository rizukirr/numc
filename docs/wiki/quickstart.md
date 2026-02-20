# Quick Start

## Include

```c
#include <numc/numc.h>   // umbrella — pulls in array.h, math.h, dtype.h, error.h
```

## Build (CMake)

```cmake
find_package(numc REQUIRED)
target_link_libraries(my_app PRIVATE numc::numc)
```

Or with the project's `run.sh`:

```bash
./run.sh debug     # build + run demo (AddressSanitizer enabled)
./run.sh release   # build + run demo (-O3 -march=native)
./run.sh test      # build + run all tests
```

## First program

```c
#include <numc/numc.h>
#include <stdio.h>

int main(void) {
    // 1. Create a context — owns all memory
    NumcCtx *ctx = numc_ctx_create();

    // 2. Create arrays
    size_t shape[] = {2, 3};
    NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *b   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

    // 3. Write data
    float da[] = {1, 2, 3, 4, 5, 6};
    float db[] = {6, 5, 4, 3, 2, 1};
    numc_array_write(a, da);
    numc_array_write(b, db);

    // 4. Compute
    numc_add(a, b, out);       // out = a + b
    numc_array_print(out);     // [[7, 7, 7], [7, 7, 7]]

    numc_mul_scalar_inplace(out, 2.0);   // out *= 2
    numc_array_print(out);     // [[14, 14, 14], [14, 14, 14]]

    // 5. Free everything at once
    numc_ctx_free(ctx);
    return 0;
}
```

## Allocating vs in-place

Every math operation has two forms:

```c
// Allocating: result goes into `out`, inputs unchanged
numc_neg(a, out);

// In-place: mutates `a` directly, no separate output needed
numc_neg_inplace(a);
```

Use in-place when you no longer need the original — it avoids an extra array allocation.

## Error checking

```c
int rc = numc_add(a, b, out);
if (rc != 0) {
    NumcError e = numc_get_error();
    fprintf(stderr, "error %d: %s\n", e.code, e.msg);
}
```

See [error.md](error.md) for all error codes.
