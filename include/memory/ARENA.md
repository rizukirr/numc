# Arena Allocator API Reference

Single-header arena (bump) allocator. Define `ARENA_IMPLEMENTATION` in exactly
one translation unit before including `arena.h` to get the implementation.

```c
// arena.c
#define ARENA_IMPLEMENTATION
#include <memory/arena.h>
```

All other files just include the header normally:

```c
#include <memory/arena.h>
```

---

## `ARENA_ALIGNOF(type)`

Portable alignment query. Expands to `alignof(type)` on C11+, falls back to
a struct-offset trick on older compilers.

```c
size_t align = ARENA_ALIGNOF(double);   // 8
size_t align = ARENA_ALIGNOF(float);    // 4
```

---

## `arena_create`

```c
Arena *arena_create(size_t default_block_size);
```

Create a new arena. No memory blocks are allocated until the first
`arena_alloc()` call (lazy initialization).

**Parameters**
- `default_block_size` — Minimum size of each internal block in bytes.
  Allocations larger than this get their own block.

**Returns** — `Arena*` on success, `NULL` if `default_block_size == 0` or
`malloc` fails.

```c
// 1 MB block size — good for large matrix workloads
Arena *arena = arena_create(1024 * 1024);
if (!arena) { /* handle error */ }
```

### Choosing block size

| Workload | Suggested size |
|---|---|
| Many small arrays (< 4 KB each) | `64 * 1024` (64 KB) |
| Medium matrices (up to ~256 KB) | `1024 * 1024` (1 MB) |
| Large matrices (> 1 MB) | `8 * 1024 * 1024` (8 MB) |

A single allocation larger than `default_block_size` always succeeds — the
arena allocates a block sized exactly for that request (plus alignment padding).

---

## `arena_alloc`

```c
void *arena_alloc(Arena *arena, size_t size, size_t alignment);
```

Bump-allocate `size` bytes with the given alignment. O(1) in the common case
(just a pointer bump). Falls back to allocating a new block when the current
one is full.

**Parameters**
- `arena` — Valid arena pointer.
- `size` — Number of bytes (must be > 0).
- `alignment` — Power of two (1, 2, 4, 8, 16, 32, 64, ...).

**Returns** — Pointer to uninitialized memory, or `NULL` on failure.

```c
// Allocate a 1000x1000 double matrix, 64-byte aligned for AVX-512
size_t n = 1000 * 1000;
double *data = arena_alloc(arena, n * sizeof(double), 64);

// Allocate shape array (natural alignment is fine)
size_t *shape = arena_alloc(arena, 2 * sizeof(size_t), ARENA_ALIGNOF(size_t));
shape[0] = 1000;
shape[1] = 1000;
```

### Alignment guide for SIMD

| Target | Alignment |
|---|---|
| Scalar / generic | `ARENA_ALIGNOF(type)` |
| SSE (128-bit) | `16` |
| AVX/AVX2 (256-bit) | `32` |
| AVX-512 (512-bit) | `64` |

### Contiguity guarantee

Each individual `arena_alloc()` call returns a **single contiguous block**.
A 1000x1000 double matrix is one flat 8 MB region — exactly what you need
for row-major ndarray storage and BLAS-style matmul.

Two separate `arena_alloc()` calls are **not** guaranteed to be contiguous
with each other (they may land in different internal blocks).

---

## `arena_reset`

```c
void arena_reset(Arena *arena);
```

Reset all allocation cursors to zero. All blocks are retained (no `free()`
calls), so subsequent allocations reuse existing capacity without hitting
`malloc`.

**Use case** — Reuse the arena across iterations without reallocating:

```c
Arena *scratch = arena_create(1024 * 1024);

for (int epoch = 0; epoch < 100; epoch++) {
    double *grad  = arena_alloc(scratch, n * sizeof(double), 32);
    double *delta = arena_alloc(scratch, n * sizeof(double), 32);
    // ... compute ...
    arena_reset(scratch);  // all allocations gone, memory retained
}

arena_free(scratch);
```

After `arena_reset()`:
- All pointers previously returned by `arena_alloc()` are **invalid**.
- The arena's capacity (total allocated bytes from the OS) is unchanged.

---

## `arena_free`

```c
void arena_free(Arena *arena);
```

Release all memory (all blocks + the arena struct itself). The pointer is
invalid after this call.

```c
arena_free(arena);
arena = NULL;  // good practice
```

---

## `arena_checkpoint`

```c
ArenaCheckpoint arena_checkpoint(Arena *arena);
```

Snapshot the current allocation position. Everything allocated **before** the
checkpoint is preserved when you later call `arena_restore()`.

**Returns** — `ArenaCheckpoint` value (stack-allocated, cheap to copy).

```c
Arena *arena = arena_create(1024 * 1024);

// Persistent data — survives restore
double *A = arena_alloc(arena, 1000 * 1000 * sizeof(double), 32);
double *B = arena_alloc(arena, 1000 * 1000 * sizeof(double), 32);

ArenaCheckpoint cp = arena_checkpoint(arena);

// Temporary scratch — freed on restore
double *tmp = arena_alloc(arena, 1000 * 1000 * sizeof(double), 32);
// ... use tmp for intermediate matmul result ...

arena_restore(arena, cp);  // tmp is freed, A and B survive
```

---

## `arena_restore`

```c
void arena_restore(Arena *arena, ArenaCheckpoint checkpoint);
```

Rewind the arena to a previous checkpoint. All allocations made after the
checkpoint are invalidated. Blocks that were allocated after the checkpoint
block are freed back to the OS.

**Parameters**
- `arena` — Same arena the checkpoint was taken from.
- `checkpoint` — Value from a prior `arena_checkpoint()` call.

**Constraints**
- The checkpoint must belong to the same arena (asserted in debug builds).
- Do not use a checkpoint after `arena_reset()` or `arena_free()` — undefined
  behavior.

### Nested checkpoints

Checkpoints nest naturally. Restore to any prior checkpoint to unwind
multiple levels:

```c
ArenaCheckpoint cp1 = arena_checkpoint(arena);
double *x = arena_alloc(arena, 1024, 32);

    ArenaCheckpoint cp2 = arena_checkpoint(arena);
    double *y = arena_alloc(arena, 1024, 32);
    arena_restore(arena, cp2);  // y freed, x survives

arena_restore(arena, cp1);      // x freed too
```

### Matrix multiplication scratch pattern

```c
Arena *perm  = arena_create(8 * 1024 * 1024);  // long-lived arrays
Arena *scratch = arena_create(4 * 1024 * 1024); // temporaries

// Allocate result matrix (permanent)
double *C = arena_alloc(perm, M * N * sizeof(double), 32);

// matmul with temporary buffer
ArenaCheckpoint cp = arena_checkpoint(scratch);
double *tmp = arena_alloc(scratch, M * K * sizeof(double), 32);
// ... tiled matmul using tmp ...
arena_restore(scratch, cp);  // release tmp

// C is still valid in perm arena
arena_free(scratch);
arena_free(perm);
```

---

## Full ndarray example

```c
#define ARENA_IMPLEMENTATION
#include <memory/arena.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    double *data;
    size_t *shape;
    size_t  ndim;
    size_t  size;  // total elements
} NDArray;

NDArray ndarray_create(Arena *arena, size_t ndim, const size_t shape[]) {
    NDArray arr;
    arr.ndim = ndim;

    // Allocate shape (small, natural alignment)
    arr.shape = arena_alloc(arena, ndim * sizeof(size_t), ARENA_ALIGNOF(size_t));
    memcpy(arr.shape, shape, ndim * sizeof(size_t));

    // Compute total elements
    arr.size = 1;
    for (size_t i = 0; i < ndim; i++)
        arr.size *= shape[i];

    // Allocate contiguous data buffer (32-byte aligned for AVX2)
    arr.data = arena_alloc(arena, arr.size * sizeof(double), 32);
    return arr;
}

void matmul(Arena *scratch,
            const NDArray *A, const NDArray *B, NDArray *C) {
    size_t M = A->shape[0], K = A->shape[1], N = B->shape[1];

    // Use checkpoint for any scratch buffers
    ArenaCheckpoint cp = arena_checkpoint(scratch);

    // Naive matmul (replace with tiled/BLAS for real use)
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++)
                sum += A->data[i * K + k] * B->data[k * N + j];
            C->data[i * N + j] = sum;
        }

    arena_restore(scratch, cp);
}

int main(void) {
    Arena *arena   = arena_create(1024 * 1024);
    Arena *scratch = arena_create(1024 * 1024);

    NDArray A = ndarray_create(arena, 2, (size_t[]){3, 4});
    NDArray B = ndarray_create(arena, 2, (size_t[]){4, 2});
    NDArray C = ndarray_create(arena, 2, (size_t[]){3, 2});

    // Fill A and B ...
    for (size_t i = 0; i < A.size; i++) A.data[i] = (double)i;
    for (size_t i = 0; i < B.size; i++) B.data[i] = (double)i * 0.5;

    matmul(scratch, &A, &B, &C);

    // Print result
    for (size_t i = 0; i < C.shape[0]; i++) {
        for (size_t j = 0; j < C.shape[1]; j++)
            printf("%8.2f ", C.data[i * C.shape[1] + j]);
        printf("\n");
    }

    arena_free(scratch);
    arena_free(arena);
    return 0;
}
```

---

## Thread safety

The arena is **not** thread-safe. For parallel workloads (OpenMP, pthreads),
use one scratch arena per thread:

```c
#pragma omp parallel
{
    Arena *local = arena_create(1024 * 1024);
    // ... thread-local allocations ...
    arena_free(local);
}
```
