# numc Post-Mortem: Complete Analysis for v2 Rewrite

> This document catalogs every bug, performance issue, architectural mistake, and missed
> optimization found in the numc v1 codebase. Use it as a checklist when building v2.

---

## Table of Contents

1. [Critical Bugs (Ship-Blocking)](#1-critical-bugs-ship-blocking)
2. [Correctness Bugs](#2-correctness-bugs)
3. [Performance Issues — Vectorization](#3-performance-issues--vectorization)
4. [Performance Issues — Parallelization](#4-performance-issues--parallelization)
5. [Performance Issues — Memory](#5-performance-issues--memory)
6. [Performance Issues — Algorithms](#6-performance-issues--algorithms)
7. [Architectural Mistakes](#7-architectural-mistakes)
8. [API Design Problems](#8-api-design-problems)
9. [Build System Issues](#9-build-system-issues)
10. [Test Suite Gaps](#10-test-suite-gaps)
11. [Benchmark Methodology Flaws](#11-benchmark-methodology-flaws)
12. [Missing Features](#12-missing-features)
13. [Lessons Learned](#13-lessons-learned)
14. [v2 Design Recommendations](#14-v2-design-recommendations)

---

## 1. Critical Bugs (Ship-Blocking)

### 1.1 Scalar Integer Division Gives Wrong Results

**File:** `src/math.c:922-940`

```c
const double recip = 1.0 / (double)*(const NUMC_INT *)scalar;
pout[i] = (NUMC_INT)((double)pa[i] * recip);
```

**Bug:** Reciprocal multiplication does not match C integer truncation semantics.
Example: `3 / 3` → `3 * 0.3333... = 0.9999...` → truncates to `0` instead of `1`.

Under `-ffast-math`, FMA contraction makes this even worse — `fma(a, 1.0/b, 0.0)`
can produce values slightly below the true quotient, changing the truncation direction.

**Impact:** `array_divide_scalar()` returns **wrong values** for INT and UINT types.
The binary division path (`div_INT`) correctly uses actual division — only the scalar
path has this bug because it attempted a reciprocal optimization.

**v2 fix:** Don't use reciprocal multiplication for integer division. Use actual
division. The performance gain is not worth the correctness loss.

---

### 1.2 `numc_type_is_unsigned()` Returns True for Signed Types

**File:** `include/numc/dtype.h:135-137`

```c
return numc_type >= NUMC_TYPE_UBYTE && numc_type <= NUMC_TYPE_ULONG;
```

**Bug:** Enum layout is BYTE(0), UBYTE(1), SHORT(2), USHORT(3), INT(4), UINT(5),
LONG(6), ULONG(7), FLOAT(8), DOUBLE(9). The range `[1, 7]` includes SHORT(2),
INT(4), and LONG(6) — all **signed** types. The function incorrectly reports them as
unsigned.

**Impact:** `array_arange()` rejects negative start/stop values for signed types
SHORT, INT, and LONG, treating them as unsigned.

**v2 fix:** Either reorder the enum to group unsigned types contiguously, or use
explicit checks:

```c
// Option A: Reorder enum — unsigned types contiguous
enum { BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, UBYTE, USHORT, UINT, ULONG };

// Option B: Explicit check
return type == NUMC_TYPE_UBYTE || type == NUMC_TYPE_USHORT ||
       type == NUMC_TYPE_UINT  || type == NUMC_TYPE_ULONG;

// Option C: Bitmask lookup
static const uint16_t unsigned_mask = (1<<UBYTE)|(1<<USHORT)|(1<<UINT)|(1<<ULONG);
return (unsigned_mask >> type) & 1;
```

---

### 1.3 `strncpy` Missing Null Terminator in Error System

**File:** `src/error.c:14`

```c
strncpy(error_msg, msg, MAX_ERROR_LEN);
```

**Bug:** When `strlen(msg) >= 256`, `strncpy` does not null-terminate. Any subsequent
read of `error_msg` as a C string is **undefined behavior** — reads past the buffer.

**v2 fix:**
```c
strncpy(error_msg, msg, MAX_ERROR_LEN - 1);
error_msg[MAX_ERROR_LEN - 1] = '\0';
// Or just use snprintf(error_msg, MAX_ERROR_LEN, "%s", msg);
```

---

### 1.4 Binary Op Does Not Check Output Contiguity

**File:** `src/math.c:481`

```c
const int contiguous = a->is_contiguous & b->is_contiguous;
// out->is_contiguous is NEVER checked
```

**Bug:** If `out` is a non-contiguous view (e.g., transposed), the kernel writes to
`out->data[0..n-1]` as if it were contiguous. This writes to the **wrong memory
locations** or out of bounds — undefined behavior.

Compare with `array_scalar_op` (line 1138) which correctly checks
`a->is_contiguous & out->is_contiguous`.

**Impact:** Silent data corruption when passing a strided view as the output array.

**v2 fix:** Always check all three arrays for contiguity. Or better: support strided
output (see Section 7.2).

---

### 1.5 CMakeLists.txt LTO References Targets Before Definition

**File:** `CMakeLists.txt:84-97`

```cmake
# Line 84-97: Sets LTO properties on targets
set_target_properties(numc_demo PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
set_target_properties(comprehensive_benchmark PROPERTIES ...)

# Line 100-134: Targets are actually DEFINED here
if(BUILD_EXAMPLES)
    add_executable(numc_demo examples/demo.c)
```

**Bug:** `numc_demo` and benchmark targets are referenced at lines 84-96 but defined
at line 101+. CMake configuration fails. If `BUILD_EXAMPLES=OFF`, targets never exist
and the LTO block unconditionally errors.

**v2 fix:** Move LTO properties inside the `if(BUILD_EXAMPLES)` block, after target
definition.

---

## 2. Correctness Bugs

### 2.1 Error State Data Race with OpenMP

**File:** `src/error.c:6-7`

```c
static char error_msg[MAX_ERROR_LEN];   // NOT _Thread_local
static int error_code = NUMC_OK;        // NOT _Thread_local
```

The library uses `omp parallel for` in math kernels. If any error triggers inside a
parallel region, or two threads call numc concurrently, both globals race — **UB** per
C11 5.1.2.4.

**v2 fix:** Use `_Thread_local` storage:
```c
static _Thread_local char error_msg[MAX_ERROR_LEN];
static _Thread_local int error_code = NUMC_OK;
```

---

### 2.2 `restrict` Aliasing UB in In-Place `astype`

**File:** `src/array_core.c:397-404, 856-868`

When narrowing conversion reuses the same buffer (`src == dst`):
```c
const src_type *restrict psrc = (const src_type *)src;  // points to array->data
dst_type *restrict pdst = (dst_type *)dst;               // ALSO points to array->data
```

Two `restrict` pointers aliasing the same memory is **UB** per C11 6.7.3.1. The
compiler may reorder loads/stores assuming no aliasing.

**Why it works today:** On x86 with current compilers, narrowing writes never clobber
future reads (forward iteration, smaller elements). But this is not guaranteed.

**v2 fix:** Remove `restrict` from astype kernels, or always allocate a separate
buffer when `src == dst`.

---

### 2.3 `const` Violations — Mutating Supposedly-Const Input

**Files:**
- `src/math.c` — `array_sum_axis` (line 1627): `array_ascontiguousarray((Array *)a)`
- `src/array_core.c` — `array_equal`, `array_allclose` (lines 904-914): same pattern

Functions accept `const Array *` but cast away const to make the input contiguous.
Callers expect their arrays to be unchanged. This silently mutates shared views.

**v2 fix:** Either: (a) make a contiguous copy internally without mutating input, or
(b) remove `const` from the API to be honest about mutation.

---

### 2.4 `array_offset` Returns 0 on Error

**File:** `src/array_core.c:646`

```c
if (!array || !indices) {
  numc_set_error(NUMC_ERR_NULL, "...");
  return NUMC_OK;   // NUMC_OK is 0 — a VALID byte offset!
}
```

Caller cannot distinguish "offset is 0 (first element)" from "error occurred."

**v2 fix:** Return `SIZE_MAX` as sentinel, or change API to return error code with
offset via output parameter.

---

### 2.5 No Error Clear Mechanism

**Files:** `src/error.c:10`, `include/numc/error.h`

`numc_set_error(NUMC_OK, ...)` returns early without clearing the old message or code.
There is no `numc_clear_error()` function. Error state is permanently "sticky" after
any failure — it can never be reset.

Also, `numc_get_error()` returns `char *` (mutable), allowing callers to corrupt the
internal error buffer. Should return `const char *`.

---

### 2.6 `std_along_axis` Missing NULL Check

**File:** `src/math.c:1515-1516`

```c
double *means = (double *)numc_malloc(NUMC_ALIGN, inner_size * sizeof(double));
// No NULL check — memcpy(NULL, ...) on next line if allocation fails
```

---

### 2.7 `numc_realloc` Leaks Memory on Edge Case

**File:** `src/alloc.c:72-77`

```c
if (ptr != NULL && old_size > 0) {
    ...
    numc_free(ptr);
}
```

If caller passes valid `ptr` with `old_size == 0`, the old pointer is never freed.

---

### 2.8 No Bounds Check on Dispatch Table Index

**Files:** `src/math.c:509, 1039, etc.`

If `a->numc_type` is corrupted (>= 10), `op_funcs[type]` is an out-of-bounds read.
No defensive `NUMC_TYPE_COUNT` sentinel exists.

---

## 3. Performance Issues — Vectorization

### 3.1 BYTE/UBYTE Vectorize Width Is Half What It Should Be

**File:** `src/math.c:246-247, 265-266, 284-285, 358-359`

```c
GENERATE_BINARY_OP_FUNC_SMALL(add, BYTE, NUMC_BYTE, +, 16)   // WRONG
// Should be 32: AVX2 YMM = 256 bits / 8 bits = 32 elements
```

**Impact:** 8 kernel functions (add/sub/mul for BYTE/UBYTE + div for BYTE/UBYTE)
running at **50% register utilization**. Using only 128-bit XMM operations instead of
256-bit YMM.

---

### 3.2 All Reduction Kernels Use Fixed `vectorize_width(8)` Regardless of Type

**File:** `src/math.c:555-769`

`GENERATE_SUM_FUNC`, `GENERATE_MIN_FUNC`, `GENERATE_MAX_FUNC`, `GENERATE_DOT_FUNC`,
`GENERATE_PROD_FUNC` all hardcode `vectorize_width(8)`.

| Type | Correct Width (AVX2) | Actual | Utilization |
|------|---------------------|--------|-------------|
| BYTE | 32 | 8 | **25%** |
| UBYTE | 32 | 8 | **25%** |
| SHORT | 16 | 8 | **50%** |
| USHORT | 16 | 8 | **50%** |
| INT/UINT/FLOAT | 8 | 8 | 100% |

**Impact:** 20 out of 35 reduction kernel instantiations (BYTE/UBYTE/SHORT/USHORT ×
5 ops) running at 25-50% of achievable SIMD throughput.

---

### 3.3 64-bit Dot Product Width Is Wrong

**File:** `src/math.c:718`

```c
NUMC_PRAGMA(clang loop vectorize_width(8) interleave_count(4))  // 64-bit dot
```

For 8-byte types, AVX2 fits **4** elements per register, not 8. Width 8 is impossible;
Clang silently falls back to a default. Compare with `GENERATE_SUM_FUNC_64BIT`
(line 580) which correctly uses `vectorize_width(4)`.

**Root cause:** Copy-pasted from 32-bit template without adjusting width.

---

### 3.4 `sum_to_double` Kernels Have Wrong Width

**File:** `src/math.c:1209`

The accumulator is `double` (8 bytes, 4 per YMM). Even when the input is BYTE (1 byte),
the accumulator width constrains vectorization to 4-wide, not 8. The `vectorize_width(8)`
hint is misleading and may cause Clang to emit suboptimal code.

---

### 3.5 Clang-Only Pragmas — GCC Gets No Vectorization Hints

**File:** `src/math.c` (throughout)

`NUMC_PRAGMA(clang loop vectorize_width(N) interleave_count(4))` is Clang-specific.
GCC silently ignores it. GCC builds lose:
- Explicit vectorization width control for reductions
- Interleave count hints for BYTE/SHORT binary ops
- The entire `GENERATE_BINARY_OP_FUNC_SMALL` optimization strategy

GCC has its own pragma syntax (`#pragma GCC ivdep`, `__attribute__((optimize(...)))`)
that is never used.

**v2 fix:** Compiler-dispatch macro:
```c
#if defined(__clang__)
  #define NUMC_SIMD_HINT(w) _Pragma(NUMC_STR(clang loop vectorize_width(w) interleave_count(4)))
#elif defined(__GNUC__)
  #define NUMC_SIMD_HINT(w) _Pragma("GCC ivdep")
#else
  #define NUMC_SIMD_HINT(w)
#endif
```

---

### 3.6 Vectorize Width Should Be Computed, Not Hardcoded

The root cause of issues 3.1-3.4 is manual width selection per callsite. A single
computed macro eliminates all these bugs:

```c
#define VEC_WIDTH(c_type) (NUMC_ALIGN / sizeof(c_type))
// BYTE: 32/1 = 32, SHORT: 32/2 = 16, INT: 32/4 = 8, DOUBLE: 32/8 = 4
```

Then use `VEC_WIDTH(c_type)` in all kernel templates. This automatically adapts to
element size and alignment guarantee. **Zero chance of mismatch.**

---

## 4. Performance Issues — Parallelization

### 4.1 ALL Reductions Are Single-Threaded

**File:** `src/math.c:555-769`

Sum, prod, min, max, and dot kernels use SIMD only — zero OpenMP. The comment at
line 549 correctly explains that OpenMP outlining kills vectorization.

**But this is a false dichotomy.** The solution is tiled parallelism:

```
┌─────────────────────────────────────────────┐
│              Large Array (100M elements)     │
├──────────┬──────────┬──────────┬────────────┤
│  Tile 0  │  Tile 1  │  Tile 2  │  Tile 3   │  ← OMP parallel for
│  (64K)   │  (64K)   │  (64K)   │  (64K)    │
│ SIMD sum │ SIMD sum │ SIMD sum │ SIMD sum  │  ← Vectorized inner loop
│ partial₀ │ partial₁ │ partial₂ │ partial₃  │
└──────────┴──────────┴──────────┴────────────┘
              │  Final reduce (4 values)  │
              └───────────────────────────┘
```

Each thread gets a tile that fits in L1/L2 cache. The inner loop is fully vectorized
(no OpenMP outlining). The final reduction over partial results is negligible.

**Impact:** For arrays exceeding L3 cache (>~10M floats), single-threaded reductions
are memory-bandwidth-bound on a single memory controller. Multi-threaded reductions
can saturate all memory channels. Expected **4-8x speedup** on typical 4-8 core
systems.

---

### 4.2 Axis Reduction Outer Loop Not Parallelized

**File:** `src/math.c:1419, 1426, 1458, 1466, 1501, 1518`

```c
for (size_t o = 0; o < outer_size; o++) {  // SERIAL
    accum_kernel(src + o * stride, out + o * inner_bytes, inner_size);
}
```

For shape `[1000, 50, 100]` reduced along axis=1: `outer_size=1000` and each iteration
is completely independent. Adding `#pragma omp parallel for` when
`outer_size > threshold` would give near-linear scaling.

---

### 4.3 Scalar Ops on Narrow Types Miss the SMALL Template

**File:** `src/math.c:843-907`

Binary ops have a specialized `GENERATE_BINARY_OP_FUNC_SMALL` template for BYTE/SHORT
with explicit vectorization hints and a higher OMP threshold (16MB). But scalar ops use
the standard `GENERATE_SCALAR_OP_FUNC` template for ALL types — BYTE/SHORT get OMP
outlining overhead for 100K elements (100KB), where the SMALL binary template would
wait until 16MB.

**Inconsistency:** `array_add(byte_a, byte_b, out)` is faster than
`array_add_scalar(byte_a, &val, out)` for the same array size, despite scalar ops
being inherently simpler.

---

### 4.4 `NUMC_OMP_FOR` Doubles Code Size

**File:** `src/internal.h:114-120`

The `if/else` pattern expands the loop body twice. With ~80 call sites, this adds
~11.5KB of redundant `.text`. Not critical for I-cache (32KB L1i), but wasteful.

**v2 alternative:** Use a function pointer approach or `__attribute__((noinline))` on
the OMP path to share code.

---

## 5. Performance Issues — Memory

### 5.1 `array_zeros` Wastes Write Bandwidth

**File:** `src/array_core.c:237-238, 243-253`

```c
// array_create with data=NULL calls array_empty (numc_malloc) then:
memset(array->data, 0, array->size * elem_size);
```

For allocations above glibc's mmap threshold (~128KB), the OS provides zero-mapped
pages for free. `calloc()` exploits this — the kernel maps zero pages that fault on
first write. `memset` forces a **full physical write pass** that is pure waste.

**Impact:** Creating a 400MB zero array writes 400MB to memory before the user ever
touches it. With `calloc`, the cost is deferred to actual use.

**v2 fix:** Use `numc_calloc` for zero-filled allocations. Implement `numc_calloc` as
`calloc` with alignment (or `mmap` + `madvise` for very large allocations).

---

### 5.2 Concatenation Zeroes Then Overwrites

**File:** `src/array_shape.c:334`

```c
Array *result = array_create(&result_create);  // .data=NULL → memset(0)
// Then immediately overwrites everything with memcpy (fast path) or strided copy
```

For 1GB concatenation: wastes 1GB of write bandwidth on zeroing.

**v2 fix:** Use `array_empty()` instead of `array_create()`.

---

### 5.3 Wasted Allocation in Non-Owning `array_create`

**File:** `src/array_core.c:228-233`

```c
if (!src->owns_data) {
  numc_free(array->data);          // Frees buffer just allocated by array_empty
  array->data = (void *)src->data;
```

One wasted `aligned_alloc` + `free` pair per view creation through this path.

**v2 fix:** Check `owns_data` before allocating.

---

### 5.4 No Arena/Pool Allocator for Temporaries

Every math operation calls `aligned_alloc` → kernel syscall for the output buffer.
In expression chains like `c = a + b; d = c * e;`, intermediate `c` is allocated
and freed immediately. A thread-local arena (pointer-bump allocator) would make
temporaries nearly free and improve cache locality.

---

### 5.5 No In-Place Operations

The API always requires pre-allocated output: `array_add(a, b, out)`. There are no
in-place variants (`a += b`). This doubles memory bandwidth for accumulation patterns:

```
array_add(a, b, out):   read a + read b + write out  (3 streams)
array_iadd(a, b):       read a + read b + write a    (2 streams, a is hot in cache)
```

---

### 5.6 Struct Padding Waste

**File:** `include/numc/array.h:38-51`

`numc_type` (4-byte enum) between 8-byte fields creates 4 bytes of padding. Two `bool`
fields before 8-byte-aligned buffers create 6 bytes of padding. ~10 bytes wasted per
Array. Minor, but easily fixed by grouping same-sized fields:

```c
// Group 8-byte fields together, then 4-byte, then 1-byte
typedef struct {
  void *data;
  size_t *shape, *strides;
  size_t ndim, elem_size, size, capacity;
  size_t _shape_buff[8], _strides_buff[8];
  NUMC_TYPE numc_type;          // 4 bytes
  bool is_contiguous, owns_data; // 2 bytes + 2 padding
} Array;  // saves 8 bytes
```

---

## 6. Performance Issues — Algorithms

### 6.1 Strided Copy Is O(n × ndim)

**File:** `src/array_core.c:562-579`

The general `strided_to_contiguous_copy_general` computes full offsets per element:

```c
for (size_t i = 0; i < count; i++) {
  size_t offset = 0;
  for (size_t d = 0; d < ndim; d++)     // O(ndim) per element
    offset += indices[d] * strides[d];
  memcpy(dst, src + offset, elem_size);
  increment_indices(...);                // Another O(ndim) amortized
}
```

For a 4D array with 10M elements: ~80M arithmetic operations just for addressing.

**v2 fix:** Flat iterator with carry propagation — amortized O(1) per element:

```c
typedef struct {
  char *ptr;
  size_t coords[MAX_NDIM];
  ssize_t backstrides[MAX_NDIM]; // strides[d] - shape[d]*strides[d+1]
  int ndim;
} FlatIter;

static inline void iter_next(FlatIter *it) {
  it->ptr += it->strides[it->ndim - 1];       // Fast: inner dim advance
  if (++it->coords[it->ndim - 1] < shape)     // Fast: no carry 99% of time
    return;
  // Carry propagation (rare — once per inner dimension exhaust)
  for (int d = it->ndim - 1; d >= 0; d--) {
    it->ptr += it->backstrides[d];
    it->coords[d] = 0;
    if (d > 0 && ++it->coords[d-1] < shape[d-1]) break;
  }
}
```

```c
char *ptr = arr.data;

size_t coords[MAX_NDIM] = {0};

while (true) {

    // use current element
    printf("%d\n", *(int*)ptr);

    // 1️⃣ move along inner-most dimension
    int d = arr.ndim - 1;
    ptr += arr.strides[d];
    coords[d]++;

    // 2️⃣ if no overflow → continue fast
    if (coords[d] < arr.shape[d])
        continue;

    // 3️⃣ carry propagation
    while (d >= 0) {

        coords[d] = 0;

        if (d == 0)
            return; // finished entire tensor

        d--;

        coords[d]++;

        // pointer correction
        ptr += arr.strides[d]
             - arr.shape[d+1] * arr.strides[d+1];

        if (coords[d] < arr.shape[d])
            break;
    }
}

```

```c
void array_fill_int(NDArray *a, int value) {
    char *ptr = a->data;

    size_t total = 1;
    for (size_t i = 0; i < a->ndim; ++i)
        total *= a->shape[i];

    for (size_t i = 0; i < total; ++i) {
        *(int*)ptr = value;
        ptr += a->strides[a->ndim - 1];
    }
}

```

---

### 6.2 No N-D Inner-Contiguous Copy Fast Path

**File:** `src/array_core.c:635`

For ndim >= 3, even when `strides[ndim-1] == elem_size` (innermost dimension is
contiguous), the code falls to element-by-element copy.

**Example:** 3D array `[100, 200, 300]` of float, transposed on axes 0,1:
- General path: 6,000,000 individual `memcpy(dst, src, 4)` calls
- Inner-contiguous path: 20,000 `memcpy(dst, src, 1200)` calls
- That is **300x fewer function calls**

**v2 fix:** Detect contiguous inner dimensions and batch-copy them:

```c
// Find how many trailing dimensions are contiguous
size_t inner_bytes = elem_size;
int inner_dims = 0;
for (int d = ndim - 1; d >= 0; d--) {
  if (strides[d] != inner_bytes) break;
  inner_bytes *= shape[d];
  inner_dims++;
}
// Copy inner_bytes at a time, iterate over outer dims only
```

---

### 6.3 No Pairwise or Kahan Summation

`sum_FLOAT` uses naive sequential accumulation. For 10M+ elements, the accumulated
value grows large enough that small additions fall below the unit of least precision
(ULP) and are silently dropped.

**Precision comparison for summing 10M random floats in [0,1]:**

| Algorithm | Relative Error |
|-----------|---------------|
| Naive sequential | O(n × epsilon) ≈ 1.2e-3 |
| Pairwise (NumPy) | O(log(n) × epsilon) ≈ 2.8e-6 |
| Kahan compensated | O(epsilon) ≈ 1.2e-7 |

NumPy uses pairwise summation. It adds minimal overhead and is still SIMD-friendly
(recursive halving naturally creates independent accumulator chains).

**Note:** `array_mean` uses a double accumulator (`sum_to_double`) which is much
better, but `array_sum` on float arrays has poor precision with no documentation.

---

### 6.4 Integer Reduction Overflow

**File:** `src/math.c:555-569`

`sum_BYTE` uses `int8_t acc = 0`. Summing 128 positive bytes overflows. With `-fwrapv`,
signed overflow wraps (defined behavior), but the mathematical result is **wrong**.
Same for all narrow types.

NumPy also wraps (documented behavior), but a widened accumulator would produce correct
results for arrays up to ~2B elements at negligible cost.

---

### 6.5 Axis Reduction Function Call Overhead

**File:** `src/math.c:1422-1434`

For shape `[1000000, 3]` reduced along axis=0: calls `accum_sum_FLOAT(ptr, out, 3)`
one million times. Each call goes through function pointer dispatch. With `n=3`,
the vectorization hint is ignored (3 < SIMD width). Function call overhead (~5ns)
exceeds kernel work (~2ns for 3 scalar adds).

The accum kernels are `static` (not `static inline`), and called through function
pointer tables, so they cannot be inlined.

**v2 fix:** For small `inner_size`, use a tiled kernel that processes multiple axis
elements per call. Or transpose the data so the reduction axis is last, then use the
fast full-reduction path.

---

### 6.6 Math Ops Require Contiguous — Forced Copy

All binary ops, scalar ops, and reductions require contiguous arrays. Users must call
`array_ascontiguousarray()` first, which means a **full array copy** before any
computation.

For transposed or sliced arrays, this means: copy to contiguous → operate → result.
If the library supported strided kernels, it could operate directly on views:

```c
// Inner-contiguous fast path (covers 90% of real use cases)
for (size_t outer = 0; outer < outer_size; outer++) {
  float *a_row = (float*)(a_base + outer * a_outer_stride);
  float *b_row = (float*)(b_base + outer * b_outer_stride);
  float *o_row = (float*)(o_base + outer * o_outer_stride);
  // This inner loop is still fully SIMD-vectorizable!
  for (size_t i = 0; i < inner_size; i++)
    o_row[i] = a_row[i] + b_row[i];
}
```

---

## 7. Architectural Mistakes

### 7.1 No Expression Fusion

The biggest performance killer in NumPy-style libraries is **temporary materialization**.

```c
// User wants: d = (a + b) * c
Array *tmp = array_empty(...);
array_add(a, b, tmp);         // Full array write to tmp
Array *d = array_empty(...);
array_multiply(tmp, c, d);    // Full array read of tmp + write to d
array_free(tmp);
```

This creates a full temporary for `a + b`. For a 100M-element array, that is 400MB of
extra memory traffic. With kernel fusion:

```c
// Fused: out[i] = (a[i] + b[i]) * c[i]  — single pass
array_fma_custom(a, b, c, d);  // One read of a,b,c + one write of d
```

**v2 approach:** At minimum, provide BLAS-style fused kernels:
- `array_fma(a, b, c, out)` — `out = a * b + c` (uses hardware FMA)
- `array_axpy(alpha, x, y, out)` — `out = alpha * x + y`
- `array_addcmul(a, b, c, out)` — `out = a + b * c`

For full generality: a lightweight expression tree with lazy evaluation (more complex
but eliminates all temporaries).

---

### 7.2 No Strided Kernel Support

The library only operates on contiguous arrays. Any non-contiguous input requires
a copy first. This is the #1 user-facing performance issue — real workloads involve
slices and transposes constantly.

**v2 design:** Every kernel should have two paths:
1. **Contiguous fast path** — current behavior, full SIMD
2. **Inner-contiguous path** — outer loop over non-contiguous dims, SIMD inner loop
3. **General strided path** — flat iterator (fallback, still correct)

---

### 7.3 No Runtime ISA Dispatch

The library compiles with `-march=native`, meaning the binary only runs on the build
machine's CPU. No runtime detection of AVX2 vs AVX-512 vs SSE4.2.

**v2 approach:** Use `__builtin_cpu_supports()` or CPUID to dispatch at init:

```c
void numc_init(void) {
  if (__builtin_cpu_supports("avx512f"))
    kernel_table = avx512_kernels;
  else if (__builtin_cpu_supports("avx2"))
    kernel_table = avx2_kernels;
  else
    kernel_table = sse2_kernels;
}
```

Or use GCC/Clang's `__attribute__((target_clones("avx512f","avx2","default")))`.

---

### 7.4 `numc_type_sizes[]` Duplicated Across Translation Units

**File:** `include/numc/dtype.h:115-126`

```c
static const size_t numc_type_sizes[] = { ... };
```

`static const` in a header means every `.c` file that includes it gets its own copy
in `.rodata`. With ~29 source files plus downstream consumers, this wastes space.

**v2 fix:** `extern const` in header, single definition in `dtype.c`.

---

### 7.5 No BLAS Backend for Matrix Multiply

For matrix multiplication, auto-vectorization achieves ~2% of peak FLOPS. A tiled
micro-kernel gets ~30-50%. OpenBLAS/MKL gets ~95%. This is the single highest-impact
missing feature for any numerical computing library.

**v2 approach:**
```c
#ifdef NUMC_HAS_BLAS
  cblas_sgemm(CblasRowMajor, ...);
#else
  // Fallback: tiled micro-kernel with register blocking
  numc_sgemm_fallback(M, N, K, ...);
#endif
```

---

## 8. API Design Problems

### 8.1 `array_arange` Takes `int` Parameters

**File:** `include/numc/array.h:124`

```c
Array *array_arange(int start, int stop, int step, NUMC_TYPE type);
```

The library supports `int64_t` (LONG) and `double` (DOUBLE), but range parameters
are `int`. Cannot express `arange(0.0, 1.0, 0.1)` for float arrays.

---

### 8.2 No `array_var()` Public API

Internal `var_funcs` exist (used by `array_std`), but `array_var()` is not exposed
in the public API. Users who want variance must compute `std²`.

---

### 8.3 Hand-Written Accessor Functions

**File:** `include/numc/array.h:202-316`

10 nearly identical `array_get_*` / `array_set_*` inline functions for each type.
These should be X-macro generated to reduce maintenance burden and ensure consistency.

---

### 8.4 No Broadcasting

Binary ops require **exact** shape match. NumPy's broadcasting rules (e.g., adding
`[3,4]` + `[4]`) are not supported. This is the most-requested NumPy feature in
array libraries.

---

## 9. Build System Issues

### 9.1 Compiler Lock in Cache

**File:** `CMakeLists.txt:4`

```cmake
set(CMAKE_C_COMPILER "clang" CACHE STRING "C compiler")
```

Locks the compiler in CMake cache. `CC=gcc ./run.sh release` only works on first
configure or after clean. Should use CMake's standard compiler detection.

---

### 9.2 Overrides User Flags

**File:** `CMakeLists.txt:30-31`

```cmake
set(CMAKE_C_FLAGS_DEBUG "...")    # Overrides, doesn't append
set(CMAKE_C_FLAGS_RELEASE "...")
```

CMake best practice is `add_compile_options()` with generator expressions, which
appends to rather than replacing user-provided flags.

---

### 9.3 ASan Compile-Only for Library Target

Debug ASan is a compile flag on the library, but no link flag. Works for static builds
(test binaries add their own link flag), but shared builds (`BUILD_SHARED=ON`) would
fail to link ASan symbols.

---

### 9.4 Redundant Flag

`-fno-math-errno` is already implied by `-ffast-math`. Harmless but cluttered.

---

### 9.5 No Parallel Build

**File:** `run.sh`

`cmake --build build` runs serial. Should use `cmake --build build -j$(nproc)`.

---

### 9.6 Linux-Only `date` Format

**File:** `run.sh:11,15,127,130`

`date +%s%3N` is a GNU extension. macOS `date` does not support `%N`.

---

## 10. Test Suite Gaps

### 10.1 False Positive Tests (Check Return Code Only, Not Values)

These tests pass even if the kernel produces garbage output:

| Test File | Function | What It Misses |
|-----------|----------|----------------|
| `test_add.c` | `test_add_all_numc_types()` | Never checks `1+1=2` |
| `test_add.c` | `test_all_ops_with_numc_types()` | 4 ops × 10 types, no value check |
| `test_reduce.c` | `test_reduce_all_numc_types()` | Never checks sum of 5 ones = 5 |
| `test_scalar.c` | `test_scalar_all_numc_types()` | Never checks `ones+1=2` |
| `test_flatten.c` | `test_flatten_all_types()` | Only checks shape, not data |

**Impact:** A broken kernel for any type would go undetected by these tests.

---

### 10.2 Zero Tests for `array_index_axis()`

Declared in `array.h:377`, never tested. Unknown if it even works correctly.

---

### 10.3 Non-Contiguous Input to Math Ops Not Tested

No test verifies that passing a strided view to `array_add()` returns an error (or
silently corrupts — see bug 1.4). The error path for contiguity rejection is
completely untested.

---

### 10.4 Missing Edge Case Coverage

| Edge Case | Tested? |
|-----------|---------|
| Empty arrays (size 0) | No |
| 0-dimensional arrays (ndim=0) | No |
| Single-element binary ops | No |
| Single-element reductions | Partial (prod only) |
| ndim > 8 (heap-allocated shape/strides) | No |
| ndim = 8 boundary | No |
| Division by zero | No |
| Integer overflow in reductions | No |
| Duplicate axes in transpose | No |
| Out-of-bounds slice parameters | No |
| Mismatched output shape in binary ops | No |

---

### 10.5 No Test Timeout

**File:** `tests/CMakeLists.txt`

No CTest timeout configured. A hanging test blocks CI indefinitely.

---

### 10.6 `array_ascontiguousarray()` and `array_print()` Have No Tests

Only indirectly exercised. If they break, no test catches it.

---

## 11. Benchmark Methodology Flaws

### 11.1 No Warmup in Comprehensive Benchmark

The per-operation benchmarks have 10-iteration warmup. The comprehensive benchmark
jumps straight to timed loops. First iterations include cold cache, lazy library
loading, and OpenMP thread pool initialization.

---

### 11.2 Inconsistent `volatile` Usage

| File | Uses volatile? | Risk |
|------|---------------|------|
| comprehensive_benchmark.c | Yes | Safe |
| benchmark_reductions.c | Yes | Safe |
| benchmark_add.c | **No** | DCE risk with LTO |
| benchmark_subtract.c | **No** | DCE risk with LTO |
| benchmark_multiply.c | **No** | DCE risk with LTO |
| benchmark_divide.c | **No** | DCE risk with LTO |
| benchmark_sum.c | **No** | DCE risk with LTO |
| benchmark_min.c | **No** | DCE risk with LTO |
| benchmark_max.c | **No** | DCE risk with LTO |
| benchmark_scalar.c | **No** | DCE risk with LTO |

With LTO enabled in release builds, the compiler could theoretically eliminate the
benchmark loop if it can prove the function has no observable side effects.

---

### 11.3 No Statistical Analysis

No standard deviation, confidence intervals, or outlier rejection. A single GC or
scheduling hiccup during the 100 iterations skews results noticeably for small arrays.

---

### 11.4 Uniform Data (`array_ones`)

All benchmarks use arrays filled with `1.0`. This doesn't exercise:
- Branch prediction with varied data
- Min/max with different data patterns (sorted, random, reverse)
- Division edge cases (small/large divisors)
- Data-dependent SIMD performance characteristics

---

### 11.5 Missing Benchmarks

Not benchmarked: `array_create`, `array_copy`, `array_slice`, `array_reshape`,
`array_transpose`, `array_flatten`, `array_astype`, `array_concatenate`,
`array_equal`, `array_allclose`, `array_ascontiguousarray`, `array_sum_axis`,
`array_prod_axis`, `array_min_axis`, `array_max_axis`.

---

### 11.6 No Non-Contiguous Benchmarks

All benchmarks create contiguous arrays. No benchmark measures strided operation
performance or the cost of contiguity conversion.

---

### 11.7 "Main Memory" Size May Still Fit in L3

16 MB for FLOAT may not exceed L3 on modern CPUs (25-36 MB L3). The 3-array working
set (48 MB) likely does exceed it, but this should be explicitly verified.

---

## 12. Missing Features

### High Priority (Expected in Any NumPy-Like Library)

| Feature | Notes |
|---------|-------|
| Broadcasting | Shape `[3,4] + [4]` should work |
| `matmul` / GEMM | Essential; needs BLAS backend |
| Unary math: `abs`, `neg`, `sqrt`, `exp`, `log` | Trivial to add with X-macros |
| `clip` (clamp) | Single-pass `min(max(x, lo), hi)` |
| `argmin` / `argmax` | Returns index, not value |
| `where` (conditional select) | `out[i] = cond[i] ? a[i] : b[i]` |
| In-place ops (`iadd`, `isub`, ...) | Halves memory bandwidth |

### Medium Priority

| Feature | Notes |
|---------|-------|
| `cumsum` / `cumprod` | Prefix operations (not trivially parallelizable) |
| `sort` / `argsort` | Radix sort for ints, merge sort for floats |
| `floor` / `ceil` / `round` | Float → float rounding |
| `array_var()` public API | Already implemented internally |
| Comparison ops → boolean output | `greater`, `less`, `equal` element-wise |
| `linspace` with float params | Current API is `int`-only |

### Low Priority

| Feature | Notes |
|---------|-------|
| `median` / `percentile` | Requires partial sort |
| `convolve` / `correlate` | 1D signal processing |
| Linear algebra (LU, QR, SVD) | Delegate to LAPACK |
| `einsum` | Generalized tensor contraction |

---

## 13. Lessons Learned

### 13.1 Auto-Vectorization Beats Hand-Written Intrinsics (For This Codebase)

Explicit AVX2 intrinsics in a separate translation unit performed **worse** than
auto-vectorization with `-O3 -march=native`. Root cause: cross-TU call overhead
prevented inlining, and LTO didn't fully recover the loss.

**Lesson:** Keep kernels as simple typed loops with alignment hints. Let the compiler
vectorize. Focus effort on removing obstacles to auto-vectorization (aliasing, loop
structure, accumulator patterns) rather than writing intrinsics.

**Exception:** Matrix multiplication. GEMM requires register tiling, cache blocking,
and micro-kernel design that no compiler can auto-generate. Use BLAS.

---

### 13.2 Hardcoded Constants Rot — Compute Them

Every hardcoded `vectorize_width(8)` or `vectorize_width(16)` was a copy-paste bug
waiting to happen (and it did — 3.1, 3.2, 3.3, 3.4). A single `(NUMC_ALIGN / sizeof(type))`
eliminates the entire class of bugs.

**Lesson:** If a value depends on another value, express that dependency in code.
Never hardcode derived constants.

---

### 13.3 Test Values, Not Just Return Codes

Five "all types" tests check `ret == 0` but never verify the computed output. They
provide zero correctness coverage. A test that doesn't check the answer is
**worse than no test** — it gives false confidence.

**Lesson:** Every test must assert on the actual output value. For type-generic
tests, use a known computation (e.g., `ones + ones = twos`) and verify all elements.

---

### 13.4 The Copy-Then-Operate Pattern Is a Performance Tax

Requiring contiguous arrays forces users into a copy-then-operate pattern for any
non-trivial workflow. The copy itself is often as expensive as the operation.

**Lesson:** Support at least inner-contiguous strided operations from day one.
The inner loop is still SIMD-vectorizable; only the outer loop changes.

---

### 13.5 `static const` in Headers Causes Duplication

`numc_type_sizes[]` in `dtype.h` is duplicated in every TU. With LTO it may merge,
but without LTO it bloats every object file.

**Lesson:** Use `extern const` in headers, define once in a `.c` file. Or use
`inline` functions that return constants (C11 `static inline` is fine).

---

### 13.6 Reciprocal Multiplication Is Not Safe for Integer Division

The reciprocal optimization `a * (1.0/b)` does not produce the same truncation as
`a / b` for integers. Under `-ffast-math` with FMA, the error grows. This bug shipped
because no test checked actual division results for edge cases.

**Lesson:** Integer division must use actual division. The ~5x throughput gain of
`vmulps` over `vdivps` is not worth incorrect results. If you want fast integer
division by a constant, use the well-known multiply-and-shift technique
(libdivide approach), which is exact.

---

### 13.7 Thread-Safety Is Not Optional When You Use OpenMP

Using `omp parallel for` makes the library multi-threaded. All shared mutable state
(error globals) must be thread-safe. `_Thread_local` is the simplest fix.

---

## 14. v2 Design Recommendations

### 14.1 Day-One Priorities (Do These First)

```
1. Computed vectorize widths         — eliminates 3.1, 3.2, 3.3, 3.4
2. Compiler-portable SIMD hints      — eliminates 3.5
3. _Thread_local error state         — eliminates 2.1
4. Correct integer division          — eliminates 1.1
5. Contiguity check on ALL operands  — eliminates 1.4
6. Value-checking tests for all types — eliminates 10.1
7. Inner-contiguous strided paths    — eliminates 6.2, 6.6
8. Tiled parallel reductions         — eliminates 4.1
```

### 14.2 Architecture Blueprint

```
┌─────────────────────────────────────────────────────┐
│                   Public API Layer                   │
│  array_add(a, b, out)  array_sum(a, &result)        │
│  Validates inputs, selects dispatch strategy          │
├─────────────────────────────────────────────────────┤
│                  Dispatch Layer                       │
│  1. Check contiguity of all operands                 │
│  2. Select: contiguous / inner-contiguous / general  │
│  3. Select: serial / tiled-parallel                  │
│  4. Lookup type-specific kernel                      │
├─────────────────────────────────────────────────────┤
│                  Kernel Layer                         │
│  Generated via X-macros with computed widths          │
│  ┌──────────────────────────────────────────────┐   │
│  │ Contiguous:     for(i) out[i] = a[i] + b[i] │   │
│  │ Inner-contig:   for(o) for(i) row[i] OP= ... │   │
│  │ General:        flat_iter_next() per element   │   │
│  └──────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│              Memory Management Layer                  │
│  Arena allocator for temporaries                      │
│  numc_calloc for zero-filled (OS zero-page exploit)  │
│  32-byte aligned for AVX2, 64-byte option for AVX512 │
├─────────────────────────────────────────────────────┤
│                Iterator Layer                         │
│  FlatIter: amortized O(1) per element                │
│  Carries coords, backstrides for efficient traversal  │
│  Powers: strided copy, strided kernels, print         │
└─────────────────────────────────────────────────────┘
```

### 14.3 Kernel Template Design

```c
// Single macro computes correct width for any type + alignment
#define VEC_WIDTH(c_type) (NUMC_ALIGN / sizeof(c_type))

// Compiler-portable SIMD hint
#if defined(__clang__)
  #define SIMD_HINT(c_type) \
    _Pragma(STRINGIFY(clang loop vectorize_width(VEC_WIDTH(c_type)) interleave_count(4)))
#elif defined(__GNUC__)
  #define SIMD_HINT(c_type) _Pragma("GCC ivdep")
#else
  #define SIMD_HINT(c_type)
#endif

// Universal kernel template — works for all types automatically
#define GENERATE_BINARY_KERNEL(op_name, NAME, C_TYPE, OP)          \
  static void op_name##_##NAME(const void *a, const void *b,       \
                                void *out, size_t n) {              \
    const C_TYPE *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN); \
    const C_TYPE *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN); \
    C_TYPE *restrict po = __builtin_assume_aligned(out, NUMC_ALIGN);     \
    SIMD_HINT(C_TYPE)                                               \
    for (size_t i = 0; i < n; i++)                                  \
      po[i] = pa[i] OP pb[i];                                      \
  }
```

### 14.4 Test Template

```c
// Every "all types" test MUST verify computed values
#define TEST_BINARY_OP_TYPE(op_func, a_val, b_val, expected, NAME, C_TYPE)  \
  do {                                                                       \
    Array *a = array_full(1, (size_t[]){1024}, NUMC_TYPE_##NAME, &(C_TYPE){a_val}); \
    Array *b = array_full(1, (size_t[]){1024}, NUMC_TYPE_##NAME, &(C_TYPE){b_val}); \
    Array *c = array_zeros(1, (size_t[]){1024}, NUMC_TYPE_##NAME);          \
    assert(op_func(a, b, c) == NUMC_OK);                                    \
    for (size_t i = 0; i < 1024; i++) {                                      \
      C_TYPE val;                                                             \
      memcpy(&val, (char*)c->data + i * sizeof(C_TYPE), sizeof(C_TYPE));     \
      assert(val == (C_TYPE)(expected));  /* VERIFY THE VALUE */              \
    }                                                                         \
    array_free(a); array_free(b); array_free(c);                             \
  } while(0)
```

### 14.5 Performance Checklist

Before shipping any kernel, verify:

- [ ] Vectorize width matches `NUMC_ALIGN / sizeof(element_type)`
- [ ] All operands (including output) checked for contiguity
- [ ] OpenMP threshold appropriate for element size (not just count)
- [ ] Reductions use tiled parallel approach for large arrays
- [ ] No wasted `memset` before overwrite
- [ ] `restrict` only on truly non-aliasing pointers
- [ ] Works on both Clang and GCC (test with both)
- [ ] Benchmark with varied data patterns, not just `array_ones`
- [ ] Test verifies actual computed values, not just return code

---

*Generated from exhaustive audit of numc v1 codebase — 2025-02-10*
*Total issues found: 12 bugs (4 critical), 18 performance issues, 11 test gaps*
