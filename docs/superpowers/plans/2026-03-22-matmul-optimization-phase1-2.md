# Matmul Optimization Phase 1-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add storage-aware dispatch, integer GEMMSUP kernels for all 8 integer dtypes, and hand-tuned AVX2 ASM micro-kernels for all 10 dtypes to numc's matmul.

**Architecture:** Phase 1 adds transpose detection in the matmul dispatch path and unpacked GEMMSUP kernels for integer types, following the existing f32/f64 GEMMSUP pattern in `gemmsup_avx2.h`. Phase 2 adds 8 new `.S` assembly micro-kernels following the existing `gemm_ukernel_f32_6x16_avx2.S` pattern, then wires them into the packed GEMM dispatch in `gemm_avx2.h`.

**Tech Stack:** C23, x86-64 GNU assembler (AT&T syntax), AVX2 intrinsics, CMake

**Spec:** `docs/superpowers/specs/2026-03-22-matmul-optimization-design.md`

**Deferred to Phase 3 plan:** Transposed packing variants (`_pack_a_t`, `_pack_b_t`) and GEMMSUP `csb` parameter. Storage detection infrastructure is added now; the actual optimized transposed packing functions will be implemented alongside the fused pack+compute work in Phase 3, since both affect the same packing code paths. The existing packed GEMM already handles arbitrary strides via its stride-aware packing routines, so transposed inputs work correctly — just without the packing speed optimization.

**Baseline benchmark (pre-optimization):**
- f32 1024²: 4047μs (1.01x vs NumPy), peak 702 GFLOPS at 512²
- f64 1024²: 8353μs (1.06x), one regression at 512² (0.94x)
- i16/u16: 28x-733x vs NumPy, peak 688 GFLOPS
- i8/u8: 4.6x-17x, peak 33 GFLOPS (20x gap vs f32 — main optimization target)
- i64/u64: 2.5x-64x, peak 48 GFLOPS

---

## File Structure

### Phase 1 files

| File | Action | Responsibility |
|------|--------|---------------|
| `src/matmul/dispatch.h` | Modify | Add `_detect_storage()` helper |
| `src/intrinsics/gemmsup_avx2.h` | Modify | Add 8 integer GEMMSUP kernels (i32, u32, i16, u16, i64, u64, i8, u8) |
| `src/matmul/matmul.c` | Modify | Wire integer GEMMSUP dispatch + per-dtype thresholds + storage detection |
| `tests/matmul/test_matmul.c` | Modify | Add integer dtype matmul tests + transposed input tests |

### Phase 2 files

| File | Action | Responsibility |
|------|--------|---------------|
| `src/intrinsics/gemm_ukernel_i32_6x16_avx2.S` | Create | ASM micro-kernel for i32 packed GEMM |
| `src/intrinsics/gemm_ukernel_u32_6x16_avx2.S` | Create | ASM micro-kernel for u32 packed GEMM |
| `src/intrinsics/gemm_ukernel_i16_6x32_avx2.S` | Create | ASM micro-kernel for i16 packed GEMM |
| `src/intrinsics/gemm_ukernel_u16_6x32_avx2.S` | Create | ASM micro-kernel for u16 packed GEMM |
| `src/intrinsics/gemm_ukernel_i64_6x8_avx2.S` | Create | ASM micro-kernel for i64 packed GEMM |
| `src/intrinsics/gemm_ukernel_u64_6x8_avx2.S` | Create | ASM micro-kernel for u64 packed GEMM |
| `src/intrinsics/gemm_ukernel_i8_6x16_avx2.S` | Create | ASM micro-kernel for i8 packed GEMM (i32 accumulation) |
| `src/intrinsics/gemm_ukernel_u8_6x16_avx2.S` | Create | ASM micro-kernel for u8 packed GEMM (u32 accumulation) |
| `src/intrinsics/gemm_avx2.h` | Modify | Declare extern ASM symbols, wire into packed GEMM dispatch |
| `src/CMakeLists.txt` | Modify | Add 8 new `.S` files to `NUMC_CORE_SOURCES` |

---

## Task 1: Add Storage Detection to dispatch.h

**Files:**
- Modify: `src/matmul/dispatch.h`

- [ ] **Step 1: Add `_detect_storage()` helper**

Add after the existing `_check_matmul()` function in `src/matmul/dispatch.h`:

```c
typedef enum {
  NUMC_STORAGE_ROW, /* strides[1] == elem_size (C-contiguous rows) */
  NUMC_STORAGE_COL, /* strides[0] == elem_size (Fortran-contiguous) */
  NUMC_STORAGE_GEN, /* general strided */
} NumcStorageOrder;

/**
 * @brief Detect storage order of a 2D array.
 */
static inline NumcStorageOrder _detect_storage(const struct NumcArray *a) {
  if (a->strides[1] == a->elem_size)
    return NUMC_STORAGE_ROW;
  if (a->strides[0] == a->elem_size)
    return NUMC_STORAGE_COL;
  return NUMC_STORAGE_GEN;
}
```

- [ ] **Step 2: Build to verify no compilation errors**

Run: `./run.sh release`
Expected: Build succeeds (no usage of the new function yet)

- [ ] **Step 3: Commit**

```bash
git add src/matmul/dispatch.h
git commit -m "feat(matmul): add storage order detection helper"
```

---

## Task 2: Add Integer GEMMSUP Kernels — i32/u32

These follow the exact same pattern as the existing `gemmsup_ukernel_f32_6x16` but use integer intrinsics. i32 and u32 share the same MR×NR=6×16 tile since `vpmulld` works for both signed and unsigned (truncated multiplication).

**Files:**
- Modify: `src/intrinsics/gemmsup_avx2.h`
- Modify: `tests/matmul/test_matmul.c`

- [ ] **Step 1: Write failing tests for i32/u32 matmul**

Add to `tests/matmul/test_matmul.c`, before `main()`:

```c
/* ── Int32 ────────────────────────────────────────────────────── */

static int test_matmul_i32_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  int32_t *da = (int32_t *)numc_array_data(a);
  int32_t *db = (int32_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (int32_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul i32 identity should succeed", ctx);
  int32_t *rc = (int32_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul i32 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_gemm_vs_naive_i32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *c_gemm = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  int32_t *da = (int32_t *)numc_array_data(a);
  int32_t *db = (int32_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (int32_t)((i % 7) + 1);
    db[i] = (int32_t)((i % 5) + 1);
  }
  int err1 = numc_matmul(a, b, c_gemm);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both matmul paths should succeed", ctx);
  int32_t *rg = (int32_t *)numc_array_data(c_gemm);
  int32_t *rn = (int32_t *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rg[i] == rn[i], "GEMM and naive i32 results should match", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── UInt32 ───────────────────────────────────────────────────── */

static int test_matmul_u32_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT32);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT32);
  uint32_t *da = (uint32_t *)numc_array_data(a);
  uint32_t *db = (uint32_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (uint32_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul u32 identity should succeed", ctx);
  uint32_t *rc = (uint32_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul u32 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}
```

Add to `main()` in `test_matmul.c`:
```c
  printf("\nInt32:\n");
  RUN_TEST(test_matmul_i32_identity);
  RUN_TEST(test_matmul_gemm_vs_naive_i32);

  printf("\nUInt32:\n");
  RUN_TEST(test_matmul_u32_identity);
```

- [ ] **Step 2: Run tests to verify they pass (they already exercise the packed GEMM path)**

Run: `./run.sh test`
Expected: All tests pass (i32/u32 already have packed GEMM kernels via `gemm_avx2.h`). This confirms the tests are valid against the existing implementation.

- [ ] **Step 3: Add i32 GEMMSUP micro-kernel to gemmsup_avx2.h**

Add before the final `#undef GEMMSUP_MIN` in `src/intrinsics/gemmsup_avx2.h`. The kernel uses `vpmulld` (AVX2 `_mm256_mullo_epi32`) for multiplication and `vpaddd` for accumulation. Same 6×16 tile as f32 but with integer ops:

```c
/* =================================================================
   Int32 unpacked 6x16 micro-kernel (vpmulld + vpaddd)
   Same tile as f32, shared by both i32 and u32 (truncated mul).
   ================================================================= */

static inline void gemmsup_ukernel_i32_6x16(const int32_t *a, const int32_t *b,
                                             int32_t *c, size_t kc, intptr_t rsa,
                                             intptr_t csa, intptr_t rsb,
                                             intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
  __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
  __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
  __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
  __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
  __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();

#define GEMMSUP_I32_K_BODY(p_off)                                              \
  do {                                                                         \
    const int32_t *bp_ = b + (p_off) * rsb;                                    \
    __m256i b0_ = _mm256_loadu_si256((const __m256i *)bp_);                     \
    __m256i b1_ = _mm256_loadu_si256((const __m256i *)(bp_ + 8));               \
    __m256i a0_, a1_;                                                          \
    a0_ = _mm256_set1_epi32(a[0 * rsa + (p_off) * csa]);                       \
    a1_ = _mm256_set1_epi32(a[1 * rsa + (p_off) * csa]);                       \
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(a0_, b0_));                  \
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(a0_, b1_));                  \
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(a1_, b0_));                  \
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(a1_, b1_));                  \
    a0_ = _mm256_set1_epi32(a[2 * rsa + (p_off) * csa]);                       \
    a1_ = _mm256_set1_epi32(a[3 * rsa + (p_off) * csa]);                       \
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(a0_, b0_));                  \
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(a0_, b1_));                  \
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(a1_, b0_));                  \
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(a1_, b1_));                  \
    a0_ = _mm256_set1_epi32(a[4 * rsa + (p_off) * csa]);                       \
    a1_ = _mm256_set1_epi32(a[5 * rsa + (p_off) * csa]);                       \
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(a0_, b0_));                  \
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(a0_, b1_));                  \
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(a1_, b0_));                  \
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(a1_, b1_));                  \
  } while (0)

  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 4) {
    GEMMSUP_I32_K_BODY(p);
    GEMMSUP_I32_K_BODY(p + 1);
    GEMMSUP_I32_K_BODY(p + 2);
    GEMMSUP_I32_K_BODY(p + 3);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++)
    GEMMSUP_I32_K_BODY(p);
#undef GEMMSUP_I32_K_BODY

  _mm256_storeu_si256((__m256i *)c, c00);
  _mm256_storeu_si256((__m256i *)(c + 8), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 8), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 8), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 8), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 8), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 8), c51);
}

static inline void gemmsup_edge_i32(const int32_t *a, const int32_t *b,
                                     int32_t *c, size_t mr, size_t nr, size_t kc,
                                     intptr_t rsa, intptr_t csa, intptr_t rsb,
                                     intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0;
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      for (size_t j = 0; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemmsup_i32_avx2(const int32_t *a, const int32_t *b,
                                     int32_t *out, size_t M, size_t K, size_t N,
                                     intptr_t rsa, intptr_t csa, intptr_t rsb,
                                     intptr_t rso) {
  const size_t MR = 6, NR = 16;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N >            \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR)
        gemmsup_ukernel_i32_6x16(a + i * rsa, b + j, out + i * rso + j, K,
                                  rsa, csa, rsb, rso);
      else
        gemmsup_edge_i32(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                          rsa, csa, rsb, rso);
    }
  }
}
```

The u32 version reuses the same kernels by casting — `vpmulld` is sign-agnostic for truncated multiplication. Add a thin wrapper:

```c
static inline void gemmsup_u32_avx2(const uint32_t *a, const uint32_t *b,
                                     uint32_t *out, size_t M, size_t K,
                                     size_t N, intptr_t rsa, intptr_t csa,
                                     intptr_t rsb, intptr_t rso) {
  gemmsup_i32_avx2((const int32_t *)a, (const int32_t *)b, (int32_t *)out, M,
                    K, N, rsa, csa, rsb, rso);
}
```

- [ ] **Step 4: Wire i32/u32 GEMMSUP into matmul.c dispatch**

In `src/matmul/matmul.c`, inside the GEMMSUP dispatch block (after the f64 `if` block at line ~310), add:

```c
      if (a->dtype == NUMC_DTYPE_INT32 || a->dtype == NUMC_DTYPE_UINT32) {
#if NUMC_HAVE_AVX2
        gemmsup_i32_avx2((const int32_t *)a->data, (const int32_t *)b->data,
                         (int32_t *)out->data, m, k, n, rsa, csa, rsb, rso);
#endif
        return 0;
      }
```

Also add per-dtype threshold for i32/u32 in the threshold selection block (after the f64 threshold):
```c
    if (a->dtype == NUMC_DTYPE_INT32 || a->dtype == NUMC_DTYPE_UINT32) {
      gemmsup_threshold = (96ULL * 96ULL * 96ULL);
    }
```

- [ ] **Step 5: Build and run tests**

Run: `./run.sh test`
Expected: All tests pass including new i32/u32 tests.

- [ ] **Step 6: Commit**

```bash
git add src/intrinsics/gemmsup_avx2.h src/matmul/matmul.c tests/matmul/test_matmul.c
git commit -m "feat(matmul): add i32/u32 GEMMSUP unpacked kernels for AVX2"
```

---

## Task 3: Add Integer GEMMSUP Kernels — i16/u16

Same pattern as Task 2, but MR×NR = 6×32 using `vpmullw` (16-bit multiply). Each YMM holds 16 i16 elements, so NR=32 needs 2 YMM loads per B row.

**Files:**
- Modify: `src/intrinsics/gemmsup_avx2.h`
- Modify: `src/matmul/matmul.c`
- Modify: `tests/matmul/test_matmul.c`

- [ ] **Step 1: Write failing tests for i16/u16 matmul**

Follow the exact same test pattern as Task 2's i32 tests but with `NUMC_DTYPE_INT16`, `int16_t`, and smaller values (mod 10 to avoid overflow). Add `test_matmul_i16_identity`, `test_matmul_gemm_vs_naive_i16`, `test_matmul_u16_identity`, and wire into `main()`.

- [ ] **Step 2: Run tests to verify they pass with existing packed GEMM**

Run: `./run.sh test`
Expected: Pass (existing packed GEMM handles i16/u16)

- [ ] **Step 3: Add i16 GEMMSUP micro-kernel**

Same 6-row structure. Key difference: `_mm256_set1_epi16()` for A broadcast, `_mm256_mullo_epi16()` for multiply, `_mm256_add_epi16()` for accumulate. NR=32 = 2 YMM vectors per B-row. Store with `_mm256_storeu_si256`.

Add `gemmsup_ukernel_i16_6x32`, `gemmsup_edge_i16`, `gemmsup_i16_avx2`, and `gemmsup_u16_avx2` (wrapper casting to i16) to `gemmsup_avx2.h`.

- [ ] **Step 4: Wire i16/u16 GEMMSUP dispatch and thresholds into matmul.c**

Threshold: `128^3` for i16/u16. Add dispatch block similar to i32.

- [ ] **Step 5: Build and run tests**

Run: `./run.sh test`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/intrinsics/gemmsup_avx2.h src/matmul/matmul.c tests/matmul/test_matmul.c
git commit -m "feat(matmul): add i16/u16 GEMMSUP unpacked kernels for AVX2"
```

---

## Task 4: Add Integer GEMMSUP Kernels — i64/u64

MR×NR = 6×8 (same as f64). Uses widening multiply sequence since AVX2 lacks `vpmullq`.

**Files:**
- Modify: `src/intrinsics/gemmsup_avx2.h`
- Modify: `src/matmul/matmul.c`
- Modify: `tests/matmul/test_matmul.c`

- [ ] **Step 1: Write failing tests for i64/u64 matmul**

Same pattern, `NUMC_DTYPE_INT64`, `int64_t`, small values (mod 10).

- [ ] **Step 2: Run tests to verify they pass with existing packed GEMM**

- [ ] **Step 3: Add i64 GEMMSUP micro-kernel**

The K-body macro uses the widening multiply helper:

```c
/* 64-bit multiply helper (no vpmullq on AVX2) */
static inline __m256i _mm256_mullo_epi64_avx2(__m256i a, __m256i b) {
  __m256i lo = _mm256_mul_epu32(a, b);
  __m256i a_hi = _mm256_srli_epi64(a, 32);
  __m256i b_hi = _mm256_srli_epi64(b, 32);
  __m256i hi1 = _mm256_mul_epu32(a_hi, b);
  __m256i hi2 = _mm256_mul_epu32(a, b_hi);
  __m256i hi_sum = _mm256_add_epi64(hi1, hi2);
  return _mm256_add_epi64(lo, _mm256_slli_epi64(hi_sum, 32));
}
```

6×8 tile: 12 accumulators (6×2) + 2 B loads + 2 A broadcasts = 16 YMM. K-unroll 2× (each K-step is ~16 instructions due to widening mul).

Add `gemmsup_ukernel_i64_6x8`, `gemmsup_edge_i64`, `gemmsup_i64_avx2`, `gemmsup_u64_avx2`.

- [ ] **Step 4: Wire i64/u64 GEMMSUP dispatch and thresholds**

Threshold: `48^3` for i64/u64 (same as f64).

- [ ] **Step 5: Build and run tests**

Run: `./run.sh test`

- [ ] **Step 6: Commit**

```bash
git add src/intrinsics/gemmsup_avx2.h src/matmul/matmul.c tests/matmul/test_matmul.c
git commit -m "feat(matmul): add i64/u64 GEMMSUP unpacked kernels for AVX2"
```

---

## Task 5: Add Integer GEMMSUP Kernels — i8/u8

MR×NR = 6×16 with i32 accumulation. Uses `_mm256_cvtepi8_epi32` (sign-extend 8 i8 → 8 i32) then `vpmulld` + `vpaddd`. Each YMM holds 8 i32 accumulators, so NR=16 = 2 YMM per row.

**Files:**
- Modify: `src/intrinsics/gemmsup_avx2.h`
- Modify: `src/matmul/matmul.c`
- Modify: `tests/matmul/test_matmul.c`

- [ ] **Step 1: Write failing tests for i8/u8 matmul**

Use small values (mod 5) to stay within i8 range. The cross-validation test against naive is critical since i8 accumulates in i32 internally.

- [ ] **Step 2: Run tests to verify they pass with existing packed GEMM**

- [ ] **Step 3: Add i8 GEMMSUP micro-kernel**

The K-body loads 8 B elements at a time (i8→i32 promotion via `_mm256_cvtepi8_epi32`). For the B load, extract a 64-bit chunk from the i8 array and sign-extend to 8×i32:

```c
/* Load 8 i8 values and sign-extend to i32 */
static inline __m256i _load_i8_to_i32(const int8_t *ptr) {
  return _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)ptr));
}
```

K-body for i8 6×16:
```c
#define GEMMSUP_I8_K_BODY(p_off)                                    \
  do {                                                              \
    const int8_t *bp_ = b + (p_off) * rsb;                          \
    __m256i b0_ = _load_i8_to_i32(bp_);                              \
    __m256i b1_ = _load_i8_to_i32(bp_ + 8);                          \
    __m256i a0_, a1_;                                                \
    a0_ = _mm256_set1_epi32((int32_t)a[0 * rsa + (p_off) * csa]);    \
    a1_ = _mm256_set1_epi32((int32_t)a[1 * rsa + (p_off) * csa]);    \
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(a0_, b0_));        \
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(a0_, b1_));        \
    /* ... rows 2-5 same pattern ... */                              \
  } while (0)
```

Store path: truncate i32→i8 using `_mm256_packs_epi32` → `_mm256_packs_epi16` → `vpermq` for cross-lane correction, then scalar store for the 16 i8 values. Alternatively, use scalar stores from the i32 accumulators (simpler, store path is not hot):

```c
/* Simple store: extract i32 accumulators and truncate to i8 */
for (size_t j = 0; j < 16; j++) {
  int32_t val;
  if (j < 8)
    val = _mm256_extract_epi32(c0X, j); // appropriate accumulator
  else
    val = _mm256_extract_epi32(c0X, j - 8);
  c_out[j] = (int8_t)val;
}
```

For the GEMMSUP path (which is not the hot path for large matrices), use the simple scalar store approach. The hot packed GEMM path (Phase 2 ASM) will use vectorized stores.

Add `gemmsup_ukernel_i8_6x16`, `gemmsup_edge_i8`, `gemmsup_i8_avx2`.

For u8: separate kernel using `_mm256_cvtepu8_epi32` instead of `_mm256_cvtepi8_epi32`. Add `gemmsup_u8_avx2`.

- [ ] **Step 4: Wire i8/u8 GEMMSUP dispatch**

Threshold: `192^3` for i8/u8.

- [ ] **Step 5: Build and run tests**

Run: `./run.sh test`

- [ ] **Step 6: Commit**

```bash
git add src/intrinsics/gemmsup_avx2.h src/matmul/matmul.c tests/matmul/test_matmul.c
git commit -m "feat(matmul): add i8/u8 GEMMSUP unpacked kernels for AVX2"
```

---

## Task 6: Refactor matmul.c GEMMSUP Dispatch

The current dispatch uses nested `if` chains for each dtype. With 10 dtypes now having GEMMSUP, refactor to a table-driven approach.

**Files:**
- Modify: `src/matmul/matmul.c`

- [ ] **Step 1: Add GEMMSUP dispatch table and threshold table**

Replace the per-dtype `if` chain with:

```c
typedef void (*GemmSupKernel)(const void *a, const void *b, void *out,
                               size_t M, size_t K, size_t N, intptr_t rsa,
                               intptr_t csa, intptr_t rsb, intptr_t rso);

/* Wrappers to cast void* to typed pointers */
#define GEMMSUP_WRAP(name, CT, fn)                                    \
  static void name(const void *a, const void *b, void *out, size_t M, \
                   size_t K, size_t N, intptr_t rsa, intptr_t csa,    \
                   intptr_t rsb, intptr_t rso) {                      \
    fn((const CT *)a, (const CT *)b, (CT *)out, M, K, N, rsa, csa,    \
       rsb, rso);                                                     \
  }

#if NUMC_HAVE_AVX2
GEMMSUP_WRAP(_gemmsup_i8,  int8_t,   gemmsup_i8_avx2)
GEMMSUP_WRAP(_gemmsup_i16, int16_t,  gemmsup_i16_avx2)
GEMMSUP_WRAP(_gemmsup_i32, int32_t,  gemmsup_i32_avx2)
GEMMSUP_WRAP(_gemmsup_i64, int64_t,  gemmsup_i64_avx2)
GEMMSUP_WRAP(_gemmsup_u8,  uint8_t,  gemmsup_u8_avx2)
GEMMSUP_WRAP(_gemmsup_u16, uint16_t, gemmsup_u16_avx2)
GEMMSUP_WRAP(_gemmsup_u32, uint32_t, gemmsup_u32_avx2)
GEMMSUP_WRAP(_gemmsup_u64, uint64_t, gemmsup_u64_avx2)
GEMMSUP_WRAP(_gemmsup_f32, float,    gemmsup_f32_avx2)
GEMMSUP_WRAP(_gemmsup_f64, double,   gemmsup_f64_avx2)
#endif

static const GemmSupKernel gemmsup_table[NUMC_DTYPE_COUNT] = {
#if NUMC_HAVE_AVX2
    [NUMC_DTYPE_INT8]    = _gemmsup_i8,
    [NUMC_DTYPE_INT16]   = _gemmsup_i16,
    [NUMC_DTYPE_INT32]   = _gemmsup_i32,
    [NUMC_DTYPE_INT64]   = _gemmsup_i64,
    [NUMC_DTYPE_UINT8]   = _gemmsup_u8,
    [NUMC_DTYPE_UINT16]  = _gemmsup_u16,
    [NUMC_DTYPE_UINT32]  = _gemmsup_u32,
    [NUMC_DTYPE_UINT64]  = _gemmsup_u64,
    [NUMC_DTYPE_FLOAT32] = _gemmsup_f32,
    [NUMC_DTYPE_FLOAT64] = _gemmsup_f64,
#endif
};

static const uint64_t gemmsup_threshold_table[NUMC_DTYPE_COUNT] = {
    [NUMC_DTYPE_INT8]    = 192ULL * 192 * 192,
    [NUMC_DTYPE_INT16]   = 128ULL * 128 * 128,
    [NUMC_DTYPE_INT32]   = 96ULL * 96 * 96,
    [NUMC_DTYPE_INT64]   = 48ULL * 48 * 48,
    [NUMC_DTYPE_UINT8]   = 192ULL * 192 * 192,
    [NUMC_DTYPE_UINT16]  = 128ULL * 128 * 128,
    [NUMC_DTYPE_UINT32]  = 96ULL * 96 * 96,
    [NUMC_DTYPE_UINT64]  = 48ULL * 48 * 48,
    [NUMC_DTYPE_FLOAT32] = 96ULL * 96 * 96,
    [NUMC_DTYPE_FLOAT64] = 48ULL * 48 * 48,
};
```

Then replace the entire GEMMSUP dispatch block in `numc_matmul()` with:

```c
  /* Small-matrix path: unpacked SIMD GEMM */
  {
    size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    uint64_t flops = (uint64_t)m * k * n;
    GemmSupKernel sup = gemmsup_table[a->dtype];
    if (sup && flops <= gemmsup_threshold_table[a->dtype]) {
      size_t elem = numc_dtype_size(a->dtype);
      intptr_t rsa = (intptr_t)(a->strides[0] / elem);
      intptr_t csa = (intptr_t)(a->strides[1] / elem);
      intptr_t rsb = (intptr_t)(b->strides[0] / elem);
      intptr_t rso = (intptr_t)(out->strides[0] / elem);
      sup(a->data, b->data, out->data, m, k, n, rsa, csa, rsb, rso);
      return 0;
    }
  }
```

- [ ] **Step 2: Build and run all tests**

Run: `./run.sh test`
Expected: All tests pass. This is a pure refactor — same behavior, cleaner dispatch.

- [ ] **Step 3: Run matmul benchmark to confirm no regression**

Run: `./run.sh bench matmul`
Expected: Similar or better numbers to baseline.

- [ ] **Step 4: Commit**

```bash
git add src/matmul/matmul.c
git commit -m "refactor(matmul): table-driven GEMMSUP dispatch with per-dtype thresholds"
```

---

## Task 7: Add Transposed Input Tests

Test that matmul produces correct results when A or B (or both) are transposed views.

**Files:**
- Modify: `tests/matmul/test_matmul.c`

- [ ] **Step 1: Write transposed matmul tests**

```c
static int test_matmul_f32_transposed_b(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b_orig = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c1 = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c2 = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);

  float *da = (float *)numc_array_data(a);
  float *db = (float *)numc_array_data(b_orig);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (float)((i % 7) + 1);
    db[i] = (float)((i % 5) + 1);
  }

  /* c1 = A @ B (normal) */
  int err1 = numc_matmul(a, b_orig, c1);
  ASSERT_MSG_CTX(err1 == 0, "normal matmul should succeed", ctx);

  /* c2 = A @ (B^T)^T (transpose B twice — same result) */
  NumcArray *bt = numc_array_transpose(b_orig);
  NumcArray *btt = numc_array_transpose(bt);
  int err2 = numc_matmul(a, btt, c2);
  ASSERT_MSG_CTX(err2 == 0, "transposed matmul should succeed", ctx);

  float *r1 = (float *)numc_array_data(c1);
  float *r2 = (float *)numc_array_data(c2);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    float diff = r1[i] - r2[i];
    if (diff < 0) diff = -diff;
    ASSERT_MSG_CTX(diff < 1e-3f, "transposed result should match normal", ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}
```

Add similar test for transposed A (`test_matmul_f32_transposed_a`) and both transposed (`test_matmul_f32_transposed_ab`). Also add one i32 transposed test.

Wire all into `main()`.

- [ ] **Step 2: Build and run tests**

Run: `./run.sh test`
Expected: All pass (the packed GEMM already handles arbitrary strides).

- [ ] **Step 3: Commit**

```bash
git add tests/matmul/test_matmul.c
git commit -m "test(matmul): add transposed input tests for storage-aware validation"
```

---

## Task 8: Add Larger Matrix Tests for Packed GEMM Coverage

The GEMMSUP tests at N=64 fall below integer thresholds (e.g., i32 threshold = 96^3), so after GEMMSUP dispatch is added, N=64 tests only exercise the GEMMSUP path. Add larger tests to ensure the packed GEMM path remains correct.

**Files:**
- Modify: `tests/matmul/test_matmul.c`

- [ ] **Step 1: Add packed GEMM cross-validation tests at N=128**

```c
static int test_matmul_packed_128_i32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {128, 128};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *c_fast = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *c_naive = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t *ad = (int32_t *)numc_array_data(a);
  int32_t *bd = (int32_t *)numc_array_data(b);
  for (size_t i = 0; i < 128 * 128; i++) {
    ad[i] = (int32_t)((i % 7) + 1);
    bd[i] = (int32_t)((i % 5) + 1);
  }
  int err1 = numc_matmul(a, b, c_fast);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0, "matmul failed", ctx);
  ASSERT_MSG_CTX(err2 == 0, "matmul_naive failed", ctx);
  const int32_t *r1 = (const int32_t *)numc_array_data(c_fast);
  const int32_t *r2 = (const int32_t *)numc_array_data(c_naive);
  for (size_t i = 0; i < 128 * 128; i++)
    ASSERT_MSG_CTX(r1[i] == r2[i], "packed vs naive i32 mismatch at 128x128", ctx);
  numc_ctx_free(ctx);
  return 0;
}
```

Add similar tests for i8, i16, i64. Also add a non-power-of-2 test (e.g., 67×67) to cover M and N remainder paths. Wire all into `main()`.

- [ ] **Step 2: Build and run tests**

Run: `./run.sh test`

- [ ] **Step 3: Commit**

```bash
git add tests/matmul/test_matmul.c
git commit -m "test(matmul): add 128x128 and 67x67 packed GEMM cross-validation for integer dtypes"
```

---

## Task 9: Phase 1 Benchmark Checkpoint

Run benchmarks to measure Phase 1 impact (integer GEMMSUP for small matrices).

**Files:** None (benchmark only)

- [ ] **Step 1: Run matmul benchmark**

Run: `./run.sh bench matmul`
Record results. Integer types at 64×64 should show improvement since they now use GEMMSUP instead of packed GEMM.

- [ ] **Step 2: Save benchmark results**

Copy terminal output to a file for comparison:
```bash
./run.sh bench matmul 2>&1 | tee bench/numc/phase1_results.txt
```

---

## Task 9: AVX2 ASM Micro-kernel — i32 6×16

The first ASM kernel. Follow the existing `gemm_ukernel_f32_6x16_avx2.S` pattern exactly, replacing `vfmadd231ps` with `vpmulld` + `vpaddd`.

**Files:**
- Create: `src/intrinsics/gemm_ukernel_i32_6x16_avx2.S`
- Modify: `src/intrinsics/gemm_avx2.h`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Create the ASM micro-kernel**

Create `src/intrinsics/gemm_ukernel_i32_6x16_avx2.S`. Key differences from f32:
- Replace `vbroadcastss` → `vpbroadcastd` (broadcast 32-bit integer)
- Replace `vfmadd231ps` → `vpmulld ymm_a, ymm_b, ymm_tmp` + `vpaddd ymm_tmp, ymm_acc, ymm_acc`
  - This needs a temp register. Since `vfmadd231ps` is 3-operand (fused), but integer doesn't have fused multiply-add, we need to restructure. Use ymm14 as multiply scratch (it's loaded with A broadcast, then used for multiply, then result added to accumulator). Sequence per row-pair:
    ```asm
    vpbroadcastd 0(%rdi), %ymm14        # A broadcast row 0
    vpmulld      %ymm14, %ymm12, %ymm15 # tmp = A[0] * B[0:7]
    vpaddd       %ymm15, %ymm0, %ymm0   # acc[0,0] += tmp
    vpmulld      %ymm14, %ymm13, %ymm15 # tmp = A[0] * B[8:15]
    vpaddd       %ymm15, %ymm1, %ymm1   # acc[0,1] += tmp
    vpbroadcastd 4(%rdi), %ymm14        # A broadcast row 1
    vpmulld      %ymm14, %ymm12, %ymm15
    vpaddd       %ymm15, %ymm2, %ymm2
    vpmulld      %ymm14, %ymm13, %ymm15
    vpaddd       %ymm15, %ymm3, %ymm3
    ```
  - Problem: we need both ymm14 AND ymm15 for scratch (A broadcast + multiply temp), but they're already used for A broadcasts. Solution: the f32 kernel alternates ymm14/ymm15 for two A broadcasts, then does FMA. For integer, we process one row at a time: load A into ymm14, multiply into ymm15, add to accumulator, repeat.
- Replace `vmovups` → `vmovdqu` for integer loads/stores
- Replace `vxorps` → `vpxord` or `vpxor` for zeroing
- K-unroll: 4× (same as f32, integer mul is same latency on modern CPUs)
- Same +128 bias, same prefetch strategy

The function signature matches f32:
```c
extern void numc_gemm_ukernel_i32_6x16_avx2(const int32_t *a, const int32_t *b,
                                              int32_t *c, uint64_t kc,
                                              int64_t rso, int first);
```

- [ ] **Step 2: Declare extern in gemm_avx2.h and wire into packed GEMM**

Add after the existing f64 extern declaration:
```c
extern void numc_gemm_ukernel_i32_6x16_avx2(const int32_t *a, const int32_t *b,
                                              int32_t *c, uint64_t kc,
                                              int64_t rso, int first);
```

In the i32 packed GEMM `gemm_i32_avx2()` function, replace the intrinsics micro-kernel call with the ASM kernel call (similar to how f32 does it — check for `NUMC_HAVE_ASM_UKERNEL`).

- [ ] **Step 3: Add .S file to CMakeLists.txt**

In `src/CMakeLists.txt`, add to the x86_64 assembly list:
```cmake
        intrinsics/gemm_ukernel_i32_6x16_avx2.S
```

- [ ] **Step 4: Build and run tests**

Run: `./run.sh test`
Expected: All tests pass. The i32 packed GEMM now uses the ASM micro-kernel.

- [ ] **Step 5: Run i32 benchmark subset**

Run: `./run.sh bench matmul` and compare i32 numbers to baseline.

- [ ] **Step 6: Commit**

```bash
git add src/intrinsics/gemm_ukernel_i32_6x16_avx2.S src/intrinsics/gemm_avx2.h src/CMakeLists.txt
git commit -m "perf(matmul): add AVX2 ASM micro-kernel for i32 6x16 packed GEMM"
```

---

## Task 10: AVX2 ASM Micro-kernel — u32 6×16

`vpmulld` is sign-agnostic for truncated 32-bit multiplication. The u32 kernel is identical to i32 except for the symbol name. Create it as a copy with renamed labels.

**Files:**
- Create: `src/intrinsics/gemm_ukernel_u32_6x16_avx2.S`
- Modify: `src/intrinsics/gemm_avx2.h`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Create u32 ASM kernel (copy i32 with renamed symbols)**

Copy `gemm_ukernel_i32_6x16_avx2.S`, rename function to `numc_gemm_ukernel_u32_6x16_avx2`, rename all labels from `.Li32_*` to `.Lu32_*`.

- [ ] **Step 2: Declare extern, wire dispatch, add to CMake**
- [ ] **Step 3: Build and run tests**
- [ ] **Step 4: Commit**

```bash
git add src/intrinsics/gemm_ukernel_u32_6x16_avx2.S src/intrinsics/gemm_avx2.h src/CMakeLists.txt
git commit -m "perf(matmul): add AVX2 ASM micro-kernel for u32 6x16 packed GEMM"
```

---

## Task 11: AVX2 ASM Micro-kernel — i16/u16 6×32

NR=32 with `vpmullw` + `vpaddw`. Each YMM holds 16 i16 elements, so 2 YMM vectors per B-row. Same 6-row structure: 12 accumulators + 2 B loads + 2 A broadcast scratch = 16 YMM.

Key differences from i32:
- `vpbroadcastw` for A broadcast (16-bit)
- `vpmullw` for multiply
- `vpaddw` for accumulate
- A pointer advances by `MR * 2 = 12` bytes per K-step (6 elements × 2 bytes)
- B pointer advances by `NR * 2 = 64` bytes per K-step (32 elements × 2 bytes)
- 4× K-unroll

**Files:**
- Create: `src/intrinsics/gemm_ukernel_i16_6x32_avx2.S`
- Create: `src/intrinsics/gemm_ukernel_u16_6x32_avx2.S`
- Modify: `src/intrinsics/gemm_avx2.h`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Create i16 ASM kernel**
- [ ] **Step 2: Create u16 ASM kernel (renamed copy)**
- [ ] **Step 3: Declare externs, wire dispatch, add to CMake**
- [ ] **Step 4: Build and run tests**
- [ ] **Step 5: Commit**

```bash
git add src/intrinsics/gemm_ukernel_i16_6x32_avx2.S src/intrinsics/gemm_ukernel_u16_6x32_avx2.S src/intrinsics/gemm_avx2.h src/CMakeLists.txt
git commit -m "perf(matmul): add AVX2 ASM micro-kernels for i16/u16 6x32 packed GEMM"
```

---

## Task 12: AVX2 ASM Micro-kernel — i64/u64 6×8

NR=8 with widening multiply (no `vpmullq` on AVX2). Same tile as f64. 2× K-unroll due to multiply complexity.

Key instructions per multiply:
```asm
# Widening 64-bit multiply: result = lo + (hi1 << 32) + (hi2 << 32)
vpmuludq    %ymm_a, %ymm_b, %ymm_lo     # lo = lower32(a) * lower32(b) → 64-bit
vpsrlq      $32, %ymm_a, %ymm_tmp1      # a_hi = a >> 32
vpmuludq    %ymm_tmp1, %ymm_b, %ymm_hi1 # hi1 = upper32(a) * lower32(b)
vpsrlq      $32, %ymm_b, %ymm_tmp2      # b_hi = b >> 32
vpmuludq    %ymm_a, %ymm_tmp2, %ymm_hi2 # hi2 = lower32(a) * upper32(b)
vpaddq      %ymm_hi1, %ymm_hi2, %ymm_hi # hi = hi1 + hi2
vpsllq      $32, %ymm_hi, %ymm_hi       # hi <<= 32
vpaddq      %ymm_lo, %ymm_hi, %ymm_res  # result = lo + hi
vpaddq      %ymm_res, %ymm_acc, %ymm_acc # accumulate
```

This uses 3 scratch registers (ymm_lo, ymm_tmp1, ymm_tmp2). Register pressure is tight:
- 12 accumulators (ymm0-ymm11)
- 2 B loads (ymm12, ymm13)
- 2 scratch for A broadcast + multiply (ymm14, ymm15)

The widening mul needs intermediate results. Solution: reuse ymm14/ymm15 for intermediates since A broadcast value is consumed immediately. The sequence becomes:
```asm
vpbroadcastq (%rdi), %ymm14            # A[row0]
vpmuludq     %ymm14, %ymm12, %ymm15   # lo = lower32(a) * lower32(b0)
vpsrlq       $32, %ymm14, %ymm14      # a_hi (reuse ymm14)
vpmuludq     %ymm14, %ymm12, %ymm14   # hi1 = upper(a) * lower(b0) (reuse ymm14)
# Need b_hi... but ymm15 holds lo. Must store lo temporarily.
# Alternative: pre-compute b_hi into a spare register before the loop.
```

This is complex. The implementation should pre-compute `b_hi = b >> 32` into a register at the start of each K-step, reusing it across all 6 rows. Since we have exactly 16 registers, we need to spill one accumulator to the stack temporarily, or reduce MR to 4 (yielding 8 accumulators + 2 B + 2 B_hi + 2 A + 2 scratch = 16). For this plan, **keep MR=6** and use a stack spill for the b_hi intermediate.

**Files:**
- Create: `src/intrinsics/gemm_ukernel_i64_6x8_avx2.S`
- Create: `src/intrinsics/gemm_ukernel_u64_6x8_avx2.S`
- Modify: `src/intrinsics/gemm_avx2.h`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Create i64 ASM kernel with widening multiply**
- [ ] **Step 2: Create u64 ASM kernel (identical math, different symbol)**
- [ ] **Step 3: Declare externs, wire dispatch, add to CMake**
- [ ] **Step 4: Build and run tests**
- [ ] **Step 5: Commit**

```bash
git add src/intrinsics/gemm_ukernel_i64_6x8_avx2.S src/intrinsics/gemm_ukernel_u64_6x8_avx2.S src/intrinsics/gemm_avx2.h src/CMakeLists.txt
git commit -m "perf(matmul): add AVX2 ASM micro-kernels for i64/u64 6x8 packed GEMM"
```

---

## Task 13: Add Packing Infrastructure for i8/u8 GEMM

**Critical prerequisite for i8/u8 ASM kernels.** The existing i8/u8 GEMM in `gemm_avx2.h` uses direct strided access with no packing and no KC-blocking (unlike f32/f64/i32/i16/i64 which all have full BLIS-style pack+compute). The new ASM kernels expect packed data. This task adds packing routines and KC-blocking to the i8/u8 GEMM path, updating NR from 8 to 16.

**Files:**
- Modify: `src/intrinsics/gemm_avx2.h`

- [ ] **Step 1: Update GEMM_I8_NR and add GEMM_I8_KC**

```c
#define GEMM_I8_NR 16
#define GEMM_I8_KC 512
```

- [ ] **Step 2: Add i8/u8 packing routines**

Add `gemm_pack_a_i8`, `gemm_pack_b_i8` following the existing f32 packing pattern. Key difference: elements are 1 byte, so NR=16 bytes per B-row (fits in one `_mm_loadu_si128`). A-packing reads MR=6 i8 elements per K-step (6 bytes) and writes them contiguously.

```c
static inline void gemm_pack_b_i8(const int8_t *b, int8_t *packed, size_t kc,
                                    size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_I8_NR <= nc; jr += GEMM_I8_NR) {
    int8_t *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const int8_t *src = b + p * rsb + jr;
      _mm_storeu_si128((__m128i *)(dest + p * GEMM_I8_NR),
                       _mm_loadu_si128((const __m128i *)src));
    }
  }
  if (jr < nc) {
    int8_t *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      memset(dest + p * GEMM_I8_NR, 0, GEMM_I8_NR);
      memcpy(dest + p * GEMM_I8_NR, b + p * rsb + jr, rem);
    }
  }
}
```

Add similar `gemm_pack_a_i8` and corresponding u8 versions (or reuse via cast since packing is byte-level).

- [ ] **Step 3: Restructure gemm_i8_avx2 with KC-blocking and packing**

Replace the direct-access IC/JR loop with the full BLIS-style 5-loop:
- JC loop (N-blocking by NC)
- PC loop (K-blocking by KC)
- Pack A and B per block
- IC × JR micro-kernel dispatch

Follow the existing `gemm_f32_avx2` structure.

- [ ] **Step 4: Build and run tests**

Run: `./run.sh test`
Expected: All i8/u8 tests pass with the new packing infrastructure.

- [ ] **Step 5: Run i8 benchmark to verify no regression**

Run: `./run.sh bench matmul`
The intrinsics micro-kernel with NR=16 may already show improvement over the old NR=8.

- [ ] **Step 6: Commit**

```bash
git add src/intrinsics/gemm_avx2.h
git commit -m "feat(matmul): add packing infrastructure for i8/u8 GEMM, update NR 8->16"
```

---

## Task 14: AVX2 ASM Micro-kernel — i8 6×16 (i32 accumulation)

**Depends on Task 13** (packing infrastructure must be in place).

NR=16 output elements (matching i32 accumulator width). B is packed as i8 but loaded via `vpmovsxbd` (sign-extend i8→i32, 8 at a time). A is broadcast as i32 via sign-extending a single i8 element.

Key sequence per row:
```asm
movsbl      (%rdi), %eax              # sign-extend i8 A[row0] to i32
vmovd       %eax, %xmm14
vpbroadcastd %xmm14, %ymm14          # ymm14 = A[row0] sign-extended to i32

vpmovsxbd   -128(%rsi), %ymm12       # B[0:7] i8→i32 (biased pointer)
vpmovsxbd   -120(%rsi), %ymm13       # B[8:15] i8→i32

vpmulld     %ymm14, %ymm12, %ymm15   # tmp = A[0] * B[0:7]
vpaddd      %ymm15, %ymm0, %ymm0     # acc[0,0] += tmp
vpmulld     %ymm14, %ymm13, %ymm15   # tmp = A[0] * B[8:15]
vpaddd      %ymm15, %ymm1, %ymm1     # acc[0,1] += tmp
```

Store path: accumulators are i32. Use saturation packing to match the existing vectorized store behavior: `vpackssdw` + `vpacksswb` + `vpermq` for cross-lane correction, then store 16 bytes via `_mm_storeu_si128`.

**Note on saturation vs truncation:** The existing i8 vectorized micro-kernel uses saturation (`_mm_packs_epi32`), while the scalar edge kernel uses truncation (`(int8_t)acc`). The ASM kernel matches the vectorized path (saturation). This existing behavioral mismatch is pre-existing and not introduced by this change.

A pointer advances by MR bytes (6) per K-step. B pointer advances by NR bytes (16) per K-step.

**Files:**
- Create: `src/intrinsics/gemm_ukernel_i8_6x16_avx2.S`
- Modify: `src/intrinsics/gemm_avx2.h`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Create i8 ASM kernel with i32 accumulation**
- [ ] **Step 2: Declare extern, wire dispatch, add to CMake**
- [ ] **Step 3: Build and run tests**
- [ ] **Step 4: Commit**

```bash
git add src/intrinsics/gemm_ukernel_i8_6x16_avx2.S src/intrinsics/gemm_avx2.h src/CMakeLists.txt
git commit -m "perf(matmul): add AVX2 ASM micro-kernel for i8 6x16 packed GEMM (i32 acc)"
```

---

## Task 15: AVX2 ASM Micro-kernel — u8 6×16 (u32 accumulation)

Same as i8 but uses `vpmovzxbd` (zero-extend u8→u32) instead of `vpmovsxbd`. Multiply with `vpmulld` (same — truncated mul is sign-agnostic). Store truncates u32→u8 using `vpackusdw` + `vpackuswb`.

**Files:**
- Create: `src/intrinsics/gemm_ukernel_u8_6x16_avx2.S`
- Modify: `src/intrinsics/gemm_avx2.h`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Create u8 ASM kernel (copy i8 with vpmovzxbd and unsigned pack)**
- [ ] **Step 2: Declare extern, wire dispatch, add to CMake**
- [ ] **Step 3: Build and run tests**
- [ ] **Step 4: Commit**

```bash
git add src/intrinsics/gemm_ukernel_u8_6x16_avx2.S src/intrinsics/gemm_avx2.h src/CMakeLists.txt
git commit -m "perf(matmul): add AVX2 ASM micro-kernel for u8 6x16 packed GEMM (u32 acc)"
```

---

## Task 16: Phase 2 Final Benchmark

Run full matmul benchmark to measure the combined Phase 1 + Phase 2 impact.

**Files:** None (benchmark only)

- [ ] **Step 1: Run full matmul benchmark**

Run: `./run.sh bench matmul`

- [ ] **Step 2: Compare with baseline**

Key metrics to check:
- i32/u32: should show improvement at all sizes (ASM replaces intrinsics)
- i16/u16: should show improvement (ASM replaces intrinsics)
- i8/u8: should show significant improvement (ASM with NR=16 vs old NR=8 intrinsics)
- i64/u64: should show improvement (ASM with tuned widening mul)
- f32/f64: should be unchanged (ASM kernels already existed)
- 64×64 sizes across all integer types: should show improvement from GEMMSUP

- [ ] **Step 3: Save results**

```bash
./run.sh bench matmul 2>&1 | tee bench/numc/phase2_results.txt
```

- [ ] **Step 4: Commit benchmark results**

```bash
git add bench/numc/phase2_results.txt
git commit -m "bench: record Phase 1+2 matmul optimization results"
```

---

## Task 17: Run Full Test Suite

Final validation that all changes are clean.

- [ ] **Step 1: Run check (format + tidy + tests)**

Run: `./run.sh check`
Expected: All checks pass.

- [ ] **Step 2: Fix any formatting or tidy issues**

If clang-format or clang-tidy reports issues, fix them and re-run.

- [ ] **Step 3: Run cross-platform tests via QEMU (if available)**

```bash
./run.sh avx512 test
./run.sh neon test
./run.sh sve test
./run.sh rvv test
```

Expected: All pass. The new `.S` files are guarded by `#if defined(__x86_64__)` so they won't compile on other ISAs. The GEMMSUP additions in `gemmsup_avx2.h` are only included when `NUMC_HAVE_AVX2` is set.

- [ ] **Step 4: Final commit if any fixes were needed**
