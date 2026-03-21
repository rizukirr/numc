# Matmul Full Optimization Design

**Date:** 2026-03-22
**Status:** Approved
**Scope:** Complete matmul optimization across all ISAs (AVX2, AVX-512, NEON, SVE/SVE2, RVV) and all 10 dtypes

## Overview

A comprehensive optimization pass for numc's matmul implementation, structured in 5 phases. Each phase produces independently benchmarkable improvements. AVX2 is implemented and validated first on native hardware (i7-13620H), then patterns are ported to all other ISAs.

## Phase Structure

```
Phase 1: Storage-Aware Dispatch + Integer GEMMSUP
Phase 2: AVX2 ASM Micro-kernels (all 10 dtypes)
Phase 3: Structural GEMM Improvements (fused pack+compute, adaptive threading, B-panel caching)
Phase 4: Cache Blocking Tuning + Benchmark Validation
Phase 5: ISA Porting (AVX-512 -> NEON -> SVE/SVE2 -> RVV)
```

**Key invariant:** Every phase must pass `./run.sh test` and show measurable improvement on `./run.sh bench matmul` before proceeding.

## Phase 1: Storage-Aware Dispatch + Integer GEMMSUP

### Storage-Aware Dispatch

Detect transpose state of A and B by inspecting strides:

- Row-major: `strides[0] = N * elem_size`, `strides[1] = elem_size`
- Col-major: `strides[0] = elem_size`, `strides[1] = M * elem_size`

Classify each input as N (normal) or T (transposed), producing 4 combinations: NN, NT, TN, TT.

**Impact on packing:**

| Combo | A packing | B packing | Benefit |
|-------|-----------|-----------|---------|
| NN | strided row-panel | strided col-panel | current path (baseline) |
| NT | strided row-panel | sequential copy | faster B-packing |
| TN | sequential column read | strided col-panel | faster A-packing |
| TT | sequential column read | sequential copy | both faster |

**Implementation:**

- Add `_detect_storage(array)` returning `NUMC_STORAGE_ROW` or `NUMC_STORAGE_COL` in `dispatch.h`
- Add `_pack_a_t()` and `_pack_b_t()` variants in `matmul.c` for column-major inputs
- Micro-kernel interface unchanged (always receives packed data)
- GEMMSUP kernels add `csb` (column stride of B) to their signature, enabling direct strided access for NT/TT cases without pre-packing B. C's column stride (`cso`) is not needed since C is always contiguous row-major

**Scope boundary:** C storage is always contiguous row-major, so we skip C-storage variants (unlike BLIS's 8-variant approach).

### Integer GEMMSUP

Add unpacked small-matrix GEMM kernels for all 8 integer dtypes. Currently only f32/f64 have GEMMSUP.

**Thresholds (M*K*N flops):**

| dtype | AVX2 threshold | Rationale |
|-------|---------------|-----------|
| i32/u32 | 96^3 (same as f32) | same element width |
| i64/u64 | 48^3 (same as f64) | same element width |
| i16/u16 | 128^3 | smaller elements, packing overhead proportionally larger |
| i8/u8 | 192^3 | 1-byte elements have worst overhead-to-compute ratio |

**AVX2 kernel specifications:**

| dtype | MR x NR | AVX2 intrinsic | Accumulator |
|-------|---------|---------------|-------------|
| i32/u32 | 6x16 | `vpmulld` + `vpaddd` | same type |
| i16/u16 | 6x32 | `vpmullw` + `vpaddw` | same type |
| i64/u64 | 6x8 | widening mul (4 insns) | same type |
| i8/u8 | 6x16 | `_mm256_cvtepi8_epi32` + `vpmulld` | promoted i32/u32 |

**i8/u8 accumulation strategy:** Use sign/zero-extension to i32 (`_mm256_cvtepi8_epi32` for i8, `_mm256_cvtepu8_epi32` for u8) followed by `vpmulld` + `vpaddd`. This avoids `vpmaddubsw` overflow: that instruction treats one operand as unsigned and one as signed, and its i16 intermediate overflows for adversarial inputs even at K=1 (max pairwise sum: 2 × 255 × 127 = 64770 > 32767). The i32-promotion approach matches the existing naive kernel's semantics. NR=16 gives 6×2=12 i32 accumulators + 2 B + 2 A = 16 YMM registers.

**i64/u64 widening multiply (no `vpmullq` on AVX2):**

```
lo = _mm256_mul_epu32(a, b)           // lower 32x32->64
a_hi = _mm256_srli_epi64(a, 32)
hi1 = _mm256_mul_epu32(a_hi, b)       // upper_a x lower_b
b_hi = _mm256_srli_epi64(b, 32)
hi2 = _mm256_mul_epu32(a, b_hi)       // lower_a x upper_b
result = lo + (hi1 << 32) + (hi2 << 32)
```

**i8/u8 rejected alternative:** `vpmaddubsw` + `vpmaddwd` was considered but rejected because it treats one operand as unsigned and one as signed, and the i16 intermediate overflows for adversarial inputs even at K=1 (max pairwise sum 2 × 255 × 127 = 64770 > i16 max 32767). The chosen approach (i32-promotion via `cvt` + `vpmulld`) is safer and matches the naive kernel's exact semantics.

**Dispatch modification:** The current `matmul.c` only applies per-dtype GEMMSUP thresholds for f32 and f64 (all others fall through to the generic `GEMMSUP_FLOPS_THRESHOLD`). This must be extended to per-dtype threshold branching for all 10 types, using a threshold table indexed by `NumcDType`.

**Files modified:**

- `src/matmul/matmul.c` — dispatch logic, storage detection, packing variants, per-dtype thresholds
- `src/matmul/dispatch.h` — `_detect_storage()` helper
- `src/intrinsics/gemmsup_avx2.h` — 8 new integer GEMMSUP kernels

## Phase 2: AVX2 ASM Micro-kernels (All 10 dtypes)

Add 8 new hand-tuned `.S` assembly micro-kernels (f32 and f64 already exist).

### Register Allocation (16 YMM registers)

| dtype | MR x NR | Accumulators | B loads | A broadcasts | K-unroll |
|-------|---------|-------------|---------|-------------|----------|
| f32 | 6x16 | 12 (6x2) | 2 | 2 | 4x (existing) |
| f64 | 6x8 | 12 (6x2) | 2 | 2 | 4x (existing) |
| i32 | 6x16 | 12 (6x2) | 2 | 2 | 4x |
| u32 | 6x16 | 12 (6x2) | 2 | 2 | 4x |
| i16 | 6x32 | 12 (6x2) | 2 | 2 | 4x |
| u16 | 6x32 | 12 (6x2) | 2 | 2 | 4x |
| i64 | 6x8 | 12 (6x2) | 2 | 2 | 2x (mul is 4 insns) |
| u64 | 6x8 | 12 (6x2) | 2 | 2 | 2x (mul is 4 insns) |
| i8 | 6x16 | 12 (6x2, i32 acc) | 2 | 2 | 2x (cvt+mullo) |
| u8 | 6x16 | 12 (6x2, u32 acc) | 2 | 2 | 2x (cvt+mullo) |

### Common ASM Patterns

1. **+128 bias addressing:** Offset A and B base pointers by -128 bytes for `disp8` encoding. Saves ~2 bytes per load instruction.

2. **B pre-load:** Load next iteration's B vectors at end of current K-iteration. Hides memory latency behind compute.

3. **Software prefetch:** `prefetcht0` for A (~10 K-iterations ahead, ~256 bytes), B (next NR panel), C (before first accumulate and before store).

4. **Beta handling:** Two entry points per kernel: `_beta0` (zero accumulators with `vpxor`) and `_beta1` (load existing C into accumulators). Packed GEMM calls `_beta0` on first KC iteration, `_beta1` on subsequent.

### Dtype-Specific Notes

- **i8/u8:** Accumulate in i32 via sign/zero-extension + `vpmulld`. Store path truncates i32 accumulators to i8/u8 with `vpackssdw` + `vpacksswb` (signed) or `vpackusdw` + `vpackuswb` (unsigned). Note: `vpack*` operates per 128-bit lane, so a cross-lane `vpermq` is needed after packing for correct element order.
- **i16/u16:** NR=32 means 2 YMM loads cover 32 elements x 2 bytes = 64 bytes = 1 cache line.
- **i64/u64:** K-unroll reduced to 2x because each K-step is ~16 instructions (widening multiply sequence).

### New Files

```
src/intrinsics/gemm_ukernel_i32_6x16_avx2.S
src/intrinsics/gemm_ukernel_u32_6x16_avx2.S
src/intrinsics/gemm_ukernel_i16_6x32_avx2.S
src/intrinsics/gemm_ukernel_u16_6x32_avx2.S
src/intrinsics/gemm_ukernel_i64_6x8_avx2.S
src/intrinsics/gemm_ukernel_u64_6x8_avx2.S
src/intrinsics/gemm_ukernel_i8_6x16_avx2.S
src/intrinsics/gemm_ukernel_u8_6x16_avx2.S
```

**CMake:** Add to `NUMC_CORE_SOURCES` under existing `x86_64|AMD64` guard in `src/CMakeLists.txt`.

### Edge-Kernel Strategy

When M % MR != 0 or N % NR != 0, remainder tiles need special handling. Strategy:
- **M remainder:** The existing approach of using a temporary MR×NR buffer with the full micro-kernel, then copying valid rows to C, is retained. ASM kernels always compute full MR×NR tiles.
- **N remainder:** Same buffer approach. This avoids needing separate ASM edge kernels for every possible remainder, keeping the ASM file count manageable.
- **Phase 5 (AVX-512):** Can use masked stores (`vmovdqu32` with `kmov` mask) to handle N remainders without a buffer, eliminating the copy overhead.

### Transposed Packing Scope

Storage-aware dispatch (Phase 1) requires transposed packing variants. These are per-dtype with SIMD optimization:
- `_pack_a_t_{dtype}()` — 10 functions (column-major A → packed MR-panels)
- `_pack_b_t_{dtype}()` — 10 functions (column-major B → packed NR-panels)

For AVX2, packing uses `vpermd`/`vpunpck*` for in-register transpose. Other ISAs use equivalent shuffles. Total: 20 new packing functions across Phase 1 + Phase 5.

## Phase 3: Structural GEMM Improvements

### Fused Pack+Compute

Merge B-packing into the micro-kernel for contiguous B inputs, eliminating one pass over B data.

**Current (two-pass):**
```
pack_b(B_raw, packed_b, KC, NR)
micro_kernel(packed_a, packed_b, C)
```

**Fused (single-pass):**
```
micro_kernel_fused(packed_a, B_raw, C, rsb, csb)
// Load B with stride, pack into registers on-the-fly
```

**Decision logic:** Use fused path when:
- B has unit column stride (elements within a row are contiguous) — this is the NN case. For NT (col-major B), rows are strided, so fusing would produce column-strided access patterns that defeat cache advantages; use explicit packing instead.
- `KC * NR * elem_size <= L1_SIZE` (32KB on target hardware)
- Otherwise fall back to explicit pack + standard micro-kernel

**A-packing is never fused.** A's packed layout is reused across all JR iterations within an IC block; fusing would mean re-packing per JR — net loss.

**New entry point:** Each ASM kernel gets a `_fused` variant alongside `_beta0` and `_beta1`.

### Adaptive JC/IC/JR Thread Factorization

Replace flat 2D `IC x JR` decomposition with shape-aware hierarchical factorization.

**Algorithm:**

```
mu = ceil(M / MR)    // number of MR-row tiles
nu = ceil(N / NR)    // number of NR-column tiles

For nthreads >= 8 (three-level):
    jc_nt = min(2, ceil(N / NC))      // L3 partitioning
    remaining = nthreads / jc_nt
    ic_nt = min(remaining, mu)        // L2 partitioning
    jr_nt = remaining / ic_nt         // L1 partitioning

For nthreads 4-7 (two-level):
    if mu >= nu:
        ic_nt = min(nthreads, mu)     // M-dominant
        jr_nt = nthreads / ic_nt
    else:
        jr_nt = min(nthreads, nu)     // N-dominant
        ic_nt = nthreads / jr_nt

For nthreads < 4:
    Keep current flat 2D decomposition
```

**Cache topology rationale:**
- JC threads: each gets own B panel (L3 residency)
- IC threads: each packs own A block (L2 residency)
- JR threads: share packed A, load own B micro-panels (L1 residency)

**Implementation:** Add `_compute_thread_factors()` in `matmul.c`. Replace flat `omp parallel for` with single `omp parallel` region where each thread computes its `(jc_id, ic_id, jr_id)` from its thread ID.

### B-Panel L3 Caching

Ring buffer of recently packed B panels tagged by `(jc_start, pc_start, nr_width)`:

```c
typedef struct {
    void *data;
    size_t jc_start, pc_start, nr_width;
    bool valid;
} BPanelCacheEntry;

#define BPANEL_CACHE_SLOTS 4
```

**Logic:**
1. Before packing B, check if `(jc_start, pc_start)` matches a cached entry
2. Hit: skip packing, use cached pointer
3. Miss: pack into oldest slot, update tag
4. Cache allocated from arena (freed with context)

**Size budget:** `slots * KC * NC * elem_size`. For f32: 4 * 512 * 4080 * 4 = ~32MB. On i7-13620H (24MB L3), limit to 2 slots via runtime detection.

**When it helps:** Iterative algorithms calling `numc_matmul` repeatedly with same B (e.g., neural net inference with fixed weights).

## Phase 4: Cache Blocking Tuning

### Method

Systematic sweep of MC/KC/NC with benchmark harness after ASM kernels are in place.

**Sweep ranges:**
- MC: {72, 96, 120, 144, 168, 192, 240, 288, 336, 480}
- KC: {64, 128, 192, 256, 384, 512}
- NC: {2048, 3072, 4080, 6144}

**Test matrix sizes:** 32^2, 64^2, 128^2, 256^2, 512^2, 1024^2, 2048^2 across all 10 dtypes.

**i7-13620H cache constraints:**
- L1d: 48KB per P-core → `KC * NR * elem_size < 48KB`
- L2: 1.25MB per P-core → `MC * KC * elem_size < ~1MB`
- L3: 24MB shared → `KC * NC * elem_size < ~24MB`

**OMP threshold tuning:** Also sweep the `GEMMSUP_FLOPS_THRESHOLD` and `OMP_BYTE_THRESHOLD` values per dtype.

## Phase 5: ISA Porting

**Order:** AVX-512 -> NEON -> SVE/SVE2 -> RVV

Each port includes: ASM micro-kernels (all 10 dtypes) + integer GEMMSUP + storage-aware dispatch + fused pack+compute. Adaptive threading and B-panel caching are ISA-agnostic.

### AVX-512 Micro-kernels

| dtype | MR x NR | Key advantage |
|-------|---------|---------------|
| f32 | 12x32 | 32 ZMM registers, wider FMA |
| f64 | 12x16 | 24 acc + 2 B + 2 A + 2 preload + 2 spare = 32 ZMM |
| i32/u32 | 12x32 | `vpmulld` 512-bit |
| i16/u16 | 12x64 | `vpmullw` 512-bit |
| i64/u64 | 12x16 | `vpmullq` (AVX-512DQ) — single instruction! |
| i8/u8 | 12x32 | `_mm512_cvtepi8_epi32` + `vpmulld` 512-bit |

**AVX-512 advantage:** `vpmullq` exists (AVX-512DQ), so i64/u64 are single-instruction vs 4-instruction AVX2 widening. Masked operations (`kmov` + predicated loads/stores) handle edge cases without separate edge kernels. f64 uses MR=12 (not 14) to leave headroom for B pre-loading and scratch registers. i8/u8 uses NR=32 with i32 accumulation: 12×(32/16)=24 accumulators + 2 B + 2 A + 4 spare = 32 ZMM.

### NEON (AArch64) Micro-kernels

| dtype | MR x NR | Key note |
|-------|---------|----------|
| f32 | 8x12 | 32 registers, lower pressure than AVX2 |
| f64 | 6x8 | `vfmaq_f64` |
| i32/u32 | 6x8 | `vmulq_s32` |
| i16/u16 | 6x16 | `vmulq_s16` |
| i64 | 6x4 | emulated (no `vmulq_s64` on NEON) |
| i8/u8 | 6x16 | `vmull_s8` + `vpadalq` |

Written in GNU assembler. 32 SIMD registers means more room for prefetch and K-unrolling.

**NEON i64 emulation (no `vmulq_s64`):** Use the same widening strategy as AVX2 i64 but with NEON instructions:
```
// a * b for 64-bit on NEON:
lo   = vmull_u32(vmovn_u64(a), vmovn_u64(b))      // lower 32x32->64
a_hi = vshrn_n_u64(a, 32)
hi1  = vmull_u32(a_hi, vmovn_u64(b))               // upper_a x lower_b
b_hi = vshrn_n_u64(b, 32)
hi2  = vmull_u32(vmovn_u64(a), b_hi)               // lower_a x upper_b
result = vaddq_u64(lo, vshlq_n_u64(vaddq_u64(hi1, hi2), 32))
```

### SVE/SVE2

Scalable vector length: MR is fixed, NR is runtime (`svcntw()` / `svcntd()`). Predicated loads for edge handling.

**SVE2-specific instructions:**
- `SMLALB`/`SMLALT` for i8->i16 widening multiply-accumulate
- `SDOT`/`UDOT` for i8/u8 (4xi8->i32 in one instruction)
- `FMLALB`/`FMLALT` for potential f16 support

### RVV

Scalable with `vsetvl`. `vfmacc.vf` takes scalar directly (no broadcast register). LMUL=m2 doubles effective vector length.

- i8/u8: `vwmul` (widening multiply) + `vwadd` (widening accumulate)
- All dtypes use predicated tail handling via `vsetvl`

### Correctness Validation

All ISA ports validated via QEMU before native benchmarking:
```bash
./run.sh neon test
./run.sh sve test
./run.sh rvv test
./run.sh avx512 test
```

## Files Modified (Summary)

**Core:**
- `src/matmul/matmul.c` — dispatch, loop structure, packing, threading, B-panel cache
- `src/matmul/dispatch.h` — storage detection
- `src/internal.h` — new OMP macros if needed

**AVX2 (Phase 1-2):**
- `src/intrinsics/gemmsup_avx2.h` — 8 new integer GEMMSUP kernels
- `src/intrinsics/gemm_ukernel_{i32,u32,i16,u16,i64,u64,i8,u8}_6xN_avx2.S` — 8 new ASM files

**AVX-512 (Phase 5):**
- `src/intrinsics/gemmsup_avx512.h` — integer GEMMSUP
- `src/intrinsics/gemm_ukernel_*_avx512.S` — 10 ASM files

**NEON (Phase 5):**
- `src/intrinsics/gemmsup_neon.h` — integer GEMMSUP
- `src/intrinsics/gemm_neon.h` — updated packed kernels

**SVE/SVE2 (Phase 5):**
- `src/intrinsics/gemmsup_sve.h` — integer GEMMSUP
- `src/intrinsics/gemm_sve.h` — updated packed kernels
- `src/intrinsics/gemm_sve2.h` — SVE2-specific kernels (currently empty)

**RVV (Phase 5):**
- `src/intrinsics/gemmsup_rvv.h` — integer GEMMSUP
- `src/intrinsics/gemm_rvv.h` — updated packed kernels

**Build:**
- `src/CMakeLists.txt` — new .S files per ISA guard

**Documentation:**
- `CLAUDE.md` — optimization rules section
