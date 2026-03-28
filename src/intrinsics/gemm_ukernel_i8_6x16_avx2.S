/*
 * numc — dedicated AVX2 i8 6x16 packed GEMM micro-kernel (i32 accumulation)
 *
 * void numc_gemm_ukernel_i8_6x16_avx2(
 *     const int8_t  *a,       // rdi — packed A micropanel (MR-column-panel)
 *     const int8_t  *b,       // rsi — packed B micropanel (NR-row-panel)
 *     int8_t        *c,       // rdx — output tile (row 0)
 *     uint64_t       kc,      // rcx — K dimension of current block
 *     int64_t        rso,     // r8  — row stride of output C (in bytes)
 *     int            first    // r9d — beta flag: 1 = zero accumulators, 0 = load C
 * );
 *
 * SysV x86-64 ABI.  Callee-saved: rbx, rbp, r12-r15.
 * All 16 YMM registers are clobbered (caller-saved on SysV).
 *
 * Register map (YMM):
 *   ymm0-ymm11  : 12 accumulators (6 rows x 2 NR-halves), i32
 *   ymm12, ymm13: B pre-load vectors (cols 0-7, 8-15), i32
 *   ymm14       : A broadcast scratch
 *   ymm15       : multiply scratch (vpmulld result)
 *
 * A elements: i8, sign-extended to i32 via movsbl + vpbroadcastd
 * B elements: i8, sign-extended to i32 via vpmovsxbd (8 bytes -> 8 i32)
 * C load:     i8, sign-extended to i32 via vpmovsxbd
 * C store:    i32 truncated to i8 via signed saturation (vpackssdw + vpacksswb)
 *
 * Per K-step: A += 6 bytes (MR*1), B += 16 bytes (NR*1)
 * Per 4-unroll: A += 24, B += 64
 *
 * Techniques:
 *   - BLIS-style pointer bias (+128 on B) for disp8 encoding
 *   - 4x K-unroll, single pointer advance per group
 *   - B pre-load at end of each K-iter for latency hiding
 *   - Software prefetch: 2x A per 4-unroll, 1x B per 4-unroll
 *   - C prefetch before compute + refresh before store
 */

#if defined(__x86_64__) || defined(_M_X64)
#include "asm_compat.inc"

    .text
    .balign 32
    .globl numc_gemm_ukernel_i8_6x16_avx2
    DECLARE_FUNC(numc_gemm_ukernel_i8_6x16_avx2)

numc_gemm_ukernel_i8_6x16_avx2:
    WIN64_SAVE
    WIN64_REMAP_ARGS6
    /* --- prologue: save callee-saved registers --- */
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13

    /* --- map SysV arguments to working registers --- */
    /* rdi = a, rsi = b, rdx = c, rcx = kc, r8 = rso, r9d = first */

    mov     %rdx, %rbx          /* rbx = c (free up rdx)                      */
    lea     (%rbx, %r8, 1), %r10 /* r10 = c + 1*rso (rso in bytes for i8)    */
    lea     (%r10, %r8, 1), %r11 /* r11 = c + 2*rso                          */
    lea     (%r11, %r8, 1), %r12 /* r12 = c + 3*rso                          */
    lea     (%r12, %r8, 1), %r13 /* r13 = c + 4*rso                          */
    lea     (%r13, %r8, 1), %rbp /* rbp = c + 5*rso                          */

    /* --- Bias B pointer by +128 for compact disp8 encoding --- */
    add     $128, %rsi

    /* --- C prefetch (bring output tile into L1 before compute) --- */
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    /* --- beta handling: zero accumulators or load existing C --- */
    test    %r9d, %r9d
    jz      .Li8_load_c

    /* first=1 (beta=0): zero all 12 accumulators */
    vpxor   %ymm0,  %ymm0,  %ymm0
    vpxor   %ymm1,  %ymm1,  %ymm1
    vpxor   %ymm2,  %ymm2,  %ymm2
    vpxor   %ymm3,  %ymm3,  %ymm3
    vpxor   %ymm4,  %ymm4,  %ymm4
    vpxor   %ymm5,  %ymm5,  %ymm5
    vpxor   %ymm6,  %ymm6,  %ymm6
    vpxor   %ymm7,  %ymm7,  %ymm7
    vpxor   %ymm8,  %ymm8,  %ymm8
    vpxor   %ymm9,  %ymm9,  %ymm9
    vpxor   %ymm10, %ymm10, %ymm10
    vpxor   %ymm11, %ymm11, %ymm11
    jmp     .Li8_preload_b

.Li8_load_c:
    /* first=0 (beta=1): load existing i8 C, sign-extend to i32 accumulators */
    vpmovsxbd   (%rbx),      %ymm0      /* C[row0, 0:7]  i8->i32  */
    vpmovsxbd   8(%rbx),     %ymm1      /* C[row0, 8:15] i8->i32  */
    vpmovsxbd   (%r10),      %ymm2      /* C[row1, 0:7]            */
    vpmovsxbd   8(%r10),     %ymm3      /* C[row1, 8:15]           */
    vpmovsxbd   (%r11),      %ymm4      /* C[row2, 0:7]            */
    vpmovsxbd   8(%r11),     %ymm5      /* C[row2, 8:15]           */
    vpmovsxbd   (%r12),      %ymm6      /* C[row3, 0:7]            */
    vpmovsxbd   8(%r12),     %ymm7      /* C[row3, 8:15]           */
    vpmovsxbd   (%r13),      %ymm8      /* C[row4, 0:7]            */
    vpmovsxbd   8(%r13),     %ymm9      /* C[row4, 8:15]           */
    vpmovsxbd   (%rbp),      %ymm10     /* C[row5, 0:7]            */
    vpmovsxbd   8(%rbp),     %ymm11     /* C[row5, 8:15]           */

.Li8_preload_b:
    /* --- Pre-load first B vectors (sign-extend i8 -> i32) --- */
    vpmovsxbd   -128(%rsi),  %ymm12     /* B[0:7]  i8->i32  */
    vpmovsxbd   -120(%rsi),  %ymm13     /* B[8:15] i8->i32  */

    /* --- Main K-loop: 4x unrolled --- */
    mov     %rcx, %rdx              /* rdx = kc (rcx freed for other use)   */
    shr     $2, %rdx                /* rdx = kc / 4 = k_iter               */
    jz      .Li8_k_left

    .p2align 5
.Li8_k_loop:

    /* ===== K-iteration 0 ===== */
    prefetcht0 64(%rdi)              /* A prefetch ahead */

    movsbl      0(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    movsbl      1(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    movsbl      2(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    movsbl      3(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    movsbl      4(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    movsbl      5(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    vpmovsxbd   -112(%rsi), %ymm12  /* reload B for K-iter 1 */
    vpmovsxbd   -104(%rsi), %ymm13

    /* ===== K-iteration 1 ===== */
    movsbl      6(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    movsbl      7(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    movsbl      8(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    movsbl      9(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    movsbl      10(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    movsbl      11(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    vpmovsxbd   -96(%rsi),  %ymm12  /* reload B for K-iter 2 */
    vpmovsxbd   -88(%rsi),  %ymm13

    /* ===== K-iteration 2 ===== */
    prefetcht0 88(%rdi)              /* A prefetch (second) */

    movsbl      12(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    movsbl      13(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    movsbl      14(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    movsbl      15(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    movsbl      16(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    movsbl      17(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    vpmovsxbd   -80(%rsi),  %ymm12  /* reload B for K-iter 3 */
    vpmovsxbd   -72(%rsi),  %ymm13

    /* ===== K-iteration 3 ===== */
    prefetcht0 128(%rsi)             /* B prefetch ahead */

    movsbl      18(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    movsbl      19(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    movsbl      20(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    movsbl      21(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    movsbl      22(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    movsbl      23(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    /* --- Pointer advance (once per 4-unroll) --- */
    add     $24, %rdi                /* A += 4 * MR * sizeof(int8_t) = 4*6*1 = 24  */
    add     $64, %rsi                /* B += 4 * NR * sizeof(int8_t) = 4*16*1 = 64 */

    /* Reload B for next group (using biased pointer) */
    vpmovsxbd -128(%rsi), %ymm12
    vpmovsxbd -120(%rsi), %ymm13

    dec     %rdx
    jnz     .Li8_k_loop

    /* --- Remainder K-loop (kc % 4) --- */
.Li8_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx                 /* rdx = kc % 4 */
    jz      .Li8_store

    .p2align 4
.Li8_k_left_loop:
    /* Single K-iteration using pre-loaded B (no prefetch) */
    movsbl      0(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    movsbl      1(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    movsbl      2(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    movsbl      3(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    movsbl      4(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    movsbl      5(%rdi), %eax
    vmovd       %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    add     $6, %rdi                 /* A += MR * sizeof(int8_t) = 6*1 = 6 */
    add     $16, %rsi                /* B += NR * sizeof(int8_t) = 16*1 = 16 */
    vpmovsxbd -128(%rsi), %ymm12     /* reload B (biased) */
    vpmovsxbd -120(%rsi), %ymm13
    dec     %rdx
    jnz     .Li8_k_left_loop

    /* --- C prefetch refresh (before store) --- */
.Li8_store:
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    /* --- Store accumulators to C: truncate i32 -> i8 via signed saturation ---
     *
     * For each row we have two YMM accumulators (acc0, acc1) with 8 i32 each.
     * Pack 16 i32 -> 16 i8 using 128-bit ops to avoid cross-lane issues:
     *   1. Extract high 128-bit halves
     *   2. vpackssdw: 4+4 i32 -> 8 i16 (per accumulator)
     *   3. vpacksswb: 8+8 i16 -> 16 i8
     *   4. Store 16 bytes (xmm)
     */

    /* Row 0: ymm0 (acc0), ymm1 (acc1) */
    vextracti128 $1, %ymm0, %xmm14
    vpackssdw    %xmm14, %xmm0, %xmm14   /* acc0: 8 i32 -> 8 i16 */
    vextracti128 $1, %ymm1, %xmm15
    vpackssdw    %xmm15, %xmm1, %xmm15   /* acc1: 8 i32 -> 8 i16 */
    vpacksswb    %xmm15, %xmm14, %xmm14  /* 16 i16 -> 16 i8 */
    vmovdqu      %xmm14, (%rbx)

    /* Row 1: ymm2 (acc0), ymm3 (acc1) */
    vextracti128 $1, %ymm2, %xmm14
    vpackssdw    %xmm14, %xmm2, %xmm14
    vextracti128 $1, %ymm3, %xmm15
    vpackssdw    %xmm15, %xmm3, %xmm15
    vpacksswb    %xmm15, %xmm14, %xmm14
    vmovdqu      %xmm14, (%r10)

    /* Row 2: ymm4 (acc0), ymm5 (acc1) */
    vextracti128 $1, %ymm4, %xmm14
    vpackssdw    %xmm14, %xmm4, %xmm14
    vextracti128 $1, %ymm5, %xmm15
    vpackssdw    %xmm15, %xmm5, %xmm15
    vpacksswb    %xmm15, %xmm14, %xmm14
    vmovdqu      %xmm14, (%r11)

    /* Row 3: ymm6 (acc0), ymm7 (acc1) */
    vextracti128 $1, %ymm6, %xmm14
    vpackssdw    %xmm14, %xmm6, %xmm14
    vextracti128 $1, %ymm7, %xmm15
    vpackssdw    %xmm15, %xmm7, %xmm15
    vpacksswb    %xmm15, %xmm14, %xmm14
    vmovdqu      %xmm14, (%r12)

    /* Row 4: ymm8 (acc0), ymm9 (acc1) */
    vextracti128 $1, %ymm8, %xmm14
    vpackssdw    %xmm14, %xmm8, %xmm14
    vextracti128 $1, %ymm9, %xmm15
    vpackssdw    %xmm15, %xmm9, %xmm15
    vpacksswb    %xmm15, %xmm14, %xmm14
    vmovdqu      %xmm14, (%r13)

    /* Row 5: ymm10 (acc0), ymm11 (acc1) */
    vextracti128 $1, %ymm10, %xmm14
    vpackssdw    %xmm14, %xmm10, %xmm14
    vextracti128 $1, %ymm11, %xmm15
    vpackssdw    %xmm15, %xmm11, %xmm15
    vpacksswb    %xmm15, %xmm14, %xmm14
    vmovdqu      %xmm14, (%rbp)

    /* --- epilogue --- */
    vzeroupper
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    WIN64_RESTORE
    ret

    END_FUNC(numc_gemm_ukernel_i8_6x16_avx2)

GNU_STACK_NOTE
#endif /* __x86_64__ || _M_X64 */
