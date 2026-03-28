/*
 * numc — dedicated AVX2 u32 6x16 packed GEMM micro-kernel
 *
 * void numc_gemm_ukernel_u32_6x16_avx2(
 *     const uint32_t *a,      // rdi — packed A micropanel (MR-column-panel)
 *     const uint32_t *b,      // rsi — packed B micropanel (NR-row-panel)
 *     uint32_t       *c,      // rdx — output tile (row 0)
 *     uint64_t        kc,     // rcx — K dimension of current block
 *     int64_t         rso,    // r8  — row stride of output C (in elements)
 *     int             first   // r9d — beta flag: 1 = zero accumulators, 0 = load C
 * );
 *
 * SysV x86-64 ABI.  Callee-saved: rbx, rbp, r12-r15.
 * All 16 YMM registers are clobbered (caller-saved on SysV).
 *
 * Register map (YMM):
 *   ymm0-ymm11  : 12 accumulators (6 rows x 2 NR-halves)
 *   ymm12, ymm13: B pre-load vectors (cols 0-7, 8-15)
 *   ymm14       : A broadcast scratch
 *   ymm15       : multiply scratch (vpmulld result)
 *
 * Note: vpmulld is sign-agnostic for truncated 32-bit multiplication, so
 * this u32 kernel is functionally identical to the i32 kernel.
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
    .globl numc_gemm_ukernel_u32_6x16_avx2
    DECLARE_FUNC(numc_gemm_ukernel_u32_6x16_avx2)

numc_gemm_ukernel_u32_6x16_avx2:
    WIN64_SAVE
    WIN64_REMAP_ARGS6
    /* --- prologue: save callee-saved registers --- */
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13

    /* --- map SysV arguments to working registers --- */
    /* rdi = a (A pointer)                              */
    /* rsi = b (B pointer)                              */
    /* rdx = c (C base pointer, row 0)                  */
    /* rcx = kc                                         */
    /* r8  = rso (row stride of C, in elements)         */
    /* r9d = first (beta flag)                          */

    mov     %rdx, %rbx          /* rbx = c (free up rdx for computation)     */
    lea     (%rbx, %r8, 4), %r10 /* r10 = c + 1*rso (in bytes, rso*4)       */
    lea     (%r10, %r8, 4), %r11 /* r11 = c + 2*rso                         */
    lea     (%r11, %r8, 4), %r12 /* r12 = c + 3*rso                         */
    lea     (%r12, %r8, 4), %r13 /* r13 = c + 4*rso                         */
    lea     (%r13, %r8, 4), %rbp /* rbp = c + 5*rso                         */

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
    jz      .Lu32_load_c

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
    jmp     .Lu32_preload_b

.Lu32_load_c:
    /* first=0 (beta=1): load existing C into accumulators */
    vmovdqu (%rbx),      %ymm0      /* C[row0, 0:7]  */
    vmovdqu 32(%rbx),    %ymm1      /* C[row0, 8:15] */
    vmovdqu (%r10),      %ymm2      /* C[row1, 0:7]  */
    vmovdqu 32(%r10),    %ymm3      /* C[row1, 8:15] */
    vmovdqu (%r11),      %ymm4      /* C[row2, 0:7]  */
    vmovdqu 32(%r11),    %ymm5      /* C[row2, 8:15] */
    vmovdqu (%r12),      %ymm6      /* C[row3, 0:7]  */
    vmovdqu 32(%r12),    %ymm7      /* C[row3, 8:15] */
    vmovdqu (%r13),      %ymm8      /* C[row4, 0:7]  */
    vmovdqu 32(%r13),    %ymm9      /* C[row4, 8:15] */
    vmovdqu (%rbp),      %ymm10     /* C[row5, 0:7]  */
    vmovdqu 32(%rbp),    %ymm11     /* C[row5, 8:15] */

.Lu32_preload_b:
    /* --- Pre-load first B vectors (BLIS-style, before loop entry) --- */
    vmovdqu -128(%rsi),  %ymm12     /* B[0:7]  */
    vmovdqu -96(%rsi),   %ymm13     /* B[8:15] */

    /* --- Main K-loop: 4x unrolled --- */
    mov     %rcx, %rdx              /* rdx = kc (rcx freed for other use)   */
    shr     $2, %rdx                /* rdx = kc / 4 = k_iter               */
    jz      .Lu32_k_left

    .p2align 5
.Lu32_k_loop:

    /* ===== K-iteration 0 ===== */
    prefetcht0 256(%rdi)             /* A prefetch: 256 bytes ahead */

    vpbroadcastd 0(%rdi),   %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    vpbroadcastd 4(%rdi),   %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    vpbroadcastd 8(%rdi),   %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    vpbroadcastd 12(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    vpbroadcastd 16(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    vpbroadcastd 20(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    vmovdqu     -64(%rsi), %ymm12  /* reload B for K-iter 1 */
    vmovdqu     -32(%rsi), %ymm13

    /* ===== K-iteration 1 ===== */
    vpbroadcastd 24(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    vpbroadcastd 28(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    vpbroadcastd 32(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    vpbroadcastd 36(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    vpbroadcastd 40(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    vpbroadcastd 44(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    vmovdqu     0(%rsi),   %ymm12  /* reload B for K-iter 2 */
    vmovdqu     32(%rsi),  %ymm13

    /* ===== K-iteration 2 ===== */
    prefetcht0 304(%rdi)             /* A prefetch: 304 bytes ahead (second) */

    vpbroadcastd 48(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    vpbroadcastd 52(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    vpbroadcastd 56(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    vpbroadcastd 60(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    vpbroadcastd 64(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    vpbroadcastd 68(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    vmovdqu     64(%rsi),  %ymm12  /* reload B for K-iter 3 */
    vmovdqu     96(%rsi),  %ymm13

    /* ===== K-iteration 3 ===== */
    prefetcht0 512(%rsi)             /* B prefetch: 512 bytes ahead (8 K-iters) */

    vpbroadcastd 72(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    vpbroadcastd 76(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    vpbroadcastd 80(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    vpbroadcastd 84(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    vpbroadcastd 88(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    vpbroadcastd 92(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    /* --- Pointer advance (once per 4-unroll) --- */
    add     $96, %rdi                /* A += 4 * MR * sizeof(uint32_t) = 4*6*4 = 96  */
    add     $256, %rsi               /* B += 4 * NR * sizeof(uint32_t) = 4*16*4 = 256 */

    /* Reload B for next group (using biased pointer) */
    vmovdqu -128(%rsi), %ymm12
    vmovdqu -96(%rsi),  %ymm13

    dec     %rdx
    jnz     .Lu32_k_loop

    /* --- Remainder K-loop (kc % 4) --- */
.Lu32_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx                 /* rdx = kc % 4 */
    jz      .Lu32_store

    .p2align 4
.Lu32_k_left_loop:
    /* Single K-iteration using pre-loaded B (no prefetch) */
    vpbroadcastd 0(%rdi),   %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm0, %ymm0
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm1, %ymm1

    vpbroadcastd 4(%rdi),   %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm2, %ymm2
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm3, %ymm3

    vpbroadcastd 8(%rdi),   %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm4, %ymm4
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm5, %ymm5

    vpbroadcastd 12(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm6, %ymm6
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm7, %ymm7

    vpbroadcastd 16(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm8, %ymm8
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm9, %ymm9

    vpbroadcastd 20(%rdi),  %ymm14
    vpmulld     %ymm14, %ymm12, %ymm15
    vpaddd      %ymm15, %ymm10, %ymm10
    vpmulld     %ymm14, %ymm13, %ymm15
    vpaddd      %ymm15, %ymm11, %ymm11

    add     $24, %rdi                /* A += MR * sizeof(uint32_t) = 6*4 = 24 */
    add     $64, %rsi                /* B += NR * sizeof(uint32_t) = 16*4 = 64 */
    vmovdqu -128(%rsi), %ymm12       /* reload B (biased) */
    vmovdqu -96(%rsi),  %ymm13
    dec     %rdx
    jnz     .Lu32_k_left_loop

    /* --- C prefetch refresh (before store) --- */
.Lu32_store:
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    /* --- Store accumulators to C --- */
    vmovdqu %ymm0,  (%rbx)
    vmovdqu %ymm1,  32(%rbx)
    vmovdqu %ymm2,  (%r10)
    vmovdqu %ymm3,  32(%r10)
    vmovdqu %ymm4,  (%r11)
    vmovdqu %ymm5,  32(%r11)
    vmovdqu %ymm6,  (%r12)
    vmovdqu %ymm7,  32(%r12)
    vmovdqu %ymm8,  (%r13)
    vmovdqu %ymm9,  32(%r13)
    vmovdqu %ymm10, (%rbp)
    vmovdqu %ymm11, 32(%rbp)

    /* --- epilogue --- */
    vzeroupper
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    WIN64_RESTORE
    ret

    END_FUNC(numc_gemm_ukernel_u32_6x16_avx2)

#endif /* __x86_64__ || _M_X64 */
