/*
 * numc — dedicated AVX2 i16 6x32 packed GEMM micro-kernel
 *
 * void numc_gemm_ukernel_i16_6x32_avx2(
 *     const int16_t *a,       // rdi — packed A micropanel (MR-column-panel)
 *     const int16_t *b,       // rsi — packed B micropanel (NR-row-panel)
 *     int16_t       *c,       // rdx — output tile (row 0)
 *     uint64_t       kc,      // rcx — K dimension of current block
 *     int64_t        rso,     // r8  — row stride of output C (in elements)
 *     int            first    // r9d — beta flag: 1 = zero accumulators, 0 = load C
 * );
 *
 * SysV x86-64 ABI.  Callee-saved: rbx, rbp, r12-r15.
 * All 16 YMM registers are clobbered (caller-saved on SysV).
 *
 * Register map (YMM):
 *   ymm0-ymm11  : 12 accumulators (6 rows x 2 NR-halves)
 *   ymm12, ymm13: B pre-load vectors (cols 0-15, 16-31)
 *   ymm14       : A broadcast scratch
 *   ymm15       : multiply scratch (vpmullw result)
 *
 * Key difference from i32: 16-bit elements → vpbroadcastw, vpmullw, vpaddw.
 * Each YMM holds 16 i16 elements, NR=32 = 2 YMM vectors per B-row.
 * A advance per K: MR * 2 = 12 bytes.  B advance per K: NR * 2 = 64 bytes.
 * C row stride scaled by 2 (sizeof(int16_t)).
 *
 * Techniques:
 *   - BLIS-style pointer bias (+128 on B) for disp8 encoding
 *   - 4x K-unroll, single pointer advance per group
 *   - B pre-load at end of each K-iter for latency hiding
 *   - Software prefetch: 2x A per 4-unroll, 1x B per 4-unroll
 *   - C prefetch before compute + refresh before store
 */

#if defined(__x86_64__) || defined(_M_X64)

    .text
    .balign 32
    .globl numc_gemm_ukernel_i16_6x32_avx2
    .type  numc_gemm_ukernel_i16_6x32_avx2, @function

numc_gemm_ukernel_i16_6x32_avx2:
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
    lea     (%rbx, %r8, 2), %r10 /* r10 = c + 1*rso (in bytes, rso*2)       */
    lea     (%r10, %r8, 2), %r11 /* r11 = c + 2*rso                         */
    lea     (%r11, %r8, 2), %r12 /* r12 = c + 3*rso                         */
    lea     (%r12, %r8, 2), %r13 /* r13 = c + 4*rso                         */
    lea     (%r13, %r8, 2), %rbp /* rbp = c + 5*rso                         */

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
    jz      .Li16_load_c

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
    jmp     .Li16_preload_b

.Li16_load_c:
    /* first=0 (beta=1): load existing C into accumulators */
    vmovdqu (%rbx),      %ymm0      /* C[row0, 0:15]  */
    vmovdqu 32(%rbx),    %ymm1      /* C[row0, 16:31] */
    vmovdqu (%r10),      %ymm2      /* C[row1, 0:15]  */
    vmovdqu 32(%r10),    %ymm3      /* C[row1, 16:31] */
    vmovdqu (%r11),      %ymm4      /* C[row2, 0:15]  */
    vmovdqu 32(%r11),    %ymm5      /* C[row2, 16:31] */
    vmovdqu (%r12),      %ymm6      /* C[row3, 0:15]  */
    vmovdqu 32(%r12),    %ymm7      /* C[row3, 16:31] */
    vmovdqu (%r13),      %ymm8      /* C[row4, 0:15]  */
    vmovdqu 32(%r13),    %ymm9      /* C[row4, 16:31] */
    vmovdqu (%rbp),      %ymm10     /* C[row5, 0:15]  */
    vmovdqu 32(%rbp),    %ymm11     /* C[row5, 16:31] */

.Li16_preload_b:
    /* --- Pre-load first B vectors (BLIS-style, before loop entry) --- */
    vmovdqu -128(%rsi),  %ymm12     /* B[0:15]  */
    vmovdqu -96(%rsi),   %ymm13     /* B[16:31] */

    /* --- Main K-loop: 4x unrolled --- */
    mov     %rcx, %rdx              /* rdx = kc (rcx freed for other use)   */
    shr     $2, %rdx                /* rdx = kc / 4 = k_iter               */
    jz      .Li16_k_left

    .p2align 5
.Li16_k_loop:

    /* ===== K-iteration 0 ===== */
    prefetcht0 128(%rdi)             /* A prefetch ahead */

    vpbroadcastw 0(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm0, %ymm0
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm1, %ymm1

    vpbroadcastw 2(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm2, %ymm2
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm3, %ymm3

    vpbroadcastw 4(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm4, %ymm4
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm5, %ymm5

    vpbroadcastw 6(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm6, %ymm6
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm7, %ymm7

    vpbroadcastw 8(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm8, %ymm8
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm9, %ymm9

    vpbroadcastw 10(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm10, %ymm10
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm11, %ymm11

    vmovdqu     -64(%rsi), %ymm12  /* reload B for K-iter 1 */
    vmovdqu     -32(%rsi), %ymm13

    /* ===== K-iteration 1 ===== */
    vpbroadcastw 12(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm0, %ymm0
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm1, %ymm1

    vpbroadcastw 14(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm2, %ymm2
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm3, %ymm3

    vpbroadcastw 16(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm4, %ymm4
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm5, %ymm5

    vpbroadcastw 18(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm6, %ymm6
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm7, %ymm7

    vpbroadcastw 20(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm8, %ymm8
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm9, %ymm9

    vpbroadcastw 22(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm10, %ymm10
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm11, %ymm11

    vmovdqu     0(%rsi),   %ymm12  /* reload B for K-iter 2 */
    vmovdqu     32(%rsi),  %ymm13

    /* ===== K-iteration 2 ===== */
    prefetcht0 152(%rdi)             /* A prefetch ahead (second) */

    vpbroadcastw 24(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm0, %ymm0
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm1, %ymm1

    vpbroadcastw 26(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm2, %ymm2
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm3, %ymm3

    vpbroadcastw 28(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm4, %ymm4
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm5, %ymm5

    vpbroadcastw 30(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm6, %ymm6
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm7, %ymm7

    vpbroadcastw 32(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm8, %ymm8
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm9, %ymm9

    vpbroadcastw 34(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm10, %ymm10
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm11, %ymm11

    vmovdqu     64(%rsi),  %ymm12  /* reload B for K-iter 3 */
    vmovdqu     96(%rsi),  %ymm13

    /* ===== K-iteration 3 ===== */
    prefetcht0 512(%rsi)             /* B prefetch ahead */

    vpbroadcastw 36(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm0, %ymm0
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm1, %ymm1

    vpbroadcastw 38(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm2, %ymm2
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm3, %ymm3

    vpbroadcastw 40(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm4, %ymm4
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm5, %ymm5

    vpbroadcastw 42(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm6, %ymm6
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm7, %ymm7

    vpbroadcastw 44(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm8, %ymm8
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm9, %ymm9

    vpbroadcastw 46(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm10, %ymm10
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm11, %ymm11

    /* --- Pointer advance (once per 4-unroll) --- */
    add     $48, %rdi                /* A += 4 * MR * sizeof(int16_t) = 4*6*2 = 48  */
    add     $256, %rsi               /* B += 4 * NR * sizeof(int16_t) = 4*32*2 = 256 */

    /* Reload B for next group (using biased pointer) */
    vmovdqu -128(%rsi), %ymm12
    vmovdqu -96(%rsi),  %ymm13

    dec     %rdx
    jnz     .Li16_k_loop

    /* --- Remainder K-loop (kc % 4) --- */
.Li16_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx                 /* rdx = kc % 4 */
    jz      .Li16_store

    .p2align 4
.Li16_k_left_loop:
    /* Single K-iteration using pre-loaded B (no prefetch) */
    vpbroadcastw 0(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm0, %ymm0
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm1, %ymm1

    vpbroadcastw 2(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm2, %ymm2
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm3, %ymm3

    vpbroadcastw 4(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm4, %ymm4
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm5, %ymm5

    vpbroadcastw 6(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm6, %ymm6
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm7, %ymm7

    vpbroadcastw 8(%rdi),   %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm8, %ymm8
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm9, %ymm9

    vpbroadcastw 10(%rdi),  %ymm14
    vpmullw     %ymm14, %ymm12, %ymm15
    vpaddw      %ymm15, %ymm10, %ymm10
    vpmullw     %ymm14, %ymm13, %ymm15
    vpaddw      %ymm15, %ymm11, %ymm11

    add     $12, %rdi                /* A += MR * sizeof(int16_t) = 6*2 = 12 */
    add     $64, %rsi                /* B += NR * sizeof(int16_t) = 32*2 = 64 */
    vmovdqu -128(%rsi), %ymm12       /* reload B (biased) */
    vmovdqu -96(%rsi),  %ymm13
    dec     %rdx
    jnz     .Li16_k_left_loop

    /* --- C prefetch refresh (before store) --- */
.Li16_store:
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
    ret

    .size numc_gemm_ukernel_i16_6x32_avx2, .-numc_gemm_ukernel_i16_6x32_avx2

#endif /* __x86_64__ || _M_X64 */
