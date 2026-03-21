/*
 * numc — dedicated AVX2 u16 6x32 packed GEMM micro-kernel
 *
 * void numc_gemm_ukernel_u16_6x32_avx2(
 *     const uint16_t *a,      // rdi — packed A micropanel (MR-column-panel)
 *     const uint16_t *b,      // rsi — packed B micropanel (NR-row-panel)
 *     uint16_t       *c,      // rdx — output tile (row 0)
 *     uint64_t        kc,     // rcx — K dimension of current block
 *     int64_t         rso,    // r8  — row stride of output C (in elements)
 *     int             first   // r9d — beta flag: 1 = zero accumulators, 0 = load C
 * );
 *
 * SysV x86-64 ABI.  Callee-saved: rbx, rbp, r12-r15.
 * All 16 YMM registers are clobbered (caller-saved on SysV).
 *
 * Identical to i16 kernel — vpmullw is sign-agnostic for truncated multiply.
 * Only symbol names differ (.Lu16_ prefix for labels).
 */

#if defined(__x86_64__) || defined(_M_X64)

    .text
    .balign 32
    .globl numc_gemm_ukernel_u16_6x32_avx2
    .type  numc_gemm_ukernel_u16_6x32_avx2, @function

numc_gemm_ukernel_u16_6x32_avx2:
    /* --- prologue: save callee-saved registers --- */
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13

    mov     %rdx, %rbx
    lea     (%rbx, %r8, 2), %r10
    lea     (%r10, %r8, 2), %r11
    lea     (%r11, %r8, 2), %r12
    lea     (%r12, %r8, 2), %r13
    lea     (%r13, %r8, 2), %rbp

    add     $128, %rsi

    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    test    %r9d, %r9d
    jz      .Lu16_load_c

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
    jmp     .Lu16_preload_b

.Lu16_load_c:
    vmovdqu (%rbx),      %ymm0
    vmovdqu 32(%rbx),    %ymm1
    vmovdqu (%r10),      %ymm2
    vmovdqu 32(%r10),    %ymm3
    vmovdqu (%r11),      %ymm4
    vmovdqu 32(%r11),    %ymm5
    vmovdqu (%r12),      %ymm6
    vmovdqu 32(%r12),    %ymm7
    vmovdqu (%r13),      %ymm8
    vmovdqu 32(%r13),    %ymm9
    vmovdqu (%rbp),      %ymm10
    vmovdqu 32(%rbp),    %ymm11

.Lu16_preload_b:
    vmovdqu -128(%rsi),  %ymm12
    vmovdqu -96(%rsi),   %ymm13

    mov     %rcx, %rdx
    shr     $2, %rdx
    jz      .Lu16_k_left

    .p2align 5
.Lu16_k_loop:

    /* ===== K-iteration 0 ===== */
    prefetcht0 128(%rdi)

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

    vmovdqu     -64(%rsi), %ymm12
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

    vmovdqu     0(%rsi),   %ymm12
    vmovdqu     32(%rsi),  %ymm13

    /* ===== K-iteration 2 ===== */
    prefetcht0 152(%rdi)

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

    vmovdqu     64(%rsi),  %ymm12
    vmovdqu     96(%rsi),  %ymm13

    /* ===== K-iteration 3 ===== */
    prefetcht0 512(%rsi)

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

    add     $48, %rdi
    add     $256, %rsi

    vmovdqu -128(%rsi), %ymm12
    vmovdqu -96(%rsi),  %ymm13

    dec     %rdx
    jnz     .Lu16_k_loop

.Lu16_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx
    jz      .Lu16_store

    .p2align 4
.Lu16_k_left_loop:
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

    add     $12, %rdi
    add     $64, %rsi
    vmovdqu -128(%rsi), %ymm12
    vmovdqu -96(%rsi),  %ymm13
    dec     %rdx
    jnz     .Lu16_k_left_loop

.Lu16_store:
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

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

    vzeroupper
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret

    .size numc_gemm_ukernel_u16_6x32_avx2, .-numc_gemm_ukernel_u16_6x32_avx2

#endif /* __x86_64__ || _M_X64 */
