/*
 * numc — dedicated AVX2 f32 6×16 packed GEMM micro-kernel
 *
 * void numc_gemm_ukernel_f32_6x16_avx2(
 *     const float *a,       // rdi — packed A micropanel (MR-column-panel)
 *     const float *b,       // rsi — packed B micropanel (NR-row-panel)
 *     float       *c,       // rdx — output tile (row 0)
 *     uint64_t     kc,      // rcx — K dimension of current block
 *     int64_t      rso,     // r8  — row stride of output C (in elements)
 *     int          first    // r9d — beta flag: 1 = zero accumulators, 0 = load C
 * );
 *
 * SysV x86-64 ABI.  Callee-saved: rbx, rbp, r12–r15.
 * All 16 YMM registers are clobbered (caller-saved on SysV).
 *
 * Register map (YMM):
 *   ymm0–ymm11  : 12 accumulators (6 rows × 2 NR-halves)
 *   ymm12, ymm13: B pre-load vectors (cols 0–7, 8–15)
 *   ymm14, ymm15: A broadcast scratch (2 alternating)
 *
 * Techniques:
 *   - BLIS-style pointer bias (+128 on B) for disp8 encoding
 *   - 4× K-unroll, single pointer advance per group
 *   - B pre-load at end of each K-iter for latency hiding
 *   - Software prefetch: 2× A per 4-unroll, 1× B per 4-unroll
 *   - C prefetch before compute + refresh before store
 */

#if defined(__x86_64__) || defined(_M_X64)

    .text
    .balign 32
    .globl numc_gemm_ukernel_f32_6x16_avx2
    .type  numc_gemm_ukernel_f32_6x16_avx2, @function

numc_gemm_ukernel_f32_6x16_avx2:
    /* ─── prologue: save callee-saved registers ─── */
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13

    /* ─── map SysV arguments to working registers ─── */
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

    /* ─── Bias B pointer by +128 for compact disp8 encoding ─── */
    add     $128, %rsi

    /* ─── C prefetch (bring output tile into L1 before compute) ─── */
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    /* ─── beta handling: zero accumulators or load existing C ─── */
    test    %r9d, %r9d
    jz      .Lf32_load_c

    /* first=1 (beta=0): zero all 12 accumulators */
    vxorps  %ymm0,  %ymm0,  %ymm0
    vxorps  %ymm1,  %ymm1,  %ymm1
    vxorps  %ymm2,  %ymm2,  %ymm2
    vxorps  %ymm3,  %ymm3,  %ymm3
    vxorps  %ymm4,  %ymm4,  %ymm4
    vxorps  %ymm5,  %ymm5,  %ymm5
    vxorps  %ymm6,  %ymm6,  %ymm6
    vxorps  %ymm7,  %ymm7,  %ymm7
    vxorps  %ymm8,  %ymm8,  %ymm8
    vxorps  %ymm9,  %ymm9,  %ymm9
    vxorps  %ymm10, %ymm10, %ymm10
    vxorps  %ymm11, %ymm11, %ymm11
    jmp     .Lf32_preload_b

.Lf32_load_c:
    /* first=0 (beta=1): load existing C into accumulators */
    vmovups (%rbx),      %ymm0      /* C[row0, 0:7]  */
    vmovups 32(%rbx),    %ymm1      /* C[row0, 8:15] */
    vmovups (%r10),      %ymm2      /* C[row1, 0:7]  */
    vmovups 32(%r10),    %ymm3      /* C[row1, 8:15] */
    vmovups (%r11),      %ymm4      /* C[row2, 0:7]  */
    vmovups 32(%r11),    %ymm5      /* C[row2, 8:15] */
    vmovups (%r12),      %ymm6      /* C[row3, 0:7]  */
    vmovups 32(%r12),    %ymm7      /* C[row3, 8:15] */
    vmovups (%r13),      %ymm8      /* C[row4, 0:7]  */
    vmovups 32(%r13),    %ymm9      /* C[row4, 8:15] */
    vmovups (%rbp),      %ymm10     /* C[row5, 0:7]  */
    vmovups 32(%rbp),    %ymm11     /* C[row5, 8:15] */

.Lf32_preload_b:
    /* ─── Pre-load first B vectors (BLIS-style, before loop entry) ─── */
    vmovups -128(%rsi),  %ymm12     /* B[0:7]  */
    vmovups -96(%rsi),   %ymm13     /* B[8:15] */

    /* ─── Main K-loop: 4× unrolled ─── */
    mov     %rcx, %rdx              /* rdx = kc (rcx freed for other use)   */
    shr     $2, %rdx                /* rdx = kc / 4 = k_iter               */
    jz      .Lf32_k_left

    .p2align 5
.Lf32_k_loop:

    /* ===== K-iteration 0 ===== */
    prefetcht0 256(%rdi)             /* A prefetch: 256 bytes = ~10.7 K-iters ahead */

    vbroadcastss 0(%rdi),   %ymm14
    vbroadcastss 4(%rdi),   %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 8(%rdi),   %ymm14
    vbroadcastss 12(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 16(%rdi),  %ymm14
    vbroadcastss 20(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    vmovups      -64(%rsi), %ymm12  /* reload B for K-iter 1 */
    vmovups      -32(%rsi), %ymm13

    /* ===== K-iteration 1 ===== */
    vbroadcastss 24(%rdi),  %ymm14
    vbroadcastss 28(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 32(%rdi),  %ymm14
    vbroadcastss 36(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 40(%rdi),  %ymm14
    vbroadcastss 44(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    vmovups      0(%rsi),   %ymm12  /* reload B for K-iter 2 */
    vmovups      32(%rsi),  %ymm13

    /* ===== K-iteration 2 ===== */
    prefetcht0 304(%rdi)             /* A prefetch: 304 bytes ahead (second) */

    vbroadcastss 48(%rdi),  %ymm14
    vbroadcastss 52(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 56(%rdi),  %ymm14
    vbroadcastss 60(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 64(%rdi),  %ymm14
    vbroadcastss 68(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    vmovups      64(%rsi),  %ymm12  /* reload B for K-iter 3 */
    vmovups      96(%rsi),  %ymm13

    /* ===== K-iteration 3 ===== */
    prefetcht0 512(%rsi)             /* B prefetch: 512 bytes ahead (8 K-iters) */

    vbroadcastss 72(%rdi),  %ymm14
    vbroadcastss 76(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 80(%rdi),  %ymm14
    vbroadcastss 84(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 88(%rdi),  %ymm14
    vbroadcastss 92(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11

    /* ─── Pointer advance (once per 4-unroll) ─── */
    add     $96, %rdi                /* A += 4 * MR * sizeof(float) = 4*6*4 = 96  */
    add     $256, %rsi               /* B += 4 * NR * sizeof(float) = 4*16*4 = 256 */

    /* Reload B for next group (using biased pointer) */
    vmovups -128(%rsi), %ymm12
    vmovups -96(%rsi),  %ymm13

    dec     %rdx
    jnz     .Lf32_k_loop

    /* ─── Remainder K-loop (kc % 4) ─── */
.Lf32_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx                 /* rdx = kc % 4 */
    jz      .Lf32_store

    .p2align 4
.Lf32_k_left_loop:
    /* Single K-iteration using pre-loaded B (no prefetch) */
    vbroadcastss 0(%rdi),   %ymm14
    vbroadcastss 4(%rdi),   %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 8(%rdi),   %ymm14
    vbroadcastss 12(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 16(%rdi),  %ymm14
    vbroadcastss 20(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    add     $24, %rdi                /* A += MR * sizeof(float) = 6*4 = 24 */
    add     $64, %rsi                /* B += NR * sizeof(float) = 16*4 = 64 */
    vmovups -128(%rsi), %ymm12       /* reload B (biased) */
    vmovups -96(%rsi),  %ymm13
    dec     %rdx
    jnz     .Lf32_k_left_loop

    /* ─── C prefetch refresh (before store) ─── */
.Lf32_store:
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    /* ─── Store accumulators to C ─── */
    vmovups %ymm0,  (%rbx)
    vmovups %ymm1,  32(%rbx)
    vmovups %ymm2,  (%r10)
    vmovups %ymm3,  32(%r10)
    vmovups %ymm4,  (%r11)
    vmovups %ymm5,  32(%r11)
    vmovups %ymm6,  (%r12)
    vmovups %ymm7,  32(%r12)
    vmovups %ymm8,  (%r13)
    vmovups %ymm9,  32(%r13)
    vmovups %ymm10, (%rbp)
    vmovups %ymm11, 32(%rbp)

    /* ─── epilogue ─── */
    vzeroupper
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret

    .size numc_gemm_ukernel_f32_6x16_avx2, .-numc_gemm_ukernel_f32_6x16_avx2

/*
 * numc_gemm_f32_6x16_avx2_loop — IR-loop fused micro-kernel
 *
 * void numc_gemm_f32_6x16_avx2_loop(
 *     const float *a,       // rdi — packed A (start of first MR panel)
 *     const float *b,       // rsi — packed B micropanel (constant across IR)
 *     float       *c,       // rdx — output C (row 0 of first MR tile)
 *     uint64_t     kc,      // rcx — K dimension
 *     int64_t      rso,     // r8  — row stride of output C (in elements)
 *     int          first,   // r9d — beta flag: 1 = zero, 0 = accumulate
 *     uint64_t     mr_iters // [rsp+8] — number of full MR (6-row) iterations
 * );
 *
 * Loops mr_iters times over the 6×16 micro-kernel body without returning
 * to C between tiles, eliminating vzeroupper + constant reload overhead.
 *
 * Per iteration:
 *   A (rdi): advances automatically by kc*MR*4 bytes from K-loop
 *   B (rsi): reset to original each iteration (same packed B strip)
 *   C (rdx): advances by MR*rso*4 bytes (6 rows)
 */
    .balign 32
    .globl numc_gemm_f32_6x16_avx2_loop
    .type  numc_gemm_f32_6x16_avx2_loop, @function

numc_gemm_f32_6x16_avx2_loop:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15

    /* 7th arg: 6 pushes (48) + ret addr (8) = offset 56 */
    mov     56(%rsp), %r14          /* r14 = mr_iters */
    test    %r14, %r14
    jz      .Lf32L_done

    mov     %rsi, %r15              /* r15 = original B pointer (saved) */

    /* Precompute C row stride = MR * rso * sizeof(float) = 6 * rso * 4
     * = 24 * rso.  Use: rso*2 = rso+rso, rso*4, rso*8, ... */
    lea     (%r8, %r8, 2), %rax     /* rax = 3 * rso */
    shl     $3, %rax                /* rax = 24 * rso (bytes) */
    push    %rax                    /* [rsp] = C stride */
    push    %rdx                    /* [rsp] = current C pointer,
                                       [rsp+8] = C stride */

    .p2align 4
.Lf32L_ir_loop:
    /* ─── Setup C row pointers ─── */
    mov     (%rsp), %rbx            /* rbx = current C (row 0) */
    lea     (%rbx, %r8, 4), %r10
    lea     (%r10, %r8, 4), %r11
    lea     (%r11, %r8, 4), %r12
    lea     (%r12, %r8, 4), %r13
    lea     (%r13, %r8, 4), %rbp

    /* ─── Bias B pointer ─── */
    lea     128(%r15), %rsi

    /* ─── C prefetch ─── */
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    /* ─── Beta handling ─── */
    test    %r9d, %r9d
    jz      .Lf32L_load_c

    vxorps  %ymm0,  %ymm0,  %ymm0
    vxorps  %ymm1,  %ymm1,  %ymm1
    vxorps  %ymm2,  %ymm2,  %ymm2
    vxorps  %ymm3,  %ymm3,  %ymm3
    vxorps  %ymm4,  %ymm4,  %ymm4
    vxorps  %ymm5,  %ymm5,  %ymm5
    vxorps  %ymm6,  %ymm6,  %ymm6
    vxorps  %ymm7,  %ymm7,  %ymm7
    vxorps  %ymm8,  %ymm8,  %ymm8
    vxorps  %ymm9,  %ymm9,  %ymm9
    vxorps  %ymm10, %ymm10, %ymm10
    vxorps  %ymm11, %ymm11, %ymm11
    jmp     .Lf32L_preload_b

.Lf32L_load_c:
    vmovups (%rbx),      %ymm0
    vmovups 32(%rbx),    %ymm1
    vmovups (%r10),      %ymm2
    vmovups 32(%r10),    %ymm3
    vmovups (%r11),      %ymm4
    vmovups 32(%r11),    %ymm5
    vmovups (%r12),      %ymm6
    vmovups 32(%r12),    %ymm7
    vmovups (%r13),      %ymm8
    vmovups 32(%r13),    %ymm9
    vmovups (%rbp),      %ymm10
    vmovups 32(%rbp),    %ymm11

.Lf32L_preload_b:
    vmovups -128(%rsi),  %ymm12
    vmovups -96(%rsi),   %ymm13

    /* ─── Main K-loop: 4× unrolled ─── */
    mov     %rcx, %rdx
    shr     $2, %rdx
    jz      .Lf32L_k_left

    .p2align 5
.Lf32L_k_loop:
    /* K-iteration 0 */
    prefetcht0 256(%rdi)

    vbroadcastss 0(%rdi),   %ymm14
    vbroadcastss 4(%rdi),   %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 8(%rdi),   %ymm14
    vbroadcastss 12(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 16(%rdi),  %ymm14
    vbroadcastss 20(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    vmovups      -64(%rsi), %ymm12
    vmovups      -32(%rsi), %ymm13

    /* K-iteration 1 */
    vbroadcastss 24(%rdi),  %ymm14
    vbroadcastss 28(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 32(%rdi),  %ymm14
    vbroadcastss 36(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 40(%rdi),  %ymm14
    vbroadcastss 44(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    vmovups      0(%rsi),   %ymm12
    vmovups      32(%rsi),  %ymm13

    /* K-iteration 2 */
    prefetcht0 304(%rdi)

    vbroadcastss 48(%rdi),  %ymm14
    vbroadcastss 52(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 56(%rdi),  %ymm14
    vbroadcastss 60(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 64(%rdi),  %ymm14
    vbroadcastss 68(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    vmovups      64(%rsi),  %ymm12
    vmovups      96(%rsi),  %ymm13

    /* K-iteration 3 */
    prefetcht0 512(%rsi)

    vbroadcastss 72(%rdi),  %ymm14
    vbroadcastss 76(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 80(%rdi),  %ymm14
    vbroadcastss 84(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 88(%rdi),  %ymm14
    vbroadcastss 92(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11

    add     $96, %rdi
    add     $256, %rsi
    vmovups -128(%rsi), %ymm12
    vmovups -96(%rsi),  %ymm13

    dec     %rdx
    jnz     .Lf32L_k_loop

    /* ─── Remainder K-loop ─── */
.Lf32L_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx
    jz      .Lf32L_store

    .p2align 4
.Lf32L_k_left_loop:
    vbroadcastss 0(%rdi),   %ymm14
    vbroadcastss 4(%rdi),   %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm0
    vfmadd231ps  %ymm14, %ymm13, %ymm1
    vfmadd231ps  %ymm15, %ymm12, %ymm2
    vfmadd231ps  %ymm15, %ymm13, %ymm3
    vbroadcastss 8(%rdi),   %ymm14
    vbroadcastss 12(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5
    vfmadd231ps  %ymm15, %ymm12, %ymm6
    vfmadd231ps  %ymm15, %ymm13, %ymm7
    vbroadcastss 16(%rdi),  %ymm14
    vbroadcastss 20(%rdi),  %ymm15
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9
    vfmadd231ps  %ymm15, %ymm12, %ymm10
    vfmadd231ps  %ymm15, %ymm13, %ymm11
    add     $24, %rdi
    add     $64, %rsi
    vmovups -128(%rsi), %ymm12
    vmovups -96(%rsi),  %ymm13
    dec     %rdx
    jnz     .Lf32L_k_left_loop

    /* ─── Store ─── */
.Lf32L_store:
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    vmovups %ymm0,  (%rbx)
    vmovups %ymm1,  32(%rbx)
    vmovups %ymm2,  (%r10)
    vmovups %ymm3,  32(%r10)
    vmovups %ymm4,  (%r11)
    vmovups %ymm5,  32(%r11)
    vmovups %ymm6,  (%r12)
    vmovups %ymm7,  32(%r12)
    vmovups %ymm8,  (%r13)
    vmovups %ymm9,  32(%r13)
    vmovups %ymm10, (%rbp)
    vmovups %ymm11, 32(%rbp)

    /* ─── Advance C for next IR iteration ─── */
    /* A (rdi) already points to next MR panel from K-loop advancement */
    mov     8(%rsp), %rax           /* rax = C stride (24 * rso) */
    add     %rax, (%rsp)            /* c_ptr += MR * rso * 4 */

    dec     %r14
    jnz     .Lf32L_ir_loop

    add     $16, %rsp               /* pop saved C ptr + stride */
.Lf32L_done:
    vzeroupper
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret

    .size numc_gemm_f32_6x16_avx2_loop, .-numc_gemm_f32_6x16_avx2_loop

#endif /* __x86_64__ || _M_X64 */
