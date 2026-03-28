/*
 * numc — dedicated AVX2 f64 6×8 packed GEMM micro-kernel
 *
 * void numc_gemm_ukernel_f64_6x8_avx2(
 *     const double *a,       // rdi — packed A micropanel (MR-column-panel)
 *     const double *b,       // rsi — packed B micropanel (NR-row-panel)
 *     double       *c,       // rdx — output tile (row 0)
 *     uint64_t      kc,      // rcx — K dimension of current block
 *     int64_t       rso,     // r8  — row stride of output C (in elements)
 *     int           first    // r9d — beta flag: 1 = zero accumulators, 0 = load C
 * );
 *
 * SysV x86-64 ABI.  Callee-saved: rbx, rbp, r12–r15.
 * All 16 YMM registers are clobbered (caller-saved on SysV).
 *
 * Register map (YMM):
 *   ymm0–ymm11  : 12 accumulators (6 rows × 2 NR-halves)
 *   ymm12, ymm13: B pre-load vectors (cols 0–3, 4–7)
 *   ymm14, ymm15: A broadcast scratch (2 alternating)
 *
 * Techniques:
 *   - BLIS-style pointer bias (+128 on both A and B) for disp8 encoding
 *   - 4× K-unroll, single pointer advance per group
 *   - B pre-load at end of each K-iter for latency hiding
 *   - Software prefetch: 2× A per 4-unroll (768/1536 bytes ahead)
 *     + 1× B per 4-unroll
 *   - C prefetch before compute + refresh before store
 */

#if defined(__x86_64__) || defined(_M_X64)
#include "asm_compat.inc"

    .text
    .balign 32
    .globl numc_gemm_ukernel_f64_6x8_avx2
    DECLARE_FUNC(numc_gemm_ukernel_f64_6x8_avx2)

numc_gemm_ukernel_f64_6x8_avx2:
    WIN64_SAVE
    WIN64_REMAP_ARGS6
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

    mov     %rdx, %rbx              /* rbx = c (free up rdx)                */
    lea     (%rbx, %r8, 8), %r10    /* r10 = c + 1*rso (in bytes, rso*8)   */
    lea     (%r10, %r8, 8), %r11    /* r11 = c + 2*rso                     */
    lea     (%r11, %r8, 8), %r12    /* r12 = c + 3*rso                     */
    lea     (%r12, %r8, 8), %r13    /* r13 = c + 4*rso                     */
    lea     (%r13, %r8, 8), %rbp    /* rbp = c + 5*rso                     */

    /* ─── Bias A and B pointers by +128 for compact disp8 encoding ─── */
    add     $128, %rdi
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
    jz      .Lf64_load_c

    /* first=1 (beta=0): zero all 12 accumulators */
    vxorpd  %ymm0,  %ymm0,  %ymm0
    vxorpd  %ymm1,  %ymm1,  %ymm1
    vxorpd  %ymm2,  %ymm2,  %ymm2
    vxorpd  %ymm3,  %ymm3,  %ymm3
    vxorpd  %ymm4,  %ymm4,  %ymm4
    vxorpd  %ymm5,  %ymm5,  %ymm5
    vxorpd  %ymm6,  %ymm6,  %ymm6
    vxorpd  %ymm7,  %ymm7,  %ymm7
    vxorpd  %ymm8,  %ymm8,  %ymm8
    vxorpd  %ymm9,  %ymm9,  %ymm9
    vxorpd  %ymm10, %ymm10, %ymm10
    vxorpd  %ymm11, %ymm11, %ymm11
    jmp     .Lf64_preload_b

.Lf64_load_c:
    /* first=0 (beta=1): load existing C into accumulators */
    vmovupd (%rbx),      %ymm0      /* C[row0, 0:3]  */
    vmovupd 32(%rbx),    %ymm1      /* C[row0, 4:7]  */
    vmovupd (%r10),      %ymm2      /* C[row1, 0:3]  */
    vmovupd 32(%r10),    %ymm3      /* C[row1, 4:7]  */
    vmovupd (%r11),      %ymm4      /* C[row2, 0:3]  */
    vmovupd 32(%r11),    %ymm5      /* C[row2, 4:7]  */
    vmovupd (%r12),      %ymm6      /* C[row3, 0:3]  */
    vmovupd 32(%r12),    %ymm7      /* C[row3, 4:7]  */
    vmovupd (%r13),      %ymm8      /* C[row4, 0:3]  */
    vmovupd 32(%r13),    %ymm9      /* C[row4, 4:7]  */
    vmovupd (%rbp),      %ymm10     /* C[row5, 0:3]  */
    vmovupd 32(%rbp),    %ymm11     /* C[row5, 4:7]  */

.Lf64_preload_b:
    /* ─── Pre-load first B vectors (BLIS-style, before loop entry) ─── */
    vmovupd -128(%rsi),  %ymm12     /* B[0:3]  */
    vmovupd -96(%rsi),   %ymm13     /* B[4:7]  */

    /* ─── Main K-loop: 4× unrolled ─── */
    mov     %rcx, %rdx              /* rdx = kc                            */
    shr     $2, %rdx                /* rdx = kc / 4 = k_iter              */
    jz      .Lf64_k_left

    .p2align 5
.Lf64_k_loop:

    /* ===== K-iteration 0 ===== */
    /* A prefetch: 768 bytes ahead = 16 K-iters * 48 bytes/iter */
    prefetcht0 640(%rdi)

    vbroadcastsd -128(%rdi),  %ymm14     /* A[0] */
    vbroadcastsd -120(%rdi),  %ymm15     /* A[1] */
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -112(%rdi),  %ymm14     /* A[2] */
    vbroadcastsd -104(%rdi),  %ymm15     /* A[3] */
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd -96(%rdi),   %ymm14     /* A[4] */
    vbroadcastsd -88(%rdi),   %ymm15     /* A[5] */
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    vmovupd      -64(%rsi),   %ymm12     /* reload B for K-iter 1 */
    vmovupd      -32(%rsi),   %ymm13

    /* ===== K-iteration 1 ===== */
    vbroadcastsd -80(%rdi),   %ymm14     /* A[6]  = offset 48-128 = -80 */
    vbroadcastsd -72(%rdi),   %ymm15     /* A[7]  = offset 56-128 = -72 */
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -64(%rdi),   %ymm14     /* A[8]  = offset 64-128 = -64 */
    vbroadcastsd -56(%rdi),   %ymm15     /* A[9]  = offset 72-128 = -56 */
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd -48(%rdi),   %ymm14     /* A[10] = offset 80-128 = -48 */
    vbroadcastsd -40(%rdi),   %ymm15     /* A[11] = offset 88-128 = -40 */
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    vmovupd      0(%rsi),     %ymm12     /* reload B for K-iter 2 */
    vmovupd      32(%rsi),    %ymm13

    /* ===== K-iteration 2 ===== */
    /* A prefetch: 1408 bytes ahead = ~29 K-iters */
    prefetcht0 1280(%rdi)

    vbroadcastsd -32(%rdi),   %ymm14     /* A[12] = offset 96-128  = -32 */
    vbroadcastsd -24(%rdi),   %ymm15     /* A[13] = offset 104-128 = -24 */
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -16(%rdi),   %ymm14     /* A[14] = offset 112-128 = -16 */
    vbroadcastsd -8(%rdi),    %ymm15     /* A[15] = offset 120-128 = -8  */
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd 0(%rdi),     %ymm14     /* A[16] = offset 128-128 = 0   */
    vbroadcastsd 8(%rdi),     %ymm15     /* A[17] = offset 136-128 = 8   */
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    vmovupd      64(%rsi),    %ymm12     /* reload B for K-iter 3 */
    vmovupd      96(%rsi),    %ymm13

    /* ===== K-iteration 3 ===== */
    prefetcht0   512(%rsi)               /* B prefetch: 512 bytes ahead */

    vbroadcastsd 16(%rdi),    %ymm14     /* A[18] = offset 144-128 = 16  */
    vbroadcastsd 24(%rdi),    %ymm15     /* A[19] = offset 152-128 = 24  */
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd 32(%rdi),    %ymm14     /* A[20] = offset 160-128 = 32  */
    vbroadcastsd 40(%rdi),    %ymm15     /* A[21] = offset 168-128 = 40  */
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd 48(%rdi),    %ymm14     /* A[22] = offset 176-128 = 48  */
    vbroadcastsd 56(%rdi),    %ymm15     /* A[23] = offset 184-128 = 56  */
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11

    /* ─── Pointer advance (once per 4-unroll) ─── */
    add     $192, %rdi               /* A += 4 * MR * sizeof(double) = 4*6*8 = 192 */
    add     $256, %rsi               /* B += 4 * NR * sizeof(double) = 4*8*8 = 256  */

    /* Reload B for next group (using biased pointer) */
    vmovupd -128(%rsi), %ymm12
    vmovupd -96(%rsi),  %ymm13

    dec     %rdx
    jnz     .Lf64_k_loop

    /* ─── Remainder K-loop (kc % 4) ─── */
.Lf64_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx                 /* rdx = kc % 4 */
    jz      .Lf64_store

    .p2align 4
.Lf64_k_left_loop:
    /* Single K-iteration using pre-loaded B (no prefetch) */
    vbroadcastsd -128(%rdi),  %ymm14     /* A[0] */
    vbroadcastsd -120(%rdi),  %ymm15     /* A[1] */
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -112(%rdi),  %ymm14     /* A[2] */
    vbroadcastsd -104(%rdi),  %ymm15     /* A[3] */
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd -96(%rdi),   %ymm14     /* A[4] */
    vbroadcastsd -88(%rdi),   %ymm15     /* A[5] */
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    add     $48, %rdi                /* A += MR * sizeof(double) = 6*8 = 48 */
    add     $64, %rsi                /* B += NR * sizeof(double) = 8*8 = 64 */
    vmovupd -128(%rsi), %ymm12       /* reload B (biased) */
    vmovupd -96(%rsi),  %ymm13
    dec     %rdx
    jnz     .Lf64_k_left_loop

    /* ─── C prefetch refresh (before store) ─── */
.Lf64_store:
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    /* ─── Store accumulators to C ─── */
    vmovupd %ymm0,  (%rbx)
    vmovupd %ymm1,  32(%rbx)
    vmovupd %ymm2,  (%r10)
    vmovupd %ymm3,  32(%r10)
    vmovupd %ymm4,  (%r11)
    vmovupd %ymm5,  32(%r11)
    vmovupd %ymm6,  (%r12)
    vmovupd %ymm7,  32(%r12)
    vmovupd %ymm8,  (%r13)
    vmovupd %ymm9,  32(%r13)
    vmovupd %ymm10, (%rbp)
    vmovupd %ymm11, 32(%rbp)

    /* ─── epilogue ─── */
    vzeroupper
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    WIN64_RESTORE
    ret

    END_FUNC(numc_gemm_ukernel_f64_6x8_avx2)

/*
 * numc_gemm_f64_6x8_avx2_loop — IR-loop fused micro-kernel (f64)
 *
 * void numc_gemm_f64_6x8_avx2_loop(
 *     const double *a,       // rdi — packed A (start of first MR panel)
 *     const double *b,       // rsi — packed B micropanel (constant across IR)
 *     double       *c,       // rdx — output C (row 0 of first MR tile)
 *     uint64_t      kc,      // rcx — K dimension
 *     int64_t       rso,     // r8  — row stride of output C (in elements)
 *     int           first,   // r9d — beta flag: 1 = zero, 0 = accumulate
 *     uint64_t      mr_iters // [rsp+8] — number of full MR iterations
 * );
 */
    .balign 32
    .globl numc_gemm_f64_6x8_avx2_loop
    DECLARE_FUNC(numc_gemm_f64_6x8_avx2_loop)

numc_gemm_f64_6x8_avx2_loop:
    WIN64_SAVE
    WIN64_REMAP_ARGS6
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15

    mov     LOOP_ARG7_OFF(%rsp), %r14  /* r14 = mr_iters */
    test    %r14, %r14
    jz      .Lf64L_done

    mov     %rsi, %r15              /* r15 = original B pointer */

    /* C stride = MR * rso * sizeof(double) = 6 * rso * 8 = 48 * rso */
    lea     (%r8, %r8, 2), %rax     /* rax = 3 * rso */
    shl     $4, %rax                /* rax = 48 * rso */
    push    %rax                    /* [rsp] = C stride */
    push    %rdx                    /* [rsp] = current C, [rsp+8] = stride */

    .p2align 4
.Lf64L_ir_loop:
    mov     (%rsp), %rbx
    lea     (%rbx, %r8, 8), %r10
    lea     (%r10, %r8, 8), %r11
    lea     (%r11, %r8, 8), %r12
    lea     (%r12, %r8, 8), %r13
    lea     (%r13, %r8, 8), %rbp

    /* Bias A and B */
    lea     128(%rdi), %rdi
    lea     128(%r15), %rsi

    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    test    %r9d, %r9d
    jz      .Lf64L_load_c

    vxorpd  %ymm0,  %ymm0,  %ymm0
    vxorpd  %ymm1,  %ymm1,  %ymm1
    vxorpd  %ymm2,  %ymm2,  %ymm2
    vxorpd  %ymm3,  %ymm3,  %ymm3
    vxorpd  %ymm4,  %ymm4,  %ymm4
    vxorpd  %ymm5,  %ymm5,  %ymm5
    vxorpd  %ymm6,  %ymm6,  %ymm6
    vxorpd  %ymm7,  %ymm7,  %ymm7
    vxorpd  %ymm8,  %ymm8,  %ymm8
    vxorpd  %ymm9,  %ymm9,  %ymm9
    vxorpd  %ymm10, %ymm10, %ymm10
    vxorpd  %ymm11, %ymm11, %ymm11
    jmp     .Lf64L_preload_b

.Lf64L_load_c:
    vmovupd (%rbx),      %ymm0
    vmovupd 32(%rbx),    %ymm1
    vmovupd (%r10),      %ymm2
    vmovupd 32(%r10),    %ymm3
    vmovupd (%r11),      %ymm4
    vmovupd 32(%r11),    %ymm5
    vmovupd (%r12),      %ymm6
    vmovupd 32(%r12),    %ymm7
    vmovupd (%r13),      %ymm8
    vmovupd 32(%r13),    %ymm9
    vmovupd (%rbp),      %ymm10
    vmovupd 32(%rbp),    %ymm11

.Lf64L_preload_b:
    vmovupd -128(%rsi),  %ymm12
    vmovupd -96(%rsi),   %ymm13

    mov     %rcx, %rdx
    shr     $2, %rdx
    jz      .Lf64L_k_left

    .p2align 5
.Lf64L_k_loop:
    /* K-iteration 0 */
    prefetcht0 640(%rdi)

    vbroadcastsd -128(%rdi),  %ymm14
    vbroadcastsd -120(%rdi),  %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -112(%rdi),  %ymm14
    vbroadcastsd -104(%rdi),  %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd -96(%rdi),   %ymm14
    vbroadcastsd -88(%rdi),   %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    vmovupd      -64(%rsi),   %ymm12
    vmovupd      -32(%rsi),   %ymm13

    /* K-iteration 1 */
    vbroadcastsd -80(%rdi),   %ymm14
    vbroadcastsd -72(%rdi),   %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -64(%rdi),   %ymm14
    vbroadcastsd -56(%rdi),   %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd -48(%rdi),   %ymm14
    vbroadcastsd -40(%rdi),   %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    vmovupd      0(%rsi),     %ymm12
    vmovupd      32(%rsi),    %ymm13

    /* K-iteration 2 */
    prefetcht0 1280(%rdi)

    vbroadcastsd -32(%rdi),   %ymm14
    vbroadcastsd -24(%rdi),   %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -16(%rdi),   %ymm14
    vbroadcastsd -8(%rdi),    %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd 0(%rdi),     %ymm14
    vbroadcastsd 8(%rdi),     %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    vmovupd      64(%rsi),    %ymm12
    vmovupd      96(%rsi),    %ymm13

    /* K-iteration 3 */
    prefetcht0   512(%rsi)

    vbroadcastsd 16(%rdi),    %ymm14
    vbroadcastsd 24(%rdi),    %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd 32(%rdi),    %ymm14
    vbroadcastsd 40(%rdi),    %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd 48(%rdi),    %ymm14
    vbroadcastsd 56(%rdi),    %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11

    add     $192, %rdi
    add     $256, %rsi
    vmovupd -128(%rsi), %ymm12
    vmovupd -96(%rsi),  %ymm13

    dec     %rdx
    jnz     .Lf64L_k_loop

.Lf64L_k_left:
    mov     %rcx, %rdx
    and     $3, %rdx
    jz      .Lf64L_store

    .p2align 4
.Lf64L_k_left_loop:
    vbroadcastsd -128(%rdi),  %ymm14
    vbroadcastsd -120(%rdi),  %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm0
    vfmadd231pd  %ymm14, %ymm13, %ymm1
    vfmadd231pd  %ymm15, %ymm12, %ymm2
    vfmadd231pd  %ymm15, %ymm13, %ymm3
    vbroadcastsd -112(%rdi),  %ymm14
    vbroadcastsd -104(%rdi),  %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm4
    vfmadd231pd  %ymm14, %ymm13, %ymm5
    vfmadd231pd  %ymm15, %ymm12, %ymm6
    vfmadd231pd  %ymm15, %ymm13, %ymm7
    vbroadcastsd -96(%rdi),   %ymm14
    vbroadcastsd -88(%rdi),   %ymm15
    vfmadd231pd  %ymm14, %ymm12, %ymm8
    vfmadd231pd  %ymm14, %ymm13, %ymm9
    vfmadd231pd  %ymm15, %ymm12, %ymm10
    vfmadd231pd  %ymm15, %ymm13, %ymm11
    add     $48, %rdi
    add     $64, %rsi
    vmovupd -128(%rsi), %ymm12
    vmovupd -96(%rsi),  %ymm13
    dec     %rdx
    jnz     .Lf64L_k_left_loop

.Lf64L_store:
    prefetcht0  (%rbx)
    prefetcht0  (%r10)
    prefetcht0  (%r11)
    prefetcht0  (%r12)
    prefetcht0  (%r13)
    prefetcht0  (%rbp)

    vmovupd %ymm0,  (%rbx)
    vmovupd %ymm1,  32(%rbx)
    vmovupd %ymm2,  (%r10)
    vmovupd %ymm3,  32(%r10)
    vmovupd %ymm4,  (%r11)
    vmovupd %ymm5,  32(%r11)
    vmovupd %ymm6,  (%r12)
    vmovupd %ymm7,  32(%r12)
    vmovupd %ymm8,  (%r13)
    vmovupd %ymm9,  32(%r13)
    vmovupd %ymm10, (%rbp)
    vmovupd %ymm11, 32(%rbp)

    /* Advance C, undo A bias for next iteration */
    sub     $128, %rdi              /* undo bias (K-loop left rdi biased) */
    mov     8(%rsp), %rax
    add     %rax, (%rsp)

    dec     %r14
    jnz     .Lf64L_ir_loop

    add     $16, %rsp
.Lf64L_done:
    vzeroupper
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    WIN64_RESTORE
    ret

    END_FUNC(numc_gemm_f64_6x8_avx2_loop)

GNU_STACK_NOTE
#endif /* __x86_64__ || _M_X64 */
