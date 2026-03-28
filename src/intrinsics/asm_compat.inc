/*
 * asm_compat.inc — Cross-platform x86-64 assembly compatibility macros
 *
 * Abstracts ELF vs COFF directive differences and SysV vs Win64 ABI
 * conventions.  Include at the top of every .S micro-kernel file.
 *
 * Modelled after OpenBLAS common_x86_64.h (PROLOGUE/EPILOGUE,
 * SAVEREGISTERS/RESTOREREGISTERS) but adapted for numc's approach:
 * kernel bodies stay in SysV register layout, and a thin ABI shim
 * remaps Win64 arguments at function entry/exit.
 *
 * ───────────────────────────────────────────────────────────────────
 * SysV x86-64 ABI (Linux, macOS, FreeBSD):
 *   args:  rdi, rsi, rdx, rcx, r8, r9, [rsp+8]
 *   callee-saved: rbx, rbp, r12-r15
 *   ALL xmm/ymm registers are caller-saved
 *
 * Win64 ABI (Windows):
 *   args:  rcx, rdx, r8, r9, [rsp+40], [rsp+48], [rsp+56]
 *   callee-saved: rbx, rbp, rdi, rsi, r12-r15, xmm6-xmm15
 *   32-byte shadow space above return address
 * ───────────────────────────────────────────────────────────────────
 */

/* ─── ELF vs COFF function directives ───
 *
 * ELF:  .type name, @function  /  .size name, .-name
 * COFF: no equivalent needed for clang (symbols are functions by default)
 *
 * cf. OpenBLAS common_x86_64.h PROLOGUE/EPILOGUE
 */

#if defined(_WIN32) || defined(_WIN64)
#define DECLARE_FUNC(name)
#define END_FUNC(name)
#define GNU_STACK_NOTE
#else
#define DECLARE_FUNC(name) .type name, @function
#define END_FUNC(name)     .size name, .-name
/* Mark stack as non-executable. Place once at the very end of each .S file,
 * AFTER the last END_FUNC and before the #endif guard. */
#define GNU_STACK_NOTE     .section .note.GNU-stack,"",@progbits
#endif

/* ─── Win64 callee-saved register save / restore ───
 *
 * On Win64, rdi, rsi, and xmm6-xmm15 are callee-saved.
 * Our kernels clobber all of these — save on entry, restore on exit.
 *
 * Stack cost: 2 pushes (16 bytes) + 160-byte xmm save area = 176 bytes.
 *
 * On SysV these macros expand to nothing.
 *
 * cf. OpenBLAS SAVEREGISTERS / RESTOREREGISTERS
 */

#if defined(_WIN32) || defined(_WIN64)

.macro WIN64_SAVE
    push    %rdi
    push    %rsi
    sub     $160, %rsp
    vmovdqu %xmm6,  0(%rsp)
    vmovdqu %xmm7,  16(%rsp)
    vmovdqu %xmm8,  32(%rsp)
    vmovdqu %xmm9,  48(%rsp)
    vmovdqu %xmm10, 64(%rsp)
    vmovdqu %xmm11, 80(%rsp)
    vmovdqu %xmm12, 96(%rsp)
    vmovdqu %xmm13, 112(%rsp)
    vmovdqu %xmm14, 128(%rsp)
    vmovdqu %xmm15, 144(%rsp)
.endm

.macro WIN64_RESTORE
    vmovdqu 0(%rsp),   %xmm6
    vmovdqu 16(%rsp),  %xmm7
    vmovdqu 32(%rsp),  %xmm8
    vmovdqu 48(%rsp),  %xmm9
    vmovdqu 64(%rsp),  %xmm10
    vmovdqu 80(%rsp),  %xmm11
    vmovdqu 96(%rsp),  %xmm12
    vmovdqu 112(%rsp), %xmm13
    vmovdqu 128(%rsp), %xmm14
    vmovdqu 144(%rsp), %xmm15
    add     $160, %rsp
    pop     %rsi
    pop     %rdi
.endm

/*
 * WIN64_REMAP_ARGS6 — Remap 6 Win64 arguments to SysV positions.
 *
 * Must be called immediately AFTER WIN64_SAVE, BEFORE any function pushes.
 * At this point rsp = entry_rsp - 176, so Win64 stack args are at:
 *   arg5 = [rsp + 176 + 40] = [rsp + 216]
 *   arg6 = [rsp + 176 + 48] = [rsp + 224]
 *
 * Move order avoids clobbering — each source is read before overwritten:
 *   rcx → rdi   rdx → rsi   r8 → rdx   r9 → rcx   [stk] → r8/r9
 */
.macro WIN64_REMAP_ARGS6
    mov     %rcx, %rdi
    mov     %rdx, %rsi
    mov     %r8,  %rdx
    mov     %r9,  %rcx
    mov     216(%rsp), %r8
    mov     224(%rsp), %r9
.endm

/*
 * LOOP_ARG7_OFF — offset from rsp to the 7th stack argument in _loop
 * variants, measured AFTER WIN64_SAVE (176 bytes) + 6 function pushes (48).
 *
 * Win64: arg7 at [entry_rsp + 56], rsp = entry_rsp - 224 → offset 280
 * SysV:  arg7 at [entry_rsp + 8],  rsp = entry_rsp - 48  → offset 56
 */
#define LOOP_ARG7_OFF 280

#else /* SysV — macros expand to nothing */

.macro WIN64_SAVE
.endm

.macro WIN64_RESTORE
.endm

.macro WIN64_REMAP_ARGS6
.endm

#define LOOP_ARG7_OFF 56

#endif /* _WIN32 || _WIN64 */
