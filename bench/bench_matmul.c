/*
 * bench_matmul.c — numc_matmul_naive benchmark
 *
 * Sections:
 *   1. Square size scaling  — float32, sizes 32..512, reports GFLOP/s
 *   2. Dtype comparison     — all 10 dtypes at fixed 256x256
 *   3. Shape variants       — tall/square/wide at float32
 */

#include <numc/numc.h>
#include <stdio.h>
#include <time.h>

/* ── Timer ─────────────────────────────────────────────────────────── */

static double time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ── Helpers ───────────────────────────────────────────────────────── */

static const char *dtype_name(NumcDType dt) {
    static const char *names[] = {
        [NUMC_DTYPE_INT8]    = "int8",    [NUMC_DTYPE_INT16]   = "int16",
        [NUMC_DTYPE_INT32]   = "int32",   [NUMC_DTYPE_INT64]   = "int64",
        [NUMC_DTYPE_UINT8]   = "uint8",   [NUMC_DTYPE_UINT16]  = "uint16",
        [NUMC_DTYPE_UINT32]  = "uint32",  [NUMC_DTYPE_UINT64]  = "uint64",
        [NUMC_DTYPE_FLOAT32] = "float32", [NUMC_DTYPE_FLOAT64] = "float64",
    };
    return names[dt];
}

/* Fill a scalar value buffer for a given dtype */
static void fill_scalar(NumcDType dt, char buf[static 8]) {
    switch (dt) {
        case NUMC_DTYPE_INT8:    *(int8_t   *)buf = 2;    break;
        case NUMC_DTYPE_INT16:   *(int16_t  *)buf = 2;    break;
        case NUMC_DTYPE_INT32:   *(int32_t  *)buf = 2;    break;
        case NUMC_DTYPE_INT64:   *(int64_t  *)buf = 2;    break;
        case NUMC_DTYPE_UINT8:   *(uint8_t  *)buf = 2;    break;
        case NUMC_DTYPE_UINT16:  *(uint16_t *)buf = 2;    break;
        case NUMC_DTYPE_UINT32:  *(uint32_t *)buf = 2;    break;
        case NUMC_DTYPE_UINT64:  *(uint64_t *)buf = 2;    break;
        case NUMC_DTYPE_FLOAT32: *(float    *)buf = 1.0f; break;
        case NUMC_DTYPE_FLOAT64: *(double   *)buf = 1.0;  break;
    }
}

/* Run warmup + timed iters, return avg time in microseconds */
static double run(NumcArray *a, NumcArray *b, NumcArray *out,
                  int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        numc_matmul_naive(a, b, out);

    double t0 = time_us();
    for (int i = 0; i < iters; i++)
        numc_matmul_naive(a, b, out);
    return (time_us() - t0) / iters;
}

/* GFLOP/s: 2 multiply-adds per output element × M*K*N output elements */
static double gflops(size_t M, size_t K, size_t N, double us) {
    return 2.0 * (double)M * (double)K * (double)N / (us * 1e3);
}

/* ── Section 1: Square size scaling ─────────────────────────────────── */

static void bench_square_scaling(void) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  SQUARE SIZE SCALING  (float32, NxN @ NxN -> NxN)\n");
    printf("\n  %6s  %6s  %8s  %8s  %8s\n",
           "N", "iters", "time(us)", "time(ms)", "GFLOP/s");
    printf("  ─────────────────────────────────────────────────────\n");

    /* Adaptive iters: fewer for larger matrices to keep runtime sane */
    static const struct { size_t n; int warmup; int iters; } sizes[] = {
        {  32, 50, 500 },
        {  64, 20, 200 },
        { 128, 10, 50  },
        { 256,  5, 20  },
        { 512,  2, 5   },
    };
    static const int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < nsizes; s++) {
        size_t n = sizes[s].n;
        size_t sh[] = {n, n};

        NumcCtx *ctx = numc_ctx_create();
        char val[8]; fill_scalar(NUMC_DTYPE_FLOAT32, val);
        NumcArray *a   = numc_array_fill(ctx, sh, 2, NUMC_DTYPE_FLOAT32, val);
        NumcArray *b   = numc_array_fill(ctx, sh, 2, NUMC_DTYPE_FLOAT32, val);
        NumcArray *out = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
        if (!a || !b || !out) { numc_ctx_free(ctx); continue; }

        double us = run(a, b, out, sizes[s].warmup, sizes[s].iters);
        printf("  %6zu  %6d  %8.2f  %8.3f  %8.3f\n",
               n, sizes[s].iters, us, us / 1e3,
               gflops(n, n, n, us));
        numc_ctx_free(ctx);
    }
}

/* ── Section 2: Dtype comparison ────────────────────────────────────── */

static void bench_dtype_comparison(void) {
    static const size_t N = 256;

    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  DTYPE COMPARISON  (%zux%zu @ %zux%zu, 20 iters)\n", N, N, N, N);
    printf("\n  %-8s  %8s  %8s\n", "dtype", "time(us)", "GFLOP/s");
    printf("  ──────────────────────────────\n");

    static const NumcDType dtypes[] = {
        NUMC_DTYPE_INT8,    NUMC_DTYPE_INT16,
        NUMC_DTYPE_INT32,   NUMC_DTYPE_INT64,
        NUMC_DTYPE_UINT8,   NUMC_DTYPE_UINT16,
        NUMC_DTYPE_UINT32,  NUMC_DTYPE_UINT64,
        NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64,
    };
    static const int ndtypes = sizeof(dtypes) / sizeof(dtypes[0]);

    size_t sh[] = {N, N};
    for (int d = 0; d < ndtypes; d++) {
        NumcDType dt = dtypes[d];
        NumcCtx *ctx = numc_ctx_create();
        char val[8]; fill_scalar(dt, val);
        NumcArray *a   = numc_array_fill(ctx, sh, 2, dt, val);
        NumcArray *b   = numc_array_fill(ctx, sh, 2, dt, val);
        NumcArray *out = numc_array_zeros(ctx, sh, 2, dt);
        if (!a || !b || !out) { numc_ctx_free(ctx); continue; }

        double us = run(a, b, out, 5, 20);
        printf("  %-8s  %8.2f  %8.3f\n",
               dtype_name(dt), us, gflops(N, N, N, us));
        numc_ctx_free(ctx);
    }
}

/* ── Section 3: Shape variants ──────────────────────────────────────── */

static void bench_shape_variants(void) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  SHAPE VARIANTS  (float32, 20 iters)\n");
    printf("\n  %-24s  %8s  %8s  %8s\n",
           "shape (M,K)@(K,N)", "time(us)", "GFLOP/s", "flops");
    printf("  ──────────────────────────────────────────────────────\n");

    static const struct { size_t M, K, N; const char *label; } shapes[] = {
        { 512,  32, 512, "wide K (512x32@32x512)"   },
        { 512, 512, 512, "square  (512x512@512x512)" },
        {  32, 512,  32, "tall K  (32x512@512x32)"   },
        { 256, 128, 512, "rect    (256x128@128x512)" },
        {   1, 256, 256, "vec-mat (1x256@256x256)"   },
        { 256, 256,   1, "mat-vec (256x256@256x1)"   },
    };
    static const int nshapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int s = 0; s < nshapes; s++) {
        size_t M = shapes[s].M, K = shapes[s].K, N = shapes[s].N;
        size_t sha[] = {M, K}, shb[] = {K, N}, sho[] = {M, N};

        NumcCtx *ctx = numc_ctx_create();
        char val[8]; fill_scalar(NUMC_DTYPE_FLOAT32, val);
        NumcArray *a   = numc_array_fill(ctx, sha, 2, NUMC_DTYPE_FLOAT32, val);
        NumcArray *b   = numc_array_fill(ctx, shb, 2, NUMC_DTYPE_FLOAT32, val);
        NumcArray *out = numc_array_zeros(ctx, sho, 2, NUMC_DTYPE_FLOAT32);
        if (!a || !b || !out) { numc_ctx_free(ctx); continue; }

        double us = run(a, b, out, 5, 20);
        double total_flops = 2.0 * (double)M * (double)K * (double)N;
        printf("  %-24s  %8.2f  %8.3f  %8.0f K\n",
               shapes[s].label, us,
               gflops(M, K, N, us),
               total_flops / 1e3);
        numc_ctx_free(ctx);
    }
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n  numc matmul_naive benchmark\n");
    printf("  build: "
#ifdef __clang__
           "clang " __clang_version__
#elif defined(__GNUC__)
           "gcc " __VERSION__
#else
           "unknown"
#endif
#ifdef _OPENMP
           " | OpenMP"
#endif
#ifdef HAVE_BLAS
           " | BLAS"
#endif
           "\n");

    bench_square_scaling();
    bench_dtype_comparison();
    bench_shape_variants();

    printf("\n");
    return 0;
}
