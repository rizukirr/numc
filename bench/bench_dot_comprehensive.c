#include <numc/numc.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

static double time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

void bench_1d(NumcCtx *ctx, size_t N, int iters) {
    size_t shape[] = {N};
    NumcArray *a = numc_array_randn(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    NumcArray *b = numc_array_randn(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, (size_t[]){1}, 1, NUMC_DTYPE_FLOAT32);

    /* Warmup */
    numc_dot(a, b, out);

    double t0 = time_us();
    for (int i = 0; i < iters; i++) {
        numc_dot(a, b, out);
    }
    double t = (time_us() - t0) / iters;
    
    double gflops = (2.0 * N) / (t * 1e-6) / 1e9;
    printf("1D Dot (N=%zu): %8.2f us | %6.2f GFLOPS\n", N, t, gflops);
}

void bench_2d(NumcCtx *ctx, size_t M, size_t K, size_t N, int iters) {
    size_t sa[] = {M, K};
    size_t sb[] = {K, N};
    size_t so[] = {M, N};
    NumcArray *a = numc_array_randn(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *b = numc_array_randn(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

    numc_dot(a, b, out);

    double t0 = time_us();
    for (int i = 0; i < iters; i++) {
        numc_dot(a, b, out);
    }
    double t = (time_us() - t0) / iters;
    
    double gflops = (2.0 * M * K * N) / (t * 1e-6) / 1e9;
    printf("2D Dot (%zux%zu . %zux%zu): %8.2f us | %6.2f GFLOPS\n", M, K, K, N, t, gflops);
}

void bench_nd(NumcCtx *ctx, int iters) {
    /* (10, 100) . (20, 100, 50) -> (10, 20, 50) */
    size_t sa[] = {10, 100};
    size_t sb[] = {20, 100, 50};
    size_t so[] = {10, 20, 50};
    NumcArray *a = numc_array_randn(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *b = numc_array_randn(ctx, sb, 3, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, so, 3, NUMC_DTYPE_FLOAT32);

    numc_dot(a, b, out);

    double t0 = time_us();
    for (int i = 0; i < iters; i++) {
        numc_dot(a, b, out);
    }
    double t = (time_us() - t0) / iters;
    
    double ops = 2.0 * 10 * 20 * 100 * 50;
    double gflops = ops / (t * 1e-6) / 1e9;
    printf("ND Dot (10x100 . 20x100x50): %8.2f us | %6.2f GFLOPS\n", t, gflops);
}

int main() {
    NumcCtx *ctx = numc_ctx_create();
    
    printf("--- numc_dot Performance Benchmark ---\n");
    
    bench_1d(ctx, 1000, 10000);
    bench_1d(ctx, 1000000, 100);
    
    bench_2d(ctx, 128, 128, 128, 1000);
    bench_2d(ctx, 512, 512, 512, 100);
    bench_2d(ctx, 1024, 1024, 1024, 10);
    
    bench_nd(ctx, 100);
    
    numc_ctx_free(ctx);
    return 0;
}
