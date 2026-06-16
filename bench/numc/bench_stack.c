/*
 * bench_stack.c — numc_array_stack benchmark
 *
 * stack is a pure memory-movement op: it copies n input arrays into a new
 * output array with an extra axis. Performance is dominated by copy
 * granularity, which is set by the axis position:
 *   - axis == 0      : one large memcpy per input array (fast path)
 *   - axis == ndim   : interleaved, copies a single element-slice at a time
 *
 * Measures:
 *   1. AXIS POSITION  — same data, vary where the new axis lands
 *   2. NUM ARRAYS     — fixed total size, vary n (copy granularity = total/n)
 *   3. DTYPE          — sweep all 10 dtypes, fast path vs interleaved
 *   4. SIZE SCALING   — float32 fast path across sizes
 *
 * Reports: min per-call time (us), throughput (Melem/s), bandwidth (GB/s).
 * Bandwidth counts read+write of the full output (2x bytes moved).
 */

#include "bench_common.h"

#define MAX_N 64

/* Build the stacked output shape: insert size `n` at position `axis`. */
static size_t make_out_shape(const size_t *in_shape, size_t in_dim, size_t axis,
                             size_t n, size_t *out_shape) {
  for (size_t d = 0; d <= in_dim; d++) {
    if (d < axis)
      out_shape[d] = in_shape[d];
    else if (d == axis)
      out_shape[d] = n;
    else
      out_shape[d] = in_shape[d - 1];
  }
  return in_dim + 1; /* out_dim */
}

static size_t elem_count(const size_t *shape, size_t dim) {
  size_t total = 1;
  for (size_t d = 0; d < dim; d++)
    total *= shape[d];
  return total;
}

/*
 * Core measurement: stack `n` arrays of `in_shape`/`dt` along `axis`.
 * Prints one table row. Returns total output elements (0 on alloc failure).
 */
static size_t run_stack(const char *label, NumcDType dt, const size_t *in_shape,
                        size_t in_dim, size_t n, size_t axis) {
  NumcCtx *ctx = numc_ctx_create();

  char val[8];
  fill_value(dt, val);

  NumcArray *arr[MAX_N];
  for (size_t i = 0; i < n; i++) {
    arr[i] = numc_array_fill(ctx, in_shape, in_dim, dt, val);
    if (!arr[i]) {
      fprintf(stderr, "  alloc failed (%s, n=%zu)\n", dtype_name(dt), n);
      numc_ctx_free(ctx);
      return 0;
    }
  }

  size_t out_shape[16];
  size_t out_dim = make_out_shape(in_shape, in_dim, axis, n, out_shape);
  NumcArray *out = numc_array_zeros(ctx, out_shape, out_dim, dt);
  if (!out) {
    fprintf(stderr, "  alloc failed for out (%s)\n", dtype_name(dt));
    numc_ctx_free(ctx);
    return 0;
  }

  size_t total = elem_count(out_shape, out_dim);
  size_t bytes = total * numc_dtype_size(dt);

  double us;
  BENCH_MIN_LOOP(numc_array_stack(arr, n, axis, out), BENCH_WARMUP,
                 BENCH_ITERS);

  double mops = total / us;
  double gbs = (2.0 * bytes) / (us * 1e3); /* read inputs + write output */
  printf("  %-22s %10.2f %12.1f %10.2f\n", label, us, mops, gbs);

  numc_ctx_free(ctx);
  return total;
}

static void header(const char *title) {
  printf("\n========================================================"
         "==================\n");
  printf("  %s\n", title);
  printf("\n  %-22s %10s %12s %10s\n", "config", "time (us)", "Melem/s",
         "GB/s");
  printf("  --------------------------------------------------------------\n");
}

/* -- 1. Axis position: n=4, 2D inputs, vary axis -------------------- */

static void bench_axis_position(void) {
  header("AXIS POSITION  (n=4, float32, 1024x1024 inputs)");
  size_t in_shape[] = {1024, 1024};
  char lbl[32];
  for (size_t axis = 0; axis <= 2; axis++) {
    snprintf(lbl, sizeof(lbl), "axis=%zu%s", axis,
             axis == 0   ? " (fast path)"
             : axis == 2 ? " (interleaved)"
                         : "");
    run_stack(lbl, NUMC_DTYPE_FLOAT32, in_shape, 2, 4, axis);
  }
}

/* -- 2. Number of arrays: fixed ~4M total, axis=0 ------------------- */

static void bench_num_arrays(void) {
  header("NUM ARRAYS  (float32, axis=0, fixed ~4M total elements)");
  size_t ns[] = {2, 4, 8, 16, 32};
  char lbl[32];
  for (size_t k = 0; k < sizeof(ns) / sizeof(ns[0]); k++) {
    size_t n = ns[k];
    size_t per = ((size_t)4 * 1024 * 1024) / n; /* per-array elements */
    size_t in_shape[] = {per};
    snprintf(lbl, sizeof(lbl), "n=%zu (%zu KiB/arr)", n,
             (per * sizeof(float)) / 1024);
    run_stack(lbl, NUMC_DTYPE_FLOAT32, in_shape, 1, n, 0);
  }
}

/* -- 3. Dtype sweep: n=4, fast path vs interleaved ------------------ */

static void bench_dtype(void) {
  size_t in_shape[] = {512, 512};

  header("DTYPE  (n=4, 512x512 inputs, axis=0 fast path)");
  for (int d = 0; d < N_DTYPES; d++)
    run_stack(dtype_name(ALL_DTYPES[d]), ALL_DTYPES[d], in_shape, 2, 4, 0);

  header("DTYPE  (n=4, 512x512 inputs, axis=2 interleaved)");
  for (int d = 0; d < N_DTYPES; d++)
    run_stack(dtype_name(ALL_DTYPES[d]), ALL_DTYPES[d], in_shape, 2, 4, 2);
}

/* -- 4. Size scaling: float32, n=4, axis=0 -------------------------- */

static void bench_scaling(void) {
  header("SIZE SCALING  (float32, n=4, axis=0)");
  size_t pers[] = {1024, 16384, 262144, 1048576, 4194304};
  char lbl[32];
  for (size_t k = 0; k < sizeof(pers) / sizeof(pers[0]); k++) {
    size_t per = pers[k];
    size_t in_shape[] = {per};
    snprintf(lbl, sizeof(lbl), "%zu elem/arr", per);
    run_stack(lbl, NUMC_DTYPE_FLOAT32, in_shape, 1, 4, 0);
  }
}

int main(void) {
  printf("\n  numc stack benchmark\n");
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
         "\n");

  bench_cpu_warmup();

  bench_axis_position();
  bench_num_arrays();
  bench_dtype();
  bench_scaling();

  printf("\n");
  return 0;
}
