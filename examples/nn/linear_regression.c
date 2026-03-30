#include "../helpers.h"
#include <stdio.h>


#define N_SAMPLES 200
#define FEATURES  2
#define EPOCHS    50
#define LR        0.01f

#define CHECK_NUMC(call)                                                  \
  do {                                                                    \
    int rc_ = (call);                                                     \
    if (rc_ < 0) {                                                        \
      NumcError err_ = numc_get_error();                                  \
      fprintf(stderr, "numc error: %s failed with code %d (%s)\n", #call, \
              err_.code, err_.msg ? err_.msg : "unknown");                \
      return -1;                                                          \
    }                                                                     \
  } while (0)

static int run(NumcCtx *ctx) {
  /* ── data generation ─────────────────────────────────────────────────── */
  /* X: shape (N, FEATURES), uniform [0, 10) */
  size_t x_shape[] = {N_SAMPLES, FEATURES};
  size_t y_shape[] = {N_SAMPLES};
  size_t w_shape[] = {FEATURES};
  size_t one_shape[] = {1};

  NumcArray *x = numc_array_rand(ctx, x_shape, 2, NUMC_DTYPE_FLOAT32);
  if (!x) {
    fprintf(stderr, "allocation failure: x\n");
    return -1;
  }
  CHECK_NUMC(numc_mul_scalar_inplace(x, 10.0));

  /* true weights: W = [3, 2], b = 1 */
  float w_true_data[] = {3.0f, 2.0f};
  float b_true_val = 1.0f;

  NumcArray *w_true = numc_array_create(ctx, w_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *y_clean = numc_array_zeros(ctx, y_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *noise = numc_array_randn(ctx, y_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *y = numc_array_zeros(ctx, y_shape, 1, NUMC_DTYPE_FLOAT32);

  if (!w_true || !y_clean || !noise || !y) {
    fprintf(stderr, "allocation failure: data arrays\n");
    return -1;
  }

  numc_array_write(w_true, w_true_data);

  /* y_clean = X . w_true  (N,F).(F,) -> (N,) */
  CHECK_NUMC(numc_dot(x, w_true, y_clean));
  /* y_clean += b_true */
  CHECK_NUMC(numc_add_scalar_inplace(y_clean, (double)b_true_val));
  /* noise *= 0.1 */
  CHECK_NUMC(numc_mul_scalar_inplace(noise, 0.1));
  /* y = y_clean + noise */
  CHECK_NUMC(numc_add(y_clean, noise, y));

  /* ── model parameters ────────────────────────────────────────────────── */
  NumcArray *w = numc_array_zeros(ctx, w_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, one_shape, 1, NUMC_DTYPE_FLOAT32);

  if (!w || !b) {
    fprintf(stderr, "allocation failure: model parameters\n");
    return -1;
  }

  /* ── pre-allocate training temporaries ───────────────────────────────── */
  /* X.T: shape (FEATURES, N) for gradient computation */
  NumcArray *x_t = numc_array_transpose_copy(x, (size_t[]){1, 0});

  /* forward pass buffers */
  NumcArray *pred = numc_array_zeros(ctx, y_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *pred_b = numc_array_zeros(ctx, y_shape, 1, NUMC_DTYPE_FLOAT32);

  /* loss intermediates */
  NumcArray *diff = numc_array_zeros(ctx, y_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *sq = numc_array_zeros(ctx, y_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *loss_out = numc_array_zeros(ctx, one_shape, 1, NUMC_DTYPE_FLOAT32);

  /* gradient accumulators */
  NumcArray *dw = numc_array_zeros(ctx, w_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *db_arr = numc_array_zeros(ctx, one_shape, 1, NUMC_DTYPE_FLOAT32);

  /* SGD scaled copies */
  NumcArray *scaled_dw = numc_array_zeros(ctx, w_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *scaled_db =
      numc_array_zeros(ctx, one_shape, 1, NUMC_DTYPE_FLOAT32);

  if (!x_t || !pred || !pred_b || !diff || !sq || !loss_out || !dw || !db_arr ||
      !scaled_dw || !scaled_db) {
    fprintf(stderr, "allocation failure: training buffers\n");
    return -1;
  }

  printf("samples: %d, features: %d, epochs: %d, lr: %.4f\n", N_SAMPLES,
         FEATURES, EPOCHS, LR);
  printf("true weights: W=[%.1f, %.1f]  b=%.1f\n\n", w_true_data[0],
         w_true_data[1], b_true_val);

  /* ── training loop ───────────────────────────────────────────────────── */
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    /* 1. pred = X . W   (N,F).(F,) -> (N,) */
    CHECK_NUMC(numc_dot(x, w, pred));
    /* 2. pred_b = pred + b  (b is shape (1,), broadcasts over N) */
    CHECK_NUMC(numc_add(pred, b, pred_b));

    /* 3. diff = pred_b - y */
    CHECK_NUMC(numc_sub(pred_b, y, diff));

    /* 4. loss = mean(diff^2) */
    CHECK_NUMC(numc_mul(diff, diff, sq));
    CHECK_NUMC(numc_mean(sq, loss_out));

    /* 5. dW = (2/N) * X.T . diff   (F,N).(N,) -> (F,) */
    CHECK_NUMC(numc_dot(x_t, diff, dw));
    CHECK_NUMC(numc_mul_scalar_inplace(dw, 2.0 / N_SAMPLES));

    /* 6. db = (2/N) * sum(diff) */
    CHECK_NUMC(numc_sum(diff, db_arr));
    CHECK_NUMC(numc_mul_scalar_inplace(db_arr, 2.0 / N_SAMPLES));

    /* 7. SGD update: W -= lr * dW,  b -= lr * db */
    CHECK_NUMC(numc_mul_scalar(dw, LR, scaled_dw));
    CHECK_NUMC(numc_mul_scalar(db_arr, LR, scaled_db));
    CHECK_NUMC(numc_sub(w, scaled_dw, w));
    CHECK_NUMC(numc_sub(b, scaled_db, b));

    const float *loss_ptr = (const float *)numc_array_data(loss_out);
    printf("epoch %2d/%d | loss %.6f\n", epoch + 1, EPOCHS, loss_ptr[0]);
  }

  /* ── final results ───────────────────────────────────────────────────── */
  const float *w_ptr = (const float *)numc_array_data(w);
  const float *b_ptr = (const float *)numc_array_data(b);

  printf("\nlearned W=[%.4f, %.4f]  b=%.4f\n", w_ptr[0], w_ptr[1], b_ptr[0]);
  printf("true    W=[%.4f, %.4f]  b=%.4f\n", w_true_data[0], w_true_data[1],
         b_true_val);

  return 0;
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "failed to create numc context\n");
    return 1;
  }

  section("Linear Regression with numc API");
  numc_manual_seed(42);

  int rc = run(ctx);
  numc_ctx_free(ctx);
  return rc == 0 ? 0 : 1;
}
