#include "../helpers.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MNIST_ROWS    28u
#define MNIST_COLS    28u
#define MNIST_PIXELS  (MNIST_ROWS * MNIST_COLS)
#define MNIST_CLASSES 10u

#define DEFAULT_TRAIN_SAMPLES 6000u
#define DEFAULT_TEST_SAMPLES  1000u
#define DEFAULT_EPOCHS        8u
#define DEFAULT_LR            0.20f

#define CHECK_NUMC(call)                                                   \
  do {                                                                     \
    int rc_ = (call);                                                      \
    if (rc_ < 0) {                                                         \
      NumcError err_ = numc_get_error();                                   \
      fprintf(stderr, "numc error: %s failed with code %d (%s)\\n", #call, \
              err_.code, err_.msg ? err_.msg : "unknown");                 \
      return -1;                                                           \
    }                                                                      \
  } while (0)

typedef struct {
  NumcArray *images;
  NumcArray *labels;
  size_t count;
} MnistSplit;

typedef struct {
  float loss;
  float acc;
} Metrics;

static uint32_t read_u32_be(FILE *f) {
  uint8_t b[4] = {0, 0, 0, 0};
  if (fread(b, 1, 4, f) != 4) {
    return 0;
  }
  return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
         ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

static int parse_size(const char *s, size_t *out) {
  char *end = nullptr;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0') {
    return -1;
  }
  *out = (size_t)v;
  return 0;
}

static int parse_float(const char *s, float *out) {
  char *end = nullptr;
  float v = strtof(s, &end);
  if (end == s || *end != '\0') {
    return -1;
  }
  *out = v;
  return 0;
}

static int build_path(char *dst, size_t cap, const char *dir,
                      const char *file) {
  int n = snprintf(dst, cap, "%s/%s", dir, file);
  if (n <= 0 || (size_t)n >= cap) {
    return -1;
  }
  return 0;
}

static int load_mnist_split(NumcCtx *ctx, const char *img_path,
                            const char *lbl_path, size_t limit,
                            MnistSplit *out) {
  FILE *fi = nullptr;
  FILE *fl = nullptr;
#if defined(_MSC_VER)
  fopen_s(&fi, img_path, "rb");
  fopen_s(&fl, lbl_path, "rb");
#else
  fi = fopen(img_path, "rb");
  fl = fopen(lbl_path, "rb");
#endif
  uint8_t *img_u8 = nullptr;
  float *img_f32 = nullptr;
  uint8_t *labels = nullptr;

  if (!fi || !fl) {
    fprintf(stderr, "failed to open MNIST files:\\n  %s\\n  %s\\n", img_path,
            lbl_path);
    if (fi) {
      fclose(fi);
    }
    if (fl) {
      fclose(fl);
    }
    return -1;
  }

  uint32_t magic_img = read_u32_be(fi);
  uint32_t n_img = read_u32_be(fi);
  uint32_t rows = read_u32_be(fi);
  uint32_t cols = read_u32_be(fi);

  uint32_t magic_lbl = read_u32_be(fl);
  uint32_t n_lbl = read_u32_be(fl);

  if (magic_img != 2051u || magic_lbl != 2049u || rows != MNIST_ROWS ||
      cols != MNIST_COLS) {
    fprintf(
        stderr,
        "invalid MNIST IDX headers (img magic=%u, lbl magic=%u, shape=%ux%u)\n",
        magic_img, magic_lbl, rows, cols);
    fclose(fi);
    fclose(fl);
    return -1;
  }

  size_t n = (size_t)(n_img < n_lbl ? n_img : n_lbl);
  if (limit > 0 && limit < n) {
    n = limit;
  }
  if (n == 0) {
    fprintf(stderr, "empty MNIST split\n");
    fclose(fi);
    fclose(fl);
    return -1;
  }

  size_t pixels_total = n * MNIST_PIXELS;
  img_u8 = (uint8_t *)malloc(pixels_total);
  img_f32 = (float *)malloc(pixels_total * sizeof(float));
  labels = (uint8_t *)malloc(n);
  if (!img_u8 || !img_f32 || !labels) {
    fprintf(stderr, "allocation failure while loading MNIST split\n");
    free(img_u8);
    free(img_f32);
    free(labels);
    fclose(fi);
    fclose(fl);
    return -1;
  }

  if (fread(img_u8, 1, pixels_total, fi) != pixels_total ||
      fread(labels, 1, n, fl) != n) {
    fprintf(stderr, "failed to read MNIST payload bytes\n");
    free(img_u8);
    free(img_f32);
    free(labels);
    fclose(fi);
    fclose(fl);
    return -1;
  }

  for (size_t i = 0; i < pixels_total; i++) {
    img_f32[i] = (float)img_u8[i] * (1.0f / 255.0f);
  }

  size_t x_shape[] = {n, MNIST_PIXELS};
  size_t y_shape[] = {n};
  NumcArray *x = numc_array_create(ctx, x_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *y = numc_array_create(ctx, y_shape, 1, NUMC_DTYPE_UINT8);
  if (!x || !y) {
    fprintf(stderr, "failed to create NumcArray for MNIST split\n");
    free(img_u8);
    free(img_f32);
    free(labels);
    fclose(fi);
    fclose(fl);
    return -1;
  }

  numc_array_write(x, img_f32);
  numc_array_write(y, labels);

  free(img_u8);
  free(img_f32);
  free(labels);
  fclose(fi);
  fclose(fl);

  out->images = x;
  out->labels = y;
  out->count = n;
  return 0;
}

static int evaluate_softmax_classifier(NumcCtx *ctx, const NumcArray *x,
                                       const NumcArray *y,
                                       const NumcArray *y_one_hot,
                                       const NumcArray *w, const NumcArray *b,
                                       Metrics *m) {
  NumcCheckpoint cp = numc_ctx_checkpoint(ctx);
  int rc = -1;

  size_t n = numc_array_size(y);
  size_t logits_shape[] = {n, MNIST_CLASSES};
  size_t row_shape[] = {n, 1};
  size_t one_shape[] = {1};
  size_t pred_shape[] = {n};

  NumcArray *logits0 =
      numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *logits =
      numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *row_max = numc_array_zeros(ctx, row_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *shifted =
      numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *exp_logits =
      numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *row_sum = numc_array_zeros(ctx, row_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *probs = numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *probs_eps =
      numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *log_probs =
      numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *mul = numc_array_zeros(ctx, logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *loss_sum = numc_array_zeros(ctx, one_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *preds = numc_array_zeros(ctx, pred_shape, 1, NUMC_DTYPE_INT64);

  if (!logits0 || !logits || !row_max || !shifted || !exp_logits || !row_sum ||
      !probs || !probs_eps || !log_probs || !mul || !loss_sum || !preds) {
    fprintf(stderr, "allocation failure in evaluate_softmax_classifier\n");
    goto cleanup;
  }

  if (numc_matmul(x, w, logits0) < 0 || numc_add(logits0, b, logits) < 0 ||
      numc_max_axis(logits, 1, 1, row_max) < 0 ||
      numc_sub(logits, row_max, shifted) < 0 ||
      numc_exp(shifted, exp_logits) < 0 ||
      numc_sum_axis(exp_logits, 1, 1, row_sum) < 0 ||
      numc_div(exp_logits, row_sum, probs) < 0 ||
      numc_add_scalar(probs, 1e-8, probs_eps) < 0 ||
      numc_log(probs_eps, log_probs) < 0 ||
      numc_mul(y_one_hot, log_probs, mul) < 0 || numc_sum(mul, loss_sum) < 0 ||
      numc_argmax_axis(probs, 1, 0, preds) < 0) {
    NumcError err = numc_get_error();
    fprintf(stderr, "numc evaluate error: (%d) %s\n", err.code,
            err.msg ? err.msg : "unknown");
    goto cleanup;
  }

  const float *loss_ptr = (const float *)numc_array_data(loss_sum);
  m->loss = -loss_ptr[0] / (float)n;

  const uint8_t *y_ptr = (const uint8_t *)numc_array_data(y);
  const int64_t *pred_ptr = (const int64_t *)numc_array_data(preds);
  size_t correct = 0;
  for (size_t i = 0; i < n; i++) {
    correct += (size_t)(y_ptr[i] == (uint8_t)pred_ptr[i]);
  }
  m->acc = (float)correct / (float)n;
  rc = 0;

cleanup:
  numc_ctx_restore(ctx, cp);
  return rc;
}

static int train_mnist_softmax(NumcCtx *ctx, const MnistSplit *train,
                               const MnistSplit *test, size_t epochs,
                               float lr) {
  size_t train_n = train->count;
  size_t train_logits_shape[] = {train_n, MNIST_CLASSES};
  size_t train_row_shape[] = {train_n, 1};
  size_t pred_shape[] = {train_n};
  size_t w_shape[] = {MNIST_PIXELS, MNIST_CLASSES};
  size_t b_shape[] = {MNIST_CLASSES};

  NumcArray *train_one_hot =
      numc_one_hot(ctx, train->labels, MNIST_CLASSES, NUMC_DTYPE_FLOAT32);
  NumcArray *test_one_hot =
      numc_one_hot(ctx, test->labels, MNIST_CLASSES, NUMC_DTYPE_FLOAT32);
  NumcArray *x_t = numc_array_transpose_copy(train->images, (size_t[]){1, 0});

  NumcArray *w = numc_array_random_xavier(ctx, w_shape, 2, NUMC_DTYPE_FLOAT32,
                                          MNIST_PIXELS, MNIST_CLASSES);
  NumcArray *b = numc_array_zeros(ctx, b_shape, 1, NUMC_DTYPE_FLOAT32);

  NumcArray *logits0 =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *logits =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *row_max =
      numc_array_zeros(ctx, train_row_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *shifted =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *exp_logits =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *row_sum =
      numc_array_zeros(ctx, train_row_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *probs =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *probs_eps =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *log_probs =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *mul =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *dlogits =
      numc_array_zeros(ctx, train_logits_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *dw = numc_array_zeros(ctx, w_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *db = numc_array_zeros(ctx, b_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *scaled_dw = numc_array_zeros(ctx, w_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *scaled_db = numc_array_zeros(ctx, b_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *loss_sum =
      numc_array_zeros(ctx, (size_t[]){1}, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *preds = numc_array_zeros(ctx, pred_shape, 1, NUMC_DTYPE_INT64);

  if (!train_one_hot || !test_one_hot || !x_t || !w || !b || !logits0 ||
      !logits || !row_max || !shifted || !exp_logits || !row_sum || !probs ||
      !probs_eps || !log_probs || !mul || !dlogits || !dw || !db ||
      !scaled_dw || !scaled_db || !loss_sum || !preds) {
    fprintf(stderr, "allocation failure while setting up training tensors\n");
    return -1;
  }

  printf("train samples: %zu, test samples: %zu, epochs: %zu, lr: %.4f\n",
         train->count, test->count, epochs, lr);
  printf(
      "model: linear(784 -> 10) + softmax (full-batch gradient descent)\n\n");

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    CHECK_NUMC(numc_matmul(train->images, w, logits0));
    CHECK_NUMC(numc_add(logits0, b, logits));
    CHECK_NUMC(numc_max_axis(logits, 1, 1, row_max));
    CHECK_NUMC(numc_sub(logits, row_max, shifted));
    CHECK_NUMC(numc_exp(shifted, exp_logits));
    CHECK_NUMC(numc_sum_axis(exp_logits, 1, 1, row_sum));
    CHECK_NUMC(numc_div(exp_logits, row_sum, probs));

    CHECK_NUMC(numc_add_scalar(probs, 1e-8, probs_eps));
    CHECK_NUMC(numc_log(probs_eps, log_probs));
    CHECK_NUMC(numc_mul(train_one_hot, log_probs, mul));
    CHECK_NUMC(numc_sum(mul, loss_sum));

    CHECK_NUMC(numc_argmax_axis(probs, 1, 0, preds));

    const float *loss_ptr = (const float *)numc_array_data(loss_sum);
    float train_loss = -loss_ptr[0] / (float)train_n;

    const uint8_t *y_ptr = (const uint8_t *)numc_array_data(train->labels);
    const int64_t *pred_ptr = (const int64_t *)numc_array_data(preds);
    size_t correct = 0;
    for (size_t i = 0; i < train_n; i++) {
      correct += (size_t)(y_ptr[i] == (uint8_t)pred_ptr[i]);
    }
    float train_acc = (float)correct / (float)train_n;

    CHECK_NUMC(numc_sub(probs, train_one_hot, dlogits));
    CHECK_NUMC(numc_mul_scalar_inplace(dlogits, 1.0 / (double)train_n));
    CHECK_NUMC(numc_matmul(x_t, dlogits, dw));
    CHECK_NUMC(numc_sum_axis(dlogits, 0, 0, db));

    CHECK_NUMC(numc_mul_scalar(dw, lr, scaled_dw));
    CHECK_NUMC(numc_mul_scalar(db, lr, scaled_db));
    CHECK_NUMC(numc_sub(w, scaled_dw, w));
    CHECK_NUMC(numc_sub(b, scaled_db, b));

    Metrics test_metrics = {0.0f, 0.0f};
    if (evaluate_softmax_classifier(ctx, test->images, test->labels,
                                    test_one_hot, w, b, &test_metrics) < 0) {
      return -1;
    }

    printf("epoch %2zu/%zu | train loss %.4f | train acc %.2f%% | test acc "
           "%.2f%%\n",
           epoch + 1, epochs, train_loss, train_acc * 100.0f,
           test_metrics.acc * 100.0f);
  }

  Metrics final_train = {0.0f, 0.0f};
  Metrics final_test = {0.0f, 0.0f};
  if (evaluate_softmax_classifier(ctx, train->images, train->labels,
                                  train_one_hot, w, b, &final_train) < 0 ||
      evaluate_softmax_classifier(ctx, test->images, test->labels, test_one_hot,
                                  w, b, &final_test) < 0) {
    return -1;
  }

  printf("\nfinal train: loss %.4f | acc %.2f%%\n", final_train.loss,
         final_train.acc * 100.0f);
  printf("final test : loss %.4f | acc %.2f%%\n", final_test.loss,
         final_test.acc * 100.0f);

  return 0;
}

int main(int argc, char **argv) {
  const char *data_dir = "data";
  size_t train_limit = DEFAULT_TRAIN_SAMPLES;
  size_t test_limit = DEFAULT_TEST_SAMPLES;
  size_t epochs = DEFAULT_EPOCHS;
  float lr = DEFAULT_LR;

  if (argc > 1) {
    data_dir = argv[1];
  }
  if (argc > 2 && parse_size(argv[2], &train_limit) != 0) {
    fprintf(stderr, "invalid train sample count: %s\n", argv[2]);
    return 1;
  }
  if (argc > 3 && parse_size(argv[3], &test_limit) != 0) {
    fprintf(stderr, "invalid test sample count: %s\n", argv[3]);
    return 1;
  }
  if (argc > 4 && parse_size(argv[4], &epochs) != 0) {
    fprintf(stderr, "invalid epoch count: %s\n", argv[4]);
    return 1;
  }
  if (argc > 5 && parse_float(argv[5], &lr) != 0) {
    fprintf(stderr, "invalid learning rate: %s\n", argv[5]);
    return 1;
  }

  char train_img_path[512];
  char train_lbl_path[512];
  char test_img_path[512];
  char test_lbl_path[512];

  if (build_path(train_img_path, sizeof(train_img_path), data_dir,
                 "train-images-idx3-ubyte") < 0 ||
      build_path(train_lbl_path, sizeof(train_lbl_path), data_dir,
                 "train-labels-idx1-ubyte") < 0 ||
      build_path(test_img_path, sizeof(test_img_path), data_dir,
                 "t10k-images-idx3-ubyte") < 0 ||
      build_path(test_lbl_path, sizeof(test_lbl_path), data_dir,
                 "t10k-labels-idx1-ubyte") < 0) {
    fprintf(stderr, "failed to build MNIST file paths\n");
    return 1;
  }

  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "failed to create numc context\n");
    return 1;
  }

  section("MNIST Softmax Classifier with numc API");
  printf("usage: demo_mnist [data_dir] [train_samples] [test_samples] [epochs] "
         "[lr]\n");
  printf("example: demo_mnist data 6000 1000 8 0.2\n\n");

  numc_manual_seed(42);

  MnistSplit train = {0};
  MnistSplit test = {0};
  if (load_mnist_split(ctx, train_img_path, train_lbl_path, train_limit,
                       &train) < 0 ||
      load_mnist_split(ctx, test_img_path, test_lbl_path, test_limit, &test) <
          0) {
    numc_ctx_free(ctx);
    return 1;
  }

  int rc = train_mnist_softmax(ctx, &train, &test, epochs, lr);
  numc_ctx_free(ctx);
  return rc == 0 ? 0 : 1;
}
