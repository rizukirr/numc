#include "../helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define N_SAMPLES  300u
#define FEATURES   2u
#define K          3u
#define MAX_ITERS  20u
#define CLUSTER_SZ (N_SAMPLES / K)

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

/* True cluster centers used for data generation */
static const float true_centers[K][FEATURES] = {
    {0.0f, 0.0f},
    {5.0f, 5.0f},
    {10.0f, 0.0f},
};

/*
 * generate_data — build a (N_SAMPLES, FEATURES) float32 array.
 *
 * Three clusters, each of size CLUSTER_SZ, centered at true_centers[k].
 * Noise is N(0,1) scaled by 0.5 via numc API.
 *
 * Layout: rows [0..CLUSTER_SZ) belong to cluster 0,
 *         rows [CLUSTER_SZ..2*CLUSTER_SZ) to cluster 1, etc.
 */
static NumcArray *generate_data(NumcCtx *ctx) {
  size_t chunk_shape[] = {CLUSTER_SZ, FEATURES};
  size_t full_shape[] = {N_SAMPLES, FEATURES};

  /* Allocate output array */
  NumcArray *data = numc_array_zeros(ctx, full_shape, 2, NUMC_DTYPE_FLOAT32);
  if (!data) {
    return nullptr;
  }

  float *dst = (float *)numc_array_data(data);

  for (size_t k = 0; k < K; k++) {
    NumcCheckpoint cp = numc_ctx_checkpoint(ctx);

    /* noise ~ N(0, 0.5) */
    NumcArray *noise =
        numc_array_randn(ctx, chunk_shape, 2, NUMC_DTYPE_FLOAT32);
    if (!noise) {
      numc_ctx_restore(ctx, cp);
      return nullptr;
    }
    if (numc_mul_scalar_inplace(noise, 0.5) < 0) {
      numc_ctx_restore(ctx, cp);
      return nullptr;
    }

    /* Copy noise into the k-th block of data, then add center offsets */
    const float *src = (const float *)numc_array_data(noise);
    size_t row_start = k * CLUSTER_SZ;
    for (size_t i = 0; i < CLUSTER_SZ; i++) {
      for (size_t f = 0; f < FEATURES; f++) {
        dst[(row_start + i) * FEATURES + f] =
            src[i * FEATURES + f] + true_centers[k][f];
      }
    }

    numc_ctx_restore(ctx, cp);
  }

  return data;
}

/*
 * init_centroids — pick first K points as initial centroids (K=3, tiny).
 *
 * Returns a (K, FEATURES) float32 array.
 */
static NumcArray *init_centroids(NumcCtx *ctx, const NumcArray *x) {
  size_t shape[] = {K, FEATURES};
  NumcArray *c = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  if (!c) {
    return nullptr;
  }
  const float *xd = (const float *)numc_array_data(x);
  float *cd = (float *)numc_array_data(c);
  /* Use points 0, 100, 200 as seeds (spread across true clusters) */
  for (size_t k = 0; k < K; k++) {
    size_t row = k * CLUSTER_SZ;
    for (size_t f = 0; f < FEATURES; f++) {
      cd[k * FEATURES + f] = xd[row * FEATURES + f];
    }
  }
  return c;
}

/*
 * assign_step — compute squared L2 distance from every point to every
 * centroid via numc API, then argmin across K to get assignments.
 *
 * dist(x_i, c_k) = sum_f (x_i[f] - c_k[f])^2
 *
 * For each centroid k we:
 *   1. Broadcast-subtract centroid row from X  → diff (N, FEATURES)
 *   2. Element-wise square                     → sq  (N, FEATURES)
 *   3. sum_axis along axis=1                   → dist_k (N,)
 *   4. Write column k of distances matrix (N, K) via raw pointer
 *      (K=3, this scatter is trivially cheap)
 *
 * Returns 0 on success, -1 on error.
 * assignments[] is an output buffer of length N_SAMPLES.
 */
static int assign_step(NumcCtx *ctx, const NumcArray *x,
                       const NumcArray *centroids, int32_t *assignments) {
  size_t diff_shape[] = {N_SAMPLES, FEATURES};
  size_t dist_k_shape[] = {N_SAMPLES};
  size_t cent_row_shape[] = {FEATURES};

  /* Allocate temporaries in a scoped checkpoint */
  NumcCheckpoint cp = numc_ctx_checkpoint(ctx);

  NumcArray *cent_row =
      numc_array_zeros(ctx, cent_row_shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *diff = numc_array_zeros(ctx, diff_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *sq = numc_array_zeros(ctx, diff_shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *dist_k =
      numc_array_zeros(ctx, dist_k_shape, 1, NUMC_DTYPE_FLOAT32);

  if (!cent_row || !diff || !sq || !dist_k) {
    numc_ctx_restore(ctx, cp);
    return -1;
  }

  /* distances[i][k] stored in row-major (N, K) temp buffer */
  float dist_buf[N_SAMPLES * K];
  memset(dist_buf, 0, sizeof(dist_buf));

  const float *cd = (const float *)numc_array_data(centroids);
  float *cent_row_d = (float *)numc_array_data(cent_row);

  for (size_t k = 0; k < K; k++) {
    /* Load centroid k into cent_row (FEATURES,) */
    for (size_t f = 0; f < FEATURES; f++) {
      cent_row_d[f] = cd[k * FEATURES + f];
    }

    /* diff = X - cent_row  (broadcast: (N,F) - (F,) → (N,F)) */
    if (numc_sub(x, cent_row, diff) < 0) {
      numc_ctx_restore(ctx, cp);
      return -1;
    }
    /* sq = diff * diff */
    if (numc_mul(diff, diff, sq) < 0) {
      numc_ctx_restore(ctx, cp);
      return -1;
    }
    /* dist_k = sum(sq, axis=1)  → (N,) */
    if (numc_sum_axis(sq, 1, 0, dist_k) < 0) {
      numc_ctx_restore(ctx, cp);
      return -1;
    }

    /* Scatter into column k of dist_buf */
    const float *dk = (const float *)numc_array_data(dist_k);
    for (size_t i = 0; i < N_SAMPLES; i++) {
      dist_buf[i * K + k] = dk[i];
    }
  }

  /* Argmin over K for each point */
  for (size_t i = 0; i < N_SAMPLES; i++) {
    int32_t best = 0;
    float best_d = dist_buf[i * K + 0];
    for (size_t k = 1; k < K; k++) {
      if (dist_buf[i * K + k] < best_d) {
        best_d = dist_buf[i * K + k];
        best = (int32_t)k;
      }
    }
    assignments[i] = best;
  }

  numc_ctx_restore(ctx, cp);
  return 0;
}

/*
 * update_centroids — recompute centroids as mean of assigned points.
 *
 * K and FEATURES are tiny (3, 2), so we accumulate via raw pointer access
 * after the expensive distance computation already used numc API.
 * The centroid array is updated in-place.
 */
static void update_centroids(const NumcArray *x, const int32_t *assignments,
                             NumcArray *centroids) {
  const float *xd = (const float *)numc_array_data(x);
  float *cd = (float *)numc_array_data(centroids);

  float accum[K * FEATURES];
  size_t counts[K];
  memset(accum, 0, sizeof(accum));
  memset(counts, 0, sizeof(counts));

  for (size_t i = 0; i < N_SAMPLES; i++) {
    int32_t k = assignments[i];
    counts[k]++;
    for (size_t f = 0; f < FEATURES; f++) {
      accum[k * (size_t)FEATURES + f] += xd[i * (size_t)FEATURES + f];
    }
  }

  for (size_t k = 0; k < K; k++) {
    if (counts[k] == 0) {
      continue;
    }
    for (size_t f = 0; f < FEATURES; f++) {
      cd[k * FEATURES + f] = accum[k * FEATURES + f] / (float)counts[k];
    }
  }
}

static int run_kmeans(NumcCtx *ctx) {
  section("K-Means Clustering (numc API)");
  printf("config: N=%u, features=%u, K=%u, max_iters=%u\n\n", N_SAMPLES,
         FEATURES, K, MAX_ITERS);

  /* --- Data generation -------------------------------------------------- */
  NumcArray *x = generate_data(ctx);
  if (!x) {
    fprintf(stderr, "failed to generate data\n");
    return -1;
  }

  /* --- Centroid init ----------------------------------------------------- */
  NumcArray *centroids = init_centroids(ctx, x);
  if (!centroids) {
    fprintf(stderr, "failed to initialise centroids\n");
    return -1;
  }

  int32_t assignments[N_SAMPLES];
  int32_t prev_assignments[N_SAMPLES];
  memset(assignments, -1, sizeof(assignments));
  memset(prev_assignments, -1, sizeof(prev_assignments));

  /* --- K-means loop ------------------------------------------------------ */
  section("Training");
  size_t iter = 0;
  for (; iter < MAX_ITERS; iter++) {
    /* Assignment step */
    if (assign_step(ctx, x, centroids, assignments) < 0) {
      NumcError err = numc_get_error();
      fprintf(stderr, "assign_step failed: (%d) %s\n", err.code,
              err.msg ? err.msg : "unknown");
      return -1;
    }

    /* Count moves */
    size_t moved = 0;
    if (iter > 0) {
      for (size_t i = 0; i < N_SAMPLES; i++) {
        moved += (size_t)(assignments[i] != prev_assignments[i]);
      }
    } else {
      moved = N_SAMPLES; /* first iteration: all "moved" */
    }

    printf("iter %2zu | points moved: %zu\n", iter + 1, moved);

    /* Update step */
    update_centroids(x, assignments, centroids);

    memcpy(prev_assignments, assignments, sizeof(assignments));

    if (iter > 0 && moved == 0) {
      printf("converged after %zu iterations\n", iter + 1);
      break;
    }
  }

  /* --- Results ----------------------------------------------------------- */
  section("Results");

  /* Cluster sizes */
  size_t sizes[K];
  memset(sizes, 0, sizeof(sizes));
  for (size_t i = 0; i < N_SAMPLES; i++) {
    sizes[assignments[i]]++;
  }

  printf("cluster sizes:\n");
  for (size_t k = 0; k < K; k++) {
    printf("  cluster %zu: %zu points\n", k, sizes[k]);
  }

  printf("\nfinal centroids vs true centers:\n");
  const float *cd = (const float *)numc_array_data(centroids);
  printf("  %-8s  %-20s  %-20s\n", "cluster", "learned centroid",
         "true center");
  for (size_t k = 0; k < K; k++) {
    printf("  %-8zu  (% .3f, % .3f)         (% .3f, % .3f)\n", k,
           cd[k * FEATURES + 0], cd[k * FEATURES + 1], true_centers[k][0],
           true_centers[k][1]);
  }

  return 0;
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "failed to create numc context\n");
    return 1;
  }

  numc_manual_seed(42);

  int rc = run_kmeans(ctx);
  numc_ctx_free(ctx);
  return rc == 0 ? 0 : 1;
}
