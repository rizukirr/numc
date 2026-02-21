#include "../helpers.h"

static void demo_broadcast(NumcCtx *ctx) {
  section("Broadcasting");

  /* ── (1,4) + (3,4) → (3,4) ─────────────────────────────────────── */
  label("Broadcast dim 0: (1,4) + (3,4)");
  {
    size_t sa[] = {1, 4}, sb[] = {3, 4}, so[] = {3, 4};
    NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

    float da[] = {1, 2, 3, 4};
    float db[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    numc_array_write(a, da);
    numc_array_write(b, db);

    printf("a:\n");
    numc_array_print(a);
    printf("b:\n");
    numc_array_print(b);

    numc_add(a, b, out);
    printf("a + b:\n");
    numc_array_print(out);
  }

  /* ── (3,1) + (1,4) → (3,4) ─────────────────────────────────────── */
  label("Both broadcast: (3,1) + (1,4)");
  {
    size_t sa[] = {3, 1}, sb[] = {1, 4}, so[] = {3, 4};
    NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

    float da[] = {1, 2, 3};
    float db[] = {10, 20, 30, 40};
    numc_array_write(a, da);
    numc_array_write(b, db);

    printf("a:\n");
    numc_array_print(a);
    printf("b:\n");
    numc_array_print(b);

    numc_add(a, b, out);
    printf("a + b:\n");
    numc_array_print(out);
  }

  /* ── (4,) + (3,4) → (3,4) ──────────────────────────────────────── */
  label("Rank mismatch: (4,) + (3,4)");
  {
    size_t sa[] = {4}, sb[] = {3, 4}, so[] = {3, 4};
    NumcArray *a = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
    NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

    float da[] = {1, 2, 3, 4};
    float db[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    numc_array_write(a, da);
    numc_array_write(b, db);

    printf("a:\n");
    numc_array_print(a);
    printf("b:\n");
    numc_array_print(b);

    numc_add(a, b, out);
    printf("a + b:\n");
    numc_array_print(out);
  }
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_broadcast(ctx);
  numc_ctx_free(ctx);
  return 0;
}
