#include "../helpers.h"

static int test_ctx_create(void) {
  NumcCtx *ctx = numc_ctx_create();
  ASSERT_MSG(ctx != NULL, "ctx should not be NULL");
  numc_ctx_free(ctx);
  return 0;
}

static int test_ctx_free_null(void) {
  numc_ctx_free(NULL); // should not crash
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_ctx ===\n\n");

  printf("Context:\n");
  RUN_TEST(test_ctx_create);
  RUN_TEST(test_ctx_free_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
