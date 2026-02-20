#include "../helpers.h"

static int test_error_set_get(void) {
  numc_set_error(NUMC_ERR_SHAPE, "test shape error");
  NumcError err = numc_get_error();
  ASSERT_MSG(err.code == NUMC_ERR_SHAPE, "error code should be SHAPE");
  ASSERT_MSG(err.msg != NULL, "error message should not be NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_error ===\n\n");

  printf("Error API:\n");
  RUN_TEST(test_error_set_get);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
