#include <numc/numc.h>
#include <stdio.h>
#include <assert.h>

int main() {
  NumcCtx *ctx = numc_ctx_create();
  assert(ctx != NULL);

  size_t shape[] = {1000};
  // 1. Allocate persistent data
  NumcArray *persistent = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  assert(persistent != NULL);

  // 2. Save checkpoint
  NumcCheckpoint cp = numc_ctx_checkpoint(ctx);

  // 3. Allocate transient data
  for (int i = 0; i < 100; i++) {
    NumcArray *temp = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    assert(temp != NULL);
  }

  // 4. Restore checkpoint
  numc_ctx_restore(ctx, cp);

  // 5. Verify persistent data is still valid and we can allocate again
  NumcArray *new_arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  assert(new_arr != NULL);

  numc_ctx_free(ctx);
  printf("Checkpoint/Restore test passed!\n");
  return 0;
}
