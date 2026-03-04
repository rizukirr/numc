#include <numc/numc.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    NumcCtx *ctx = numc_ctx_create();
    assert(ctx != NULL);

    size_t shape[] = {4};
    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_b[] = {2.0f, 3.0f, 4.0f, 5.0f};
    
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    NumcArray *out = numc_array_zeros(ctx, (size_t[]){1}, 1, NUMC_DTYPE_FLOAT32);
    
    numc_array_write(a, data_a);
    numc_array_write(b, data_b);
    
    int err = numc_dot(a, b, out);
    assert(err == 0);
    
    float result = *(float *)numc_array_data(out);
    float expected = 1.0f*2.0f + 2.0f*3.0f + 3.0f*4.0f + 4.0f*5.0f; // 2 + 6 + 12 + 20 = 40
    
    assert(fabsf(result - expected) < 1e-6f);

    printf("Dot Product test passed!\n");
    numc_ctx_free(ctx);
    return 0;
}
