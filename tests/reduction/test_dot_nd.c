#include <numc/numc.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

int main() {
    NumcCtx *ctx = numc_ctx_create();
    assert(ctx != NULL);

    /* 1. Scalar (0D) . ND */
    {
        float val = 2.0f;
        float data[] = {1, 2, 3, 4};
        NumcArray *a = numc_array_fill(ctx, (size_t[]){0}, 0, NUMC_DTYPE_FLOAT32, &val);
        NumcArray *b = numc_array_create(ctx, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32);
        numc_array_write(b, data);
        NumcArray *out = numc_array_create(ctx, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32);
        
        int err = numc_dot(a, b, out);
        assert(err == 0);
        float *res = (float *)numc_array_data(out);
        for(int i=0; i<4; i++) assert(res[i] == data[i] * 2.0f);
        printf("0D-1D dot passed\n");
    }

    /* 2. 1D-1D Inner Product */
    {
        float d1[] = {1, 2, 3};
        float d2[] = {4, 5, 6};
        NumcArray *a = numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_FLOAT32);
        NumcArray *b = numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_FLOAT32);
        numc_array_write(a, d1);
        numc_array_write(b, d2);
        NumcArray *out = numc_array_zeros(ctx, (size_t[]){1}, 1, NUMC_DTYPE_FLOAT32);
        
        int err = numc_dot(a, b, out);
        assert(err == 0);
        float res = *(float *)numc_array_data(out);
        assert(res == (1*4 + 2*5 + 3*6));
        printf("1D-1D dot passed\n");
    }

    /* 3. 2D-2D Matrix Multiply */
    {
        /* A (2, 3) @ B (3, 2) -> (2, 2) */
        float d1[] = {1, 2, 3, 
                      4, 5, 6};
        float d2[] = {7, 8, 
                      9, 10, 
                      11, 12};
        NumcArray *a = numc_array_create(ctx, (size_t[]){2, 3}, 2, NUMC_DTYPE_FLOAT32);
        NumcArray *b = numc_array_create(ctx, (size_t[]){3, 2}, 2, NUMC_DTYPE_FLOAT32);
        numc_array_write(a, d1);
        numc_array_write(b, d2);
        NumcArray *out = numc_array_create(ctx, (size_t[]){2, 2}, 2, NUMC_DTYPE_FLOAT32);
        
        int err = numc_dot(a, b, out);
        assert(err == 0);
        float *res = (float *)numc_array_data(out);
        assert(res[0] == 58); assert(res[1] == 64);
        assert(res[2] == 139); assert(res[3] == 154);
        printf("2D-2D dot passed\n");
    }

    /* 4. ND-1D */
    {
        /* (2, 3) . (3) -> (2) */
        float d1[] = {1, 2, 3, 
                      4, 5, 6};
        float d2[] = {1, 1, 1};
        NumcArray *a = numc_array_create(ctx, (size_t[]){2, 3}, 2, NUMC_DTYPE_FLOAT32);
        NumcArray *b = numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_FLOAT32);
        numc_array_write(a, d1);
        numc_array_write(b, d2);
        NumcArray *out = numc_array_create(ctx, (size_t[]){2}, 1, NUMC_DTYPE_FLOAT32);
        
        int err = numc_dot(a, b, out);
        assert(err == 0);
        float *res = (float *)numc_array_data(out);
        assert(res[0] == 6); assert(res[1] == 15);
        printf("2D-1D dot passed\n");
    }

    /* 5. High-D Dot: (2, 3) . (4, 3, 5) -> (2, 4, 5) */
    {
        size_t s_a[] = {2, 3};
        size_t s_b[] = {4, 3, 5};
        size_t s_o[] = {2, 4, 5};
        NumcArray *a = numc_array_fill(ctx, s_a, 2, NUMC_DTYPE_FLOAT32, &(float){1.0f});
        NumcArray *b = numc_array_fill(ctx, s_b, 3, NUMC_DTYPE_FLOAT32, &(float){1.0f});
        NumcArray *out = numc_array_zeros(ctx, s_o, 3, NUMC_DTYPE_FLOAT32);
        
        int err = numc_dot(a, b, out);
        assert(err == 0);
        float *res = (float *)numc_array_data(out);
        /* sum(1 * 1) over K=3 => each element should be 3.0 */
        for (size_t i = 0; i < 2*4*5; i++) {
            if (res[i] != 3.0f) {
                printf("Error at index %zu: expected 3.0, got %f\n", i, res[i]);
                assert(res[i] == 3.0f);
            }
        }
        printf("High-D dot passed\n");
    }

    numc_ctx_free(ctx);
    return 0;
}
