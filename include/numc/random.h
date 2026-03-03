#ifndef NUMC_RANDOM_H
#define NUMC_RANDOM_H

#include "numc/array.h"
#include "numc/dtype.h"
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Seed the thread-local xoshiro256** PRNG.
 *
 * Call before any numc_array_rand* function for reproducible results.
 * If never called, a fixed default seed is used automatically.
 *
 * @param seed 64-bit seed value.
 */
void numc_manual_seed(uint64_t seed);

/**
 * @brief Create an array filled with uniform random values.
 *
 * Float dtypes: uniform [0, 1) via IEEE 754 bit trick.
 * Integer dtypes: full-range random bits masked to type width.
 *
 * @param ctx   Context.
 * @param shape Array of dimensions.
 * @param dim   Number of dimensions.
 * @param dtype Data type of elements.
 * @return Pointer to the newly created array, or NULL on failure.
 */
NumcArray *numc_array_rand(NumcCtx *ctx, const size_t *shape,
                           size_t dim, NumcDType dtype);

/**
 * @brief Create an array filled with standard normal N(0,1) values.
 *
 * Uses Box-Muller transform. Float dtypes produce true N(0,1) samples.
 * Integer dtypes produce N(0,1) doubles rounded and cast to the type.
 *
 * @param ctx   Context.
 * @param shape Array of dimensions.
 * @param dim   Number of dimensions.
 * @param dtype Data type of elements.
 * @return Pointer to the newly created array, or NULL on failure.
 */
NumcArray *numc_array_randn(NumcCtx *ctx, const size_t *shape,
                            size_t dim, NumcDType dtype);

/**
 * @brief He (Kaiming) initialization: N(0, sqrt(2 / fan_in)).
 *
 * Recommended for layers followed by ReLU activations.
 *
 * @param ctx    Context.
 * @param shape  Array of dimensions.
 * @param dim    Number of dimensions.
 * @param dtype  Data type of elements.
 * @param fan_in Number of input units (e.g. in_channels * kH * kW).
 * @return Pointer to the newly created array, or NULL on failure.
 */
NumcArray *numc_array_random_he(NumcCtx *ctx, const size_t *shape,
                                size_t dim, NumcDType dtype,
                                size_t fan_in);

/**
 * @brief Xavier (Glorot) initialization: uniform [-limit, limit),
 *        where limit = sqrt(6 / (fan_in + fan_out)).
 *
 * Recommended for tanh / sigmoid activations.
 *
 * @param ctx     Context.
 * @param shape   Array of dimensions.
 * @param dim     Number of dimensions.
 * @param dtype   Data type of elements.
 * @param fan_in  Number of input units.
 * @param fan_out Number of output units.
 * @return Pointer to the newly created array, or NULL on failure.
 */
NumcArray *numc_array_random_xavier(NumcCtx *ctx, const size_t *shape,
                                    size_t dim, NumcDType dtype,
                                    size_t fan_in, size_t fan_out);

#endif /* NUMC_RANDOM_H */
