#ifndef NUMC_DTYPE_H
#define NUMC_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#define NUMC_ALIGNOF(type) _Alignof(type)

#define NUMC_INT8 int8_t
#define NUMC_INT16 int16_t
#define NUMC_INT32 int32_t
#define NUMC_INT64 int64_t
#define NUMC_UINT8 uint8_t
#define NUMC_UINT16 uint16_t
#define NUMC_UINT32 uint32_t
#define NUMC_UINT64 uint64_t
#define NUMC_FLOAT32 float
#define NUMC_FLOAT64 double

#define GENERATE_NUMC_TYPES(NUMC_DTYPE)                                        \
  NUMC_DTYPE(NUMC_DTYPE_INT8, NUMC_INT8)                                       \
  NUMC_DTYPE(NUMC_DTYPE_INT16, NUMC_INT16)                                     \
  NUMC_DTYPE(NUMC_DTYPE_INT32, NUMC_INT32)                                     \
  NUMC_DTYPE(NUMC_DTYPE_INT64, NUMC_INT64)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT8, NUMC_UINT8)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT16, NUMC_UINT16)                                   \
  NUMC_DTYPE(NUMC_DTYPE_UINT32, NUMC_UINT32)                                   \
  NUMC_DTYPE(NUMC_DTYPE_UINT64, NUMC_UINT64)                                   \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)                                 \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64)

#define GENERATE_NONINT_NUMC_TYPES(NUMC_DTYPE)                                 \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)                                 \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64)

#define GENERATE_INT8_INT16_NUMC_TYPES(NUMC_DTYPE)                             \
  NUMC_DTYPE(NUMC_DTYPE_INT8, NUMC_INT8)                                       \
  NUMC_DTYPE(NUMC_DTYPE_INT16, NUMC_INT16)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT8, NUMC_UINT8)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT16, NUMC_UINT16)

#define GENERATE_INT32(NUMC_DTYPE)                                             \
  NUMC_DTYPE(NUMC_DTYPE_INT32, NUMC_INT32)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT32, NUMC_UINT32)

#define GENERATE_32BIT_NUMC_TYPES(NUMC_DTYPE)                                  \
  NUMC_DTYPE(NUMC_DTYPE_INT32, NUMC_INT32)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT32, NUMC_UINT32)                                   \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)

#define GENERATE_64BIT_NUMC_TYPES(NUMC_DTYPE)                                  \
  NUMC_DTYPE(NUMC_DTYPE_INT64, NUMC_INT64)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT64, NUMC_UINT64)                                   \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64)

#define GENERATE_8BIT_AND_64BIT_NUMC_TYPES(NUMC_DTYPE)                         \
  NUMC_DTYPE(NUMC_DTYPE_INT8, NUMC_INT8)                                       \
  NUMC_DTYPE(NUMC_DTYPE_UINT8, NUMC_UINT8)                                     \
  NUMC_DTYPE(NUMC_DTYPE_INT16, NUMC_INT16)                                     \
  NUMC_DTYPE(NUMC_DTYPE_UINT16, NUMC_UINT16)                                   \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)                                 \
  NUMC_DTYPE(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64)

typedef enum {
  NUMC_DTYPE_INT8,
  NUMC_DTYPE_INT16,
  NUMC_DTYPE_INT32,
  NUMC_DTYPE_INT64,
  NUMC_DTYPE_UINT8,
  NUMC_DTYPE_UINT16,
  NUMC_DTYPE_UINT32,
  NUMC_DTYPE_UINT64,
  NUMC_DTYPE_FLOAT32,
  NUMC_DTYPE_FLOAT64,
} NumcDType;

static const size_t numc_type_size[] = {
    [NUMC_DTYPE_INT8] = sizeof(NUMC_INT8),
    [NUMC_DTYPE_INT16] = sizeof(NUMC_INT16),
    [NUMC_DTYPE_INT32] = sizeof(NUMC_INT32),
    [NUMC_DTYPE_INT64] = sizeof(NUMC_INT64),
    [NUMC_DTYPE_UINT8] = sizeof(NUMC_UINT8),
    [NUMC_DTYPE_UINT16] = sizeof(NUMC_UINT16),
    [NUMC_DTYPE_UINT32] = sizeof(NUMC_UINT32),
    [NUMC_DTYPE_UINT64] = sizeof(NUMC_UINT64),
    [NUMC_DTYPE_FLOAT32] = sizeof(NUMC_FLOAT32),
    [NUMC_DTYPE_FLOAT64] = sizeof(NUMC_FLOAT64),
};

static const size_t numc_type_align[] = {
    [NUMC_DTYPE_INT8] = NUMC_ALIGNOF(NUMC_INT8),
    [NUMC_DTYPE_INT16] = NUMC_ALIGNOF(NUMC_INT16),
    [NUMC_DTYPE_INT32] = NUMC_ALIGNOF(NUMC_INT32),
    [NUMC_DTYPE_INT64] = NUMC_ALIGNOF(NUMC_INT64),
    [NUMC_DTYPE_UINT8] = NUMC_ALIGNOF(NUMC_UINT8),
    [NUMC_DTYPE_UINT16] = NUMC_ALIGNOF(NUMC_UINT16),
    [NUMC_DTYPE_UINT32] = NUMC_ALIGNOF(NUMC_UINT32),
    [NUMC_DTYPE_UINT64] = NUMC_ALIGNOF(NUMC_UINT64),
    [NUMC_DTYPE_FLOAT32] = NUMC_ALIGNOF(NUMC_FLOAT32),
    [NUMC_DTYPE_FLOAT64] = NUMC_ALIGNOF(NUMC_FLOAT64),
};

#endif
