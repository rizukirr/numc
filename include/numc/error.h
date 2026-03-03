#ifndef NUMC_ERROR_H
#define NUMC_ERROR_H

#include "numc/export.h"
#include <stddef.h>

/**
 * @brief Error codes for numc operations.
 */
typedef enum {
  NUMC_ERR_NONE = 0,
  NUMC_ERR_NULL = -1,    /* NULL pointer where not allowed */
  NUMC_ERR_MALLOC = -2,  /* Memory allocation failure */
  NUMC_ERR_SHAPE = -3,   /* Shape/dimension mismatch */
  NUMC_ERR_TYPE = -4,    /* Dtype mismatch or unsupported type */
  NUMC_ERR_BOUNDS = -5,  /* Index out of bounds */
  NUMC_ERR_INTERNAL = -99 /* Unspecified internal error */
} NumcErrorCode;

/**
 * @brief Error information for the current thread.
 */
typedef struct {
  int code;
  const char *msg;
} NumcError;

/**
 * @brief Set the error for the current thread with formatting.
 *
 * Usually called via the NUMC_SET_ERROR macro.
 */
NUMC_API int numc_set_error_v(int code, const char *func, const char *file,
                             int line, const char *fmt, ...);

/**
 * @brief Set a simple error message for the current thread.
 */
NUMC_API int numc_set_error(int code, const char *msg);

/**
 * @brief Get the last error that occurred on the current thread.
 */
NUMC_API NumcError numc_get_error(void);

/**
 * @brief Log an error structure to stderr.
 */
NUMC_API void numc_log_error(const NumcError *err);

/**
 * @brief Macro to set an error with automatic context (function, file, line).
 */
#define NUMC_SET_ERROR(code, fmt, ...)                                         \
  numc_set_error_v(code, __func__, __FILE__, __LINE__, fmt __VA_OPT__(, ) __VA_ARGS__)

#endif /* NUMC_ERROR_H */
