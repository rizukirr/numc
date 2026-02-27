#ifndef NUMC_ERROR_H
#define NUMC_ERROR_H

#define NUMC_ERR_NULL -1
#define NUMC_ERR_SHAPE -3
#define NUMC_ERR_TYPE -4

typedef struct {
  int code;
  char *msg;
} NumcError;

/**
 * @brief Low-level function to format and set an error with context.
 *
 * @param code Error code.
 * @param func Function name where the error occurred.
 * @param file File name where the error occurred.
 * @param line Line number where the error occurred.
 * @param fmt  Format string for the error message.
 * @param ...  Arguments for the format string.
 * @return The error code.
 */
int numc_set_error_v(int code,
                     const char *func,
                     const char *file,
                     int line,
                     const char *fmt, ...);

/* Capture caller context automatically */
#define NUMC_SET_ERROR(code, fmt, ...) \
  numc_set_error_v(code, __func__, __FILE__, __LINE__, fmt __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Set an error message with a specific code.
 *
 * Backward-compatible convenience function.
 *
 * @param code Error code.
 * @param msg  Error message.
 * @return The error code.
 */
int numc_set_error(int code, const char *msg);

/**
 * @brief Retrieve the current thread-local error.
 *
 * @return A copy of the current NumcError struct.
 */
NumcError numc_get_error(void);

/**
 * @brief Print an error to stderr.
 *
 * Decorated for CLI/demo as "[ERROR] numc:<message>".
 *
 * @param err Pointer to the NumcError struct.
 */
void numc_log_error(const NumcError *err);

#endif
