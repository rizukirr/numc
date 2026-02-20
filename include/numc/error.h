#ifndef NUMC_ERROR_H
#define NUMC_ERROR_H

#define NUMC_ERR_NULL -1
#define NUMC_ERR_SHAPE -3
#define NUMC_ERR_TYPE -4

typedef struct {
  int code;
  char *msg;
} NumcError;

/* Low-level: format + set error with context (func/file/line) */
int numc_set_error_v(int code,
                     const char *func,
                     const char *file,
                     int line,
                     const char *fmt, ...);

/* Capture caller context automatically.
 * Use standard variadic macro form (no GNU `##__VA_ARGS__` token-pasting) so
 * compilers that warn about zero-length __VA_ARGS__ won't emit a warning.
 * Callers must provide at least one variadic argument (the format string).
 */
#define NUMC_SET_ERROR(code, ...) \
  numc_set_error_v(code, __func__, __FILE__, __LINE__, __VA_ARGS__)

/* Backward-compatible convenience (keeps existing API surface) */
int numc_set_error(int code, const char *msg);

/* Retrieve current thread-local error (copy of struct) */
NumcError numc_get_error(void);

/* Print error to stderr (decorated for CLI/demo): "[ERROR] numc:<message>" */
void numc_log_error(const NumcError *err);

#endif
