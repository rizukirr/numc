/* Error handling implementation: formatted, context-aware messages */
#include "internal.h"
#include <numc/error.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define NUMC_ERROR_MSG_SIZE 256

static _Thread_local char _error_buf[NUMC_ERROR_MSG_SIZE];
static _Thread_local NumcError _error = {0, NULL};

int numc_set_error_v(int code,
                     const char *func,
                     const char *file,
                     int line,
                     const char *fmt, ...) {
  char formatted[NUMC_ERROR_MSG_SIZE / 2];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(formatted, sizeof(formatted), fmt, ap);
  va_end(ap);

#ifdef NUMC_DEBUG_ERROR_CONTEXT
  /* final stored form: func@file:line: message */
  snprintf(_error_buf, NUMC_ERROR_MSG_SIZE, "%s@%s:%d: %s",
           func ? func : "(unknown)", file ? file : "(unknown)", line, formatted);
#else
  /* compact stored form: func: message */
  snprintf(_error_buf, NUMC_ERROR_MSG_SIZE, "%s: %s",
           func ? func : "(unknown)", formatted);
#endif

  _error.code = code;
  _error.msg = _error_buf;
  return code;
}

/* Backward-compatible wrapper */
int numc_set_error(int code, const char *msg) {
  if (msg)
    return numc_set_error_v(code, "numc_set_error", __FILE__, __LINE__, "%s", msg);
  return numc_set_error_v(code, "numc_set_error", __FILE__, __LINE__, "%s", "");
}

void numc_log_error(const NumcError *err) {
  if (!err) return;
  if (err->code == 0) return;
  fprintf(stderr, "[ERROR] numc:%s\n", err->msg ? err->msg : "(null)");
}

NumcError numc_get_error(void) { return _error; }
