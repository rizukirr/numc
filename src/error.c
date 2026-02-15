#include <numc/error.h>
#include <string.h>

#define NUMC_ERROR_MSG_SIZE 256

static _Thread_local char _error_buf[NUMC_ERROR_MSG_SIZE];
static _Thread_local NumcError _error = {0, NULL};

int numc_set_error(int code, const char *msg) {
  _error.code = code;
  if (msg) {
    size_t len = strlen(msg);
    if (len >= NUMC_ERROR_MSG_SIZE)
      len = NUMC_ERROR_MSG_SIZE - 1;
    memcpy(_error_buf, msg, len);
    _error_buf[len] = '\0';
    _error.msg = _error_buf;
  } else {
    _error_buf[0] = '\0';
    _error.msg = _error_buf;
  }
  return code;
}

NumcError numc_get_error(void) { return _error; }
