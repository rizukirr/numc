#include "numc_error.h"
#include <string.h>

static NumcError _error = {0, NULL};

int numc_set_error(int code, const char *msg) {
  memset(&_error, 0, sizeof(NumcError));

  _error.code = code;
  memcpy(_error.msg, msg, strlen(msg) + 1);
  _error.msg[strlen(msg)] = '\0';
  return code;
}

NumcError numc_get_error(void) { return _error; }
