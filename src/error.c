#include <numc/error.h>
#include <string.h>

#define MAX_ERROR_LEN 256

static char error_msg[MAX_ERROR_LEN];
static int error_code = NUMC_OK;

void numc_set_error(NUMC_ERROR err, const char *msg) {
  if (err == NUMC_OK)
    return;

  if (msg)
    strncpy(error_msg, msg, MAX_ERROR_LEN);
  else
    error_msg[0] = '\0';

  error_code = err;
}

char *numc_get_error(void) { return error_msg; }

int numc_get_error_code(void) { return error_code; }
