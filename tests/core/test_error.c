#include <assert.h>
#include <string.h>
#include <numc/error.h>

int main(void) {
  /* Use macro to set an error with formatting */
  NUMC_SET_ERROR(NUMC_ERR_NULL, "unit test error %d", 42);
  NumcError e = numc_get_error();
  assert(e.code == NUMC_ERR_NULL);
  assert(e.msg != NULL && e.msg[0] != '\0');
  /* message should include function name (macro captures caller) */
  assert(strstr(e.msg, "main") || strstr(e.msg, "numc_set_error"));
#ifdef NUMC_DEBUG_ERROR_CONTEXT
  /* debug context should include '@' (func@file:line) or a file:line pattern */
  assert(strchr(e.msg, '@') != NULL || strstr(e.msg, ".c:") != NULL);
#endif
  /* Exercise the logging helper (prints to stderr) */
  numc_log_error(&e);
  return 0;
}
