#ifndef NUMC_ERROR_H
#define NUMC_ERROR_H

#define NUMC_ERR_NULL -1
#define NUMC_ERR_SHAPE -3
#define NUMC_ERR_TYPE -4

typedef struct {
  int code;
  char *msg;
} NumcError;

int numc_set_error(int code, const char *msg);

NumcError numc_get_error(void);

#endif
