#ifndef NUMC_ERROR_H
#define NUMC_ERROR_H

#define NUMC_ERR_NULL -1
#define NUMC_ERR_ALLOC -2
#define NUMC_ERR_SHAPE -3
#define NUMC_ERR_TYPE -4
#define NUMC_ERR_CONTIGUOUS -5
#define NUMC_ERR_INVALID -6
#define NUMC_ERR_SIZE -7
#define NUMC_ERR_OVERFLOW -8
#define NUMC_ERR_BOUNDS -9
#define NUMC_ERR_AXIS -10

#define NUMC_ERR_DIM -11
typedef struct {
  int code;
  char *msg;
} NumcError;

int numc_set_error(int code, const char *msg);

NumcError numc_get_error(void);

#endif
