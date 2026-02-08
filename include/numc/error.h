#ifndef NUMC_ERROR_H
#define NUMC_ERROR_H

/**
 * @file error.h
 * @brief Error codes for numc library.
 *
 * All functions returning int use these codes.
 * Functions returning Array* return NULL on error.
 */

typedef enum {
  NUMC_OK = 0,              // Success
  NUMC_ERR_NULL = -1,       // NULL argument passed
  NUMC_ERR_ALLOC = -2,      // Memory allocation failed
  NUMC_ERR_SHAPE = -3,      // Shape mismatch between arrays
  NUMC_ERR_TYPE = -4,       // Type mismatch between arrays
  NUMC_ERR_CONTIGUOUS = -5, // Array must be contiguous
  NUMC_ERR_INVALID = -6,  // Invalid argument (zero ndim, zero shape, bad slice)
  NUMC_ERR_SIZE = -7,     // Size mismatch (e.g. reshape total differs)
  NUMC_ERR_OVERFLOW = -8, // Integer overflow in size computation
  NUMC_ERR_BOUNDS = -9,   // Index out of bounds
  NUMC_ERR_AXIS = -10,    // Invalid or duplicate axis
} NUMC_ERROR;

/**
 * @brief Set the error message.
 *
 * @param err   Error code.
 * @param msg   Error message.
 */
void numc_set_error(NUMC_ERROR err, const char *msg);

/**
 * @brief Get the error message.
 *
 * @return Error message.
 */
char *numc_get_error(void);

/**
 * @brief Get the error code.
 *
 * @return Error code.
 */
int numc_get_error_code(void);

#endif
