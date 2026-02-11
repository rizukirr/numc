#ifndef NUMC_ALLOC_H
#define NUMC_ALLOC_H

#include <stddef.h>

/* Aligned malloc. Returns NULL on failure. */
void *numc_malloc(size_t alignment, size_t size);

/* Free memory from numc_malloc(). NULL-safe. */
void numc_free(void *ptr);

#endif
