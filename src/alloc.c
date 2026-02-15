#include "internal.h"
#include <stdlib.h>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h> // For _aligned_malloc() on Windows
#endif

void *numc_malloc(size_t alignment, size_t size) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return NULL; // Alignment must be power of 2
  }

  // Round size up to multiple of alignment (required by aligned_alloc)
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

#if defined(_WIN32) || defined(_WIN64)
  // Windows: Use _aligned_malloc()
  return _aligned_malloc(aligned_size, alignment);
#else
  // POSIX: Use aligned_alloc()
  return aligned_alloc(alignment, aligned_size);
#endif
}

void numc_free(void *ptr) {
  if (ptr == NULL) {
    return;
  }

#if defined(_WIN32) || defined(_WIN64)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}
