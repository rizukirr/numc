/**
 * @file memory.c
 * @brief Implementation of cross-platform aligned memory allocation.
 */

#include "memory.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h> // For _aligned_malloc() on Windows
#endif

void *aligned_calloc(size_t alignment, size_t size) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return NULL; // Alignment must be power of 2
  }

  // Round size up to multiple of alignment (required by aligned_alloc)
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

#if defined(_WIN32) || defined(_WIN64)
  // Windows: Use _aligned_malloc()
  void *ptr = _aligned_malloc(aligned_size, alignment);
  if (ptr != NULL) {
    memset(ptr, 0, aligned_size); // Zero-initialize
  }
  return ptr;
#else
  // POSIX: Use aligned_alloc()
  void *ptr = aligned_alloc(alignment, aligned_size);
  if (ptr != NULL) {
    memset(ptr, 0, aligned_size); // Zero-initialize
  }
  return ptr;
#endif
}

void aligned_free(void *ptr) {
  if (ptr == NULL) {
    return;
  }

#if defined(_WIN32) || defined(_WIN64)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

void *aligned_realloc(void *ptr, size_t alignment, size_t old_size,
                      size_t new_size) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return NULL; // Alignment must be power of 2
  }

  // Allocate new aligned memory
  void *new_ptr = aligned_calloc(alignment, new_size);
  if (new_ptr == NULL) {
    return NULL;
  }

  // Copy old data if it exists
  if (ptr != NULL && old_size > 0) {
    size_t copy_size = (old_size < new_size) ? old_size : new_size;
    memcpy(new_ptr, ptr, copy_size);
    aligned_free(ptr);
  }

  return new_ptr;
}
