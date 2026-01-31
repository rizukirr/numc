/**
 * @file memory.h
 * @brief Cross-platform aligned memory allocation for SIMD optimization.
 *
 * Provides portable wrappers for aligned memory allocation to ensure
 * data buffers meet SIMD alignment requirements (16-byte for SSE/NEON).
 */

#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

/**
 * @brief Default alignment for array data buffers (16 bytes for SSE/NEON).
 */
#define NUMC_ALIGN 16

/**
 * @brief Allocate aligned memory with zero initialization.
 *
 * Cross-platform wrapper for aligned allocation. Uses aligned_alloc()
 * on POSIX systems and _aligned_malloc() on Windows MSVC.
 *
 * @param alignment Alignment in bytes (must be power of 2).
 * @param size      Size in bytes (must be multiple of alignment).
 * @return Pointer to aligned memory, or NULL on failure.
 */
void *aligned_calloc(size_t alignment, size_t size);

/**
 * @brief Free memory allocated by aligned_calloc().
 *
 * Cross-platform wrapper for freeing aligned memory.
 *
 * @param ptr Pointer to memory allocated by aligned_calloc().
 */
void aligned_free(void *ptr);

/**
 * @brief Reallocate aligned memory while preserving alignment.
 *
 * Allocates new aligned memory, copies old data, and frees the old pointer.
 * There is no true aligned realloc on most platforms, so this allocates fresh.
 *
 * @param ptr       Pointer to existing aligned memory (or NULL).
 * @param alignment Alignment in bytes (must be power of 2).
 * @param old_size  Size of old allocation in bytes.
 * @param new_size  Size of new allocation in bytes.
 * @return Pointer to new aligned memory, or NULL on failure.
 */
void *aligned_realloc(void *ptr, size_t alignment, size_t old_size,
                      size_t new_size);

#endif
