#include "types.h"

#define GENERATE_DTYPE_SIZE(dtype_name, c_type)                                \
  static inline size_t dtype_size_##dtype_name(void) { return sizeof(c_type); }

// Generate all data type size functions: dtype_size_BYTE, etc.
FOREACH_DTYPE(GENERATE_DTYPE_SIZE)
#undef GENERATE_DTYPE_SIZE

typedef size_t (*dtype_size_func)(void);

#define DTYPE_SIZE_ENTRY(dtype_name, c_type)                                   \
  [DTYPE_##dtype_name] = dtype_size_##dtype_name,
static const dtype_size_func dtype_size_funcs[] = {
    FOREACH_DTYPE(DTYPE_SIZE_ENTRY)};
#undef DTYPE_SIZE_ENTRY

size_t dtype_size(DType dtype) { return dtype_size_funcs[dtype](); }
