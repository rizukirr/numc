#pragma once

#include <numc/numc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ASSERT_MSG(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg);           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

/* Like ASSERT_MSG but frees a NumcCtx before returning on failure.
 * Use in test functions that hold a ctx local and would leak on early exit. */
#define ASSERT_MSG_CTX(cond, msg, ctx)                                         \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg);           \
      numc_ctx_free(ctx);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define RUN_TEST(fn)                                                           \
  do {                                                                         \
    printf("  %-60s", #fn);                                                    \
    int _r = fn();                                                             \
    if (_r) {                                                                  \
      printf("FAIL\n");                                                        \
      fails++;                                                                 \
    } else {                                                                   \
      printf("OK\n");                                                          \
      passes++;                                                                \
    }                                                                          \
  } while (0)
