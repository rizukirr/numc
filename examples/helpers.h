#pragma once

#include <numc/numc.h>
#include <stdint.h>
#include <stdio.h>

static void section(const char *title) {
  printf("\n══════════════════════════════════════════\n");
  printf("  %s\n", title);
  printf("══════════════════════════════════════════\n\n");
}

static void label(const char *name) { printf("--- %s ---\n", name); }
