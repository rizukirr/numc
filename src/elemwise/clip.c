#include "dispatch.h"
#include <numc/math.h>

/* ── Stamp out clip loop kernels ────────────────────────────────────────*/

#define STAMP_CLIP(TE, CT) DEFINE_CLIP_KERNEL(TE, CT)
GENERATE_NUMC_TYPES(STAMP_CLIP)
#undef STAMP_CLIP

/* ── Dispatch table (dtype -> kernel) ──────────────────────────────── */

static const NumcClipKernel _clip_table[] = {
    E(clip, NUMC_DTYPE_INT8),    E(clip, NUMC_DTYPE_INT16),
    E(clip, NUMC_DTYPE_INT32),   E(clip, NUMC_DTYPE_INT64),
    E(clip, NUMC_DTYPE_UINT8),   E(clip, NUMC_DTYPE_UINT16),
    E(clip, NUMC_DTYPE_UINT32),  E(clip, NUMC_DTYPE_UINT64),
    E(clip, NUMC_DTYPE_FLOAT32), E(clip, NUMC_DTYPE_FLOAT64),
};

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — Clip ops
 * ═══════════════════════════════════════════════════════════════════════ */

int numc_clip(NumcArray *a, NumcArray *out, double min, double max) {
  int err = _check_unary(a, out);
  if (err)
    return err;

  NumcClipKernel kern = _clip_table[a->dtype];

  if (a->is_contiguous && out->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, (char *)out->data, a->size, es, es, min, max);
  } else {
    size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
        po[NUMC_MAX_DIMENSIONS];
    _sort_axes_unary(a->dim, a->shape, a->strides, out->strides, ps, pa, po);
    _elemwise_clip_nd(kern, (const char *)a->data, pa, (char *)out->data, po,
                      ps, a->dim, min, max);
  }
  return 0;
}

int numc_clip_inplace(NumcArray *a, double min, double max) {
  if (!a) {
    NUMC_SET_ERROR(NUMC_ERR_NULL, "clip_inplace: NULL pointer (a=%p)", a);
    return NUMC_ERR_NULL;
  }

  NumcClipKernel kern = _clip_table[a->dtype];

  if (a->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, (char *)a->data, a->size, es, es, min, max);
  } else {
    size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
        po[NUMC_MAX_DIMENSIONS];
    _sort_axes_unary(a->dim, a->shape, a->strides, a->strides, ps, pa, po);
    _elemwise_clip_nd(kern, (const char *)a->data, pa, (char *)a->data, po, ps,
                      a->dim, min, max);
  }
  return 0;
}
