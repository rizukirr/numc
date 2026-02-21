macro(numc_demo name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE numc::numc)
endmacro()

# ── core ─────────────────────────────────────────────────────────────────────
numc_demo(demo_creation        core/demo_creation.c)
numc_demo(demo_properties      core/demo_properties.c)
numc_demo(demo_shape           core/demo_shape.c)
numc_demo(demo_error           core/demo_error.c)

# ── elemwise ─────────────────────────────────────────────────────────────────
numc_demo(demo_binary          elemwise/demo_binary.c)
numc_demo(demo_scalar          elemwise/demo_scalar.c)
numc_demo(demo_neg             elemwise/demo_neg.c)
numc_demo(demo_abs             elemwise/demo_abs.c)
numc_demo(demo_log             elemwise/demo_log.c)
numc_demo(demo_exp             elemwise/demo_exp.c)
numc_demo(demo_clip            elemwise/demo_clip.c)
numc_demo(demo_maximum_minimum elemwise/demo_maximum_minimum.c)

# ── reduction ─────────────────────────────────────────────────────────────────
numc_demo(demo_sum             reduction/demo_sum.c)
numc_demo(demo_mean            reduction/demo_mean.c)
numc_demo(demo_max_min         reduction/demo_max_min.c)
numc_demo(demo_argmax_argmin   reduction/demo_argmax_argmin.c)
