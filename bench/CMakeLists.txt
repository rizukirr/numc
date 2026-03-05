add_executable(bench_elemwise numc/bench_elemwise.c)
target_link_libraries(bench_elemwise PRIVATE numc::numc)

add_executable(bench_scalar numc/bench_scalar.c)
target_link_libraries(bench_scalar PRIVATE numc::numc)

add_executable(bench_unary numc/bench_unary.c)
target_link_libraries(bench_unary PRIVATE numc::numc)

add_executable(bench_pow numc/bench_pow.c)
target_link_libraries(bench_pow PRIVATE numc::numc)

add_executable(bench_random numc/bench_random.c)
target_link_libraries(bench_random PRIVATE numc::numc)

add_executable(bench_reduction numc/bench_reduction.c)
target_link_libraries(bench_reduction PRIVATE numc::numc)

add_executable(bench_fma numc/bench_fma.c)
target_link_libraries(bench_fma PRIVATE numc::numc)

add_executable(bench_dot numc/bench_dot.c)
target_link_libraries(bench_dot PRIVATE numc::numc)

add_executable(bench_dot_comprehensive numc/bench_dot_comprehensive.c)
target_link_libraries(bench_dot_comprehensive PRIVATE numc::numc)

add_executable(bench_matmul numc/bench_matmul.c)
target_link_libraries(bench_matmul PRIVATE numc::numc)
if(NUMC_VENDOR_BLIS AND BLAS_FOUND)
    target_compile_definitions(bench_matmul PRIVATE HAVE_BLAS=1 NUMC_BLIS_OPTIMIZED=1)
elseif(NUMC_USE_BLAS AND BLAS_FOUND)
    target_compile_definitions(bench_matmul PRIVATE HAVE_BLAS=1)
    if(BLAS_INCLUDE_DIRS)
        target_include_directories(bench_matmul PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()
endif()

# Comprehensive CSV benchmark
add_executable(bench_numc_csv numc/bench.c)
target_link_libraries(bench_numc_csv PRIVATE numc::numc)
