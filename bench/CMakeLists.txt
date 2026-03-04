add_executable(bench_elemwise bench_elemwise.c)
target_link_libraries(bench_elemwise PRIVATE numc::numc)

add_executable(bench_scalar bench_scalar.c)
target_link_libraries(bench_scalar PRIVATE numc::numc)

add_executable(bench_unary bench_unary.c)
target_link_libraries(bench_unary PRIVATE numc::numc)

add_executable(bench_pow bench_pow.c)
target_link_libraries(bench_pow PRIVATE numc::numc)

add_executable(bench_reduction bench_reduction.c)
target_link_libraries(bench_reduction PRIVATE numc::numc)

add_executable(bench_fma bench_fma.c)
target_link_libraries(bench_fma PRIVATE numc::numc)

add_executable(bench_dot bench_dot.c)
target_link_libraries(bench_dot PRIVATE numc::numc)

add_executable(bench_matmul bench_matmul.c)
target_link_libraries(bench_matmul PRIVATE numc::numc)
if(NUMC_VENDOR_BLIS AND BLAS_FOUND)
    target_compile_definitions(bench_matmul PRIVATE HAVE_BLAS=1 NUMC_BLIS_OPTIMIZED=1)
elseif(NUMC_USE_BLAS AND BLAS_FOUND)
    target_compile_definitions(bench_matmul PRIVATE HAVE_BLAS=1)
    if(BLAS_INCLUDE_DIRS)
        target_include_directories(bench_matmul PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()
endif()
