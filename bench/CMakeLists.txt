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

add_executable(bench_matmul bench_matmul.c)
target_link_libraries(bench_matmul PRIVATE numc::numc)
