cmake_minimum_required(VERSION 4.0)
project(numc LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fsanitize=address -fno-omit-frame-pointer -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# --- numc library ---
set(NUMC_SOURCES
    src/array_core.c
    src/array_print.c
    src/alloc.c
    src/math_elemwise.c
    src/error.c
)

add_library(numc ${NUMC_SOURCES})

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(numc PUBLIC OpenMP::OpenMP_C)
endif()

target_include_directories(numc
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

# --- demo executable ---
add_executable(demo examples/demo.c)
target_link_libraries(demo PRIVATE numc)

# --- benchmarks ---
add_executable(bench_elemwise bench/bench_elemwise.c)
target_link_libraries(bench_elemwise PRIVATE numc)

add_executable(bench_scalar bench/bench_scalar.c)
target_link_libraries(bench_scalar PRIVATE numc)

# --- tests ---
enable_testing()

add_executable(test_array_core tests/test_array_core.c)
target_link_libraries(test_array_core PRIVATE numc)
add_test(NAME test_array_core COMMAND test_array_core)
