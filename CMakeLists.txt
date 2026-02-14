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
add_library(numc
    src/array/array_core.c
    src/array/array_print.c
    src/memory/numc_alloc.c
    src/math/math_elemwise.c
    src/numc_error.c
)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(numc PUBLIC OpenMP::OpenMP_C)
endif()

target_include_directories(numc PUBLIC include PRIVATE src)

# --- main executable ---
add_executable(main main.c)
target_link_libraries(main PRIVATE numc)

# --- benchmarks ---
add_executable(bench_array_add benchmark/array_add.c)
target_link_libraries(bench_array_add PRIVATE numc)
