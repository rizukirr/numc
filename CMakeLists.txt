cmake_minimum_required(VERSION 3.10)

# --- Compiler ---
set(CMAKE_C_COMPILER "clang" CACHE STRING "C compiler")

project(numc
        VERSION 0.0.0
        DESCRIPTION "N Dimensional Array in C"
        LANGUAGES C)

# --- C Standard ---
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Output directories ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Build options ---
option(BUILD_SHARED "Build shared library" OFF)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# --- Compiler warnings (all build types) ---
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Per-build-type flags ---
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math -fwrapv -fno-math-errno -ffp-contract=fast -Wno-pass-failed")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

# --- Link-Time Optimization (LTO) for Release builds ---
# Same optimization NumPy uses - enables cross-TU inlining and optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    message(STATUS "LTO enabled for Release build")
  else()
    message(STATUS "LTO not supported: ${ipo_error}")
  endif()
endif()

# --- OpenMP ---
find_package(OpenMP)

# --- Library ---
set(NUMC_SOURCES
    src/array_core.c
    src/array_shape.c
    src/array_print.c
    src/math.c
    src/alloc.c
    src/error.c
)

if(BUILD_SHARED)
    add_library(numc SHARED ${NUMC_SOURCES})
else()
    add_library(numc STATIC ${NUMC_SOURCES})
endif()

target_include_directories(numc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(numc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(numc PUBLIC m)

if(OpenMP_C_FOUND)
    target_link_libraries(numc PUBLIC OpenMP::OpenMP_C)
    message(STATUS "OpenMP enabled (${OpenMP_C_VERSION})")
else()
    message(STATUS "OpenMP not found — building without parallelization")
endif()

set_target_properties(numc PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME "numc"
)

# Enable LTO for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND ipo_supported)
  set_target_properties(numc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(numc_demo PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(comprehensive_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_add PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_subtract PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_multiply PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_divide PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_sum PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_min PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_max PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_scalar PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(benchmark_reductions PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# --- Examples ---
if(BUILD_EXAMPLES)
    add_executable(numc_demo examples/demo.c)
    target_link_libraries(numc_demo numc)

    add_executable(comprehensive_benchmark examples/comprehensive_benchmark.c)
    target_link_libraries(comprehensive_benchmark numc)

    # Separate operation benchmarks (avoid thermal effects from sequential testing)
    add_executable(benchmark_add examples/benchmark_add.c)
    target_link_libraries(benchmark_add numc)

    add_executable(benchmark_subtract examples/benchmark_subtract.c)
    target_link_libraries(benchmark_subtract numc)

    add_executable(benchmark_multiply examples/benchmark_multiply.c)
    target_link_libraries(benchmark_multiply numc)

    add_executable(benchmark_divide examples/benchmark_divide.c)
    target_link_libraries(benchmark_divide numc)

    add_executable(benchmark_sum examples/benchmark_sum.c)
    target_link_libraries(benchmark_sum numc)

    add_executable(benchmark_min examples/benchmark_min.c)
    target_link_libraries(benchmark_min numc)

    add_executable(benchmark_max examples/benchmark_max.c)
    target_link_libraries(benchmark_max numc)

    add_executable(benchmark_scalar examples/benchmark_scalar.c)
    target_link_libraries(benchmark_scalar numc)

    add_executable(benchmark_reductions examples/benchmark_reductions.c)
    target_link_libraries(benchmark_reductions numc)
endif()

# --- Tests ---
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
