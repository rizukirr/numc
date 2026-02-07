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
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math -Wno-pass-failed")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

# --- OpenMP ---
find_package(OpenMP)

# --- Library ---
set(NUMC_SOURCES
    src/array.c
    src/shape.c
    src/create.c
    src/math.c
    src/alloc.c
    src/error.c
    src/io.c
)

if(BUILD_SHARED)
    add_library(numc SHARED ${NUMC_SOURCES})
else()
    add_library(numc STATIC ${NUMC_SOURCES})
endif()

target_include_directories(numc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
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

# --- Examples ---
if(BUILD_EXAMPLES)
    add_executable(numc_demo examples/demo.c)
    target_link_libraries(numc_demo numc)

    add_executable(comprehensive_benchmark examples/comprehensive_benchmark.c)
    target_link_libraries(comprehensive_benchmark numc)
endif()

# --- Tests ---
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
