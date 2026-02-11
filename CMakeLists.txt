cmake_minimum_required(VERSION 4.0)
project(numc LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra -Wpedantic")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- numc library ---
set(NUMC_SOURCES
    src/array/array_core.c
    src/array/array_print.c
    src/memory/numc_alloc.c
)

if(NUMC_SOURCES)
    add_library(numc ${NUMC_SOURCES})
    target_include_directories(numc PUBLIC include PRIVATE src)
    target_compile_options(numc PRIVATE -Wall -Wextra -O2 -march=native)
endif()

# --- main executable ---
add_executable(main main.c)

if(TARGET numc)
    target_link_libraries(main PRIVATE numc)
    target_include_directories(main PRIVATE include)
else()
    target_include_directories(main PRIVATE include)
endif()
