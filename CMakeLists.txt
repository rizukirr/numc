cmake_minimum_required(VERSION 4.0)
project(numc
        VERSION 0.1
        DESCRIPTION "N Dimensional Array in C"
        LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_DEBUG
   "-g -O0 -Wall -Wextra -Wpedantic -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wextra -Wpedantic")
set(CMAKE_C_COMPILER clang)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(BUILD_SHARED "Build shared library" OFF)

if(BUILD_SHARED)
    add_library(numc_src SHARED src/array.c src/arrio.c)
else()
    add_library(numc_src STATIC src/array.c src/arrio.c)
endif()

target_include_directories(numc_src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(numc_src PUBLIC m)

add_executable(numc main.c)
target_link_libraries(numc numc_src)
