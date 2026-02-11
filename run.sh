#!/bin/bash

set -e

CC="${CC:-clang}"
CMAKE_OPTS="-DCMAKE_C_COMPILER=$CC"

build() {
    local build_type=$1
    echo "Building in $build_type mode (compiler: $CC)"
    build_start=$(date +%s%3N)
    cmake -S . -B build -DCMAKE_BUILD_TYPE="$build_type" $CMAKE_OPTS
    cmake --build build
    build_end=$(date +%s%3N)
    elapsed=$(awk "BEGIN { printf \"%.3f\", ($build_end - $build_start)/1000 }")
    echo "Build took ${elapsed} seconds"
}

case $1 in
    "debug")
        build Debug
        ;;
    "release")
        build Release
        ;;
    "test")
        echo "Running tests..."
        build Debug
        echo ""
        ctest --test-dir build
        exit 0
        ;;
    "clean")
        echo "Cleaning build directory..."
        rm -rf build
        echo "Clean complete"
        exit 0
        ;;
    "rebuild")
        echo "Rebuilding from scratch..."
        rm -rf build
        ./run.sh debug
        exit 0
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  debug              Build in debug mode with AddressSanitizer and run demo"
        echo "  release            Build in release mode with optimizations and run demo"
        echo "  test               Build in debug mode and run all tests"
        echo ""
        echo "  clean              Remove build directory"
        echo "  rebuild            Clean and rebuild in debug mode"
        echo "  help               Show this help message"
        echo "  (no args)          Run demo without rebuilding"
        echo ""
        echo "Isolated benchmarks avoid thermal effects from sequential testing."
        echo ""
        echo "Environment:"
        echo "  CC=gcc ./run.sh release   Use GCC instead of Clang"
        exit 0
        ;;
    "")
        echo "Running demo without building..."
        ;;
    *)
        echo "Unknown argument: $1"
        echo "Usage: $0 [debug|release|test|benchmark|clean|rebuild|help]"
        echo "Run '$0 help' for more information"
        exit 1
        ;;
esac

if [ ! -f "./build/bin/main" ]; then
    echo "Error: ./build/bin/main not found. Please build first."
    exit 1
fi

# Suppress known OpenMP thread pool leak in LeakSanitizer
export LSAN_OPTIONS="suppressions=$(pwd)/tests/lsan_suppressions.txt"

start=$(date +%s%3N)
./build/bin/main
end=$(date +%s%3N)
elapsed=$(awk "BEGIN { printf \"%.3f\", ($end - $start)/1000 }")
echo "Run took ${elapsed} seconds"
