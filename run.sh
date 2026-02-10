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
    "benchmark")
        echo "Building in release mode and running benchmarks..."
        build Release
        echo ""

        echo "╔═══════════════════════════════════════════════════════════════════╗"
        echo "║                    Running All Benchmarks                         ║"
        echo "╚═══════════════════════════════════════════════════════════════════╝"
        echo ""

        if [ -f "./build/bin/comprehensive_benchmark" ]; then
            echo "Comprehensive Benchmark"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ./build/bin/comprehensive_benchmark
            echo ""
        else
            echo "Comprehensive benchmark not found."
        fi

        exit 0
        ;;
    "benchmark_add"|"benchmark_subtract"|"benchmark_multiply"|"benchmark_divide"|"benchmark_sum"|"benchmark_min"|"benchmark_max"|"benchmark_scalar"|"benchmark_reductions")
        echo "Building in release mode and running $1..."
        build Release
        echo ""
        if [ -f "./build/bin/$1" ]; then
            ./build/bin/$1
        else
            echo "Benchmark $1 not found."
        fi
        exit 0
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  debug              Build in debug mode with AddressSanitizer and run demo"
        echo "  release            Build in release mode with optimizations and run demo"
        echo "  test               Build in debug mode and run all tests"
        echo "  benchmark          Build in release mode and run comprehensive benchmark"
        echo ""
        echo "  Isolated Operation Benchmarks (avoid thermal effects):"
        echo "    benchmark_add        ADD operation"
        echo "    benchmark_subtract   SUBTRACT operation"
        echo "    benchmark_multiply   MULTIPLY operation"
        echo "    benchmark_divide     DIVIDE operation"
        echo "    benchmark_sum        SUM reduction"
        echo "    benchmark_min        MIN reduction"
        echo "    benchmark_max        MAX reduction"
        echo "    benchmark_scalar     SCALAR operations (add/sub/mul/div with scalar)"
        echo "    benchmark_reductions OTHER reductions (prod/dot/mean/std)"
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

if [ ! -f "./build/bin/numc_demo" ]; then
    echo "Error: ./build/bin/numc_demo not found. Please build first."
    exit 1
fi

# Suppress known OpenMP thread pool leak in LeakSanitizer
export LSAN_OPTIONS="suppressions=$(pwd)/tests/lsan_suppressions.txt"

start=$(date +%s%3N)
./build/bin/numc_demo
end=$(date +%s%3N)
elapsed=$(awk "BEGIN { printf \"%.3f\", ($end - $start)/1000 }")
echo "Run took ${elapsed} seconds"
