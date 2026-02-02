#!/bin/bash

set -e

args=$1

case $args in
    "debug")
        echo "Building in debug mode"
        build_start=$(date +%s%3N)
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
        cmake --build build
        build_end=$(date +%s%3N)
        elapsed=$(awk "BEGIN { printf \"%.3f\", ($build_end - $build_start)/1000 }")
        echo "Build took ${elapsed} seconds"
        ;;
    "release")
        echo "Building in release mode"
        build_start=$(date +%s%3N)
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
        cmake --build build
        build_end=$(date +%s%3N)
        elapsed=$(awk "BEGIN { printf \"%.3f\", ($build_end - $build_start)/1000 }")
        echo "Build took ${elapsed} seconds"
        ;;
    "test")
        echo "Running tests..."
        build_start=$(date +%s%3N)
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
        cmake --build build
        build_end=$(date +%s%3N)
        elapsed=$(awk "BEGIN { printf \"%.3f\", ($build_end - $build_start)/1000 }")
        echo "Build took ${elapsed} seconds"
        echo ""
        ctest --test-dir build 
        exit 0;
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
        build_start=$(date +%s%3N)
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
        cmake --build build
        build_end=$(date +%s%3N)
        elapsed=$(awk "BEGIN { printf \"%.3f\", ($build_end - $build_start)/1000 }")
        echo "Build took ${elapsed} seconds"
        echo ""
        
        # Run all benchmarks
        echo "╔═══════════════════════════════════════════════════════════════════╗"
        echo "║                    Running All Benchmarks                         ║"
        echo "╚═══════════════════════════════════════════════════════════════════╝"
        echo ""
        
        if [ -f "./build/bin/benchmark_fill" ]; then
            echo "▶ Fill Operations Benchmark"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ./build/bin/benchmark_fill
            echo ""
        fi
        
        if [ -f "./build/bin/simd_benchmark" ]; then
            echo "▶ SIMD Operations Benchmark"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ./build/bin/simd_benchmark
            echo ""
        fi
        
        exit 0
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  debug       Build in debug mode with AddressSanitizer and run demo"
        echo "  release     Build in release mode with optimizations and run demo"
        echo "  test        Build in debug mode and run all tests"
        echo "  benchmark   Build in release mode and run all benchmarks"
        echo "  clean       Remove build directory"
        echo "  rebuild     Clean and rebuild in debug mode"
        echo "  help        Show this help message"
        echo "  (no args)   Run demo without rebuilding"
        exit 0
        ;;
    "")
        echo "Running demo without building..."
        ;;
    *)
        echo "Unknown argument: $args"
        echo "Usage: $0 [debug|release|test|benchmark|clean|rebuild|help]"
        echo "Run '$0 help' for more information"
        exit 1
        ;;
esac

if [ ! -f "./build/bin/numc_demo" ]; then
    echo "Error: ./build/bin/numc_demo not found. Please build first."
    exit 1
fi

start=$(date +%s%3N)
./build/bin/numc_demo
end=$(date +%s%3N)
elapsed=$(awk "BEGIN { printf \"%.3f\", ($end - $start)/1000 }")
echo "Run took ${elapsed} seconds"
