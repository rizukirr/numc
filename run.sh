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
    "bench")
        echo "Running benchmarks..."
        build Release
        echo ""
        echo "=== Binary element-wise benchmark ==="
        ./build/bin/bench_elemwise
        echo ""
        echo "=== Scalar element-wise benchmark ==="
        ./build/bin/bench_scalar
        echo ""
        echo "=== Unary element-wise benchmark ==="
        ./build/bin/bench_unary
        echo ""
        echo "=== Pow benchmark ==="
        ./build/bin/bench_pow
        echo ""
        echo "=== Reduction benchmark ==="
        ./build/bin/bench_reduction
        exit 0
        ;;
    "bench-elemwise")
        build Release
        ./build/bin/bench_elemwise
        exit 0
        ;;
    "bench-scalar")
        build Release
        ./build/bin/bench_scalar
        exit 0
        ;;
    "bench-unary")
        build Release
        ./build/bin/bench_unary
        exit 0
        ;;
    "bench-pow")
        build Release
        ./build/bin/bench_pow
        exit 0
        ;;
    "bench-reduction")
        build Release
        ./build/bin/bench_reduction
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
        echo "  bench              Build release and run all benchmarks"
        echo "  bench-elemwise     Build release and run binary element-wise benchmark"
        echo "  bench-scalar       Build release and run scalar element-wise benchmark"
        echo "  bench-unary        Build release and run unary element-wise benchmark"
        echo "  bench-pow          Build release and run pow benchmark"
        echo "  bench-reduction    Build release and run reduction benchmark"
        echo ""
        echo "  clean              Remove build directory"
        echo "  rebuild            Clean and rebuild in debug mode"
        echo "  help               Show this help message"
        echo "  (no args)          Run demo without rebuilding"
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

demos=(
  demo_creation demo_properties demo_shape demo_error
  demo_binary demo_scalar demo_neg demo_abs demo_log demo_exp
  demo_clip demo_maximum_minimum demo_sum
)

for demo in "${demos[@]}"; do
    if [ ! -f "./build/bin/$demo" ]; then
        echo "Error: ./build/bin/$demo not found. Please build first."
        exit 1
    fi
done

# Suppress known OpenMP thread pool leak in LeakSanitizer
export LSAN_OPTIONS="suppressions=$(pwd)/tests/lsan_suppressions.txt"

start=$(date +%s%3N)
for demo in "${demos[@]}"; do
    ./build/bin/$demo
done
end=$(date +%s%3N)
elapsed=$(awk "BEGIN { printf \"%.3f\", ($end - $start)/1000 }")
echo "Run took ${elapsed} seconds"
