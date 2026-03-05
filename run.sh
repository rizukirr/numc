#!/bin/bash

# run.sh — numc development helper script
# Supports: debug, release, test, bench, cross-arm, clean, rebuild

set -e

# --- Configuration ---
BUILD_DIR="build"
CC="${CC:-clang}"
PYTHON_VENV="/tmp/npvenv2/bin/python3"
NUMC_USE_BLAS="${NUMC_USE_BLAS:-ON}"
NUMC_VENDOR_BLIS="${NUMC_VENDOR_BLIS:-ON}"
BLIS_CONFIG="${BLIS_CONFIG:-auto}"

# --- Colors ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Helpers ---
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

build() {
    local build_type=$1
    local asan=$2
    local extra_opts=$3
    
    info "Configuring ${build_type} build (CC=${CC}, ASan=${asan}, BLAS=${NUMC_USE_BLAS}, VendorBLIS=${NUMC_VENDOR_BLIS})..."
    
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_C_COMPILER="$CC" \
        -DNUMC_ENABLE_ASAN="$asan" \
        -DNUMC_USE_BLAS="$NUMC_USE_BLAS" \
        -DNUMC_VENDOR_BLIS="$NUMC_VENDOR_BLIS" \
        -DBLIS_CONFIG="$BLIS_CONFIG" \
        $extra_opts
        
    info "Building..."
    start_time=$(date +%s)
    cmake --build "$BUILD_DIR" -j$(nproc)
    end_time=$(date +%s)
    
    success "Build finished in $((end_time - start_time))s"
}

run_demos() {
    info "Running demos..."
    for demo in "$BUILD_DIR"/bin/demo_*; do
        [[ -x "$demo" ]] || continue
        echo -e "\n${YELLOW}>>> $(basename "$demo")${NC}"
        "$demo"
    done
}

# --- Main Logic ---
COMMAND=$1
# Priority: Argument $2 > Environment $CC > Default clang
COMPILER=${2:-${CC:-clang}}
CC=$COMPILER

case $COMMAND in
    "debug")
        build Debug ON
        export LSAN_OPTIONS="suppressions=$(pwd)/tests/lsan_suppressions.txt"
        run_demos
        ;;
        
    "release")
        build Release OFF
        run_demos
        ;;
        
    "test")
        build Debug ON
        info "Running tests..."
        export LSAN_OPTIONS="suppressions=$(pwd)/tests/lsan_suppressions.txt"
        ctest --test-dir "$BUILD_DIR" --output-on-failure
        ;;
        
    "bench")
        build Release OFF
        info "Running all benchmarks..."
        for b in elemwise scalar unary pow random reduction matmul; do
            echo -e "\n${GREEN}=== numc $b benchmark ===${NC}"
            "./$BUILD_DIR/bin/bench_$b"
            
            if [[ -x "$PYTHON_VENV" ]]; then
                echo -e "\n${BLUE}=== numpy $b comparison ===${NC}"
                LD_LIBRARY_PATH="/tmp:$LD_LIBRARY_PATH" "$PYTHON_VENV" "bench/numpy_bench_$b.py" 2>/dev/null || warn "NumPy bench for $b failed"
            fi
        done
        ;;

    "bench-matmul")
        build Release OFF
        info "Running matmul showdown..."
        echo -e "\n${GREEN}=== numc matmul ===${NC}"
        "./$BUILD_DIR/bin/bench_matmul"
        if [[ -x "$PYTHON_VENV" ]]; then
            echo -e "\n${BLUE}=== numpy matmul ===${NC}"
            LD_LIBRARY_PATH="/tmp:$LD_LIBRARY_PATH" "$PYTHON_VENV" "bench/numpy_bench_matmul.py"
        fi
        ;;

    "cross-arm")
        info "Cross-compiling for AArch64 (ARM64)..."
        ARM_BUILD="build_aarch64"
        cmake -S . -B "$ARM_BUILD" \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_C_FLAGS="--target=aarch64-linux-gnu --sysroot=/usr/aarch64-linux-gnu" \
            -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
            -DCMAKE_C_COMPILER_TARGET=aarch64-linux-gnu \
            -DNUMC_USE_BLAS=OFF \
            -DCMAKE_BUILD_TYPE=Release
        cmake --build "$ARM_BUILD" -j$(nproc)
        
        info "Running matmul bench via QEMU..."
        /usr/bin/qemu-aarch64-static -L /usr/aarch64-linux-gnu "./$ARM_BUILD/bin/bench_matmul"
        ;;

    "clean")
        info "Cleaning build directories..."
        rm -rf build build_aarch64 build_test
        success "Clean complete"
        ;;

    "rebuild")
        rm -rf build
        build Debug ON
        ;;

    "help"|*)
        echo -e "${YELLOW}Usage:${NC} $0 {command} [compiler]"
        echo ""
        echo -e "${BLUE}Commands:${NC}"
        echo "  debug [cc]     Build Debug + ASan and run demos"
        echo "  release [cc]   Build Release and run demos"
        echo "  test [cc]      Build Debug + ASan and run ctest"
        echo "  bench [cc]     Build Release and run all benchmarks vs NumPy"
        echo "  bench-matmul   Run specialized matmul benchmark vs NumPy"
        echo "  cross-arm      Cross-build for AArch64 and run via QEMU"
        echo "  clean          Remove all build directories"
        echo "  rebuild [cc]   Fresh debug build"
        echo ""
        echo -e "${BLUE}Examples:${NC}"
        echo "  $0 release          # Build with default (clang)"
        echo "  $0 release gcc      # Build with gcc"
        echo "  CC=gcc $0 release   # Also works via env var"
        ;;
esac
