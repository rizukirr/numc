#!/bin/bash

# run.sh — numc development helper script
# Supports: debug, release, test, bench, cross-arm, clean, rebuild

set -e

# --- Configuration ---
BUILD_DIR="build"
CC="${CC:-clang}"
PYTHON_VENV="bench/numpy/.venv/bin/python3"
NUMC_USE_BLAS="${NUMC_USE_BLAS:-ON}"
NUMC_VENDOR_BLIS="${NUMC_VENDOR_BLIS:-ON}"
NUMC_BLAS_BACKEND="${NUMC_BLAS_BACKEND:-BLIS}"
NUMC_VENDOR_OPENBLAS="${NUMC_VENDOR_OPENBLAS:-OFF}"
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
    
    info "Configuring ${build_type} build (CC=${CC}, ASan=${asan}, BLAS=${NUMC_USE_BLAS}, Backend=${NUMC_BLAS_BACKEND})..."

    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_C_COMPILER="$CC" \
        -DNUMC_ENABLE_ASAN="$asan" \
        -DNUMC_USE_BLAS="$NUMC_USE_BLAS" \
        -DNUMC_VENDOR_BLIS="$NUMC_VENDOR_BLIS" \
        -DNUMC_BLAS_BACKEND="$NUMC_BLAS_BACKEND" \
        -DNUMC_VENDOR_OPENBLAS="$NUMC_VENDOR_OPENBLAS" \
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
        BENCH_OUT="bench/numc/results.csv"
        NUMPY_OUT="bench/numpy/results.csv"

        export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
        export OMP_PLACES="${OMP_PLACES:-cores}"

        info "Running numc CSV benchmark..."
        "./$BUILD_DIR/bin/bench_numc_csv" > "$BENCH_OUT"
        success "numc results -> $BENCH_OUT ($(wc -l < "$BENCH_OUT") rows)"

        if [[ -x "$PYTHON_VENV" ]]; then
            info "Running numpy CSV benchmark..."
            LD_LIBRARY_PATH="/tmp:$LD_LIBRARY_PATH" "$PYTHON_VENV" "bench/numpy/bench.py" > "$NUMPY_OUT"
            success "numpy results -> $NUMPY_OUT ($(wc -l < "$NUMPY_OUT") rows)"
        else
            warn "Python venv not found at $PYTHON_VENV — skipping numpy benchmark"
        fi

        GRAPH_VENV="bench/graph/.venv/bin/python3"
        if [[ -x "$GRAPH_VENV" ]] && [[ -s "$BENCH_OUT" ]] && [[ -s "$NUMPY_OUT" ]]; then
            info "Generating comparison charts..."
            "$GRAPH_VENV" bench/graph/plot.py
            success "Charts saved to bench/graph/output/"
        else
            warn "Graph venv not found or CSV files missing — skipping chart generation"
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

    "neon"|"sve"|"sve2"|"rvv")
        TARGET=$COMMAND
        SUBCMD=${2:-build}

        case $TARGET in
            neon) CROSS_BUILD="build-aarch64";     TOOLCHAIN="cmake/toolchain-aarch64.cmake" ;;
            sve)  CROSS_BUILD="build-aarch64-sve";  TOOLCHAIN="cmake/toolchain-aarch64-sve.cmake" ;;
            sve2) CROSS_BUILD="build-aarch64-sve2"; TOOLCHAIN="cmake/toolchain-aarch64-sve2.cmake" ;;
            rvv)  CROSS_BUILD="build-riscv64";      TOOLCHAIN="cmake/toolchain-riscv64.cmake" ;;
        esac

        if [[ "$SUBCMD" == "clean" ]]; then
            info "Cleaning $CROSS_BUILD..."
            rm -rf "$CROSS_BUILD"
            success "Removed $CROSS_BUILD"
            exit 0
        fi

        info "Cross-compiling for $TARGET (toolchain: $TOOLCHAIN)..."
        cmake -S . -B "$CROSS_BUILD" \
            -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
            -DCMAKE_BUILD_TYPE=Release \
            -DNUMC_USE_BLAS=OFF
        cmake --build "$CROSS_BUILD" -j$(nproc)
        success "Build complete: $CROSS_BUILD"

        if [[ "$SUBCMD" == "test" ]]; then
            info "Running tests via QEMU..."
            ctest --test-dir "$CROSS_BUILD" --output-on-failure
        elif [[ "$SUBCMD" == "bench" ]]; then
            BENCH_BIN="$CROSS_BUILD/bin/bench_numc"
            if [[ -f "$BENCH_BIN" ]]; then
                info "Running benchmarks via QEMU..."
                "$BENCH_BIN"
            else
                error "Benchmark binary not found at $BENCH_BIN"
            fi
        fi
        ;;

    "bench-blas")
        mkdir -p bench/blas_ab

        info "=== Building with BLIS backend ==="
        NUMC_BLAS_BACKEND=BLIS NUMC_VENDOR_BLIS=ON BUILD_DIR=build-blis build Release OFF
        export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
        export OMP_PLACES="${OMP_PLACES:-cores}"

        info "Running BLIS matmul benchmark..."
        ./build-blis/bin/bench_numc_csv > bench/blas_ab/blis.csv
        success "BLIS results -> bench/blas_ab/blis.csv"

        info "=== Building with OpenBLAS backend ==="
        NUMC_BLAS_BACKEND=OPENBLAS NUMC_VENDOR_OPENBLAS=ON NUMC_VENDOR_BLIS=OFF BUILD_DIR=build-openblas build Release OFF

        info "Running OpenBLAS matmul benchmark..."
        ./build-openblas/bin/bench_numc_csv > bench/blas_ab/openblas.csv
        success "OpenBLAS results -> bench/blas_ab/openblas.csv"

        info "A/B results saved to bench/blas_ab/"
        info "Compare matmul rows: grep matmul bench/blas_ab/*.csv"
        ;;

    "clean")
        info "Cleaning build directories..."
        rm -rf build build_aarch64 build_test build-blis build-openblas
        success "Clean complete"
        ;;
"rebuild")
    rm -rf build
    build Debug ON
    ;;

    "check")
        info "Running CI-simulation check..."
        
        # 1. Formatting
        info "Step 1/3: Checking code formatting (clang-format)..."
        if ! command -v clang-format &> /dev/null; then
            warn "clang-format not found, skipping..."
        else
            # Use -P for parallel execution; exclude vendored/venv dirs
            FORMAT_ERRS=$(find src include tests examples bench -type f \( -name '*.c' -o -name '*.h' \) -not -path '*/\.venv/*' -not -path '*/vendor/*' | xargs -P "$(nproc)" clang-format --dry-run --Werror 2>&1 || true)
            if [ -n "$FORMAT_ERRS" ]; then
                error "Formatting check failed! Please run: find src include tests examples bench -type f \( -name '*.c' -o -name '*.h' \) | xargs clang-format -i"
            fi
            success "Formatting is correct"
        fi

        # 2. Static Analysis
        info "Step 2/3: Running static analysis (clang-tidy)..."
        if ! command -v clang-tidy &> /dev/null; then
            warn "clang-tidy not found, skipping..."
        else
            # Ensure compilation database exists
            cmake -S . -B "$BUILD_DIR" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNUMC_BUILD_TESTS=ON > /dev/null
            # Run clang-tidy in parallel using xargs
            TIDY_ERRS=$(find src -name "*.c" | xargs -P "$(nproc)" clang-tidy -p "$BUILD_DIR" --warnings-as-errors='*' 2>&1 || true)
            if echo "$TIDY_ERRS" | grep -q "error:"; then
                echo -e "$TIDY_ERRS"
                error "Static analysis (clang-tidy) failed!"
            fi
            success "Static analysis passed"
        fi

    # 3. Comprehensive Tests
    info "Step 3/3: Running all tests with ASan..."
    build Debug ON "-DNUMC_BUILD_TESTS=ON"
    export LSAN_OPTIONS="suppressions=$(pwd)/tests/lsan_suppressions.txt"
    ctest --test-dir "$BUILD_DIR" --output-on-failure --parallel $(nproc)

    success "ALL CHECKS PASSED"
    ;;

"help"|*)
    echo -e "${YELLOW}Usage:${NC} $0 {command} [compiler|subcommand]"
    echo ""
    echo -e "${BLUE}Native Commands:${NC}"
    echo "  check [cc]     Run CI simulation (format, tidy, test+ASan)"
    echo "  debug [cc]     Build Debug + ASan and run demos"
    echo "  release [cc]   Build Release and run demos"
    echo "  test [cc]      Build Debug + ASan and run ctest"
    echo "  bench [cc]     Build Release and run CSV benchmarks"
    echo "  bench-blas     A/B benchmark: BLIS vs OpenBLAS matmul"
    echo "  cross-arm      Cross-build for AArch64 and run via QEMU"
    echo "  clean          Remove all build directories"
    echo "  rebuild [cc]   Fresh debug build"
    echo ""
    echo -e "${BLUE}Cross-compile Targets (QEMU):${NC}"
    echo "  neon [test|bench|clean]   AArch64 NEON (armv8-a baseline)"
    echo "  sve  [test|bench|clean]   AArch64 SVE  (armv8-a+sve)"
    echo "  sve2 [test|bench|clean]   AArch64 SVE2 (armv9-a)"
    echo "  rvv  [test|bench|clean]   RISC-V RVV   (rv64gcv)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 check            # Run CI simulation"
    echo "  $0 release          # Build with default (clang)"
    echo "  $0 release gcc      # Build with gcc"
    echo "  CC=gcc $0 release   # Also works via env var"
    echo "  $0 neon             # Cross-compile for NEON"
    echo "  $0 neon test        # Cross-compile + run tests via QEMU"
    echo "  $0 sve2 bench       # Cross-compile + run benchmarks via QEMU"
    echo "  $0 rvv clean        # Remove RISC-V build directory"
    ;;
esac
