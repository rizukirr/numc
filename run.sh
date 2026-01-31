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
    "")
        echo "Running without building..."
        ;;
    *)
        echo "Unknown argument: $args"
        echo "Usage: $0 [debug|release]"
        exit 1
        ;;
esac

if [ ! -f "./build/bin/numc" ]; then
    echo "Error: ./build/bin/numc not found. Please build first."
    exit 1
fi

start=$(date +%s%3N)
./build/bin/numc
end=$(date +%s%3N)
elapsed=$(awk "BEGIN { printf \"%.3f\", ($end - $start)/1000 }")
echo "Run took ${elapsed} seconds"
