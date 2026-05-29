#!/usr/bin/env bash
# build.sh — builds the optimised spawn_sim from code/main.cpp
# Run from the project root:
#   bash build.sh
#
# Override the compiler:
#   CXX=clang++-18 bash build.sh

set -euo pipefail

CXX="${CXX:-g++-14}"
CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra"
OUTPUT="${OUTPUT:-spawn_sim}"
# Default: code/main.cpp.  Set TARGET=opt for code/main_opt.cpp.
case "${TARGET:-}" in
    opt)  SOURCE="code/main.cpp" ;;
    *)    SOURCE="code/main_v11.cpp" ;;
esac

echo "Building $SOURCE with $CXX ..."
"$CXX" $CXXFLAGS "$SOURCE" -o "$OUTPUT" -lpthread
echo "Done: $OUTPUT"