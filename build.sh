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

echo "Building with $CXX ..."
"$CXX" $CXXFLAGS code/main.cpp -o "$OUTPUT" -lpthread
echo "Done: $OUTPUT"
