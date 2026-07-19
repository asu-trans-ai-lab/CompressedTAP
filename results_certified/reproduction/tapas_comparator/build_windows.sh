#!/bin/sh
# Windows (MinGW g++) build of TAsK with the boost_shim headers.
# The upstream src/ is UNMODIFIED; the only additions are the two
# header shims in boost_shim/boost/ (foreach.hpp -> C++11 range-for,
# tokenizer.hpp -> minimal tokenizer), which remove the Boost
# dependency entirely. Flags mirror the upstream Makefile
# (-Wall -O3 -DUSE_EXTENDED_PRECISION); -static avoids the
# libstdc++-off-PATH ABI issue documented elsewhere in this project.
set -e
cd "$(dirname "$0")"
# NOTE: -std=gnu++11 is required: the 2015-era sources use
# make_pair<int,int>(lvalue, lvalue), ill-formed under C++11 move
# semantics with modern libstdc++ (GCC 16 rejects it under gnu++11).
g++ -O3 -Wall -DUSE_EXTENDED_PRECISION -std=gnu++11 \
    -I boost_shim src/*.cpp -static -o task.exe
echo "BUILD OK: $(ls -la task.exe | awk '{print $5}') bytes"
