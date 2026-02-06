#!/bin/bash
set -e

echo "=== BUILD: RAMULATOR2 + WRAPPER (ROBUST) ==="

# 1) Check environment
if [ -z "$CONDA_PREFIX" ]; then
  echo "[Error] Please run: conda activate base"
  exit 1
fi

# 2) Clone if missing
if [ ! -d "ramulator2/.git" ]; then
  echo "[Setup] Cloning ramulator2..."
  git clone https://github.com/CMU-SAFARI/ramulator2.git
else
  echo "[Setup] ramulator2 exists. Skipping clone."
fi

cd ramulator2
git submodule update --init --recursive

# 3) Configure + build
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_INCLUDE_PATH="$CONDA_PREFIX/include" \
  -DCMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib"

# Build everything first
cmake --build . -j"$(nproc)"

# 4) Try to find libramulator.* (shared/static)
find_libramulator() {
  find . -maxdepth 8 -type f \( \
    -name "libramulator.so" -o -name "libramulator.a" -o -name "libramulator.dylib" \
  \) 2>/dev/null | head -n 1
}

LIB_CANDIDATE="$(find_libramulator || true)"

# If not found, try building common library target explicitly
if [ -z "$LIB_CANDIDATE" ]; then
  echo "[Info] libramulator.* not found yet. Trying to build library target 'ramulator'..."
  if cmake --build . --target help | grep -qi "ramulator"; then
    cmake --build . --target ramulator -j"$(nproc)" || true
  fi
  LIB_CANDIDATE="$(find_libramulator || true)"
fi

if [ -z "$LIB_CANDIDATE" ]; then
  echo "[Error] Could not find libramulator.(so|a|dylib) under ramulator2/build."
  echo "Debug commands:"
  echo "  find . -maxdepth 10 -type f | grep -i ramulator"
  echo "  cmake --build . --target help | grep -i ramulator"
  exit 1
fi

LIB_PATH="$(realpath "$LIB_CANDIDATE")"
echo "[OK] Found Ramulator library: $LIB_PATH"

cd ../..

# 5) Compile wrapper
mkdir -p build
echo "[Build] Compiling wrapper -> build/libramulator_wrapper.so"

g++ -std=c++20 -shared -fPIC -O3 \
  -o build/libramulator_wrapper.so \
  src/wrapper.cpp \
  -I ramulator2/src \
  -I "$CONDA_PREFIX/include" \
  "$LIB_PATH" \
  -lyaml-cpp \
  -Wl,-rpath,"$(dirname "$LIB_PATH")"

echo "=== SUCCESS ==="
echo "Wrapper:  $(pwd)/build/libramulator_wrapper.so"
echo "Ramulator: $LIB_PATH"
