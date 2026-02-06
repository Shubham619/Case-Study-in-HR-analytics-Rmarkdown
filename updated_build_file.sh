#!/bin/bash
set -e

echo "=== BUILD: RAMULATOR2 + WRAPPER (NO RE-CLONE IF EXISTS) ==="

# ---------- 1) Check environment ----------
if [ -z "$CONDA_PREFIX" ]; then
  echo "[Error] Please run: conda activate base"
  exit 1
fi

# ---------- 2) Get / update ramulator2 repo ----------
if [ ! -d "ramulator2/.git" ]; then
  echo "[Setup] ramulator2 not found. Cloning..."
  git clone https://github.com/CMU-SAFARI/ramulator2.git
else
  echo "[Setup] ramulator2 already exists. Skipping clone."
fi

cd ramulator2

# Ensure submodules are present (safe to run repeatedly)
echo "[Setup] Updating submodules..."
git submodule update --init --recursive

# ---------- 3) Configure + build ramulator2 normally ----------
echo "[Build] Building Ramulator2..."
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_INCLUDE_PATH="$CONDA_PREFIX/include" \
  -DCMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib"

make -j"$(nproc)"

# ---------- 4) Locate Ramulator library ----------
# Ramulator2 commonly builds: libramulator.so (NOT libramulator2.so)
echo "[Build] Locating built Ramulator library..."
LIB_PATH=""

# Linux: .so or static .a
for f in $(find . -maxdepth 8 -type f \( -name "libramulator.so" -o -name "libramulator.a" \) 2>/dev/null); do
  LIB_PATH="$(realpath "$f")"
  break
done

# macOS fallback: .dylib
if [ -z "$LIB_PATH" ]; then
  for f in $(find . -maxdepth 8 -type f -name "libramulator.dylib" 2>/dev/null); do
    LIB_PATH="$(realpath "$f")"
    break
  done
fi

if [ -z "$LIB_PATH" ]; then
  echo "[Error] Could not find libramulator.(so|a|dylib) under ramulator2/build."
  echo "Try running manually:"
  echo "  find . -type f | grep -i libramulator"
  exit 1
fi

echo "[OK] Found: $LIB_PATH"

# Go back to project root (where your src/wrapper.cpp is)
cd ../..

# ---------- 5) Compile your wrapper ----------
echo "[Build] Compiling wrapper -> build/libramulator_wrapper.so"
mkdir -p build

g++ -std=c++20 -shared -fPIC -O3 \
  -o build/libramulator_wrapper.so \
  src/wrapper.cpp \
  -I ramulator2/src \
  -I "$CONDA_PREFIX/include" \
  "$LIB_PATH" \
  -lyaml-cpp \
  -Wl,-rpath,"$(dirname "$LIB_PATH")"

echo "=== SUCCESS ==="
echo "Wrapper: $(pwd)/build/libramulator_wrapper.so"
echo "Ramulator: $LIB_PATH"
