#!/bin/bash
set -e

echo "=== CLEAN BUILD: RAMULATOR2 + WRAPPER ==="

# 1) Check Environment
if [ -z "$CONDA_PREFIX" ]; then
  echo "[Error] Please run 'conda activate base' first."
  exit 1
fi

# 2) Clean clone
rm -rf ramulator2
git clone https://github.com/CMU-SAFARI/ramulator2.git
cd ramulator2
git submodule update --init --recursive

# 3) Build ramulator2 normally (NO sed hacks)
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_INCLUDE_PATH="$CONDA_PREFIX/include" \
  -DCMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib"

make -j"$(nproc)"

# 4) Locate ramulator2 library (static or shared)
# Common possibilities: libramulator2.so, libramulator2.a, libRamulator2.so, etc.
LIB_PATH=""
for f in $(find . -maxdepth 4 -type f \( -name "libramulator2.so" -o -name "libramulator2.a" -o -name "libramulator2.so" -o -name "libramulator2.a" -o -name "*ramulator2*.so" -o -name "*ramulator2*.a" \) ); do
  LIB_PATH="$(realpath "$f")"
  break
done

if [ -z "$LIB_PATH" ]; then
  echo "[Error] Could not find a built ramulator2 library (.so/.a)."
  echo "Try: find build -maxdepth 5 -type f | grep -i ramulator"
  exit 1
fi

echo "[OK] Found Ramulator2 library: $LIB_PATH"

cd ../..

# 5) Build your wrapper as shared library linked to ramulator2
mkdir -p build

g++ -std=c++20 -shared -fPIC -O3 \
  -o build/libramulator_wrapper.so \
  src/wrapper.cpp \
  -I ramulator2/src \
  -I "$CONDA_PREFIX/include" \
  "$LIB_PATH" \
  -L "$(dirname "$LIB_PATH")" \
  -Wl,-rpath,"$(dirname "$LIB_PATH")"

echo "=== SUCCESS ==="
echo "Wrapper: $(pwd)/build/libramulator_wrapper.so"
