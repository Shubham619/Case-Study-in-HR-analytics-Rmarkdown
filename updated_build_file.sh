#!/bin/bash
set -e

echo "=== FIXED LINUX BUILD STARTED ==="

# 1. Check for Conda
if [ -z "$CONDA_PREFIX" ]; then
    echo "[Error] You are not inside a Conda environment."
    echo "Please run 'conda activate base' first."
    exit 1
fi
echo "[Setup] Using Conda environment at: $CONDA_PREFIX"

# 2. Prepare Ramulator Directory
if [ ! -d "ramulator2" ]; then
    echo "[Setup] Cloning Ramulator2..."
    git clone https://github.com/CMU-SAFARI/ramulator2.git
    cd ramulator2
    git submodule update --init --recursive
    cd ..
else
    echo "[Setup] Ramulator2 directory found."
fi

# 3. Inject Helper Function (Idempotent)
if grep -q "ramulator_create_system" ramulator2/src/memory_system/memory_system.cpp; then
    echo "[Patch] Helper function already injected."
else
    echo "[Patch] Injecting C helper into Ramulator source..."
    cat <<EOF >> ramulator2/src/memory_system/memory_system.cpp

// --- INJECTED HELPER FOR PYTHON WRAPPER ---
#include "base/factory.h"

extern "C" {
    void* ramulator_create_system(const char* impl_name, const char* config_str) {
        try {
            YAML::Node config = YAML::Load(config_str);
            auto& factory = Ramulator::Factory<Ramulator::IMemorySystem>::instance();
            return (void*) factory.create(impl_name, config, nullptr);
        } catch (...) {
            return nullptr;
        }
    }
    
    bool ramulator_send(void* sys_ptr, long addr, int is_write) {
        auto* sys = (Ramulator::IMemorySystem*)sys_ptr;
        Ramulator::Request req(addr, is_write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read);
        return sys->send(req);
    }
    
    void ramulator_tick(void* sys_ptr) {
        auto* sys = (Ramulator::IMemorySystem*)sys_ptr;
        sys->tick();
    }
}
EOF
fi

# 4. FORCE LIBRARY CREATION (The Fix)
# We need to tell CMake to build a shared library, not just an executable.
if grep -q "add_library(ramulator2 SHARED" ramulator2/src/CMakeLists.txt; then
    echo "[Patch] CMakeLists already patched."
else
    echo "[Patch] Modifying CMakeLists.txt to force shared library creation..."
    # We append a rule to build all source files into a shared library named 'ramulator2'
    cat <<EOF >> ramulator2/src/CMakeLists.txt

# --- PATCH FOR PYTHON WRAPPER ---
file(GLOB_RECURSE ALL_SOURCES "*.cpp")
# Filter out main.cpp so it doesn't conflict
list(FILTER ALL_SOURCES EXCLUDE REGEX "main.cpp")
add_library(ramulator2 SHARED \${ALL_SOURCES})
target_link_libraries(ramulator2 PUBLIC yaml-cpp spdlog::spdlog)
target_include_directories(ramulator2 PUBLIC \${CMAKE_CURRENT_SOURCE_DIR})
EOF
fi

# 5. Build Ramulator Library
echo "[Build] Building Ramulator2 Core..."
mkdir -p ramulator2/build
cd ramulator2/build

cmake .. \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_INCLUDE_PATH="$CONDA_PREFIX/include" \
    -DCMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib"

make -j$(nproc)
cd ../..

# 6. Verify Library Exists
LIB_FILE="ramulator2/build/src/libramulator2.so"
if [ ! -f "$LIB_FILE" ]; then
    echo "[Error] Library file not found at $LIB_FILE"
    echo "Build failed to generate the .so file."
    exit 1
fi
echo "[Build] Library found at: $LIB_FILE"

# 7. Compile Wrapper
echo "[Build] Compiling Wrapper and Linking..."
mkdir -p build

g++ -shared -fPIC -o build/libramulator_wrapper.so \
    src/wrapper.cpp \
    -I ramulator2/src \
    "$LIB_FILE" \
    -I "$CONDA_PREFIX/include" \
    -Wl,-rpath,$(pwd)/ramulator2/build/src \
    -O3

echo "=== BUILD SUCCESSFUL ==="
echo "Output: $(pwd)/build/libramulator_wrapper.so"
