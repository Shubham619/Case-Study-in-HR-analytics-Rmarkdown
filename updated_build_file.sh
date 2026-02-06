#!/bin/bash
set -e # Exit immediately on error

echo "=== LINUX BUILD STARTED ==="

# 1. Install Dependencies (Debian/Ubuntu)
if [ -x "$(command -v apt)" ]; then
    echo "[Setup] Installing dependencies via apt..."
    sudo apt update
    sudo apt install -y build-essential cmake libyaml-cpp-dev libspdlog-dev git
fi

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

# 3. Inject Helper Function (Idempotent: checks if already injected)
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

# 4. Build Ramulator Library
echo "[Build] Building Ramulator2 Core..."
mkdir -p ramulator2/build
cd ramulator2/build
# Linux requires standard cmake flags
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

# 5. Compile Wrapper
echo "[Build] Compiling Wrapper and Linking..."
mkdir -p build
g++ -shared -fPIC -o build/libramulator_wrapper.so \
    src/wrapper.cpp \
    -I ramulator2/src \
    -L ramulator2/build -lramulator2 \
    -Wl,-rpath,$(pwd)/ramulator2/build \
    -O3

echo "=== BUILD SUCCESSFUL ==="
echo "Output: $(pwd)/build/libramulator_wrapper.so"
