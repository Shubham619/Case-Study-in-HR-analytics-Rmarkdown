#!/bin/bash
set -e

echo "=== INTEGRATED BUILD STARTED ==="

# 1. Check Conda
if [ -z "$CONDA_PREFIX" ]; then
    echo "[Error] Please run 'conda activate base' first."
    exit 1
fi

# 2. Reset Ramulator (Fresh Start)
if [ -d "ramulator2" ]; then
    echo "[Reset] Removing old ramulator2 folder..."
    rm -rf ramulator2
fi

echo "[Setup] Cloning Ramulator2..."
git clone https://github.com/CMU-SAFARI/ramulator2.git
cd ramulator2
git submodule update --init --recursive

# 3. THE TROJAN HORSE: Drop our wrapper code directly into src/
echo "[Setup] Injecting wrapper source code..."
cat <<EOF > src/lib_wrapper.cpp
#include "base/base.h"
#include "base/request.h"
#include "base/config.h"
#include "base/factory.h"
#include "memory_system/memory_system.h"
#include <iostream>
#include <map>
#include <vector>
#include <cstdlib>
#include <ctime>

// --- WRAPPER LOGIC ---
// Since we are inside the source tree, we can use headers directly!

static Ramulator::IMemorySystem* system_ptr = nullptr;

// Physics Globals
static std::map<long, int> physical_cell_data; 
static std::map<int, int> hammer_counters; 

struct FaultConfig {
    double p_sf = 0.01; double p_rdf = 0.01; double p_drdf = 0.01; double p_wdf = 0.01;
    double p_tcf = 0.05; double p_scf = 0.05; double p_dccf = 0.05; double p_irf = 0.01; double p_icf = 0.05;
    int hammer_thresh = 5;
};
static FaultConfig f_cfg;

extern "C" {
    void init_simulator(const char* config_path) {
        try {
            YAML::Node config = YAML::LoadFile(config_path);
            
            // Standard Factory Call (Safe inside src tree)
            auto& factory = Ramulator::Factory<Ramulator::IMemorySystem>::instance();
            std::string impl = config["MemorySystem"]["impl"].as<std::string>();
            
            system_ptr = factory.create(impl, config["MemorySystem"], nullptr);
            
            // Reset Physics
            physical_cell_data.clear();
            hammer_counters.clear();
            std::srand(std::time(nullptr));
            std::cout << "[Wrapper] Initialized (Integrated Mode)." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Wrapper Error] " << e.what() << std::endl;
            exit(1);
        }
    }

    long get_aggressor(long victim) { return victim ^ 0x1; }

    int check_physics(int type, long addr, int data_in) {
        if ((rand() % 1000000) < (f_cfg.p_sf * 1000000)) { physical_cell_data[addr] = rand() % 2; return 1; }
        
        int curr_val = physical_cell_data[addr];
        long agg_addr = get_aggressor(addr);

        if (type == 0) { // READ
            if ((rand() % 1000000) < (f_cfg.p_rdf * 1000000)) { physical_cell_data[addr] = !curr_val; return 2; }
        } else { // WRITE
            if (curr_val == data_in && (rand() % 1000000) < (f_cfg.p_wdf * 1000000)) { physical_cell_data[addr] = !data_in; return 5; }
        }
        
        // RowHammer
        int bank = (addr >> 14) & 0xF;
        int row = (addr >> 18) & 0xFFFF;
        int row_id = (bank << 16) | row;
        hammer_counters[row_id]++;
        if (hammer_counters[row_id] > f_cfg.hammer_thresh) { hammer_counters[row_id] = 0; return 10; }

        if (type == 1) physical_cell_data[addr] = data_in;
        return 0; 
    }

    int step_simulator(int type, long addr, int data) {
        if (!system_ptr) return -1;
        Ramulator::Request req(addr, type == 0 ? Ramulator::Request::Type::Read : Ramulator::Request::Type::Write);
        if (!system_ptr->send(req)) return -1;
        system_ptr->tick();
        return check_physics(type, addr, data);
    }
}
EOF

# 4. MODIFY CMAKE: Tell it to build our wrapper
echo "[Setup] Modifying CMakeLists.txt..."

# We append instructions to build 'libramulator_wrapper.so' containing ALL sources + our wrapper
cat <<EOF >> src/CMakeLists.txt

# --- INTEGRATED WRAPPER TARGET ---
# 1. Gather all source files
file(GLOB_RECURSE ALL_SRCS "*.cpp")
# 2. Exclude main.cpp so we don't get 'multiple main definitions' errors
list(FILTER ALL_SRCS EXCLUDE REGEX "main.cpp")
# 3. Create the Shared Library
add_library(ramulator_wrapper SHARED \${ALL_SRCS})
# 4. Link Dependencies
target_link_libraries(ramulator_wrapper PUBLIC yaml-cpp spdlog::spdlog)
target_include_directories(ramulator_wrapper PUBLIC \${CMAKE_CURRENT_SOURCE_DIR})
EOF

# 5. BUILD
echo "[Build] Compiling..."
mkdir -p build
cd build

cmake .. \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_INCLUDE_PATH="$CONDA_PREFIX/include" \
    -DCMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib"

make -j$(nproc)

# 6. EXTRACT
echo "[Build] Verifying output..."
if [ -f "src/libramulator_wrapper.so" ]; then
    cp src/libramulator_wrapper.so ../../build/libramulator_wrapper.so
    echo "=== SUCCESS ==="
    echo "Library: $(pwd)/../../build/libramulator_wrapper.so"
else
    echo "[Error] Build finished but library not found in src/."
    ls -R src/
    exit 1
fi
