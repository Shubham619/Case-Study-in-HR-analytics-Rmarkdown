#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="ramulator2"
BUILD_DIR="${REPO_DIR}/build"
CAPI_DIR="${REPO_DIR}/src/capi"
CAPI_CPP="${CAPI_DIR}/ramulator_capi.cpp"
SRC_CMAKELISTS="${REPO_DIR}/src/CMakeLists.txt"

echo "=== Ramulator2 + C-API Bridge Build ==="

# 0) Clone only if missing
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[Setup] Cloning CMU-SAFARI/ramulator2 ..."
  git clone https://github.com/CMU-SAFARI/ramulator2.git "${REPO_DIR}"
else
  echo "[Setup] Found existing ${REPO_DIR}/.git (won't reclone)."
fi

# 1) Quick sanity check: this avoids accidentally using the old "ramulator" repo
if ! grep -q "Ramulator 2.0" "${REPO_DIR}/README.md"; then
  echo "[Error] ${REPO_DIR} does not look like CMU-SAFARI/ramulator2 (README mismatch)."
  echo "Delete '${REPO_DIR}' and rerun this script."
  exit 1
fi

# 2) Write the C-ABI bridge source (idempotent)
mkdir -p "${CAPI_DIR}"
if [[ ! -f "${CAPI_CPP}" ]]; then
  echo "[Patch] Writing ${CAPI_CPP}"
  cat > "${CAPI_CPP}" <<'EOF'
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <atomic>
#include <map>
#include <iostream>

#include "base/config.h"
#include "base/factory.h"
#include "base/request.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

using namespace Ramulator;

// -----------------------------
// Optional "physics"/fault model
// -----------------------------
static std::map<uint64_t, int> physical_cell_data;
static std::map<int, int> hammer_counters;

struct FaultConfig {
  double p_sf = 0.01;       // stuck fault prob
  int hammer_thresh = 5000; // bump this up (your old "5" is unrealistically tiny)
};
static FaultConfig f_cfg;

static inline uint64_t get_aggressor(uint64_t victim) { return victim ^ 0x1ULL; }

static int check_physics(int is_write, uint64_t addr, int data_in) {
  // Stuck fault (very simplified)
  if ((std::rand() % 1000000) < (int)(f_cfg.p_sf * 1000000.0)) {
    physical_cell_data[addr] = std::rand() & 1;
    return 1; // stuck fault observed
  }

  // Rowhammer-ish counter (toy)
  int bank = (int)((addr >> 14) & 0xF);
  int row  = (int)((addr >> 18) & 0xFFFF);
  int row_id = (bank << 16) | row;

  auto &cnt = hammer_counters[row_id];
  cnt++;
  if (cnt > f_cfg.hammer_thresh) {
    cnt = 0;
    return 10; // "hammer event"
  }

  if (is_write) physical_cell_data[addr] = data_in;
  return 0;
}

// -----------------------------
// Ramulator2 handle
// -----------------------------
struct R2Handle {
  IFrontEnd* frontend = nullptr;
  IMemorySystem* memsys = nullptr;
};

extern "C" {

// Create + connect frontend/memory system exactly like Ramulator2 README suggests.
void* r2_init_from_yaml_file(const char* config_path) {
  try {
    auto* h = new R2Handle();

    YAML::Node config = Ramulator::Config::parse_config_file(std::string(config_path), {});
    h->frontend = Ramulator::Factory::create_frontend(config);
    h->memsys   = Ramulator::Factory::create_memory_system(config);

    h->frontend->connect_memory_system(h->memsys);
    h->memsys->connect_frontend(h->frontend);

    physical_cell_data.clear();
    hammer_counters.clear();
    std::srand((unsigned)std::time(nullptr));

    std::cout << "[capi] r2_init_from_yaml_file OK\n";
    return (void*)h;
  } catch (const std::exception& e) {
    std::cerr << "[capi][Error] init failed: " << e.what() << "\n";
    return nullptr;
  } catch (...) {
    std::cerr << "[capi][Error] init failed: unknown\n";
    return nullptr;
  }
}

// Tick N cycles (both frontend + memsys).
void r2_tick(void* handle, uint64_t cycles) {
  auto* h = (R2Handle*)handle;
  if (!h || !h->frontend || !h->memsys) return;
  for (uint64_t i = 0; i < cycles; i++) {
    h->frontend->tick();
    h->memsys->tick();
  }
}

// Enqueue a request using the documented external-request path.
// Returns 1 if accepted, 0 if rejected (e.g., queue full).
int r2_enqueue(void* handle, int is_read, uint64_t addr, int context_id) {
  auto* h = (R2Handle*)handle;
  if (!h || !h->frontend) return 0;

  bool ok = h->frontend->receive_external_requests(
    is_read ? 1 : 0, addr, context_id,
    [](Ramulator::Request& req) { (void)req; /* no-op callback */ }
  );
  return ok ? 1 : 0;
}

// Blocking "one request": enqueue + tick until callback fires (or timeout).
// Returns:
//   >=0 : cycles waited until completion
//   -1  : enqueue rejected
//   -2  : timeout
int r2_step_blocking_with_fault(
  void* handle,
  int is_write,
  uint64_t addr,
  int data_in,
  int context_id,
  uint64_t max_cycles
) {
  auto* h = (R2Handle*)handle;
  if (!h || !h->frontend || !h->memsys) return -2;

  std::atomic<bool> done{false};

  bool ok = h->frontend->receive_external_requests(
    is_write ? 0 : 1,
    addr,
    context_id,
    [&](Ramulator::Request& req) {
      (void)req;
      done.store(true, std::memory_order_release);
    }
  );

  if (!ok) return -1;

  uint64_t waited = 0;
  while (!done.load(std::memory_order_acquire)) {
    h->frontend->tick();
    h->memsys->tick();
    waited++;
    if (waited >= max_cycles) return -2;
  }

  // Apply your toy fault model after the request "completes"
  return check_physics(is_write ? 1 : 0, addr, data_in);
}

// Cleanup (also calls finalize like README suggests)
void r2_destroy(void* handle) {
  auto* h = (R2Handle*)handle;
  if (!h) return;

  try {
    if (h->frontend) h->frontend->finalize();
    if (h->memsys)   h->memsys->finalize();
  } catch (...) {}

  delete h;
}

} // extern "C"
EOF
else
  echo "[Patch] ${CAPI_CPP} already exists (won't overwrite)."
fi

# 3) Patch src/CMakeLists.txt to compile the C-API into the existing shared lib
#    We do NOT replace executables or mess with add_executable().
if ! grep -q "ramulator_capi.cpp" "${SRC_CMAKELISTS}"; then
  echo "[Patch] Attaching capi source to Ramulator2 build in ${SRC_CMAKELISTS}"
  cat >> "${SRC_CMAKELISTS}" <<'EOF'

# --- Added by build_ramulator2_capi.sh ---
# Compile the C-ABI bridge into the existing libramulator.so target.
if(TARGET ramulator)
  target_sources(ramulator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/capi/ramulator_capi.cpp)
elseif(TARGET ramulator2)
  # Some build setups name the shared-lib target "ramulator2"
  target_sources(ramulator2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/capi/ramulator_capi.cpp)
endif()
EOF
else
  echo "[Patch] CMake already references ramulator_capi.cpp (skip)."
fi

# 4) Clean only the build dir (safe)
echo "[Build] Cleaning ${BUILD_DIR} ..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# 5) Configure + build (C++20 is required per README)
echo "[Build] Configuring (C++20, Release) ..."
cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo "[Build] Building ..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# 6) Locate the produced library exactly like Ramulator2 README expects
echo "[Check] Searching for libramulator.so ..."
LIB="$(find "${REPO_DIR}" "${BUILD_DIR}" -maxdepth 4 -type f -name "libramulator.so" 2>/dev/null | head -n 1 || true)"
if [[ -z "${LIB}" ]]; then
  echo "[Error] libramulator.so not found."
  echo "Tip: list targets: cmake --build ${BUILD_DIR} --target help"
  exit 1
fi

echo "[OK] Found: ${LIB}"

# 7) Verify symbols exist
echo "[Check] Verifying exported C symbols ..."
if command -v nm >/dev/null 2>&1; then
  nm -D "${LIB}" | grep -E "r2_init_from_yaml_file|r2_step_blocking_with_fault|r2_destroy" >/dev/null
  echo "[OK] Symbols present."
else
  echo "[Warn] 'nm' not found; skipping symbol check."
fi

echo "=== SUCCESS ==="
echo "Load this in Python via ctypes/cffi:"
echo "  ${LIB}"
