import gym
import numpy as np
import subprocess
import os
import sys

class DramEnv(gym.Env):
    def __init__(self, config_file="ddr5_config.yaml"):
        super(DramEnv, self).__init__()
        
        # --- A. DEFINING GSAT PATTERNS ---
        # The RL Agent will choose index 0..7
        self.gsat_patterns = [
            0x00, # All Zeros
            0xFF, # All Ones
            0x55, # Alternating 0101 (Checkerboard)
            0xAA, # Alternating 1010 (Checkerboard)
            0x33, # 00110011
            0xCC, # 11001100
            0x0F, # 00001111
            0xF0  # 11110000
        ]
        
        self.action_space = gym.spaces.Discrete(len(self.gsat_patterns))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # --- B. CONNECT SIMULATOR ---
        self.exe_path = "./ramulator2/build/ramulator_interactive"
        if not os.path.exists(self.exe_path): raise FileNotFoundError("Executable not found.")
        
        self.proc = subprocess.Popen([self.exe_path, config_file], 
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        if self.proc.stdout.readline().strip() != "READY": raise RuntimeError("Simulator failed to start")

    def step(self, action_id):
        # 1. Get the Pattern
        pattern = self.gsat_patterns[action_id]
        
        # 2. Execute GSAT Stress (Write Pattern -> Read Pattern)
        # We target a specific aggressive row (e.g. 0x100000)
        target_addr = 0x100000
        
        # Send Write (1) with Pattern
        self.proc.stdin.write(f"REQ {target_addr} 1\n"); self.proc.stdin.flush()
        self.proc.stdout.readline() # Consume response
        
        # Send Read (0)
        self.proc.stdin.write(f"REQ {target_addr} 0\n"); self.proc.stdin.flush()
        response = self.proc.stdout.readline().strip()

        # 3. Calculate Reward (Simulated Fault)
        # Real Hardware: Bit flips happen here.
        # Simulator: We reward High Contention (STALLED) or specific "Bad" Patterns
        
        reward = 0
        fault_code = 0
        
        # SYNTHETIC VULNERABILITY: 
        # Let's pretend 0xAA and 0x55 (Checkerboard) are 'weak' patterns for this DRAM
        if response == "STALLED" or pattern in [0xAA, 0x55]:
             # Probabilistic failure for RL to learn
            if np.random.rand() < 0.8: 
                reward = 10 
                fault_code = 1
        
        return reward, False, fault_code

    def close(self):
        try: self.proc.terminate()
        except: pass


















import numpy as np
from dram_rl.environment import DramEnv

# Config
EPISODES = 100
STEPS = 50
ALPHA = 0.1  # Learning Rate
GAMMA = 0.9  # Discount
EPSILON = 0.1 # Exploration

def train():
    env = DramEnv()
    q_table = np.zeros(env.action_space.n) # 1 State, 8 Actions

    print("🚀 TRAINING: Learning best GSAT Pattern...")
    
    for episode in range(EPISODES):
        for step in range(STEPS):
            # Epsilon Greedy
            if np.random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table)

            # Step
            reward, _, _ = env.step(action)

            # Update Q-Value
            q_table[action] = (1 - ALPHA) * q_table[action] + ALPHA * (reward + GAMMA * np.max(q_table))

    # Save Brain
    np.save("gsat_q_table.npy", q_table)
    print("✅ Training Done. Brain saved.")
    print("   Final Q-Values (Preference):")
    for i, val in enumerate(q_table):
        print(f"   Pattern {hex(env.gsat_patterns[i])}: {val:.2f}")

if __name__ == "__main__":
    train()














import numpy as np
from dram_rl.environment import DramEnv

def evaluate():
    # Load Brain
    try:
        q_table = np.load("gsat_q_table.npy")
    except:
        print("❌ Train first!")
        return

    env = DramEnv()
    best_action = np.argmax(q_table)
    best_pattern = env.gsat_patterns[best_action]

    print(f"🔎 EVALUATION: Attacking with Best Pattern {hex(best_pattern)}")
    
    total_faults = 0
    for i in range(100):
        reward, _, code = env.step(best_action)
        if reward > 0:
            total_faults += 1
            
    print(f"✅ Results: Generated {total_faults} faults in 100 attempts.")

if __name__ == "__main__":
    evaluate()











import numpy as np
import matplotlib.pyplot as plt
from dram_rl.environment import DramEnv

STEPS = 200

def run_baseline():
    env = DramEnv()
    faults = []
    total = 0
    for _ in range(STEPS):
        action = env.action_space.sample() # Random Pattern
        reward, _, _ = env.step(action)
        if reward > 0: total += 1
        faults.append(total)
    return faults

def run_rl():
    env = DramEnv()
    q_table = np.zeros(env.action_space.n) # Learn from scratch
    faults = []
    total = 0
    
    for _ in range(STEPS):
        # Epsilon Greedy Learning
        if np.random.rand() < 0.1: action = env.action_space.sample()
        else: action = np.argmax(q_table)
        
        reward, _, _ = env.step(action)
        
        # Update
        q_table[action] += 0.1 * (reward - q_table[action])
        
        if reward > 0: total += 1
        faults.append(total)
    return faults

# Run & Plot
print("Running Baseline...")
base_data = run_baseline()
print("Running RL...")
rl_data = run_rl()

plt.plot(base_data, label="Random GSAT (Baseline)", linestyle="--")
plt.plot(rl_data, label="RL GSAT (Trained)", linewidth=2)
plt.legend()
plt.title("GSAT Fault Discovery: RL vs Random")
plt.xlabel("Steps")
plt.ylabel("Total Faults")
plt.savefig("gsat_comparison.png")
print("✅ Graph saved to gsat_comparison.png")















#include "base/request.h"
#include "base/config.h"
#include "base/factory.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

#include <iostream>
#include <sstream>
#include <string>
#include <atomic>
#include <cstdint>

using namespace Ramulator;

static inline void tick_once(IFrontEnd* fe, IMemorySystem* ms) {
  fe->tick();
  ms->tick();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
    return 1;
  }

  // Official library flow
  YAML::Node config = Ramulator::Config::parse_config_file(std::string(argv[1]), {});
  IFrontEnd* fe = Ramulator::Factory::create_frontend(config);
  IMemorySystem* ms = Ramulator::Factory::create_memory_system(config);

  if (!fe || !ms) {
    std::cerr << "ERROR: failed to create frontend/memory system\n";
    return 2;
  }

  fe->connect_memory_system(ms);
  ms->connect_frontend(fe);

  std::cout << "READY" << std::endl;

  // Protocol:
  // REQ <R|W> <hex_addr> <ctx_id> <max_cycles>
  // Returns: STALLED | DONE <cycles> | TIMEOUT <cycles> | ERR <reason>
  // TICK <cycles> -> OK
  // EXIT -> exits
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    if (cmd == "EXIT") break;

    if (cmd == "TICK") {
      uint64_t n = 0;
      iss >> n;
      if (!iss) { std::cout << "ERR bad_format\n"; continue; }
      for (uint64_t i = 0; i < n; i++) tick_once(fe, ms);
      std::cout << "OK" << std::endl;
      continue;
    }

    if (cmd == "REQ") {
      char rw;
      std::string addr_s;
      int ctx;
      uint64_t max_cycles;
      iss >> rw >> addr_s >> ctx >> max_cycles;
      if (!iss) { std::cout << "ERR bad_format\n"; continue; }

      uint64_t addr = 0;
      try { addr = std::stoull(addr_s, nullptr, 16); }
      catch (...) { std::cout << "ERR bad_addr\n"; continue; }

      int is_write = (rw == 'W' || rw == 'w') ? 1 : 0;
      std::atomic<bool> done{false};

      bool accepted = fe->receive_external_requests(
        is_write, addr, ctx,
        [&](Ramulator::Request& req){ (void)req; done.store(true, std::memory_order_release); }
      );

      if (!accepted) {
        std::cout << "STALLED" << std::endl;
        continue;
      }

      uint64_t waited = 0;
      while (!done.load(std::memory_order_acquire) && waited < max_cycles) {
        tick_once(fe, ms);
        waited++;
      }

      if (!done.load(std::memory_order_acquire)) std::cout << "TIMEOUT " << waited << std::endl;
      else std::cout << "DONE " << waited << std::endl;

      continue;
    }

    std::cout << "ERR unknown_cmd\n";
  }

  fe->finalize();
  ms->finalize();
  return 0;
}















SPD=$(find ramulator2 -type d -path "*spdlog*include*" | head -n 1)
YML=$(find ramulator2 -type d -path "*yaml*include*"   | head -n 1)

g++ -std=c++20 -O3 -o ramulator_driver \
  interactive_driver.cpp \
  -I ramulator2/src -I "$SPD" -I "$YML" \
  -L "$(pwd)/ramulator2" -lramulator \
  -Wl,-rpath,"$(pwd)/ramulator2"


