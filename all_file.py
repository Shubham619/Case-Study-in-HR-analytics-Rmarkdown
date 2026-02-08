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



