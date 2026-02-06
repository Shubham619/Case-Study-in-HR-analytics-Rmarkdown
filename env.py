import gym
import numpy as np
import ctypes
import os
import sys

class DramEnv(gym.Env):
    def __init__(self, config_file="configs/ddr4_config.yaml"):
        super(DramEnv, self).__init__()
        
        # --- 1. ROBUST LIBRARY LOADING ---
        # We search specific paths where the build might have landed
        possible_paths = [
            "./ramulator2/build/src/libramulator_wrapper.so", # Integrated Build (Most Likely)
            "./ramulator2/src/libramulator_wrapper.so",       # Integrated Build (Alternative)
            "./build/libramulator_wrapper.so",                # Standalone Build
            "./libramulator_wrapper.so"                       # Current Dir
        ]
        
        lib_path = None
        for p in possible_paths:
            if os.path.exists(p):
                lib_path = p
                print(f"[DramEnv] Loading C++ Library from: {lib_path}")
                break
        
        if lib_path is None:
            raise FileNotFoundError("[Critical] Could not find 'libramulator_wrapper.so'. Did the build finish?")

        # Load the C++ Library
        self.lib = ctypes.CDLL(lib_path)

        # --- 2. DEFINE C++ ARGUMENT TYPES ---
        # init_simulator(const char* config_path)
        self.lib.init_simulator.argtypes = [ctypes.c_char_p]
        self.lib.init_simulator.restype = None

        # step_simulator(int type, long addr, int data) -> returns fault_code
        self.lib.step_simulator.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_int]
        self.lib.step_simulator.restype = ctypes.c_int

        # --- 3. INITIALIZE SIMULATOR ---
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        print(f"[DramEnv] Initializing Ramulator2 with {config_file}...")
        c_config = config_file.encode('utf-8')
        self.lib.init_simulator(c_config)
        
        # --- 4. RL SETUP ---
        self.action_space = gym.spaces.Discrete(100) # Placeholder
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.fault_history = set()
        
        # Legacy GSAT Patterns (Simple definition for Phase 1)
        self.gsat = type('GSAT', (object,), {
            'patterns': [0x00, 0xFF, 0xAA, 0x55, 0x33, 0xCC, 0x0F, 0xF0, 0x69, 0x96]
        })()

    def step(self, phase, action_id):
        """
        phase 1: action_id is Pattern ID (Legacy)
        phase 2: action_id is Swap Pair (Discovery)
        """
        # A. Determine Address & Data based on Phase
        addr = 0x10000 # Base address
        
        if phase == 1:
            # Legacy: Test a fixed pattern at a fixed address
            data_pattern = self.gsat.patterns[action_id % len(self.gsat.patterns)]
            # We perform a WRITE then a READ
            self.lib.step_simulator(1, addr, data_pattern) # Write
            fault_code = self.lib.step_simulator(0, addr, 0) # Read (expecting data_pattern)
            
        else:
            # Phase 2: Topology Discovery (RowHammer / Coupling)
            # The agent is trying to find 'aggressor' addresses
            # For this demo, we map action_id to an address offset
            offset = action_id * 64
            target = addr + offset
            
            # Simple Hammer Attempt: Write-Read-Write-Read rapid sequence
            self.lib.step_simulator(1, target, 0xFF)
            self.lib.step_simulator(0, target, 0)
            self.lib.step_simulator(1, target, 0xFF)
            fault_code = self.lib.step_simulator(0, target, 0)

        # B. Calculate Reward
        is_new = False
        reward = 0
        
        if fault_code > 0:
            # Construct a unique signature for this fault
            fault_sig = (phase, action_id, fault_code)
            
            if fault_sig not in self.fault_history:
                is_new = True
                self.fault_history.add(fault_sig)
                reward = 100 # Big reward for new discovery
                
                # Bonus for Critical Faults (RowHammer=10)
                if fault_code == 10:
                    reward += 500
            else:
                reward = 1 # Small reward for re-triggering known fault

        return reward, is_new, fault_code
