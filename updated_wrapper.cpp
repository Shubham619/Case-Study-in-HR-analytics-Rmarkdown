#include <iostream>
#include <map>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstdio>

// --- IMPORT INJECTED FUNCTIONS ---
extern "C" {
    void* ramulator_create_system(const char* impl_name, const char* config_str);
    bool ramulator_send(void* sys_ptr, long addr, int is_write);
    void ramulator_tick(void* sys_ptr);
}

// Global System Pointer
static void* system_ptr = nullptr;

// Physics Globals (Shadow Memory)
static std::map<long, int> physical_cell_data; 
static std::map<int, int> hammer_counters; 

// --- FULL FAULT CONFIGURATION ---
struct FaultConfig {
    double p_sf   = 0.01; // Stuck Fault
    double p_rdf  = 0.01; // Read Destructive
    double p_drdf = 0.01; // Deceptive Read Destructive
    double p_wdf  = 0.01; // Write Destructive
    double p_tcf  = 0.05; // Transition Coupling
    double p_scf  = 0.05; // Static Coupling
    double p_dccf = 0.05; // Disturb Cell Coupling
    double p_irf  = 0.01; // Incorrect Read
    double p_icf  = 0.05; // Idempotent Coupling
    int hammer_thresh = 5; // RowHammer Threshold
};
static FaultConfig f_cfg;

extern "C" {
    
    // Helper to read file content
    char* read_file(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return nullptr;
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* string = (char*)malloc(fsize + 1);
        fread(string, 1, fsize, f);
        fclose(f);
        string[fsize] = 0;
        return string;
    }

    void init_simulator(const char* config_path) {
        char* config_str = read_file(config_path);
        if (!config_str) {
            std::cerr << "[Wrapper Error] Could not read config file: " << config_path << std::endl;
            exit(1);
        }

        system_ptr = ramulator_create_system("DRAM", config_str);
        free(config_str);
        
        if (!system_ptr) {
            std::cerr << "[Wrapper Error] Failed to create system." << std::endl;
            exit(1);
        }
        
        physical_cell_data.clear();
        hammer_counters.clear();
        std::srand(std::time(nullptr));
        std::cout << "[Wrapper] Initialized with FULL 10-FAULT MODEL." << std::endl;
    }

    long get_aggressor(long victim) { return victim ^ 0x1; }

    // --- THE COMPLETE PHYSICS ENGINE ---
    int check_physics(int type, long addr, int data_in) {
        
        // 1. Stuck Fault (SF)
        if ((rand() % 1000000) < (f_cfg.p_sf * 1000000)) {
            physical_cell_data[addr] = rand() % 2; 
            return 1; 
        }

        int curr_val = physical_cell_data[addr];
        long agg_addr = get_aggressor(addr);
        int agg_val = physical_cell_data[agg_addr];

        // --- READ OPERATION FAULTS (Type == 0) ---
        if (type == 0) { 
            // 2. Read Destructive (RDF)
            if ((rand() % 1000000) < (f_cfg.p_rdf * 1000000)) { 
                physical_cell_data[addr] = !curr_val; 
                return 2; 
            }
            // 3. Deceptive Read Destructive (DRDF)
            if ((rand() % 1000000) < (f_cfg.p_drdf * 1000000)) { 
                physical_cell_data[addr] = !curr_val; 
                return 0; // Latent error
            }
            // 4. Incorrect Read Fault (IRF)
            if ((rand() % 1000000) < (f_cfg.p_irf * 1000000)) {
                return 4; // Returns wrong data, cell unchanged
            }
        } 
        // --- WRITE OPERATION FAULTS (Type == 1) ---
        else { 
            // 5. Write Destructive (WDF)
            if (curr_val == data_in) {
                if ((rand() % 1000000) < (f_cfg.p_wdf * 1000000)) { 
                    physical_cell_data[addr] = !data_in; 
                    return 5; 
                }
            }
            // 6. Transition Coupling (TCF)
            if (agg_val == 1 && curr_val == 0 && data_in == 1) { 
                if ((rand() % 1000000) < (f_cfg.p_tcf * 1000000)) {
                    physical_cell_data[addr] = 0; // Transition blocked
                    return 6; 
                }
            }
            // 7. Idempotent Coupling (ICF)
            if (curr_val != data_in) {
                if ((rand() % 1000000) < (f_cfg.p_icf * 1000000)) {
                    physical_cell_data[agg_addr] = !physical_cell_data[agg_addr]; // Flip neighbor
                    return 7;
                }
            }
        }

        // --- BACKGROUND COUPLING (Always Checked) ---
        // 8. Static Coupling (SCF)
        if (agg_val == 1) {
             if ((rand() % 1000000) < (f_cfg.p_scf * 1000000)) {
                 physical_cell_data[addr] = 0; 
                 return 8;
             }
        }
        // 9. Disturb Cell Coupling (DCCF)
        if ((rand() % 1000000) < (f_cfg.p_dccf * 1000000)) {
            physical_cell_data[agg_addr] = !physical_cell_data[agg_addr];
            return 9;
        }

        // 10. ROW HAMMER
        int bank = (addr >> 14) & 0xF;
        int row = (addr >> 18) & 0xFFFF;
        int row_id = (bank << 16) | row;
        
        hammer_counters[row_id]++;
        if (hammer_counters[row_id] > f_cfg.hammer_thresh) {
            hammer_counters[row_id] = 0;
            return 10; 
        }

        if (type == 1) physical_cell_data[addr] = data_in;
        return 0; 
    }

    int step_simulator(int type, long addr, int data) {
        if (!system_ptr) return -1;
        
        bool accepted = ramulator_send(system_ptr, addr, type);
        ramulator_tick(system_ptr);
        
        if (!accepted) return -1; 
        return check_physics(type, addr, data);
    }
}
