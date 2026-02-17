/*
    Kinetic Monte Carlo (KMC) diffusion simulation
    For simulating vacancy-mediated diffusion at low temperatures
    
    Usage in run.in:
    kmc_diffusion <temperature> <max_time> <Ef_Zr> <Em_Zr> <Ef_Cr> <Em_Cr> [dump_interval]
    
    Example:
    kmc_diffusion 800 1e-6 1.35 1.8 1.55 0.8 1000
*/

#ifndef KMC_DIFFUSION_H
#define KMC_DIFFUSION_H

#include "../utilities/common.cuh"

class KMCDiffusion {
public:
    // Parameters
    double temperature;     // K
    double max_time;        // s
    double Ef_Zr, Em_Zr;    // eV
    double Ef_Cr, Em_Cr;    // eV
    int dump_interval;      // steps
    
    // Simulation state
    double current_time;
    long long step_count;
    
    // Statistics
    double msd_Zr, msd_Cr;
    
    // Methods
    KMCDiffusion(void);
    ~KMCDiffusion(void);
    
    void parse(char **param, int num_param);
    void compute(void);
    
private:
    void initialize(void);
    void identify_events(void);
    void select_and_execute_event(void);
    void update_statistics(void);
    void output_results(void);
};

#endif
