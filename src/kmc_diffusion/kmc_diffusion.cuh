/*
    KMC Diffusion Command - Improved Version
    支持任意溶质原子，从参数文件读取
    
    Usage:
    kmc_diffusion <temperature> <max_time> [param_file]
    
    Example:
    kmc_diffusion 800 1e-6                    # 默认使用kmc_params.txt
    kmc_diffusion 800 1e-6 my_params.txt      # 使用自定义参数文件
    
    kmc_params.txt format:
    # species  Ef(eV)  Em(eV)  [mass(amu)]
    Zr         1.35    1.8     91.22
    Cr         1.55    0.8     52.00
*/

#ifndef KMC_DIFFUSION_H
#define KMC_DIFFUSION_H

#include <string>
#include <vector>
#include "../utilities/common.cuh"

// 溶质参数结构
struct SoluteParams {
    std::string species;
    double Ef;      // 空位形成能
    double Em;      // 迁移能
    double mass;    // 原子质量
    int count;      // 原子数量
    int* indices;   // 原子索引（GPU）
};

class KMCDiffusion {
public:
    // 基本参数
    double temperature;     // K
    double max_time;        // s
    std::string param_file; // 参数文件名
    int dump_interval;      // 输出间隔
    
    // 溶质参数
    std::vector<SoluteParams> solutes;
    int num_solutes;
    
    // 模拟状态
    double current_time;
    long long step_count;
    
    // 统计数据
    double* msd_per_solute;  // 每种溶质的MSD
    
    // 方法
    KMCDiffusion(void);
    ~KMCDiffusion(void);
    
    void parse(char **param, int num_param);
    void compute(void);
    
private:
    // 参数读取
    bool read_param_file(const char* filename);
    bool validate_params(void);
    
    // 溶质识别
    void identify_solutes(void);
    std::string get_species_from_type(int type);
    
    // KMC核心
    void initialize(void);
    void identify_events(void);
    void select_and_execute_event(void);
    void update_statistics(void);
    void output_results(void);
    
    // 辅助函数
    double calculate_hop_rate(double Ef, double Em, double T);
    void apply_pbc(double* pos, double* box);
};

#endif
