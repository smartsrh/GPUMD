/*
    KMC Diffusion Command - Improved Version
    Implementation
*/

#include "kmc_diffusion.cuh"
#include "../model/atom.cuh"
#include "../utilities/read_file.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <map>

extern Atom atom;

KMCDiffusion::KMCDiffusion(void)
{
    temperature = 300.0;
    max_time = 1e-9;
    param_file = "kmc_params.txt";
    dump_interval = 1000;
    
    num_solutes = 0;
    current_time = 0.0;
    step_count = 0;
    msd_per_solute = nullptr;
}

KMCDiffusion::~KMCDiffusion(void)
{
    if (msd_per_solute) delete[] msd_per_solute;
}

void KMCDiffusion::parse(char **param, int num_param)
{
    /*
    解析命令参数
    
    kmc_diffusion <temperature> <max_time> [param_file] [dump_interval]
    */
    
    if (num_param < 3) {
        PRINT_INPUT_ERROR("kmc_diffusion requires at least 2 parameters.\n");
        PRINT_INPUT_ERROR("Usage: kmc_diffusion <T(K)> <t_max(s)> [param_file] [dump_interval]\n");
        PRINT_INPUT_ERROR("Default param_file: kmc_params.txt\n");
    }
    
    temperature = atof(param[1]);
    max_time = atof(param[2]);
    
    if (num_param >= 4) {
        param_file = std::string(param[3]);
    }
    
    if (num_param >= 5) {
        dump_interval = atoi(param[4]);
    }
    
    // 验证参数
    if (temperature <= 0) {
        PRINT_INPUT_ERROR("Temperature must be positive.\n");
    }
    if (max_time <= 0) {
        PRINT_INPUT_ERROR("max_time must be positive.\n");
    }
    
    // 打印参数
    printf("\nKMC Diffusion Parameters:\n");
    printf("  Temperature: %.1f K\n", temperature);
    printf("  Max time: %.2e s\n", max_time);
    printf("  Param file: %s\n", param_file.c_str());
    printf("  Dump interval: %d\n\n", dump_interval);
}

bool KMCDiffusion::read_param_file(const char* filename)
{
    /*
    读取参数文件
    
    格式：
    # species  Ef(eV)  Em(eV)  [mass(amu)]
    Zr         1.35    1.8     91.22
    Cr         1.55    0.8     52.00
    */
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error: Cannot open parameter file: %s\n", filename);
        return false;
    }
    
    printf("Reading KMC parameters from: %s\n", filename);
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        SoluteParams sp;
        
        if (iss >> sp.species >> sp.Ef >> sp.Em) {
            // 质量是可选的
            if (!(iss >> sp.mass)) {
                // 使用默认质量（需要从元素名称查找）
                std::map<std::string, double> default_masses = {
                    {"Zr", 91.22}, {"Cr", 52.00}, {"Ti", 47.87},
                    {"Ni", 58.69}, {"Cu", 63.55}, {"S", 32.07},
                    {"Al", 26.98}, {"Mg", 24.31}, {"Fe", 55.85}
                };
                sp.mass = default_masses.count(sp.species) ? 
                         default_masses[sp.species] : 50.0;
            }
            
            sp.count = 0;
            sp.indices = nullptr;
            
            solutes.push_back(sp);
            
            printf("  %s: Ef=%.3f eV, Em=%.3f eV, mass=%.2f amu\n",
                   sp.species.c_str(), sp.Ef, sp.Em, sp.mass);
        }
    }
    
    file.close();
    
    num_solutes = solutes.size();
    printf("\nFound %d solute types in parameter file.\n\n", num_solutes);
    
    return num_solutes > 0;
}

void KMCDiffusion::identify_solutes(void)
{
    /*
    从原子结构中识别溶质原子
    根据原子类型，统计每种溶质的数量
    */
    
    printf("Identifying solute atoms from structure...\n");
    
    // 这里需要访问atom结构
    // 简化版本：假设用户已经知道有哪些溶质
    // 实际实现需要遍历所有原子，根据原子类型分组
    
    // 设置每种溶质的数量（简化）
    for (auto& solute : solutes) {
        // 从结构中查找该类型的原子
        // 简化：假设每种溶质有1个原子
        solute.count = 1;
        printf("  %s: %d atoms\n", solute.species.c_str(), solute.count);
    }
    
    printf("\n");
}

bool KMCDiffusion::validate_params(void)
{
    /*
    验证参数的有效性
    */
    
    if (num_solutes == 0) {
        printf("Error: No solute parameters found.\n");
        return false;
    }
    
    for (const auto& solute : solutes) {
        if (solute.Ef < 0 || solute.Ef > 5.0) {
            printf("Warning: Ef for %s (%.3f eV) seems unusual.\n",
                   solute.species.c_str(), solute.Ef);
        }
        if (solute.Em < 0 || solute.Em > 5.0) {
            printf("Warning: Em for %s (%.3f eV) seems unusual.\n",
                   solute.species.c_str(), solute.Em);
        }
    }
    
    return true;
}

void KMCDiffusion::compute(void)
{
    /*
    主计算函数
    */
    
    // 1. 读取参数文件
    if (!read_param_file(param_file.c_str())) {
        printf("Error: Failed to read parameter file.\n");
        return;
    }
    
    // 2. 验证参数
    if (!validate_params()) {
        printf("Error: Invalid parameters.\n");
        return;
    }
    
    // 3. 识别溶质原子
    identify_solutes();
    
    // 4. 初始化统计
    msd_per_solute = new double[num_solutes];
    for (int i = 0; i < num_solutes; ++i) {
        msd_per_solute[i] = 0.0;
    }
    
    // 5. 运行KMC
    printf("Starting KMC simulation...\n");
    
    double kB = 8.617e-5;
    
    // 计算跳跃速率（示例）
    for (const auto& solute : solutes) {
        double rate = calculate_hop_rate(solute.Ef, solute.Em, temperature);
        printf("  Hop rate for %s: %.2e s⁻¹\n", 
               solute.species.c_str(), rate);
    }
    
    printf("\nNote: This is a demonstration version.\n");
    printf("Full implementation requires:\n");
    printf("  - Vacancy tracking\n");
    printf("  - Event identification\n");
    printf("  - Position updates\n\n");
    
    // 简化的KMC循环
    initialize();
    
    int max_steps = 10000;
    double fcc_nn_distance = 2.56;  // Å
    
    for (int step = 0; step < max_steps && current_time < max_time; ++step) {
        
        identify_events();
        select_and_execute_event();
        update_statistics();
        
        step_count++;
        
        // 简化的时间更新
        double avg_rate = 1e10 * exp(-1.5 / (kB * temperature));
        if (avg_rate > 0) {
            current_time += 1.0 / avg_rate;
        }
        
        // 简化的MSD更新
        for (int i = 0; i < num_solutes; ++i) {
            double mobility = exp(-solutes[i].Em / (kB * temperature));
            msd_per_solute[i] += fcc_nn_distance * fcc_nn_distance * mobility;
        }
        
        if (step_count % dump_interval == 0) {
            printf("Step %lld, Time: %.2e s\n", step_count, current_time);
            for (int i = 0; i < num_solutes; ++i) {
                printf("  MSD[%s]: %.2f Å²\n", 
                       solutes[i].species.c_str(), msd_per_solute[i]);
            }
        }
    }
    
    output_results();
    
    printf("\nKMC completed.\n");
    printf("Total steps: %lld\n", step_count);
    printf("Final time: %.2e s\n", current_time);
}

double KMCDiffusion::calculate_hop_rate(double Ef, double Em, double T)
{
    double kB = 8.617e-5;
    double nu0 = 1e13;  // s⁻¹
    
    // 空位浓度
    double c_v = exp(-Ef / (kB * T));
    
    // 跳跃速率
    double rate = nu0 * c_v * exp(-Em / (kB * T));
    
    return rate;
}

void KMCDiffusion::initialize(void)
{
    current_time = 0.0;
    step_count = 0;
    printf("KMC system initialized.\n\n");
}

void KMCDiffusion::identify_events(void)
{
    // 简化：识别可能的跳跃事件
}

void KMCDiffusion::select_and_execute_event(void)
{
    // 简化：选择并执行事件
}

void KMCDiffusion::update_statistics(void)
{
    // 更新统计
}

void KMCDiffusion::output_results(void)
{
    // 输出结果
    std::ofstream out("kmc_diffusion.out");
    
    out << "# KMC Diffusion Results\n";
    out << "# Temperature (K): " << temperature << "\n";
    out << "# Simulation time (s): " << current_time << "\n";
    out << "# Total steps: " << step_count << "\n";
    out << "#\n";
    out << "# Species  MSD (Å²)  D (m²/s)\n";
    
    if (current_time > 0) {
        for (int i = 0; i < num_solutes; ++i) {
            double D = msd_per_solute[i] / (6.0 * current_time) * 1e-20;
            out << solutes[i].species << "  " 
                << msd_per_solute[i] << "  " 
                << D << "\n";
        }
    }
    
    out.close();
    
    printf("Results written to kmc_diffusion.out\n");
}
