/*
    改进的KMC diffusion设计
    通用化，支持任意溶质原子
*/

// ============ 方案1：参数文件方式 ============

/*
run.in:
kmc_diffusion 800 1e-6

kmc_params.txt:
# species  Ef(eV)  Em(eV)  mass(amu)
Zr         1.35    1.8     91.22
Cr         1.55    0.8     52.00
*/

class KMCDiffusionImproved {
public:
    // 参数结构
    struct SoluteParams {
        std::string species;
        double Ef;     // 空位形成能
        double Em;     // 迁移能
        double mass;   // 原子质量
        int count;     // 原子数量
    };
    
    std::vector<SoluteParams> solutes;
    
    // 从文件读取参数
    void read_params(const char* filename);
    
    // 自动识别溶质原子
    void identify_solutes();
    
    // 验证参数
    bool validate_params();
};

// ============ 方案2：自动计算方式 ============

/*
run.in:
kmc_diffusion 800 1e-6 computeEf=1 EmMode=auto

流程：
1. 自动识别溶质原子（从model.xyz）
2. 计算每个溶质的Ef（MD短时间模拟）
3. 从数据库或估算获取Em
4. 运行KMC
*/

class KMCDiffusionAuto {
public:
    // 自动计算空位形成能
    void compute_vacancy_formation_energy();
    
    // 估算迁移能（从尺寸失配）
    double estimate_migration_energy(
        double solute_radius,
        double solvent_radius
    );
    
    // 从势函数数据库查找
    double lookup_migration_energy(
        const char* solute,
        const char* solvent
    );
};

// ============ 方案3：混合方式 ============

/*
命令格式：
kmc_diffusion <T> <t_max> [options]

选项：
• param_file=xxx.txt  参数文件
• compute_Ef=0/1      是否自动计算Ef
• Em_mode=auto/file/manual
• Em_file=xxx.txt     迁移能数据库
• validate=1          验证参数

示例：
# 完全自动（计算Ef，估算Em）
kmc_diffusion 800 1e-6 compute_Ef=1 Em_mode=auto

# 使用参数文件
kmc_diffusion 800 1e-6 param_file=my_params.txt

# 部分自动（计算Ef，手动Em）
kmc_diffusion 800 1e-6 compute_Ef=1 Em_mode=manual Zr.Em=1.8 Cr.Em=0.8
*/

class KMCDiffusionHybrid {
public:
    // 配置
    struct Config {
        double temperature;
        double max_time;
        
        bool compute_Ef;
        int Em_mode;  // 0=auto, 1=file, 2=manual
        std::string param_file;
        std::string Em_file;
        
        bool validate;
        int verbose;
    };
    
    void parse_config(int argc, char** argv);
    void auto_detect_solutes();
    void compute_all_parameters();
    void run_kmc();
};
