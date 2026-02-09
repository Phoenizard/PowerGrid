# 电网稳定性研究代码库（Phase 1 + Phase 2）

本仓库围绕 swing equation 电网动力学模型，分两阶段推进：

- **Phase 1（`paper_reproduction/`）**：复现 Smith et al. (2022) Figure 1C/1D，研究**均匀功率分配**下不同节点组成与网络拓扑对稳定性的影响。
- **Phase 2（`direction3_nonuniform/`）**：扩展到**非均匀功率分配**，研究功率异质性与发电集中化对临界耦合强度的影响。

---

## 目录

- [1. 研究背景](#1-研究背景)
- [2. 公共数学模型](#2-公共数学模型)
- [3. 仓库结构](#3-仓库结构)
- [4. 环境准备](#4-环境准备)
- [5. Phase 1：论文复现（均匀功率分配）](#5-phase-1论文复现均匀功率分配)
- [6. Phase 2：方向③ 非均匀功率分配](#6-phase-2方向③-非均匀功率分配)
- [7. 常见问题（FAQ）](#7-常见问题faq)
- [8. 参考文献](#8-参考文献)

---

## 1. 研究背景

电力网络中的节点可抽象为三类：

- **发电节点**（`n_+`）：`P > 0`
- **负载节点**（`n_-`）：`P < 0`
- **被动节点**（`n_0`）：`P = 0`

满足约束 `n_+ + n_- + n_0 = N`。核心问题是：

- 在给定网络拓扑下，系统达到同步稳态所需的最小耦合强度 `kappa_c` 如何随节点组成、网络随机性与功率分配策略变化？

---

## 2. 公共数学模型

### 2.1 Swing Equation

$$
\frac{d^2\theta_i}{dt^2} + \gamma \frac{d\theta_i}{dt} = P_i - \kappa \sum_j A_{ij}\sin(\theta_i-\theta_j)
$$

代码中的一阶形式：

$$
\frac{d\theta_i}{dt}=\omega_i, \quad
\frac{d\omega_i}{dt}=P_i-\gamma\omega_i-\kappa\sum_j A_{ij}\sin(\theta_i-\theta_j)
$$

### 2.2 临界耦合强度 `kappa_c`

通过二分搜索求解：

1. 设定区间 `[kappa_min, kappa_max]`
2. 取中点 `kappa_mid`
3. 数值积分 ODE，判定是否收敛
4. 收缩区间并迭代

稳定判据采用频率收敛阈值：`max_i |omega_i| < epsilon`。

### 2.3 网络模型

两阶段均使用 Watts-Strogatz 小世界网络：`N=50, K=4`，重连概率 `q` 视实验而定。

---

## 3. 仓库结构

```text
PowerGrid/
├── paper_reproduction/          # Phase 1：论文 Fig.1C/1D 复现
│   ├── config.py
│   ├── model.py
│   ├── run_sweep.py
│   ├── plot_fig1c.py
│   ├── plot_fig1d.py
│   └── plot_combined.py
├── direction3_nonuniform/       # Phase 2：非均匀功率分配实验
│   ├── power_allocation.py
│   ├── run_experiment_2A.py
│   ├── run_experiment_2C.py
│   ├── plot_results.py
│   ├── test_power_allocation.py
│   ├── test_integration.py
│   └── self_check.py
├── Phase1_Research_Report.md
├── Phase2_Research_Report.md
└── README.md
```

---

## 4. 环境准备

```bash
# 建议使用 conda
conda create -n PowerGrid python=3.10
conda activate PowerGrid

# 基础依赖
pip install numpy scipy networkx matplotlib tqdm numba
```

依赖建议版本：

- `numpy >= 1.21`
- `scipy >= 1.7`
- `networkx >= 2.6`
- `matplotlib >= 3.4`
- `tqdm >= 4.62`
- `numba >= 0.54`（可选）

---

## 5. Phase 1：论文复现（均匀功率分配）

### 5.1 内容

Phase 1 对应 `paper_reproduction/`，目标是复现论文 Figure 1C/1D：

- **Fig. 1C**：三元单纯形热力图（`q=0`）
- **Fig. 1D**：固定被动节点截面下，多 `q` 值曲线

核心脚本：

- `paper_reproduction/model.py`：网络生成、ODE、`kappa_c` 计算
- `paper_reproduction/run_sweep.py`：扫描入口（默认 fast，支持 `--production`）
- `paper_reproduction/plot_fig1c.py` / `plot_fig1d.py` / `plot_combined.py`：作图

### 5.2 使用说明

```bash
cd paper_reproduction

# 快速模式（默认）
python run_sweep.py

# 生产模式
python run_sweep.py --production

# 绘图
python plot_combined.py
# 或
python plot_fig1c.py
python plot_fig1d.py
```

### 5.3 关键参数（`paper_reproduction/config.py`）

| 类别 | 参数 | 当前值 |
|------|------|--------|
| 物理参数 | `N, K, GAMMA, P_MAX` | `50, 4, 1.0, 1.0` |
| 二分搜索 | `KAPPA_RANGE, BISECTION_STEPS, CONV_TOL` | `(0.001, 3.0), 20, 1e-3` |
| Fast 模式 | `ENSEMBLE_SIZE, STEP_SIZE, T_INTEGRATE` | `50, 3, 100` |
| Production 模式 | `ENSEMBLE_SIZE_FINAL, STEP_SIZE_FINAL` | `200, 2` |
| 截面扫描 | `Q_VALUES, NP_CROSS, N_MINUS_RANGE` | `[0.0,0.1,0.4,1.0], 16, (1,33)` |

### 5.4 输出

默认写入 `paper_reproduction/output/`：

- `data_simplex_q0.0.npz`
- `data_crosssec.npz`
- `fig1c.png`
- `fig1d.png`
- `fig1cd_combined.png`

---

## 6. Phase 2：方向③ 非均匀功率分配

### 6.1 内容

Phase 2 对应 `direction3_nonuniform/`，包含两个实验：

- **实验 2A**：`sigma_ratio` 扫描，比较
  - `2A-gen`：发电侧异质、负荷侧均匀
  - `2A-con`：负荷侧异质、发电侧均匀
- **实验 2C**：集中式 vs 分布式发电，扫描大站占比 `r`

核心脚本：

- `direction3_nonuniform/power_allocation.py`：异质分配与集中式分配函数
- `direction3_nonuniform/run_experiment_2A.py`：2A 生产脚本
- `direction3_nonuniform/run_experiment_2C.py`：2C 生产脚本
- `direction3_nonuniform/plot_results.py`：生成 `fig_2A.png` / `fig_2C.png`
- `direction3_nonuniform/test_power_allocation.py`：单元测试
- `direction3_nonuniform/test_integration.py`：与现有 `model.py` 接口集成测试
- `direction3_nonuniform/self_check.py`：交付完整性检查

### 6.2 使用说明

#### 1) 运行测试

```bash
cd direction3_nonuniform
python test_power_allocation.py
python test_integration.py
```

#### 2) 快速验证（建议先跑）

```bash
python run_experiment_2A.py --n_ensemble 50 --output results_n50/results_2A.csv
python run_experiment_2C.py --n_ensemble 50 --output results_n50/results_2C.csv
```

#### 3) 生产运行

```bash
python run_experiment_2A.py --n_ensemble 200 --output results/results_2A.csv
python run_experiment_2C.py --n_ensemble 200 --output results/results_2C.csv
```

#### 4) 绘图与自检

```bash
python plot_results.py \
  --input_2a results_n50/results_2A.csv \
  --input_2c results_n50/results_2C.csv \
  --output_2a results_n50/fig_2A.png \
  --output_2c results_n50/fig_2C.png

python self_check.py --results_dir results_n50
```

### 6.3 阶段 2 参数（脚本内固定）

| 参数 | 值 |
|------|----|
| `N, N_PLUS, N_MINUS` | `50, 25, 25` |
| `K, Q, GAMMA, P_MAX` | `4, 0.1, 1.0, 1.0` |
| `kappa_range` | `(0.001, 3.0)` |
| `bisection_steps` | `5` |
| `t_integrate` | `20` |
| `conv_tol` | `5e-3` |
| `max_step` | `5.0` |
| `sigma_ratios (2A)` | `[0, 0.1, ..., 0.8]` |
| `r_values (2C)` | `[0.04, 0.1, ..., 0.95]` |

### 6.4 输出

- 默认生产输出：`direction3_nonuniform/results/`
- 当前开发规模产物：`direction3_nonuniform/results_n50/`

CSV 列定义：

- `results_2A.csv`
  - `sigma_ratio, kappa_c_mean_gen, kappa_c_std_gen, kappa_c_mean_con, kappa_c_std_con`
- `results_2C.csv`
  - `r, kappa_c_mean, kappa_c_std`

附：阶段 2 报告见 `Phase2_Research_Report.md`。

---

## 7. 常见问题（FAQ）

### 7.1 Phase 1 相关

**Q1: 为什么复现结果和论文有偏差？**  
A: 随机种子、积分容差、收敛判据与硬件都会影响绝对值；通常看趋势是否一致。

**Q2: 如何加速 Phase 1？**  
A: 降低 `ENSEMBLE_SIZE`、增大 `STEP_SIZE`、缩短 `T_INTEGRATE`（注意可能影响精度）。

**Q3: `KAPPA_SCALE_FACTOR` 如何设置？**  
A: 当前代码中为 `1.0`（见 `paper_reproduction/config.py`）。如需与特定论文图数值区间对齐，可在后处理阶段调整。

### 7.2 Phase 2 相关

**Q4: Phase 2 为什么同样支持断点续跑？**  
A: `run_experiment_2A.py` 和 `run_experiment_2C.py` 按参数点追加 CSV；启动时会读取已完成点并跳过。

**Q5: 看到个别实例失败怎么办？**  
A: 脚本已实现单实例容错，会记录 `WARN` 并继续，不影响整轮完成。

**Q6: 建议先用多少 `n_ensemble`？**  
A: 开发/验证建议 `50`；最终统计建议 `200` 及以上。

**Q7: 无图形界面环境能否出图？**  
A: 可以。`plot_results.py` 使用 `matplotlib.use('Agg')`。

---

## 8. 参考文献

1. Smith, P. J., et al. *Science Advances* 8, eabj6734 (2022).  
2. Rohden, M., et al. *Phys. Rev. Lett.* 109, 064101 (2012).  
3. Watts, D. J., and Strogatz, S. H. *Nature* 393, 440-442 (1998).

---

如有问题或建议，欢迎提交 Issue 或 PR。
