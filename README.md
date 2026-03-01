# ⚡ 微电网韧性研究 — Swing Equation 与 Cascade Failure

> MATH3060 课程项目  
> 基于 Smith et al. (2022) *Science Advances* 8, eabj6734  
> 报告截止：2026-03-09 ｜ 答辩：2026-03-16

---

## 研究概述

本项目围绕一个核心问题：**高比例光伏接入的微电网，其拓扑结构如何影响同步稳定性与级联韧性？**

我们复现并扩展了 Smith et al. 的 swing equation 电网模型，通过四个递进的子问题（SQ1→SQ4）揭示了一个关键发现：**同一拓扑干预（PCC 边添加）在 swing 稳定性和 cascade 韧性两个模型之间产生相反效果** —— 一种跨模型的类 Braess 悖论。

### 网络模型

- Watts-Strogatz 小世界网络：`n=50, K̄=4, q=0.1`
- 额外 1 个 PCC（公共耦合点）节点，连接外部电网
- 英国实测家庭用电（LCL 数据集）+ 光伏发电（Sheffield Solar）时序驱动

---

## 研究路线

```
Phase 0: 论文复现
   │  复现 Smith Fig 1C/1D — 均匀功率下的 simplex 热力图
   │  验证 swing equation 求解器与二分搜索框架
   ▼
SQ1: 功率异质性 → κ_c 的影响
   │  实验 2A: σ_ratio 扫描（异质程度）→ 零效应
   │  实验 2C: 发电集中化 r 扫描 → κ_c 上升 139%
   │  结论: 异质性本身不影响稳定性，集中化才是关键
   ▼
SQ2: 时序动力学 — 微电网的昼夜振荡
   │  SQ2-A: 用实测数据驱动 simplex 轨迹 (η⁺, η⁻, ηₚ)
   │  SQ2-B: κ_c 时序（56 个时间点 × 50 ensemble）
   │  结论: κ_c 昼夜比达 22.6×，PCC 是瓶颈所在
   ▼
SQ4: 拓扑优化 — 能否通过加边降低 κ_c？
   │  Exp 4A: 重连概率 q 扫描 → 无效（负结果）
   │  Exp 4B: 5 种边添加策略对比 → pcc_direct 最优
   │  m-sweep: 添加 m 条 PCC 边 → κ_c 降低 50%–83%
   │  Theorem 1: 解析下界 κ_c ≥ |P_PCC|/[P_max·(d₀+m)]
   │  结论: 定向增强 PCC 连通性是最有效的单一干预
   ▼
SQ3: Cascade 验证 — 稳定性 ≠ 韧性
   │  直接在含 PCC 网络上跑 DC cascade sweep
   │  S vs α/α* 曲线呈阶梯形（非 Smith 预期的 sigmoid）
   │  PCC 边的流量集中度 ~11× → 级联首先沿 PCC 边传播
   │  对照组（无 PCC, n=100）→ sigmoid + ρ_mean=0.84
   │  结论: SQ4 的 PCC 加边在 cascade 模型下反而有害
   ▼
SQ5: 阻尼敏感性分析 — 拓扑优化的鲁棒性验证
   │  SQ5-A: gamma 扫描 [0.1, 5.0]，16 个阻尼值 × 50 ensemble
   │  kc 随 gamma 单调递减（高阻尼 → 更易同步）
   │  m=4 加边的相对收益恒定 ~50%（阻尼不变性）
   │  低阻尼时绝对收益最大（3.79 vs 2.96）
   │  结论: 拓扑优化效果与阻尼参数乘法可分离
   ▼
核心发现: 跨模型 Braess-like 悖论
   加边 → swing 稳定性 ✓（分散同步负载）
   加边 → cascade 韧性 ✗（集中边流量）
```

---

## 各阶段结果摘要

### Phase 0 — 论文复现

| 内容 | 结果 |
|------|------|
| Fig 1C (q=0 simplex) | 成功复现，趋势一致 |
| Fig 1D (多 q 截面) | 成功复现 |
| 求解器验证 | 功率平衡误差 < 1e-10 |

### SQ1 — 功率异质性

| 实验 | 关键数值 | 解读 |
|------|---------|------|
| 2A: σ_ratio 扫描 | κ_c 变化 < 2% | 异质性对稳定性无显著影响 |
| 2C: 集中化 r 扫描 | r=0.95 时 κ_c 上升 139% | 发电集中化显著削弱稳定性 |

### SQ2 — 时序动力学

| 指标 | 数值 |
|------|------|
| κ_c 昼夜比 (noon/dawn) | 22.6× |
| noon κ_c (mean) | ~6.7 |
| dawn κ_c (mean) | ~0.3 |
| PCC 瓶颈 | PCC 功率占全网 >50% |

### SQ4 — 拓扑优化

| 实验 | 关键数值 | 解读 |
|------|---------|------|
| q-sweep (Exp 4A) | κ_c 变化 < 1% | 拓扑随机化无效 |
| 策略对比 (m=4) | pcc_direct −49.8% | 直连 PCC 最优 |
| m-sweep (m=20) | κ_c 降低 ~83% | 收益随 m 递减 |
| 解析下界 gap | ~23% | 保守但有效 |
| 1/(d₀+m) scaling | 残差 < 5% | 经验拟合良好 |

### SQ3 — Cascade 验证

| 指标 | m=0 | m=4 (pcc) | m=8 (pcc) | 对照组 (无 PCC) |
|------|-----|-----------|-----------|----------------|
| ρ_mean (noon) | 0.18 | 0.23 | 0.40 | 0.84 |
| α* (noon) | 6.16 | 3.42 | 2.37 | — |
| S vs α/α* 形状 | 阶梯 | 阶梯 | 阶梯 | sigmoid |
| PCC/non-PCC 流量比 | ~11× | ~8× | ~6× | — |

### SQ5 — 阻尼敏感性分析

| 指标 | gamma = 0.1 | gamma = 1.0 | gamma = 5.0 |
|------|-------------|-------------|-------------|
| κ_c (m=0) | 7.66 | 7.18 | 5.97 |
| κ_c (m=4) | 3.87 | 3.60 | 3.01 |
| 绝对降低 | 3.79 | 3.59 | 2.96 |
| 相对降低 | 49.5% | 49.9% | 49.5% |

**核心发现**：拓扑优化的相对效果（~50%）在全阻尼范围内保持不变，说明拓扑效应与阻尼效应近似乘法可分离。低阻尼（逆变器主导的可再生能源电网）时绝对收益最大，拓扑优化尤为重要。

### 方法论发现

Smith et al. 的 cascade 分析流程存在一个未经验证的迁移假设：

1. 在**无 PCC 均匀功率**网络上计算 ρ 的 log-normal 分布
2. 将该分布直接应用于**含 PCC 异质功率**的微电网
3. 通过 α_c = ρ × α* 估算线路所需额定容量

我们的直接计算表明，含 PCC 网络的 S vs α/α* 曲线呈阶梯形而非 sigmoid，说明 ρ 在 PCC 网络上可能不是一个 well-defined 的连续随机变量，Smith 的 log-normal 拟合前提不成立。这影响了论文 Fig 6（α_c 风险评估）和 Fig 7（电池效果评估）的定量可靠性。

---

## 仓库结构

```
PowerGrid/
├── paper_reproduction/          # Phase 0: Smith Fig 1C/1D 复现
│   ├── model.py                 #   swing equation 求解器 + κ_c 二分搜索
│   ├── config.py                #   参数配置
│   ├── run_sweep.py             #   simplex 扫描入口
│   └── plot_combined.py         #   Fig 1C/1D 绘图
│
├── direction3_nonuniform/       # SQ1: 非均匀功率实验
│   ├── power_allocation.py      #   异质/集中式功率分配
│   ├── run_experiment_2A.py     #   σ_ratio 扫描
│   ├── run_experiment_2C.py     #   集中化 r 扫描
│   └── plot_results.py          #   Fig 2A/2C 绘图
│
├── sq2_data/                    # SQ2: 数据驱动时序分析
│   ├── data_loader.py           #   LCL + PV 数据加载管道
│   ├── network.py               #   含 PCC 的网络构建
│   ├── simplex.py               #   连续密度 (η⁺, η⁻, ηₚ) 计算
│   ├── run_trajectory.py        #   simplex 轨迹生成
│   ├── run_kappa_timeseries.py  #   κ_c 56点时序计算
│   └── plot_results.py          #   Fig 3A/3B 绘图
│
├── sq4_data/                    # SQ4: 拓扑优化
│   ├── edge_strategies.py       #   5 种边添加策略实现
│   ├── kappa_pipeline.py        #   κ_c 计算管道（含边添加）
│   ├── run_q_sweep.py           #   Exp 4A: q 扫描
│   ├── run_strategy_comparison.py  # Exp 4B-S1: 策略对比
│   ├── run_m_sweep.py           #   Exp 4B-S2: m 扫描
│   ├── run_sq4b_proof.py        #   解析下界验证
│   └── plot_sq4.py              #   Fig 4A–4D + proof 绘图
│
├── sq5_data/                    # SQ5: 阻尼敏感性分析
│   ├── run_sq5a.py              #   gamma 扫描实验
│   ├── plot_sq5a.py             #   Fig 5A1–5A3 绘图
│   └── REPORT_SQ5A.md           #   实验报告
│
├── sq3_data/                    # SQ3: Cascade 验证（exp/SQ3 分支）
│
├── data/                        # 原始数据
│   ├── LCL/                     #   London 家庭用电（30min 分辨率）
│   └── PV/                      #   Sheffield Solar 光伏发电（10min）
│
├── CLAUDE.md                    # AI 协作工作流配置
├── Phase1_Research_Report.md    # Phase 1 报告
├── Phase2_Research_Report.md    # Phase 2 报告
└── README.md                    # ← 你在这里
```

> **注意**：SQ3 的实验代码位于 `exp/SQ3` Git 分支，`sq3_data/` 目录在主分支上为空。

---

## 环境与运行

### 依赖

```bash
conda create -n PowerGrid python=3.10
conda activate PowerGrid
pip install numpy scipy networkx matplotlib tqdm numba pandas
```

### 快速验证各阶段

```bash
# Phase 0: 论文复现
cd paper_reproduction && python run_sweep.py && python plot_combined.py

# SQ1: 非均匀功率
cd direction3_nonuniform
python run_experiment_2A.py --n_ensemble 50 --output results_n50/results_2A.csv
python run_experiment_2C.py --n_ensemble 50 --output results_n50/results_2C.csv

# SQ2: 时序分析
cd sq2_data && python run_trajectory.py && python run_kappa_timeseries.py

# SQ4: 拓扑优化
cd sq4_data
python run_q_sweep.py           # Exp 4A
python run_strategy_comparison.py  # Exp 4B-S1
python run_m_sweep.py           # Exp 4B-S2

# SQ5: 阻尼敏感性
cd sq5_data && python run_sq5a.py
```

生产运行请将 `n_ensemble` 设为 200（SQ1）或 50（SQ2/SQ4/SQ5）。

---

## 参考文献

1. Smith, P. J., et al. "The effect of renewable energy incorporation on power grid stability and resilience." *Science Advances* 8, eabj6734 (2022).
2. Rohden, M., et al. "Self-organized synchronization in decentralized power grids." *Phys. Rev. Lett.* 109, 064101 (2012).
3. Watts, D. J., and Strogatz, S. H. "Collective dynamics of 'small-world' networks." *Nature* 393, 440–442 (1998).
4. Braess, D. "Über ein Paradoxon aus der Verkehrsplanung." *Unternehmensforschung* 12, 258–268 (1968).
