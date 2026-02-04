# 电网稳定性与节点组成关系的复现实验

本项目复现了 Smith et al. 发表于 *Science Advances* 的论文 [Sci. Adv. 8, eabj6734 (2022)] 中的 Figure 1C（三元相图热力图）和 Figure 1D（截面曲线图），研究电力网络中不同类型节点组成对系统稳定性的影响。

## 目录

- [研究背景](#研究背景)
- [数学模型](#数学模型)
- [关键概念](#关键概念)
- [代码结构](#代码结构)
- [参数说明](#参数说明)
- [快速开始](#快速开始)
- [输出文件](#输出文件)
- [常见问题](#常见问题)

## 研究背景

电力系统中的节点可分为三类：
- **发电机节点** ($n_+$)：向电网注入功率（$P > 0$）
- **负载节点** ($n_-$)：从电网消耗功率（$P < 0$）
- **被动节点** ($n_0$)：不产生也不消耗功率（$P = 0$），如变电站

这三类节点的数量满足约束 $n_+ + n_- + n_0 = N$，因此可以用**三元单纯形**（ternary simplex）来表示所有可能的节点组成方案。

本实验的核心问题是：**不同的节点组成如何影响电网的同步稳定性？**

## 数学模型

### 摆动方程（Swing Equation）

每个节点 $i$ 的动力学由二阶微分方程描述：

$$\frac{d^2\theta_i}{dt^2} + \gamma \frac{d\theta_i}{dt} = P_i - \kappa \sum_{j} A_{ij} \sin(\theta_i - \theta_j)$$

等价的一阶形式（代码实现）：

$$\frac{d\theta_i}{dt} = \omega_i$$

$$\frac{d\omega_i}{dt} = P_i - \gamma \omega_i - \kappa \sum_{j} A_{ij} \sin(\theta_i - \theta_j)$$

### 变量说明

| 符号 | 含义 | 说明 |
|------|------|------|
| $\theta_i$ | 节点 $i$ 的相位角 | 表示发电机转子角度或电压相位 |
| $\omega_i = d\theta_i/dt$ | 角频率 | 相位变化率，稳态时趋近于0 |
| $\gamma$ | 阻尼系数 | 模拟机械摩擦和电气阻尼 |
| $P_i$ | 节点功率 | 发电机 $P_i > 0$，负载 $P_i < 0$，被动节点 $P_i = 0$ |
| $\kappa$ | 耦合强度 | 表征传输线容量，越大越容易同步 |
| $A_{ij}$ | 邻接矩阵 | $A_{ij} = 1$ 若节点 $i,j$ 直接相连，否则为 0 |

### 物理直觉

- **右边第一项** $P_i$：净功率输入/输出，驱动相位变化
- **右边第二项** $-\gamma \omega_i$：阻尼项，抑制频率偏离
- **右边第三项** $-\kappa \sum A_{ij} \sin(\theta_i - \theta_j)$：耦合项，相邻节点相互拉扯趋于同步

当 $\kappa$ 足够大时，耦合力足以克服功率差异，系统达到**同步稳态**（所有 $\omega_i \to 0$）。

## 关键概念

### 临界耦合强度 $\kappa_c$

**定义**：使系统从不稳定变为稳定的最小耦合强度。

- $\kappa < \kappa_c$：系统无法同步，频率持续波动
- $\kappa \geq \kappa_c$：系统收敛到同步稳态

$\kappa_c$ 越小，说明该节点组成更容易实现同步，**稳定性越好**。

### 二分法搜索 $\kappa_c$

由于没有解析公式，我们用**二分法**数值求解：

```
1. 设定搜索区间 [κ_min, κ_max]
2. 取中点 κ_mid = (κ_min + κ_max) / 2
3. 数值积分 ODE，检查系统是否稳定
4. 若稳定：κ_max = κ_mid（说明临界值更小）
5. 若不稳定：κ_min = κ_mid（说明临界值更大）
6. 重复直到收敛
```

**稳定性判据**：$\max_i |\omega_i| < \epsilon$（所有频率趋于零）

### Watts-Strogatz 小世界网络

网络拓扑采用 Watts-Strogatz 模型生成：

1. 初始为 $N$ 个节点的环形晶格，每个节点连接 $K$ 个最近邻
2. 以概率 $q$ 重连每条边到随机节点

| 参数 $q$ | 网络特性 |
|----------|----------|
| $q = 0$ | 规则环形网络，高聚类，长平均路径 |
| $q = 1$ | 完全随机网络，低聚类，短平均路径 |
| $0 < q < 1$ | 小世界网络，兼具高聚类和短路径 |

### 三元单纯形可视化

由于 $n_+ + n_- + n_0 = N$，三个变量只有两个自由度，可用等边三角形表示：

```
                    被动节点 (n₀ = N)
                         /\
                        /  \
                       /    \
                      /      \
                     /   (i)  \    ← 截面线
                    /          \
                   /____________\
          发电机                 负载
         (n₊ = N)              (n₋ = N)
```

- 三角形内部每个点对应唯一的 $(n_+, n_-, n_0)$ 组合
- 热力图颜色表示该组成对应的 $\bar{\kappa}_c$（集成平均值）
- 虚线 (i) 表示固定 $n_0 = 16$ 的截面

## 代码结构

| 文件 | 功能描述 |
|------|----------|
| `config.py` | 全局参数配置（网络规模、物理参数、计算参数等） |
| `model.py` | 核心物理模型：摆动方程 ODE、网络生成、$\kappa_c$ 计算 |
| `sweep.py` | 参数扫描模块（支持并行计算） |
| `run_sweep.py` | 数值计算入口脚本，生成 `.npz` 数据文件 |
| `plot_fig1c.py` | 绘制 Fig. 1C 三元热力图 |
| `plot_fig1d.py` | 绘制 Fig. 1D 截面曲线图 |
| `plot_combined.py` | 合并 Fig. 1C 和 1D 为一张图 |

### 代码流程

```
run_sweep.py
    │
    ├─► 三元单纯形扫描 (q=0.0)
    │       │
    │       ├─► 对每个 (n₊, n₋) 组合
    │       │       │
    │       │       └─► 重复 ENSEMBLE_SIZE 次
    │       │               │
    │       │               ├─► generate_network() → 生成随机网络
    │       │               ├─► assign_power() → 分配节点功率
    │       │               └─► compute_kappa_c() → 二分法求 κ_c
    │       │
    │       └─► 输出 data_simplex_q0.0.npz
    │
    └─► 截面扫描 (q = 0, 0.1, 0.4, 1.0)
            │
            └─► 输出 data_crosssec.npz

plot_combined.py
    │
    ├─► 读取 data_simplex_q0.0.npz → 绘制热力图
    ├─► 读取 data_crosssec.npz → 绘制曲线图
    └─► 输出 fig1cd_combined.png
```

## 参数说明

### 物理参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `N` | 50 | 网络节点总数 |
| `K` | 4 | Watts-Strogatz 平均度（每节点连接数） |
| `GAMMA` | 1.0 | 阻尼系数 |
| `P_MAX` | 1.0 | 归一化最大功率 |
| `KAPPA_RANGE` | (0.001, 3.0) | 二分法搜索区间 |

### 计算参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `ENSEMBLE_SIZE` | 50 | 每个配置的集成平均次数（不同随机种子） |
| `STEP_SIZE` | 3 | 三元单纯形采样步长 |
| `T_INTEGRATE` | 100 | ODE 积分时间 |
| `BISECTION_STEPS` | 20 | 二分法迭代次数 |
| `CONV_TOL` | 1e-3 | 收敛判据阈值 |
| `Q_VALUES` | [0, 0.1, 0.4, 1.0] | 网络重连概率（Fig. 1D） |

### 截面参数（Fig. 1D）

| 参数 | 值 | 含义 |
|------|-----|------|
| `NP_CROSS` | 16 | 截面处的被动节点数 |
| `N_MINUS_RANGE` | (1, 33) | 负载节点数扫描范围 |

## 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n PowerGrid python=3.10
conda activate PowerGrid

# 安装依赖
pip install numpy scipy networkx matplotlib tqdm numba
```

依赖版本参考：
- numpy >= 1.21
- scipy >= 1.7
- networkx >= 2.6
- matplotlib >= 3.4
- tqdm >= 4.62
- numba >= 0.54（可选，用于加速 ODE 计算）

### 2. 运行数值计算

```bash
cd paper_reproduction
python run_sweep.py
```

运行时间取决于硬件配置和参数设置，默认配置下约需 15-30 分钟。

运行过程中会显示进度条：
```
============================================================
SIMPLEX SWEEP (Fig. 1C)
============================================================
Parameters: q=0.0, realizations=50, step=3
Total simplex points: 272
Processing 272 configurations...
Simplex: 100%|██████████████████████████| 272/272 [12:34<00:00]
```

### 3. 生成可视化图片

```bash
# 生成合并图（推荐）
python plot_combined.py

# 或分别生成
python plot_fig1c.py
python plot_fig1d.py
```

### 4. 可选：交互查看图片

```bash
python plot_combined.py --show
```

## 输出文件

运行完成后，`output/` 目录下会生成以下文件：

### 数据文件

| 文件 | 内容 |
|------|------|
| `data_simplex_q0.0.npz` | 三元单纯形扫描数据（$q=0$） |
| `data_crosssec.npz` | 截面扫描数据（多个 $q$ 值） |

数据文件结构（NumPy 压缩格式）：
```python
# data_simplex_q0.0.npz
n_plus     # 发电机节点数
n_minus    # 负载节点数
n_passive  # 被动节点数
mean_kappa # κ_c 集成平均值
std_kappa  # κ_c 标准差

# data_crosssec.npz
n_minus    # 负载节点数
q_values   # 重连概率数组
mean_kappa # κ_c 均值矩阵 [q_index, n_minus_index]
std_kappa  # κ_c 标准差矩阵
```

### 图片文件

| 文件 | 内容 |
|------|------|
| `fig1c.png` | 三元热力图（对应论文 Fig. 1C） |
| `fig1d.png` | 截面曲线图（对应论文 Fig. 1D） |
| `fig1cd_combined.png` | 合并图（推荐使用） |

## 常见问题

### Q: Numba 编译慢或报错？

首次运行时 Numba 会 JIT 编译 ODE 函数，可能需要等待几秒。如果 Numba 不可用，代码会自动退化为纯 NumPy 实现。

在 `config.py` 中设置 `USE_NUMBA = False` 可禁用 Numba。

### Q: 如何加速计算？

1. **减少集成次数**：在 `config.py` 中减小 `ENSEMBLE_SIZE`（如 20）
2. **增大采样步长**：增大 `STEP_SIZE`（如 5）
3. **缩短积分时间**：减小 `T_INTEGRATE`（但可能影响收敛判断）

### Q: 结果与论文有差异？

- 本实现使用了校准因子 `KAPPA_SCALE_FACTOR = 0.5` 以匹配论文数值范围
- 由于随机种子和数值精度差异，细节可能略有不同
- 主要趋势应与论文一致

### Q: 如何读取数据文件？

```python
import numpy as np

# 加载数据
data = np.load('output/data_simplex_q0.0.npz')

# 查看所有键
print(data.files)  # ['n_plus', 'n_minus', 'n_passive', 'mean_kappa', 'std_kappa']

# 读取数组
mean_kappa = data['mean_kappa']
print(f"κ_c 范围: [{mean_kappa.min():.4f}, {mean_kappa.max():.4f}]")
```

## 参考文献

Smith, P. J., et al. "The role of passive agents in power grid resilience." *Science Advances* 8, eabj6734 (2022).

---

如有问题或建议，欢迎提出 Issue 或 Pull Request。
