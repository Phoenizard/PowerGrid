# 方向③：非均匀功率分配对电网稳定性的影响

## 项目计划书

---

## 1. 研究背景

### 1.1 核心模型：Swing Equation

电网中 $n$ 个节点的动力学由 swing equation 描述：

$$\frac{d^2\theta_k}{dt^2} + \gamma \frac{d\theta_k}{dt} = P_k - \kappa \sum_{l=1}^{n} A_{kl} \sin(\theta_k - \theta_l), \quad k = 1 \ldots n$$

其中：
- $\theta_k(t)$：节点 $k$ 的相角偏差
- $\gamma$：惯性阻尼系数
- $P_k$：节点 $k$ 的功率（发电为正，消费为负）
- $\kappa$：全局耦合强度
- $A_{kl}$：网络邻接矩阵

### 1.2 参考论文的简化假设

Smith et al. (2022, Science Advances) 假设**均匀功率分配**：
- 所有发电节点：$P_i = P_{\max} / n_+$
- 所有消费节点：$P_i = -P_{\max} / n_-$

现实电网中功率分配高度异质：大型发电站 vs 屋顶光伏，高耗能工厂 vs 低耗能住宅。

### 1.3 本方向的核心问题

**当功率分配从均匀变为异质时，临界耦合强度 $\kappa_c$ 如何变化？发电端和消费端的异质性，哪个对稳定性影响更大？**

### 1.4 文献支持

- Rohden et al. (PRL, 2012)：$n_+ = n_-$ 等分是swing equation研究中的标准基线设置
- Smith et al. (2022)：$n_+ = n_- = 25$ 对应 simplex 底边中心，$\bar{\kappa}_c$ 最小
- Watts-Strogatz 网络（$\bar{K}=4$, $q=0.1$）是小世界电网的标准模型

---

## 2. 已完成的工作（Phase 1）

Phase 1 已完成论文 Fig. 1 的复现，代码仓库可用。核心组件：

| 文件 | 功能 |
|------|------|
| `model.py` | swing equation ODE、Watts-Strogatz 网络生成、`assign_power()` 函数、`compute_kappa_c()` via bisection |
| `run_sweep.py` | 参数扫描框架 |

关键接口——`assign_power()` 当前实现均匀分配，是本方向需要修改的核心函数。

参考代码仓库：Smith et al. 官方代码 https://doi.org/10.5281/zenodo.5702877

---

## 3. 实验设计

### 3.0 全局固定参数

| 参数 | 值 | 理由 |
|------|-----|------|
| 网络规模 $n$ | 50 | 与 Smith et al. 一致 |
| 节点组成 | $n_+ = n_- = 25$, $n_p = 0$ | simplex 底边中心，最稳定基线，文献标准设置 |
| 平均度 $\bar{K}$ | 4 | 文献标准 |
| 拓扑 | Watts-Strogatz, $q = 0.1$ | 小世界网络 |
| 阻尼 | $\gamma = 1$ | 与 Smith et al. 一致 |
| 总功率 | $P_{\max} = 1.0$ | 归一化 |
| 均值功率 | $\bar{P} = P_{\max}/n_+ = 0.04$ | 每个发电/消费节点的均值 |
| 集成规模 | 每个参数点 200 个网络实例 | 统计充分性 |
| $\kappa_c$ 计算 | 数值积分 swing equation + bisection 搜索 | 与论文方法一致 |

### 3.1 实验 2A — 异质程度扫描

#### 目标
$\bar{\kappa}_c$ 如何随功率分配异质程度 $\sigma$ 变化？发电端和消费端的异质性效应是否不同？

#### 设计

**分两组子实验**：

**2A-gen（发电端异质）：**
- 发电节点：$P_i = \bar{P} + \epsilon_i$，其中 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$，约束 $\sum_{i \in \text{gen}} \epsilon_i = 0$
- 消费节点：均匀分配 $P_i = -\bar{P} = -0.04$
- 截断处理：$P_i = \max(\bar{P} + \epsilon_i, \delta)$，$\delta = 10^{-4} \bar{P} = 4 \times 10^{-6}$
- **截断后归一化**：确保 $\sum_{i \in \text{gen}} P_i = P_{\max}$（等比例缩放所有发电节点的功率）

**2A-con（消费端异质）：**
- 消费节点：$P_i = -\bar{P} + \epsilon_i$，其中 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$，约束 $\sum_{i \in \text{con}} \epsilon_i = 0$
- 发电节点：均匀分配 $P_i = \bar{P} = 0.04$
- 截断处理：$P_i = -\max(\bar{P} - \epsilon_i, \delta)$（确保消费功率为负）
- **截断后归一化**：确保 $\sum_{i \in \text{con}} P_i = -P_{\max}$

#### 扫描参数

$\sigma / \bar{P} \in \{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$（共 9 个点）

#### 对每个 $\sigma$ 值，每个网络实例的流程

1. 生成 Watts-Strogatz 网络（$n=50$, $\bar{K}=4$, $q=0.1$）
2. 随机选择 25 个节点为发电节点，25 个为消费节点
3. 按上述规则分配功率（含截断和归一化）
4. 验证 $\sum P_k = 0$（功率平衡）
5. 通过 bisection 计算 $\kappa_c$
6. 重复 200 次，取 $\bar{\kappa}_c$ 和标准差

#### 输出

- **图 1**：$\bar{\kappa}_c$ vs $\sigma/\bar{P}$ 曲线，2A-gen 和 2A-con 两条线叠在同一张图上，含误差带（±1 SD）
- **数据**：CSV 文件，列为 `sigma_ratio, kappa_c_mean_gen, kappa_c_std_gen, kappa_c_mean_con, kappa_c_std_con`

---

### 3.2 实验 2C — 集中式 vs 分布式过渡

#### 目标
从集中式发电向分布式发电过渡时，$\kappa_c$ 如何变化？是否存在最优的集中化比例？

#### 设计

固定 $n_+ = 25$ 个发电节点。其中 **1 个"大站"** 承担总发电功率的比例 $r$，其余 **24 个"小站"** 均分剩余功率 $(1-r)$。

具体功率分配：
- 大站：$P_{\text{big}} = r \cdot P_{\max}$
- 小站（每个）：$P_{\text{small}} = (1 - r) \cdot P_{\max} / 24$
- 消费端：均匀分配 $P_i = -P_{\max}/25 = -0.04$

当 $r = 1/25 = 0.04$ 时，退化为完全均匀分配（baseline）。
当 $r \to 1$ 时，一个大站承担几乎全部发电。

#### 扫描参数

$r \in \{0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95\}$（共 11 个点）

#### 对每个 $r$ 值的流程

1. 生成 Watts-Strogatz 网络
2. 随机选择 25 个发电节点，其中随机指定 1 个为大站
3. 大站功率 $P_{\text{big}} = r$，其余 24 个小站各 $(1-r)/24$
4. 消费节点均匀 $P_i = -0.04$
5. 验证 $\sum P_k = 0$
6. Bisection 计算 $\kappa_c$
7. 重复 200 次

#### 输出

- **图 2**：$\bar{\kappa}_c$ vs $r$ 曲线，含误差带
- **数据**：CSV 文件，列为 `r, kappa_c_mean, kappa_c_std`

---

## 4. 截断归一化算法详述

这是实验 2A 中保证物理合理性的关键步骤。

### 算法（以发电端为例）

```
输入：n_+ 个发电节点的均值 P_bar, 标准差 sigma, 最小值 delta
输出：功率向量 P_gen，满足 P_gen[i] > 0 且 sum(P_gen) = P_max

1. 采样 epsilon_i ~ N(0, sigma^2), i = 1, ..., n_+
2. 中心化：epsilon_i <- epsilon_i - mean(epsilon_i)  # 确保 sum = 0
3. 原始分配：P_i = P_bar + epsilon_i
4. 截断：P_i = max(P_i, delta)
5. 归一化：P_i = P_i * (P_max / sum(P_gen))  # 等比例缩放使总功率恢复
6. 返回 P_gen
```

消费端类似，注意符号（功率为负值）。

---

## 5. 预期结果与分析

### 5.1 实验 2A 预期

- $\bar{\kappa}_c$ 应随 $\sigma$ 单调递增（异质性越大，同步越难）
- 可能存在非线性特征：低 $\sigma$ 时影响较小，高 $\sigma$ 时急剧增加
- 发电端 vs 消费端的异质性效应可能不同（由于 swing equation 的非线性耦合项 $\sin(\theta_k - \theta_l)$）

### 5.2 实验 2C 预期

- $\bar{\kappa}_c$ 应随 $r$ 增加而增大（越集中，稳定性越差）
- 与 Rohden et al. (2012) 的结论一致：去中心化有利于同步
- 可能存在临界 $r$ 值，超过后 $\kappa_c$ 急剧上升

### 5.3 关联论文核心议题

这两个实验直接回答项目任务书中的第二个核心问题："How does variation in generation or usage influence grid function?"

---

## 6. 时间线

| 阶段 | 任务 | 预计耗时 |
|------|------|----------|
| 代码实现 | 修改 `assign_power()` + 新增扫描脚本 | 1-2 天 |
| 实验 2A 运行 | 2 组 × 9 点 × 200 实例 | 数小时（视机器性能） |
| 实验 2C 运行 | 11 点 × 200 实例 | 数小时 |
| 可视化 | 绘制图表 | 半天 |
| 分析与撰写 | 解读结果 + 写入报告 | 1-2 天 |

---

## 7. 参考文献

1. Smith, O., Cattell, O., Farcot, E., O'Dea, R.D. & Hopcraft, K.I. (2022). The effect of renewable energy incorporation on power grid stability and resilience. *Science Advances*, 8, eabj6734.
2. Rohden, M., Sorge, A., Timme, M. & Witthaut, D. (2012). Self-organized synchronization in decentralized power grids. *Phys. Rev. Lett.*, 109, 064101.
3. Watts, D.J. & Strogatz, S.H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393, 440-442.
