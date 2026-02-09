# SQ2 实验规格书 — 数据驱动的Simplex轨迹与κ_c时间演化

**签发**：Research Lead (Claude Chat)  
**执行**：Claude Code  
**日期**：2026-02-09  
**优先级**：HIGH — 本周完成

---

## 1. 目标

用伦敦真实家庭用电 + PV发电数据驱动50节点微电网模型，追踪：
1. 微电网在configuration simplex上的日轨迹（η+, η-, ηp）
2. κ_c随时间的变化（日振荡特征）

核心预期：电网大部分时间处于simplex边缘（弹性最差区域），仅在晨昏过渡时短暂经过中心（弹性最优区域）。

---

## 2. 数据规格

### 2.1 LCL家庭用电
- **路径**：`data/LCL/Small LCL Data/`
- **格式**：CSV，列 = `LCLid, stdorToU, DateTime, KWH/hh (per half hour)`
- **分辨率**：30 min
- **可用户数**：5566
- **日期范围**：2011-11 至 2014-02
- **单位**：KWH per half hour → 转换为KW需 ×2

### 2.2 PV发电
- **路径**：`data/PV/PV Data/2014-11-28 Cleansed and Processed/EXPORT TenMinData/EXPORT TenMinData - Customer Endpoints.csv`
- **关键列**：`Substation`, `DateTime`, `P_GEN`
- **可用站点**：6
- **分辨率**：10 min → 重采样至30 min（取均值）
- **日期范围**：2014-06 至 2014-11（夏+秋）
- **单位**：确认P_GEN单位，可能需要转换为KW

### 2.3 日期对齐策略
LCL和PV日期不重叠。按Smith et al.做法：
- 按季节分类（夏季：Jun-Aug，秋季：Sep-Nov，冬季：Dec-Feb）
- 从LCL抽取对应季节的一周数据
- 从PV抽取对应季节的一周数据
- 独立配对到每个节点（不要求同一天）

**本次实验选择**：
- **夏季场景**（PV输出最高，效果最显著）：PV取Jul某周，LCL取Jul某周（2012或2013）
- **冬季场景**（对比用）：PV=0（无数据/忽略），LCL取Dec某周

---

## 3. 微电网模型

### 3.1 网络结构
- n = 50 个house节点 + 1 个PCC节点 = **51个节点总计**
- 50个house节点：Watts-Strogatz(n=50, K̄=4, q=0.1)
- PCC节点：第51个节点，连接到网络中随机选择的3-4个节点（模拟微电网与外部电网的接口）

### 3.2 功率分配（每个时间步t）
- **100% PV uptake**：全部50个house都装有PV
- 每个house i 的净功率：P_i(t) = g_i(t) - c_i(t)
  - c_i(t)：从LCL随机抽取的一户的消费时间序列（KW）
  - g_i(t)：从6条PV时间序列中**有放回随机抽取**一条（KW）
- PCC功率：P_PCC(t) = -Σ P_i(t)，确保总功率平衡

### 3.3 节点分类（每个时间步t）
- P_i(t) > threshold → 发电节点（generator）
- P_i(t) < -threshold → 消费节点（consumer）  
- |P_i(t)| ≤ threshold → 被动节点（passive）
- threshold = 0.01 * max(|P(t)|)（Smith et al.用类似的小阈值）

### 3.4 连续密度计算
按Smith et al. Eq. 3-4：

```
η+(t) = (1 / (n * max(P(t)))) * Σ_{x ∈ P+} x
η-(t) = (1 / (n * min(P(t)))) * Σ_{x ∈ P-} x  
ηp(t) = 1 - η+(t) - η-(t)
```

其中 P+ 和 P- 分别为正和负的功率分量。注意这里n=50（不含PCC）。

---

## 4. 实验流程

### 实验3A — Simplex轨迹

**对每个微电网实例**：
1. 生成WS网络(n=50, K̄=4, q=0.1) + PCC节点
2. 为50个house随机分配LCL消费序列（从5566户中抽50户，无放回）
3. 为50个house随机分配PV发电序列（从6站点中抽50次，有放回）
4. 对一周内每个30min时间步（共336步）：
   a. 计算 P_i(t) = g_i(t) - c_i(t)
   b. 计算 P_PCC(t) = -Σ P_i(t)
   c. 分类节点 → 计算 (η+, η-, ηp)
   d. 记录simplex坐标
5. 重复 n_ensemble=50 次
6. 计算平均轨迹和标准差带

**输出**：
- CSV：`results_sq2/trajectory_summer_100pct.csv`
  - 列：`timestep, hour, eta_plus_mean, eta_plus_std, eta_minus_mean, eta_minus_std, eta_p_mean, eta_p_std`
- 图：simplex上的平均轨迹（类似Smith et al. Fig.4 D-G）

### 实验3B — κ_c时间序列

**对每个时间步t，对每个微电网实例**：
1. 使用当时的功率向量 P(t)（含PCC）
2. 通过bisection计算κ_c
3. 记录

**关键简化**：不需要每个时间步都跑bisection。选择代表性时间点：
- 每天选8个时间点：00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00
- 一周 = 7天 × 8点 = 56个时间点
- 每个时间点 × 50个ensemble = 2800次bisection
- 预计运行时间：~1.5小时

**输出**：
- CSV：`results_sq2/kappa_c_timeseries_summer_100pct.csv`
  - 列：`day, hour, kappa_c_mean, kappa_c_std`
- 图：κ_c vs time曲线（类似Smith et al. Fig.5B），含±1SD阴影

---

## 5. 数据预处理模块

需要新建 `sq2_data/data_loader.py`：

```python
def load_lcl_households(data_dir, season, n_households=50, seed=None):
    """
    从LCL数据中随机抽取n_households户的一周30min消费序列。
    
    Args:
        data_dir: LCL数据目录
        season: 'summer' | 'winter' | 'autumn'
        n_households: 抽取户数
        seed: 随机种子
    
    Returns:
        consumption: ndarray, shape (n_households, 336), 单位KW
        时间索引: 一周的datetime序列
    """

def load_pv_generation(data_path, season, n_panels=50, seed=None):
    """
    从PV数据中有放回抽取n_panels条发电序列，重采样至30min。
    
    Args:
        data_path: PV CSV文件路径
        season: 'summer' | 'autumn'
        n_panels: 抽取数量（有放回）
        seed: 随机种子
    
    Returns:
        generation: ndarray, shape (n_panels, 336), 单位KW
    """

def compute_net_power(consumption, generation):
    """
    计算净功率和PCC功率。
    
    Returns:
        P: ndarray, shape (n_nodes+1, 336)  # 最后一行是PCC
    """
```

---

## 6. 网络模型修改

需要修改或新建网络生成函数，支持PCC节点：

```python
def generate_network_with_pcc(n_houses=50, k=4, q=0.1, n_pcc_links=4, seed=None):
    """
    生成WS网络 + PCC节点。
    
    PCC是第n_houses个节点（index=50），连接到网络中随机选择的n_pcc_links个节点。
    
    Returns:
        A_csr: (n_houses+1, n_houses+1) 稀疏邻接矩阵
    """
```

---

## 7. 可视化规格

### 图3A：Simplex轨迹图
- 三角形simplex，三轴为 η+（generators）, η-（consumers）, ηp（passive）
- 平均轨迹用实线，±1SD用半透明阴影
- 标注 (i) midnight 和 (ii) midday 位置
- 参考Smith et al. Fig.4 C-G的风格
- **英文标注**

### 图3B：κ_c时间序列图
- x轴：Days (1-7)
- y轴：κ̄_c / P_max
- 均值实线 + ±1SD阴影带
- 参考Smith et al. Fig.5B的风格
- **英文标注**

---

## 8. 文件组织

```
codebase/PowerGrid/
├── sq2_data/
│   ├── data_loader.py      # 数据加载与预处理
│   ├── run_trajectory.py   # 实验3A主脚本
│   ├── run_kappa_timeseries.py  # 实验3B主脚本
│   ├── plot_results.py     # 可视化
│   └── results_sq2/        # 输出目录
│       ├── trajectory_summer_100pct.csv
│       ├── kappa_c_timeseries_summer_100pct.csv
│       ├── fig_3A_simplex_trajectory.png
│       └── fig_3B_kappa_timeseries.png
├── paper_reproduction/
│   └── model.py            # 复用：compute_kappa_c, generate_network
└── data/
    ├── LCL/
    └── PV/
```

---

## 9. 依赖与注意事项

1. **复用**paper_reproduction/model.py的compute_kappa_c()和相关函数
2. PV数据单位需首先确认（P_GEN的单位是KW还是W？）——在data_loader中加断言检查
3. 缺失值处理：LCL中0.003%缺失 → 线性插值；PV中缺失 → 同样插值
4. 30min对齐：PV 10min → 30min取均值
5. 夜间PV = 0是正常的，不是缺失
6. bisection参数与SQ1一致：kappa_range=(0.001, 3.0), bisection_steps=20, t_integrate=100, conv_tol=1e-3

---

## 10. 计算量估算

| 实验 | 计算量 | 预计时间（HP暗影精灵） |
|------|--------|----------------------|
| 3A（轨迹） | 50 ensemble × 336 timesteps × 分类计算 | ~5分钟（无bisection） |
| 3B（κ_c序列） | 50 ensemble × 56 时间点 × bisection | ~1.5小时 |
| **总计** | | **~2小时** |

---

## 11. 验收标准

- [ ] trajectory CSV: 336行，8列，无NaN
- [ ] kappa_c CSV: 56行，4列，无NaN
- [ ] simplex轨迹图显示边缘震荡特征
- [ ] κ_c时间序列显示日振荡特征
- [ ] 所有图使用英文标注
- [ ] 功率平衡验证：每个时间步 |Σ P_k| < 1e-10
