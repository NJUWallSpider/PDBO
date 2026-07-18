# PDBO 算法改进研究：谱尺度适配、轨迹舍入与增量精修

## 1. 结论摘要

本轮实验使用 `conda pdbo` 环境，固定 paper-aligned simultaneous GDA，重点比较同一实例、同一随机种子下的 paired 结果。最稳定的改进组合是：

1. 用谱前沿自动设置 uniform dual；
2. 用谱宽自动设置 primal step；
3. 将初始 threshold cut 纳入 archive；
4. 保存全轨迹最好的 multilinear expectation，并用条件期望确定性舍入；
5. 对输出执行稀疏增量 one-flip refinement。

推荐参数为

$$
y_0=-\lambda_{\min}(W)+\varepsilon+\frac{\beta T_{\rm burn}}4,
$$

$$
\alpha=\frac{c}{2(\lambda_{\max}(W)+y_0)},
\qquad 0<c<1.
$$

实验中 `c=0.5` 是较稳健的跨实例折中；固定 3000 步预算下，`T_burn=0--500` 较稳健，过长的 `T_burn=1000` 系统性浪费搜索预算。

## 2. 新增保证

### 2.1 谱尺度初始化

取保守的最小特征值估计和 `dual_psd_margin >= 0`，则初始矩阵满足名义上的

$$
W+y_0I\succeq \left(\varepsilon+\frac{\beta T_{\rm burn}}4\right)I.
$$

谱步长使最坏的初始 multiplier 为

$$
1-2\alpha(\lambda_{\max}+y_0)=1-c>0.
$$

因此不同权重尺度的实例不再共享一个无量纲意义不同的 `dual_init` 和 `lr_x`，并直接满足全谱低通定理所需的正 multiplier 条件。

### 2.2 初始 archive

求解器现在在第一次 primal-dual step 前评估所有初始 threshold cuts，同时保留原来的全零候选。对非负权 Max-Cut，这使代码真正实现理论中的

$$
\mathbb E C_{\rm archive}\ge M/2\ge \operatorname{OPT}/2.
$$

该改动对任意实例都不会使 incumbent 变差。

### 2.3 全轨迹条件期望舍入

对每个连续状态 $x$，先将 QUBO 对角项 multilinearize，计算独立 Bernoulli 分布下的精确期望 $\widetilde F(x)$。求解器保存

$$
x_\star\in\arg\min_{t,b}\widetilde F(x_b^t),
$$

最后逐坐标使用条件期望法，确定性构造二元点 $s_{\rm CE}$，满足

$$
F(s_{\rm CE})\le \widetilde F(x_\star).
$$

对 Max-Cut 即

$$
C(s_{\rm CE})\ge
\max_{t,b}\mathbb E C(B\mid z_b^t).
$$

这把原先关于 $\delta_q$ 的期望质量接口转成了确定性输出保证，并避免只对最终 relaxed state 舍入。

### 2.4 增量 one-flip

对一般 QUBO 维护

$$
h_i=c_i+Q_{ii}+\sum_{j\ne i}(Q_{ij}+Q_{ji})x_j,
$$

则翻转 $i$ 的精确增量为

$$
\Delta_i=(1-2x_i)h_i.
$$

接受一次翻转后只更新邻居的 $h_j$。这保持原 sequential one-flip 的单调性，但不再为每个候选重新执行完整稀疏矩阵乘。G70 上同一局部最优结果的 refinement 时间从约 `0.9s` 降到 `0.005s`。

## 3. 初始化消融

设置为 batch 10、3000 步、$\alpha=.002,\beta=.02,\rho=.05$，每项 5 个 seed。表中为 one-flip 后的平均 cut。

| 实例 | fixed $y_0=15$ | spectral burn 0 | burn 250 | burn 500 |
|---|---:|---:|---:|---:|
| G1 | 11606.2 | 11598.4 | 11602.8 | 11608.4 |
| G11 | 450.4 | 523.6 | 532.0 | 536.8 |
| G14 | 3018.0 | 3029.0 | 3028.8 | 3026.2 |
| G22 | 13213.4 | 13268.0 | 13276.6 | 13274.6 |
| G70 | 8663.0 | 9249.4 | 9298.6 | 9330.0 |

G70 的 $\lambda_{\min}\approx-3.56$。固定 `dual_init=15` 需要约 2289 个理想中心步才到 PSD 边界，因此在 3000 步预算内大部分时间都用于无效 burn-in。谱尺度初始化消除了这一失配。

## 4. 完整改进组合的大图验证

基线为 `dual_init=15, alpha=.002`、原 threshold archive，再做同一个 one-flip；改进配置为 spectral burn 0、spectral step fraction 0.5、trajectory CE 和增量 one-flip。batch 10、3000 步、3 个 paired seeds。

| 实例 | 基线 refined | 改进 refined | 平均提升 | 改进 solve 时间 | 谱初始化时间 |
|---|---:|---:|---:|---:|---:|
| G67 | 5436.7 | 6810.7 | +1374.0 | 3.93s | 0.08s |
| G70 | 8662.0 | 9475.0 | +813.0 | 3.51s | 0.04s |
| G72 | 5514.0 | 6852.0 | +1338.0 | 3.95s | 0.05s |
| G77 | 7804.7 | 9727.3 | +1922.7 | 5.54s | 0.13s |
| G81 | 11034.7 | 13732.7 | +2698.0 | 8.32s | 0.17s |

在这五个大图上关闭 CE 后的均值分别约为 6810、9475、6852、9727、13731，所以主要提升来自谱尺度参数，而不是更激进的输出采样。CE 在自动步长已经产生高度离散状态时只额外贡献 0--4，但在固定小步长的 G70 上曾贡献约 34，因此仍是低成本的 no-regression 输出层。

## 5. 负结果与未采用方案

### 5.1 单一 $v_1$ warm start

已有大图结果中，纯最小特征向量初始化在 G67/G70/G72/G77/G81 上均显著弱于随机中心 batch。一般图上 `sign(v1)` 可任意差，因此没有将其作为默认初始化；最多可作为 ensemble 中的一条额外轨迹。

### 5.2 trajectory Gram/GW kernel

将多个时刻和 batch 状态组成 Gram 矩阵，确实得到合法 $X\succeq0,\operatorname{diag}X=1$；独立 Gaussian hyperplane rounding 也具有精确 arcsine 期望，非负权时满足 GW 下界。但在 G1/G3/G14/G22/G70 上使用 256 次 rounding 后，实际 cut 普遍低于 CE+one-flip。因此保留研究脚本，不加入生产求解器。

### 5.3 延迟 anisotropy

前 500/1000 步只更新 dual mean 能得到更干净的早期多项式滤波，但在 G1/G3/G14/G22/G70 上没有稳定质量提升；1500 步延迟开始出现退化。因此没有保留该实验接口。

### 5.4 过大的谱步长和 burn-in

`c=.75` 在 G70 上略优于 `.5`，但在 G3/G22 回落；`c=.9` 更不稳定。`T_burn=1000` 在固定 3000 步预算下也普遍回落。因此不按单个实例选择激进默认值。

## 6. 推荐命令

```bash
conda run -n pdbo python main.py \
  --task mc --graph Gset --Gset_id 70 \
  --batch 10 --max_iters 3000 --lr_y 0.02 --rho 0.05 \
  --dual_init_mode spectral --dual_burn_in 0 \
  --primal_lr_mode spectral --spectral_step_fraction 0.5 \
  --conditional_rounding --refine
```

若需要更长的严格凸筛选阶段，可增加 `dual_burn_in`，但它应当与总迭代预算一起调整，而不是固定使用某个绝对 `dual_init`。

## 7. 理论边界

这些改进显著增强了尺度稳健性、有限轨迹舍入保证和实际质量，但仍没有给 plain PDBO 一个一般图上强于 $1/2$ 的统一事前近似比。当前严格新增的是：

1. 正确落到代码的随机初始 archive $1/2$ 基线；
2. 初始 PSD/正 multiplier 的谱尺度条件；
3. 条件期望输出不差于全轨迹最佳 Bernoulli 期望；
4. one-flip 单调精修及其非负 Max-Cut $1/2$ 终点保证；
5. 任意输出仍可使用 diagonal-shift upper bound 做实例级后验认证。

此外，当前大规模最小特征值由普通 `eigsh` 数值计算。它适合参数尺度估计，但若论文声称机器可验证的严格 upper bound，仍需经过验证的特征值区间、稀疏 LDL/inertia 或保守 Gershgorin 安全模式；不能把普通 Ritz residual 自动称为严格证书。
