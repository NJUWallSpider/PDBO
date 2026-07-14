# PDBO 实验效果的理论解释：严格审计、质量证书、反例与谱动力学

日期：2026-07-11
研究对象：Liu et al., *Smoothing Binary Optimization: A Primal-Dual Perspective*, arXiv:2509.21064v2，以及当前 PDBO 仓库实现。

本文使用以下标签区分证据强度：**[定理]**、**[已证命题]**、**[条件结果]**、**[反例]**、**[猜想]**、**[启发解释]**、**[实验观察]**。数值计算仅用于核对恒等式、枚举紧例和复现反例；所有核心结论均另给数学证明。

---

## 结论摘要与最强结果

1. **论文算法、论文证明所分析的递推和仓库默认实现不是同一个算法。** 论文 Eq. (6)/Algorithm 1 是固定步长、用旧状态同步更新的 projected GDA，并带确定性的中心扰动；仓库默认使用 RMSProp、先更新并投影 $x$，再用新 $x$ 更新 $y$，且论文中的扰动默认不存在。仓库返回的还是全轨迹、全 batch 的历史最佳取整解，而不是最终连续迭代点。因此，论文的 binarity 结论不能直接覆盖当前代码输出，最终 $y$ 也不能与历史 incumbent 的 flip gain 配对解释。

2. **[主定理 A：PDBO 的 dual-slack one-flip 证书]** 对任意 $F:\{0,1\}^n\to\mathbb R$ 及其 multilinear extension $f$、任意二进制点 $z$ 和 $g(s)=s^2-s$，记

   $$
   D_i(z):=F(z\oplus e_i)-F(z).
   $$

   若 $(z,y)$ 是论文 projected primal map 的固定点，则精确有

   $$
   \boxed{D_i(z)\ge y_i\quad(\forall i).}
   $$

   因而 $y_i\ge0$ 足以推出 one-flip local minimum；若 $y_i<0$，则负 dual 只给出允许违反的 slack，只有在额外严格条件下才形成正势垒，不能直接推出离散局部最优。对 Max-Cut，若 $\Delta_i$ 是单翻 cut 增益，则

   $$
   \boxed{\Delta_i(z)\le -y_i.}
   $$

3. **[主定理 B：非负权 Max-Cut 的标准紧质量界及 PDBO specialization]** 若边权非负、总边权为 $M$，且 $\Delta_i(z)\le\varepsilon_i$，则

   $$
   \boxed{C(z)\ge \frac{M}{2}-\frac14\sum_i\varepsilon_i
   \ge \frac{\mathrm{OPT}}2-\frac14\sum_i\varepsilon_i.}
   $$

   exact one-flip local optimum 因而是 $1/2$-approximation。常数 $1/2$ 在四环 $C_4$ 上严格紧；即使要求 strict one-flip local optimum，统一常数仍不能提高。对用户报告中**经逐实例核查为非负权**的算例，浅违反只带来式 (17) 的小加性修正；signed Gset 不适用该界。无论如何，$1/2$ 本身远不足以解释接近 best-known 的结果。

4. **[最强负面结论]** 不存在一般 QUBO 的非平凡质量保证：严格 one-flip local minimum 也可比全局最优任意差。对 signed Max-Cut，严格 one-flip local optimum 的近似比甚至可为 0。更强的是，论文 Algorithm 1 在单边图 $K_2$ 上，从精确凸阈值以上的 dual 初始化和完全对称的 $x^0=(1/2,1/2)$ 出发，确定性中心扰动会把轨迹送到 cut $0$，而最优值为 1。因此“凸初始化”本身不蕴含任何统一正近似比。

5. **[主定理 C：中心轨道的瞬时谱收缩阈值]** 对满足正文小步长条件 (26)、且扰动前可用谱窗口非空的 Max-Cut plain-GDA，令 $x^t=\frac12\mathbf1+z^t$。在投影和扰动未激活时，中心轨道

   $$
   z^t=0,\qquad y^t=\bar y_t\mathbf1,\qquad
   \bar y_t=\bar y_0-\frac{\beta t}{4}
   $$

   是精确轨道。其第 $k$ 个 $W$-特征模态满足线性变分

   $$
   a_k^{t+1}=\bigl[1-2\alpha(\lambda_k(W)+\bar y_t)\bigr]a_k^t.
   $$

   因而 $\bar y=-\lambda_{\min}(W)$ 不只是凸性阈值，也是瞬时 primal 线性映射首次失去收缩性的精确阈值；随后最小特征空间相对其他模态优先放大。固定 $\alpha$ 时，减小 $\beta$ 会增加轨迹在可用谱窗口内的离散步数；在小 $\alpha$ 展开下，累计相对选择强度呈 $\alpha/\beta$ 尺度。这给出了“convex initialization $\to$ early spectral symmetry breaking”的可证局部机制，但没有把该阶段与后续 binary recovery 接起来，也不自动给出目标值保证。

6. **[启发解释/待检验] 对 64/71 one-flip 现象，一个可证伪的候选联合解释是：** 随机多启动和谱型早期分盆；当严格 dual-slack 条件成立时，负 dual 势垒可能锁定浅违反坐标；历史 best-of-rounded archive 可能覆盖了许多候选及其邻域。当前没有 matched-history/coverage 数据，不能把这三项排序为已证因果。其余 7 例的 gain 全为 1，首先反映整数权下“最小正格点”的量化，而非已证明的动力学常数。

7. **研究价值判断。** dual-slack 证书与瞬时谱收缩/模态选择定理足以形成一段有实质内容的新增理论；若配套实验验证，它们可成为论文的重要理论解释。它们尚不足以构成“PDBO 对一般 QUBO 有质量保证”的核心定理。要形成核心理论贡献，还需在规则图、近二分图、随机图或 planted model 上，把首次谱模态、非线性饱和、舍入与最终 cut 质量真正连接起来。

---

## 1. 算法与现有理论的严格审计

### 1.1 三个必须分开的算法对象

#### 论文 Eq. (6)：同步 projected GDA

论文 Sec. 3.2.1, Eq. (6), PDF pp. 4--5 写的是

$$
\begin{aligned}
x^{t+1}&=\Pi_{[0,1]^n}\bigl(x^t-\alpha\nabla_xL(x^t,y^t)\bigr),\\
y^{t+1}&=y^t+\beta g(x^t),
\end{aligned}
\tag{P-GDA}
$$

其中 $g(x^t)=(g(x_1^t),\ldots,g(x_n^t))$。两式都使用旧状态 $(x^t,y^t)$。

#### 论文 Algorithm 1：带中心扰动的版本

Algorithm 1 的 Require 是 $x^0\in[0,1]^n$、$y^0\in\mathbb R_{++}^n$、$\alpha,\beta>0$、$\delta>0$ 和最大迭代数 $T_{\max}$。

对每个坐标，若

$$
|x_i^t-1/2|\le\delta,\qquad
|\partial_iL(x^t,y^t)|\le2\delta,\qquad
y_i^t\le0,
$$

则用

$$
x_i^{t+1}=\begin{cases}
1/2-\delta,&x_i^t\le1/2,\\
1/2+\delta,&x_i^t>1/2
\end{cases}
\tag{P-perturb}
$$

代替普通 primal 步；dual 仍用旧 $x_i^t$ 更新。后续关于 Proposition 3.3 的证明依赖这一具体触发规则，而不是任意“加点噪声”。

#### 当前仓库默认实现

`solver_jax.py:324--343` 实际执行：

1. 在 $(x^t,y^t)$ 上计算 raw primal gradient；
2. 用带动量的 RMSProp/Adam 产生自适应更新；
3. clip 得到 $x^{t+1}$；
4. 用 **新** $x^{t+1}$ 更新

   $$
   y^{t+1}=y^t+\beta g(x^{t+1});
   $$
5. 每步 round 全部 batch，并与旧 incumbent 比较，保留历史目标最优者。

`main.py:115--125` 将优化器硬编码为 RMSProp。虽然代码创建了 dual optimizer/state，`solver_jax.py:258--261,327--343` 从未用它更新 dual。故这是“自适应 primal + 原始 dual 累加”的 Gauss--Seidel 型工程算法，不是 Eq. (6)。

代码中的可选扰动 `solver_jax.py:345--352,382--394` 也完全不同：它由 incumbent 停滞触发，随机选取 $|x_i-1/2|<0.45$ 的部分坐标，随机踢到 $0.5\pm\text{strength}$，不检查 raw gradient 或 $y_i\le0$，而且默认关闭（`main.py:54`，README 的 Options 部分）。用户给出的 Gset 参数未启用该选项，因此 64/71 现象不能归因于论文中心扰动。

### 1.2 输出规则的决定性差异

仓库返回

$$
z^{\mathrm{inc}}
=\arg\min\{F(\operatorname{round}(x^{t,b})):
t\le T,\ b\le B\},
$$

并始终把旧 incumbent 放回候选集合（`solver_jax.py:334--343`）。若启用 `rounding_samples`，还会对最终 relaxed batch 做额外 Bernoulli sampling（`solver_jax.py:402--423`）。

因此：

- 返回点天然是 binary，哪怕连续迭代从未接近 binary；
- 返回点不一定等于任何最终 batch 的 round；
- 代码没有保存 incumbent 诞生的时间、batch、当时的 $x,y$ 或优化器状态；
- 用户报告中的 final `y_mean` 和 `y_closest_batch` 与 incumbent 没有同步 KKT 或固定点关系。

这也解释了为什么 `gain` 与 final $|y|$ 的 Spearman 相关接近 0 并不构成对 dual 机制的反证：理论给的是单侧包络，而且现有 $y$ 根本没有与输出配对。

### 1.3 论文已经证明、只给直觉、尚未证明的内容

| 层次 | 内容 | 审计结论 |
|---|---|---|
| 已有等价性 | Proposition 3.1 / Theorem 3.2 的 minimax 值等价 | 对真正 multilinear $f$ 成立；此特殊问题可直接以 $(x^\star,0)$ 构造 saddle |
| 已有可行性 | Proposition 3.3、Corollaries 3.4--3.5、Theorem 3.7 | 只针对论文固定步长旧状态递推；证明还有下述缺口；不覆盖仓库默认实现和 archive 输出 |
| 论文直觉 | convex initialization、gradual nonconvexification、binary recovery | Hessian 阈值系数正确；“single basin”和质量因果关系没有建立 |
| 未有质量理论 | one-flip、$k$-flip、近似比、global gap、basin selection | 本报告的主要填补对象 |

### 1.4 关键逻辑与假设问题

#### (a) $g(x_i^t)\to0$ 不推出 $x_i^t$ 收敛到固定 bit

Corollary 3.5 最多推出

$$
\operatorname{dist}(x_i^t,\{0,1\})\to0.
$$

**[最小反例]** 取 $n=1,f\equiv0,g(x)=x^2-x,x^0=0$，并令 $y^0>1/\alpha$。因为端点上 $g=0$，dual 恒定；普通 projected step 满足

$$
0\mapsto1\mapsto0\mapsto1\mapsto\cdots.
$$

于是 $g(x^t)\equiv0$、梯度有界，但 $x^t$ 不收敛。论文“取充分小 $\alpha$ 可避免振荡”没有给出量化条件。一个可修复版本需要显式控制单步位移，例如证明扰动最终关闭且

$$
\sup_t\alpha\|\nabla_xL(x^t,y^t)\|_\infty<1,
$$

再结合端点邻域最终分离来排除两端切换。对 entropy $g$，端点梯度奇异，甚至上述 uniform bound 也不能直接使用。

#### (b) Proposition 3.3 的证明存在未闭合步骤

- Algorithm 1 只写 $\delta>0$，但 $g'(1/2-\delta)$ 和证明区间要求 $0<\delta<1/2$。
- Appendix A.4 声称扰动“防止 $x_i^t$ 在中心区连续两步停滞”，但这与伪代码本身不符：若 $x_i^t=1/2-\delta$ 且触发条件成立，分支仍把它设为同一个 $1/2-\delta$；本报告的 $K_2$ 轨迹正会连续多步如此。若梯度条件不成立而普通步很小，坐标也可在中心区停留多步。由此得到的 dual 下界中“至多两个 $\beta g(1/2)$ 降幅”没有被现有论证支持。
- 证明到达端点后把 $x_i,y_i$ 视为不再变化，需要端点 $g'$ 存在且有限。论文允许的 entropy

  $$
  g(x)=x\log x+(1-x)\log(1-x)
  $$

  在 $0,1$ 的导数发散，Algorithm 1 的端点 gradient 未定义。

本报告没有宣称 Proposition 3.3 本身已被反例否定；结论是当前证明不足，且不能作为后续质量定理的无条件前提。

#### (c) Theorem 3.7 是 hitting/existence bound，不是末迭代或代码返回点保证

Appendix A.6 的 telescoping 实际给出

$$
\frac1T\sum_{t<T}\sum_i[-g(x_i^t)]
\le\frac{\|y^0-y^\star\|_1}{\beta T}.
$$

因此在相应 $T$ 前至少存在一个 $\epsilon$-binary iterate；若算法逐步监控，可解释成 first-hitting bound。它不直接说明 $x^T$ 或历史 rounded incumbent 是该点。

#### (d) strong max--min equality 不自动“使 simultaneous GDA 收敛”

Remark 2.6 把有限值的 strong max--min equality 一般性地表述为 saddle existence，缺少 attainment 条件。不过在本问题、$f$ 真正 multilinear 时可直接修复：取任意全局二进制最优 $x^\star$，由 Lemma 2.3 它也是盒上 $f$ 的全局最优，故 $(x^\star,0)$ 是 saddle。即便如此，strong duality 也不构成 simultaneous GDA 在非凸--线性系统中的收敛定理。

#### (e) landscape Hessian 系数正确，但叙述过强

对论文 Max-Cut 规范

$$
f(x)=x^\top Wx-\mathbf1^\top Wx,
$$

$W=W^\top$、对角为零，确有

$$
\nabla_x^2L=2(W+\operatorname{Diag}y),
$$

所以 $\bar y\ge-\lambda_{\min}(W)$ 的系数正确。等号只保证 PSD，不保证唯一 minimizer 或真正的“single basin”；严格大于阈值才统一给出强凸。

#### (f) 当前 API 并不保证使用 multilinear extension

README 把 solver 描述为一般 $x^\top Qx+c^\top x$。若 $Q$ 含对角项，连续目标含 $Q_{ii}x_i^2$，而正确 multilinear extension 应把二进制恒等式 $x_i^2=x_i$ 折入线性项。`custom` 路径同样没有检查 minimum-preserving/multilinear 性。此时 Lemma 2.3 和 Theorem 3.2 的证明不能原样套用。

#### (g) MaxSAT 的论文与代码也不一致

论文 Sec. 4.3.2 声称 Max-k-SAT 使用 entropy $g$；当前 `MAXSAT_JAX` 仍使用 $x^2-x$（`solver_jax.py:721--737`）。

#### (h) 核心 PDBO 没有 one-flip 检查

唯一显式离散局部搜索在 `pdbo/refinement.py:38--68`，并由 `--refine` 可选启用；默认关闭，且 refined solution 与 solver incumbent 分开报告。因此用户观察到的 one-flip 性不是后处理的定义性结果。

#### (i) “逐坐标凸扩展对应 SDP”和 Appendix D.5 dual certificate 尚不可复核

对任意固定 $y$，真正的 Lagrangian dual 值是

$$
d(y):=\min_{x\in[0,1]^n}L(x,y)\le\mathrm{OPT}.
$$

当 $W+\operatorname{Diag}y\succeq0$ 时，该盒约束二次子问题是凸的，若能求得其**全局最小值并控制求解误差**，才得到有效 lower bound。PDBO 的单个 primal iterate 值 $L(x^t,y)$ 只满足 $L(x^t,y)\ge d(y)$，与 $\mathrm{OPT}$ 没有确定的上下界关系，不能单独称为 dual certificate。

论文 Sec. 4.5 关于“逐坐标 diagonal shift 的最紧凸扩展对应 SDP relaxation”的说法，需要在规范 multilinear QUBO、精确求解盒上凸子问题、再对可行 diagonal shifts 最大化的框架下给出等价推导；一条 PDBO 轨迹或一次 gradient step 本身不建立该等价。Appendix D.5 Table 16 没有说明 convex subproblem 的求解器、容差及 bound 方向验证，当前仓库也没有 dual-bound 实现。因此表中数值在现有材料下不可复现，后续论文必须补充计算流程和证书精度。

---

## 2. 候选理论结果地图

评分 1--5：越高表示越强、越可能、越新颖、假设越自然；“难度”越高越难。

| 排名 | 候选结论 | 解释力度 | 成立可能 | 新颖性 | 假设自然 | 难度 | 当前状态 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | binary fixed/near-fixed state 的 dual-slack one-flip 证书；Max-Cut 加性 $1/2$ 界 | 4 | 5 | 4 | 5 | 2 | **本报告已证** |
| 2 | 中心轨道在 $-\lambda_{\min}(W)$ 瞬时失去收缩性，最小特征空间获得相对模态放大 | 4 | 5 | 4 | 5 | 3 | **本报告已证小步长局部线性版本** |
| 3 | PDBO + greedy FLIP 对非负权 Max-Cut 的 exact one-flip 与 $1/2$ 保证 | 3 | 5 | 2 | 5 | 1 | **已证，但属于标准后处理保证** |
| 4 | archive 的邻居覆盖率推出 incumbent one-flip；多 batch 提升覆盖 | 3 | 5 | 3 | 5 | 1 | **条件结果**，需记录历史候选 |
| 5 | 小 $\beta/\alpha$ 下跟踪强凸 minimizer path，直到首个 Hessian 奇异点 | 3 | 4 | 3 | 4 | 3 | **条件结果**；不能跨 bifurcation |
| 6 | 逐坐标 dual 比单一 penalty 的优势可由 diagonal-shift dual bound/适应性说明 | 3 | 4 | 3 | 4 | 3 | 静态 bound 可做；尚未连接 primal 质量 |
| 7 | 规则、近二分或 planted 图上，最小特征模态 + rounding 给出质量保证 | 5 | 3 | 5 | 3 | 5 | 最值得继续的结构化方向 |
| 8 | 随机初始化/多启动的 basin success probability | 5 | 2 | 5 | 4 | 5 | 需 basin 测度或随机图模型 |
| 9 | PDBO binary limit 自动 one-flip 或 $k$-flip | 5 | 1 | 4 | 5 | 2 | **一般命题已被反例否定** |
| 10 | 一般 QUBO 的统一 approximation ratio/global gap | 5 | 1 | 5 | 1 | 2 | **不可能；见反例** |

优先级判断：最稳健的论文增量是“dual-slack 局部质量定理 + 瞬时谱收缩/模态选择机制 + 反例边界”；最有潜力成为核心贡献的是第 7 项，但它需要新增结构假设和实质证明。

---

## 3. 主结果 I：dual-slack 与离散 one-flip 的精确关系

### 3.1 定义

令 $F:\{0,1\}^n\to\mathbb R$，$f$ 为其 multilinear extension。对 $z\in\{0,1\}^n$，定义单翻差

$$
D_i(z):=F(z\oplus e_i)-F(z).
$$

这是 minimization 记号：$D_i(z)<0$ 表示翻转 $i$ 能改进。令

$$
s_i:=2z_i-1\in\{-1,1\},\qquad
q_i(z,y):=\partial_iL(z,y).
$$

定义 one-sided primal residual

$$
r_i(z,y):=[s_iq_i(z,y)]_+.
$$

在边界点，projected gradient 固定条件恰为 $s_iq_i\le0$，即 $r_i=0$。这一定义包含等号情形。

### 3.2 主定理 A

**[定理 1：binary point 的精确 dual-slack 恒等式与质量证书]**
设 $f$ 是 $F$ 的 multilinear extension，$g(u)=u^2-u$，

$$
L(x,y)=f(x)+\sum_i y_i(x_i^2-x_i).
$$

则对任意 $z\in\{0,1\}^n$、$y\in\mathbb R^n$：

1. 精确恒等式

   $$
   \boxed{D_i(z)=y_i-s_iq_i(z,y).}
   \tag{7}
   $$

2. 单翻改进量 $I_i(z):=[-D_i(z)]_+$ 精确满足

   $$
   I_i(z)=[s_iq_i(z,y)-y_i]_+,
   $$

   并有上界

   $$
   \boxed{I_i(z)\le[r_i(z,y)-y_i]_+.}
   \tag{8}
   $$

3. 若 $\alpha>0$ 且 $(z,y)$ 是 projected primal map

   $$
   x\mapsto\Pi_{[0,1]^n}(x-\alpha\nabla_xL(x,y))
   $$

   的固定点，则 $r_i=0$，从而

   $$
   \boxed{D_i(z)\ge y_i,\qquad I_i(z)\le[-y_i]_+.}
   \tag{9}
   $$

   特别地，$y\ge0$ 足以推出 $z$ 是 one-flip local minimum。

4. 若论文 Algorithm 1 满足 $0<\delta<1/2$、$x^t\to z\in\{0,1\}^n$、$y^t\to y^\star$，则扰动最终关闭，且极限满足 (9)。另外

   $$
   y_i^\star
   =y_i^0-\beta\sum_{t=0}^{\infty}x_i^t(1-x_i^t).
   \tag{10}
   $$

   因而任何 one-flip 违反 $I_i(z)>0$ 都要求累计 fractionality exposure 满足

   $$
   \sum_t x_i^t(1-x_i^t)>\frac{y_i^0}{\beta},
   $$

   并有

   $$
   I_i(z)\le
   \left[\beta\sum_t x_i^t(1-x_i^t)-y_i^0\right]_+.
   \tag{11}
   $$

#### 证明

由于 $f$ 对第 $i$ 坐标仿射，固定 $z_{-i}$ 后

$$
\partial_if(z)=f(1,z_{-i})-f(0,z_{-i}).
$$

若 $z_i=0$，则 $D_i(z)=\partial_if(z)$；若 $z_i=1$，则 $D_i(z)=-\partial_if(z)$。统一写成

$$
D_i(z)=-s_i\partial_if(z).
\tag{12}
$$

又因为 $g'(z_i)=2z_i-1=s_i$，

$$
q_i(z,y)=\partial_if(z)+y_is_i.
$$

两边乘 $s_i$，使用 $s_i^2=1$ 与 (12)：

$$
s_iq_i=-D_i+y_i,
$$

即得 (7)。由 $s_iq_i\le r_i$ 得 $D_i\ge y_i-r_i$，取负部即得 (8)。

若 $z$ 为 projected fixed point：当 $z_i=0$ 时，固定条件等价于 $q_i\ge0$；当 $z_i=1$ 时，等价于 $q_i\le0$。二者统一为 $s_iq_i\le0$，故 $r_i=0$，得到 (9)。投影边界和等号均已包含。

最后，若 $x^t\to z$ 且 $0<\delta<1/2$，则对充分大 $t$，每个 $x_i^t$ 都位于 0 或 1 的小邻域，因而不满足中心扰动的 $|x_i^t-1/2|\le\delta$。普通更新和投影连续，取极限得到 projected fixed condition。dual 递推写成

$$
y_i^{t+1}=y_i^t-\beta x_i^t(1-x_i^t),
$$

telescoping 并令 $t\to\infty$ 得 (10)；将 (9) 代入即得 (11)。证毕。

### 3.3 负 dual 是“势垒”，不是 one-flip 证书

令 $y_i=-\rho_i<0$，固定其余坐标在二进制 $z_{-i}$，沿从 $z_i$ 到其 flip 的线段参数化 $t\in[0,1]$。由 multilinearity，

$$
L(z+t(1-2z_i)e_i,y)-L(z,y)
=tD_i(z)+\rho_i t(1-t).
\tag{13}
$$

**[已证命题：坐标势垒]** 若 $D_i(z)=-\gamma_i<0$ 且 $0<\gamma_i<\rho_i$，则 $z_i$ 虽然不是离散局部最优，却是该连续坐标剖面的严格单侧局部极小端点；势垒最高点位于

$$
t_i^\star=\frac{\rho_i-\gamma_i}{2\rho_i},
$$

相对 $z$ 的势垒高度为

$$
\frac{(\rho_i-\gamma_i)^2}{4\rho_i}.
$$

边界情形 $\gamma_i=\rho_i$ 仍满足一阶 projected fixed-point 等式，但剖面增量为 $-\rho_i t^2<0$（$t>0$），所以它不是连续局部极小点。这一区分再次说明一阶固定性弱于离散或连续局部最优性。

这给出一个比“penalty 强迫 binary”更精确的条件性解释：负 dual 只有在上述 binary 端点同时满足严格 dual-slack（这里即 $0<\gamma_i<\rho_i$）时才形成正势垒并可产生鲁棒锁定；$\gamma_i=\rho_i$ 无势垒，$\gamma_i>\rho_i$ 则沿内侧立即下降。连续 stationarity 与 one-flip optimality 因而不能混同。

### 3.4 对当前 $y_i$ 统计的直接结论

- 理论预测是单侧 envelope $\Delta_i+y_i\le0$，不是 rank monotonicity；很负的 $y_i$ 只是宽松上界，可以对应 gain (0) 或负 gain。
- 当前 final $y$ 未与 incumbent 配对，因此连该 envelope 也不应直接检查。
- 最小有效记录应是：incumbent 首次产生的 $t,b$、对应 relaxed $x^{t,b}$、$y^{t,b}$、raw gradient、优化器方向及累计 $x_i(1-x_i)$。

---

## 4. Max-Cut / QUBO 专门化与全局质量

### 4.1 QUBO 的精确单翻公式

对规范 multilinear QUBO

$$
F(z)=\kappa+\sum_i a_iz_i+\sum_{i<j}b_{ij}z_iz_j,
$$

有

$$
\partial_if(x)=a_i+\sum_{j\ne i}b_{ij}x_j,
\qquad
\nabla^2_{ij}f=b_{ij}\ (i\ne j),
\qquad
\nabla^2_{ii}f=0,
$$

以及

$$
D_i(z)=(1-2z_i)\partial_if(z).
$$

因此

$$
z\text{ 是 one-flip local minimum}
\iff D_i(z)\ge0\quad\forall i.
$$

### 4.2 Max-Cut 的梯度、Hessian 与 flip gain

令 $W=W^\top$ 为零对角加权邻接矩阵，$d=W\mathbf1$。当前仓库 `problem_parser.py:108--125` 使用

$$
f(x)=x^\top Wx-d^\top x,
$$

它在 binary 点上等于负 cut 值：$f(z)=-C(z)$。于是

$$
\nabla f(x)=2Wx-d,
\qquad
\nabla^2f=2W,
\qquad
\nabla_x^2L=2(W+\operatorname{Diag}y).
$$

定义顶点 $i$ 当前 incident cut weight

$$
c_i(z)=\sum_jw_{ij}\mathbf1\{z_i\ne z_j\},
$$

单翻 cut 增益为

$$
\Delta_i(z)
:=C(z\oplus e_i)-C(z)
=d_i-2c_i(z)
=(2z_i-1)\partial_if(z).
\tag{14}
$$

所以

$$
z\text{ 是 one-flip local max-cut}
\iff\Delta_i(z)\le0\ \forall i
\iff c_i(z)\ge d_i/2\ \forall i.
\tag{15}
$$

最后一个条件在 signed 权下仍是代数等价，但不再有非负权近似解释。

### 4.3 主定理 B：标准 Max-Cut 质量恒等式与 PDBO dual-slack specialization

**[定理 2]** 设 $w_e\ge0$，$M=\sum_ew_e$，$z\in\{0,1\}^n$。对任意 $y$，令

$$
r_i=[(2z_i-1)\partial_iL(z,y)]_+.
$$

则：

1.
   $$
   \Delta_i(z)\le r_i-y_i.
   \tag{16}
   $$

2. 令 $\varepsilon_i=[r_i-y_i]_+$，则

   $$
   \boxed{C(z)\ge\frac M2-\frac14\sum_i\varepsilon_i
   \ge\frac{\mathrm{OPT}}2-\frac14\sum_i\varepsilon_i.}
   \tag{17}
   $$

3. 还有更直接的 signed-residual 下界

   $$
   \boxed{C(z)\ge\frac M2+\frac14\sum_i(y_i-r_i).}
   \tag{18}
   $$

4. 若 $\alpha>0$ 且 $(z,y)$ 是论文 projected primal fixed point，则 $r=0$，

   $$
   \Delta_i(z)\le-y_i,
   \qquad
   C(z)\ge\frac M2+\frac14\sum_i y_i.
   \tag{19}
   $$

   若进一步 $y\ge0$，则 $z$ 是 one-flip local optimum，并满足 $C(z)\ge\mathrm{OPT}/2$。

#### 证明

由定理 1，Max-Cut 中 $\Delta_i=-D_i=s_iq_i-y_i\le r_i-y_i$，得 (16)。另一方面，对任意 cut 恒有

$$
\sum_i\Delta_i(z)=2M-4C(z).
\tag{20}
$$

这是因为每条未 cut 边在其两个端点的 flip gain 中各贡献 $+w_e$，每条已 cut 边各贡献 $-w_e$。由 $\Delta_i\le\varepsilon_i$ 和 (20) 得 (17)；由 $\Delta_i\le r_i-y_i$ 直接求和得 (18)。fixed-point 情形使用 $r=0$。证毕。

其中 (20) 及 exact one-flip 的 $1/2$ 结论属于标准 Max-Cut 局部搜索代数；本报告与 PDBO 直接相关的新增部分是把可选的 $\varepsilon_i$ 具体化为 matched dual/residual 证书 $[r_i-y_i]_+$，并在 projected fixed point 下得到 (19)。

### 4.4 紧性与边界

**[紧例 1：$1/2$]** 无权四环 $C_4$，取相邻两点在同一侧的 cut (0011)。cut 值为 2，每个顶点恰有一条 crossing 和一条 noncrossing edge，故所有 $\Delta_i=0$；最优值为 4。one-flip 的 $1/2$ 因而紧。

**[紧例 2：strict local 仍趋近 $1/2$]** 取 $\eta>0$，把上述当前 cut 的两条边赋权 $1+\eta$，未 cut 的两条边赋权 1，则每点 $\Delta_i=-\eta<0$，但当 $\eta\downarrow0$ 时

$$
\frac{C(z)}{\mathrm{OPT}}
=\frac{2+2\eta}{4+2\eta}\to\frac12.
$$

**[加性常数 $1/4$ 紧]** 取 $0<\eta<1$，把当前 cut 边赋权 $1-\eta$，未 cut 边赋权 1，并在定理 2 中取 $y=0$。此时 $r_i=\Delta_i=\eta$、$\varepsilon_i=\eta$，且 (17) 取等号。

### 4.5 对用户 Gset one-flip 报告的严格解读

本节把用户给出的 71-instance 汇总视为 **[实验观察]**。当前仓库只含 G1，且没有 `summary.csv` / `per_vertex.csv`，因此本报告无法独立重跑或逐实例核验全部权重符号与 24 个违反点。

若某个实例边权非负，令 $r$ 为违反顶点数、$\gamma$ 为最大正 gain，则取 $\varepsilon_i=[\Delta_i]_+$ 得

$$
C(z)\ge\frac{\mathrm{OPT}}2-\frac{r\gamma}{4}.
\tag{21}
$$

对其中**逐实例核实为非负权**的算例，式 (21) 说明违反点少且 gain 为 1 只造成很小的加性修正。对 signed 实例，这些数字只能称为数值上的浅违反，不能解释成相对 $\mathrm{OPT}/2$ 的理论损失。还需注意：

- 该式必须逐实例应用，不能把 71 个实例的 $r$ 合并成一个优化问题；
- Gset 含 signed-weight 系列，非负权假设必须逐文件核查；
- 即使 exact one-flip，最坏也只有紧的 $1/2$，不能解释接近 best-known；
- 整数权图上正 flip gain 是整数。所有违反恰为 1 首先说明它们已经落在最小正格点。若未来能由 matched dual/residual 证明 $\Delta_i<2$，才可把“gain 必为 1”升级成动力学结论。

### 4.6 可选 FLIP 后处理的正式保证

**[命题]** 对任意初始 binary cut，反复接受严格改进的单点翻转，有限步后必到达 one-flip local optimum。对非负整数权，若每个正改进至少为 1，则接受翻转数不超过 $M-C(z^0)$，最终 cut 至少为 $\mathrm{OPT}/2$。

这与 `pdbo/refinement.py` 的严格下降逻辑一致，但它是“PDBO + local search”的保证，不是默认 PDBO 的保证。一般二进制编码大整数权下，该过程只是伪多项式。Schäffer--Yannakakis (1991) 的 weighted MAX-CUT/FLIP 结果以二进制编码整数边权为输入、以 cut 为解、以“移动一个顶点”为邻域，证明寻找该邻域局部最优是 PLS-complete；这正是不能把“有限终止”自动升级为输入位长多项式保证的复杂性边界。单位或小整数权时，目标值范围较小，简单严格改进仍可有伪多项式甚至按 $M$ 计的界。

---

## 5. 不可能性结果与最小反例

### 5.1 一般 QUBO：strict one-flip 也可任意差

**[反例]** 对 $M>1$，令

$$
F(x_1,x_2)=M+x_1+x_2-(M+1)x_1x_2.
$$

则

$$
F(00)=M,\quad F(10)=F(01)=M+1,\quad F(11)=1.
$$

所以 (00) 是 strict one-flip local minimum，但与全局最优的比值为 $M$，可任意大；两点联合翻转立即到全局最优。这同时否定了从 one-flip 推 $k$-flip 或全局质量的一般命题。

该反例不是单纯依赖系数尺度。把整个目标除以 $M+1$ 后，常数、线性和二次系数的绝对值均不超过 1；$00$、单翻点和 $11$ 的值分别为 $M/(M+1),1,1/(M+1)$，局部/全局比值仍为 $M$，归一化加性 gap 还趋于 1。

### 5.2 Signed Max-Cut：strict one-flip 的比值可为 0

取四点图，

$$
w_{13}=w_{14}=w_{23}=w_{24}=1,
\qquad
w_{12}=w_{34}=-(2+\eta),\quad\eta>0.
$$

全零 cut 值为 0；翻转任一点会切开两条 (+1) 边和一条 $-(2+\eta)$ 边，gain 为 $-\eta<0$，故它是 strict one-flip local optimum。cut $\{1,2\}\mid\{3,4\}$ 切开全部四条正边并避开两条负边，值为 4。

为严格验证全局最优，令 $a=\mathbf1\{z_1\ne z_2\}$、$b=\mathbf1\{z_3\ne z_4\}$。若 $a=b=0$，总 cut 至多为四条正边总和 4，且上述分割取等；若 $a+b=1$，正边贡献为 2、恰切开一条负边，总值为 $-\eta$；若 $a=b=1$，四条正边中恰有两条被切开、两条负边均被切开，总值为 $2-2(2+\eta)=-2-2\eta$。故 $\mathrm{OPT}=4$，不存在正 approximation ratio。

### 5.3 任意 binary 点都可嵌入论文 projected-GDA fixed point

由定理 1，对任意 $\alpha>0$，$(z,y)$ 为 binary projected primal fixed point 当且仅当

$$
y_i\le D_i(z)\qquad\forall i.
\tag{22}
$$

给定任意 $z$，只需把每个 $y_i$ 取到足够负即可满足 (22)。若再取 $0<\delta<1/2$，binary 点不触发 Algorithm 1 的中心扰动，而 $g(z_i)=0$ 使 dual 不变，所以它也是论文 full-state Algorithm 1 的固定点。该结论不自动覆盖带非零 optimizer momentum 的仓库 RMSProp 状态。

因此“论文 PDBO 收敛到 binary fixed point”本身完全没有质量含义；真正有信息的是 dual 是否仍非负、one-sided residual 是否小，以及输出是否与该状态配对。

前两节的坏局部点甚至可以带**严格正 dual**：两变量 QUBO 的 $00$ 有 $D_i=1$，可取任意 $0<y_i\le1$；signed Max-Cut 全零点在 minimization 形式下有 $D_i=\eta$，可取任意 $0<y_i\le\eta$。从这些满足 Algorithm 1 Require 的正 dual 状态出发，primal 和 dual 都永久固定，但它们分别具有任意差的 QUBO ratio 和 signed Max-Cut ratio 0。这说明“positive dual $\Rightarrow$ one-flip”是正确的局部证书，但在一般 QUBO/signed Max-Cut 上仍不蕴含全局质量。

### 5.4 主反例：凸初始化也可能得到零质量 Max-Cut

**[定理 3：论文 Algorithm 1 的 $K_2$ 坏轨迹]**
考虑一条单位权边，

$$
W=\begin{pmatrix}0&1\\1&0\end{pmatrix},
\qquad
f(x)=2x_1x_2-x_1-x_2.
$$

取任意 $\alpha,\beta>0$、$0<\delta<1/2$，

$$
x^0=(1/2,1/2),\qquad y^0=\bar y_0\mathbf1,\qquad\bar y_0\ge1.
$$

因为 $\lambda_{\min}(W)=-1$，该 dual 初始化满足论文的凸性阈值。若无限运行论文 Algorithm 1，则轨迹最终在有限步到达 $x=(0,0)$ 并永久固定，cut 值为 0，而 $\mathrm{OPT}=1$。整个轨迹始终满足 $x_1^t=x_2^t$，故任何对两坐标使用同一阈值的 deterministic rounding 在所有时刻只能产生 $00$ 或 $11$，历史 rounded archive 也始终是零 cut。

#### 证明

对称性保证始终有 $x_1^t=x_2^t=:u_t$、$y_1^t=y_2^t=:v_t$。在 $u_t=1/2$ 时，

$$
\partial_iL=(1+v_t)(2u_t-1)=0,
$$

故只要 $v_t>0$，primal 保持在中心，而

$$
v_{t+1}=v_t-\beta/4.
$$

有限步后 $v_t\le0$，中心扰动把两个坐标同时置为 $u_{t+1}=1/2-\delta$。当 $u=1/2-\delta$ 且 $v\in[-2,0]$ 时，

$$
|\partial_iL|=2\delta|1+v|\le2\delta,
$$

所以扰动分支反复把 $u$ 保持在 $1/2-\delta$，而 $v$ 每步严格下降

$$
\beta[(1/2-\delta)^2-(1/2-\delta)]
=-\beta(1/4-\delta^2).
$$

若首次中心 kick 后已经有 $v<-2$，则直接进入下一阶段；否则上述严格下降保证有限步后 $v<-2$。此后对任意 $u\le1/2-\delta$，

$$
\partial_iL=(1+v)(2u-1)\ge2\delta,
$$

普通 projected step 使 $u$ 每步至少下降 $2\alpha\delta$，有限步到达 0。端点处 $g(0)=0$，且 $\partial_iL=-(1+v)>0$，故投影保持 $u=0$、dual 保持不变。证毕。

该反例分析的是论文的确定性扰动，不是仓库默认的随机可选扰动。它说明 convex initialization 的质量解释必须加入“非零且方向合适的非对称种子”或随机初始化假设。

---

## 6. 主结果 II：convex-to-nonconvex 的谱动力学

### 6.1 中心化后的精确递推

对 Max-Cut 令

$$
x=\frac12\mathbf1+z.
$$

因为 $d=W\mathbf1$，

$$
\nabla_xL(x,y)=2(W+\operatorname{Diag}y)z.
$$

当投影和扰动未激活时，论文 Eq. (6) 精确化为

$$
\begin{aligned}
z^{t+1}&=\bigl[I-2\alpha(W+\operatorname{Diag}y^t)\bigr]z^t,\\
y_i^{t+1}&=y_i^t+\beta[(z_i^t)^2-1/4].
\end{aligned}
\tag{23}
$$

这不是近似式；非线性只在 dual 的 $z_i^2$ 中出现。

### 6.2 主定理 C

**[定理 4：中心轨道与瞬时谱收缩阈值]**
设 $W=W^\top\ne0$、对角为零，$\alpha,\beta>0$、$0<\delta<1/2$，特征值

$$
\lambda_1\le\cdots\le\lambda_n.
$$

由于 $\operatorname{tr}(W)=0$ 且 $W\ne0$ 为对称矩阵，必有 $\lambda_1<0<\lambda_n$。

取

$$
x^0=\frac12\mathbf1,\qquad y^0=\bar y_0\mathbf1,\qquad
\bar y_0>-\lambda_1,
$$

并令

$$
\bar y_t=\bar y_0-\frac{\beta t}{4},\qquad
T_0=\min\{t:\bar y_t\le0\}.
$$

则：

1. 对 $0\le t\le T_0$，

   $$
   x^t=\frac12\mathbf1,\qquad y^t=\bar y_t\mathbf1.
   $$

   在 $t<T_0$ 时因 $y_i^t>0$ 不触发 Algorithm 1 的扰动；在 $T_0$ 的下一次 primal update 会触发中心扰动。

2. 在 $t<T_0$，算法映射沿该轨道的一阶变分为

   $$
   \begin{pmatrix}u^{t+1}\\v^{t+1}\end{pmatrix}
   =
   \begin{pmatrix}
   I-2\alpha(W+\bar y_tI)&0\\
   0&I
   \end{pmatrix}
   \begin{pmatrix}u^t\\v^t\end{pmatrix}.
   \tag{24}
   $$

3. 若 $Wv_k=\lambda_kv_k$ 且 $u^t=\sum_ka_k^tv_k$，则

   $$
   a_k^{t+1}=\mu_k(t)a_k^t,\qquad
   \mu_k(t)=1-2\alpha(\lambda_k+\bar y_t).
   \tag{25}
   $$

4. 若小步长条件

   $$
   2\alpha(\lambda_n+\bar y_0)<1,
   \tag{26}
   $$

   成立，则对每个 $t<T_0$：当 $\bar y_t>-\lambda_1$ 时所有 primal multiplier 位于 $(0,1)$，该步的 primal 线性映射横向收缩；当 $\bar y_t=-\lambda_1$ 时最小特征空间 multiplier 为 1；当 $0<\bar y_t<-\lambda_1$ 时该空间 multiplier 大于 1。因此

   $$
   \boxed{\bar y=-\lambda_{\min}(W)}
   $$

   是沿未扰动中心轨道的精确 **one-step primal contraction threshold**。这里不把单步 multiplier 大于 1 等同于非自治轨道的 Lyapunov instability。

5. 若 $\lambda_1=\cdots=\lambda_r<\lambda_{r+1}$，定义 Algorithm 1 扰动前的可用谱窗口下端与宽度

   $$
   \ell:=\max\{0,-\lambda_{r+1}\},\qquad
   \omega:=-\lambda_1-\ell
   =\min\{-\lambda_1,\lambda_{r+1}-\lambda_1\}>0.
   $$

   令

   $$
   \mathcal I:=\{t<T_0:\ell<\bar y_t<-\lambda_1\}.
   $$

   若 $\mathcal I\ne\varnothing$，令 $t_c:=\min\mathcal I$。对任意 $t>t_c$，只要每个 $s=t_c,\ldots,t-1$ 均属于 $\mathcal I$，且 $P_{E_{\min}}u^{t_c}\ne0$，则

   $$
   \frac{\|P_{E_{\min}^\perp}u^t\|}
        {\|P_{E_{\min}}u^t\|}
   \le
   \frac{\|P_{E_{\min}^\perp}u^{t_c}\|}
        {\|P_{E_{\min}}u^{t_c}\|}
   \prod_{s=t_c}^{t-1}
   \frac{1-2\alpha(\lambda_{r+1}+\bar y_s)}
        {1-2\alpha(\lambda_1+\bar y_s)}.
   \tag{27}
   $$

   每个乘子比严格小于 1，因此这是最小特征空间相对其正交补的放大界。一个保证离散 dual schedule 在中心扰动前不跳过整个可用窗口、即 $\mathcal I\ne\varnothing$ 的充分条件是

   $$
   \boxed{\beta/4<\omega
   =\min\{-\lambda_1,\lambda_{r+1}-\lambda_1\}.}
   \tag{28}
   $$

#### 证明

在中心 $z=0$ 处，primal gradient 为 0，且 $g(1/2)=-1/4$，由归纳得到精确中心轨道。中心点严格位于盒内部，故局部投影 Jacobian 为恒等。primal map 对 $y$ 的交叉导数含 $\operatorname{Diag}(z)$，在中心为 0；dual map 对 $z$ 的一阶导数为 $2\beta\operatorname{Diag}(z)$，同样为 0，得到 (24)。在 $W$ 的正交特征基中对角化得到 (25)。条件 (26) 保证在扰动前所有相关 multiplier 为正；其与 1 的大小完全由 $\lambda_k+\bar y_t$ 的符号决定，故得到瞬时收缩阈值。

在 $\mathcal I$ 内，最小特征空间 multiplier 大于 1，而其正交补的最大 multiplier 是

$$
\mu_{r+1}(s)=1-2\alpha(\lambda_{r+1}+\bar y_s)\in(0,1).
$$

逐步使用

$$
\|P_{E_{\min}^\perp}u^{s+1}\|
\le\mu_{r+1}(s)\|P_{E_{\min}^\perp}u^s\|,
\qquad
\|P_{E_{\min}}u^{s+1}\|
=\mu_1(s)\|P_{E_{\min}}u^s\|
$$

即得 (27)。dual 每步下降 $\beta/4$；若该步长小于可用窗口宽度 $\omega$，序列不能从窗口上方一步跳到下方，故 $\mathcal I$ 非空。证毕。

### 6.3 这一定理能解释什么，不能解释什么

能解释：

- 在 (26) 成立、偏差足够小且投影/扰动未激活时，$y^0$ 超过阈值会使中心附近的 primal 变分逐步收缩；
- 在小步长条件 (26) 下穿越阈值后，最小特征空间的单步 multiplier 首先超过 1，并在可用谱窗口中相对其他模态放大；
- 固定 $\alpha$ 时减小 $\beta$ 会增加窗口内步数；在小 $\alpha$ 展开下，累计相对选择强度呈 $\alpha/\beta$ 尺度；
- 最小特征值有大间隙时，应观察到更清晰的低维早期轨迹。

不能解释：

- 精确中心点即使线性不稳定也不会自行离开；必须有随机初始化、数值噪声或非对称扰动；
- 最小特征向量的符号 cut 对任意未归一化、signed 或不规则图没有统一高质量保证；
- 当 $z$ 变大、$y_i$ 因 $z_i^2$ 分裂、投影激活后，线性定理停止；
- 当前代码的 RMSProp 动量会改变 multiplier，必须另做 optimizer-state 线性化。

而且，小步长前提对用户给定参数并非自动成立。以仓库自带的 G1 为例，$n=800,m=19176$ 且边权全为 1。Rayleigh quotient 给出

$$
\lambda_{\max}(W)\ge\frac{\mathbf1^\top W\mathbf1}{\|\mathbf1\|^2}
=\frac{2m}{n}=47.94.
$$

用户参数 $\alpha=0.02,\bar y_0=5$ 因而满足

$$
\alpha(\lambda_{\max}+\bar y_0)\ge1.0588>1,
$$

所以 plain GDA 的最大特征模态在初始步已有

$$
|\mu_n(0)|
=|1-2\alpha(\lambda_{\max}+\bar y_0)|
\ge1.1176.
$$

这不仅违反 (26)，也违反普通离散梯度步的线性收缩条件。实际代码使用 RMSProp，不能由该 plain-GDA multiplier 直接判断稳定性。因此定理 4 是一个需要通过版本对齐和步长归一化检验的机制定理，不能直接当作现有 64/71 结果的既成解释。

用户报告还给出 G1 cut 值 $C=11624$。令该 cut 的符号向量为 $s\in\{\pm1\}^{800}$，则

$$
s^\top Ws=2(m-2C)=-8144,
\qquad
\frac{s^\top Ws}{\|s\|^2}=-10.18.
$$

由 Rayleigh 原理，$\lambda_{\min}(W)\le-10.18$，所以用户的 `dual_init=5` 在 G1 上**确定没有**达到凸阈值。再者，当前 `primal_init=uniform` 从整个盒中采样，并非中心的无穷小扰动。除非另证 RMSProp 会先把这些初值吸入中心邻域，否则中心轨道定理不能直接解释该实际设置。

在 $K_2$ 反例中，正确 cut 对应反对称最小特征模态；但精确中心没有该模态种子，论文 tie rule 在 $y\le0$ 时反而同时把两个坐标踢向同一侧，播种了错误的对称模态。这精确展示了谱解释所需的随机非对称性假设。

---

## 7. 理论与实验之间的可证伪连接

以下均是最小可执行设计，不要求 GPU 以外的新算法；完整 Gset 测试需用户已有实例。

### 实验 A：版本对齐与因果拆分

**对照组**：

1. 论文 Eq. (6)：plain SGD、old-$x$ dual；
2. plain SGD、new-$x$ dual；
3. RMSProp、old-$x$ dual；
4. 当前 RMSProp、new-$x$ dual；
5. 各组再分别开/关论文确定性扰动与代码随机扰动。

**记录**：incumbent、当前 rounded batch、integrality、flip violations、dual、raw gradient、optimizer update、扰动次数。

**判别**：若 one-flip 主要来自理论 projected-GDA fixed-state 机制，plain/old-$x$ 组应最符合 $\Delta_i+y_i\le0$；若主要来自 RMSProp/archive，版本切换将显著改变 envelope 而 archive incumbent 仍保持局部质量。

### 实验 B：精确谱阈值与首次模态

**实例**：先用 $K_2,C_4$、随机 $d$-regular 小图，并从 $x^0=\frac12\mathbf1+\sigma\xi$ 的小扰动开始；再选可算若干极端特征值/向量的 Gset。默认 uniform initialization 另作外推对照，不与局部定理混在一起。

**记录**：

$$
z^t=x^t-\tfrac12\mathbf1,\quad
\lambda_{\min}(W+\operatorname{Diag}y^t),\quad
\|P_{E_k}z^t\|,\quad
\text{batch covariance}.
$$

**对照**：先核对 (26)；令 $y^0$ 位于 $-\lambda_{\min}(W)$ 下方、等号附近和上方。分别做两组网格：(i) 固定 $\alpha$ 改变 $\beta$，测窗口步数；(ii) 固定较小 $\beta$ 改变 $\alpha$，测单步与累计 mode ratio。所有组都单独报告是否满足 (26)/(28)。

**支持模式**：满足小步长条件时，阈值以上的中心小扰动先收缩，最小 eigenmode 在预测时刻首先失去单步收缩；固定 $\alpha$ 减小 $\beta$ 增加窗口步数，小步长区间中的累计 log-mode-ratio 近似按 $\alpha/\beta$ 增强。
**反驳模式**：首次增长与谱阈值/模态系统无关，或 RMSProp 下完全由动量状态决定。

### 实验 C：matched dual envelope，而非 Spearman

在每次 incumbent 改善时保存 source `(iteration, batch, x, y, raw_grad, optimizer_state)`，计算

$$
E_i:=\Delta_i(z^{\mathrm{inc}})+y_i,\qquad
R_i:=(2z_i^{\mathrm{inc}}-1)\partial_iL(z^{\mathrm{inc}},y).
$$

plain projected fixed-state 机制预测 $E_i=R_i\le0$。当前实现不一定满足，但应检查违反量是否随后续稳定时间减小。最终 $y$ 的 batch mean 不再作为主要统计量。

### 实验 D：累计 fractionality 与锁定

为每个 batch/坐标记录

$$
A_i^T:=\sum_{t<T}x_i^t(1-x_i^t),
$$

以及第一次进入端点邻域、最后一次 round bit 改变、最终 flip gain。论文 old-$x$ 版本预测

$$
y_i^T=y_i^0-\beta A_i^T.
$$

只有对 matched、old-$x$、并收敛到 binary projected fixed limit 的轨迹，定理 1 才预测违反点必须有 $A_i^\infty>y_i^0/\beta$，且 gain 落在 $-y_i^\star$ envelope 内。当前有限时 archive incumbent 不满足这些前提；若实验仍观察到相同关系，它是支持证据，而不是先验必要条件。

### 实验 E：archive 邻居覆盖

保存所有 distinct rounded candidates 的哈希。对最终 incumbent 的每个一翻邻居 $z\oplus e_i$，检查它是否曾被访问。

**严格事实**：若一个邻居曾进入候选集合，历史 best incumbent 不可能比它差；若全部 $n$ 个邻居都被覆盖，则 incumbent 必为 one-flip local optimum。

**支持 archive 解释**：one-flip 性随 neighbor coverage 增长，并在关闭 archive、只返回最终 round 时明显下降。
**支持 fixed-state 解释**：即便覆盖率低，最终/current batch 也满足 envelope 和 one-flip。

### 实验 F：整数格点解释 gain $=1$

对 7 个违反实例记录 raw flip gain 的精确整数、matched $y$、$R_i$ 以及浮点误差。若所有 $R_i-y_i<2$ 且权重整数，则正 gain 只能为 1；若没有该严格上界，则“全为 1”只能视为浅违反的经验观察。

### 实验 G：随机非对称性是否避免 $K_2$ 坏轨迹

从 $x^0=\frac12\mathbf1+\sigma\xi$ 开始，$\sigma$ 取 $0,10^{-8},10^{-6},10^{-4},10^{-2}$，比较确定性同向扰动、随机独立符号扰动和无扰动。记录 antisymmetric/symmetric mode ratio 与成功 cut 概率。谱机制预测非零 antisymmetric seed 在足够慢 dual schedule 下会获得相对模态放大；是否足以克服此前收缩并主导非线性阶段仍需实验。$\sigma=0$ 配合同向 tie rule 则复现零 cut。

---

## 8. 可进入论文的最终表述

### 8.1 最值得新增的 theorem/proposition

建议以定理 1 + 定理 2 的组合为正式版本：

> **Theorem (dual-slack discrete optimality certificate).** Let $f$ be the multilinear extension of $F$, let $g(u)=u^2-u$, let $\alpha>0$, and let $z\in\{0,1\}^n$. If $(z,y)$ is a fixed point of the projected primal update with stepsize $\alpha$, then $F(z\oplus e_i)-F(z)\ge y_i$ for every coordinate. Hence nonnegative dual variables at such a fixed point certify one-flip local optimality, while negative dual variables quantify the largest admissible one-flip improvement. For nonnegative weighted Max-Cut, this yields $C(z)\ge M/2+\frac14\sum_i y_i$, and in particular a $1/2$-approximation whenever $y\ge0$.

正文必须紧接着说明：仓库历史 incumbent 不是该 fixed point，需 matched-state instrumentation 或显式 one-flip 后处理才能把 theorem 施加到实际返回解。

### 8.2 可直接放入论文、不过度声称的理论解释

> 对满足小步长条件的 Max-Cut plain-GDA，均匀 dual 初始化产生一条精确中心轨道；当 dual 穿越 $-\lambda_{\min}(W)$ 时，最小特征空间的瞬时 multiplier 首先超过 1，并在扰动前的可用谱窗口中相对其他模态放大。该结论为随机初始化残差的早期方向选择提供了局部机制，但不保证轨迹一定离开中心，也不直接覆盖 RMSProp 实现。随后若逐坐标 dual 变负且满足严格 dual-slack 条件，连续坐标剖面可形成势垒。该势垒并不等价于离散局部最优：在二进制 projected fixed point，单翻差仅满足 $F(z\oplus e_i)-F(z)\ge y_i$。因此非负 dual 给出 one-flip 证书，而负 dual 只给出允许浅违反的上界。该结论与近 one-flip 观察相容，但不构成一般 QUBO 的全局质量保证；对非负权 Max-Cut，它最多推出紧的 $1/2$ 型保证。

### 8.3 投稿前必须补齐

1. 修正或收缩 Proposition 3.3 / Corollary 3.5 的假设与证明，明确 entropy 端点处理。
2. 明确区分 paper GDA、实验代码和 archive 输出；若声称理论解释代码，必须分析 RMSProp 状态和 new-$x$ dual 时序。
3. 保存 matched incumbent state，重新做 dual envelope、累计 fractionality、锁定时间和 neighbor coverage 实验。
4. 将 Gset 按非负权与 signed 权分组；只在前者报告 $1/2$ 型保证。
5. 验证谱阈值、最小模态、固定 $\alpha$ 时的 $\beta$ 窗口步数，以及小步长下的 $\alpha/\beta$ 累计模态比预测；否则“convex initialization 负责 basin selection”仍只是叙述。
6. 若需要无条件 one-flip 结论，启用并报告 `--refine`，同时把它清楚标为后处理。
7. 若希望形成核心而非辅助理论贡献，需要在规则/近二分/随机/planted 图上证明谱模态到最终 cut 的结构化质量定理。

### 8.4 诚实的研究价值判断

- **一般 QUBO 的统一全局保证：不可得。** 两变量反例已否定从 strict one-flip 或 binary fixed point 推出非平凡 ratio/gap。
- **一般 PDBO 自动 one-flip：不可得。** dual-slack 固定点边界和 $K_2$ 轨迹给出直接反例。
- **非负权 Max-Cut：可得到正确但偏弱的 $1/2$ 局部搜索型保证。** 适合作为安全下界，不解释近最优性能。
- **dual-slack + 瞬时谱收缩/模态选择：有论文价值。** 本报告把 PDBO 动力学、局部离散质量和 convex-to-nonconvex 机制用可核查公式连接起来；在当前阶段更适合作为重要辅助理论。若结构化图类定理和配套实验成立，可升级为核心贡献。

---

## 9. 计算核查与复现

CPU-only 脚本：`research/pdbo_theory_checks.py`。

运行：

```bash
python3 research/pdbo_theory_checks.py
```

它执行：

- 随机 QUBO 上逐点核对 gradient/flip 恒等式；
- 穷举所有 $n\le4$ 无权图及所有 cuts，自动找到 one-flip 的最坏比值 $1/2$；
- 核对 $C_4$ exact/strict 紧例；
- 复现四点 signed Max-Cut ratio-0 反例；
- 复现两变量 QUBO 任意 gap；
- 复现 $g(x^t)\equiv0$ 但 $x^t$ 二周期；
- 核对 G1 的 $\lambda_{\max}$ average-degree 下界与由用户报告 cut 值导出的 $\lambda_{\min}$ Rayleigh 上界；
- 复现从严格凸初始化出发的 $K_2$ 零 cut 轨迹。

当前运行输出：

```text
all checks passed
worst unweighted n<=4 one-flip ratio: 0.500
enumerated witness bits: 0101
enumerated witness edges: 4
K2 convex-init bad trajectory reached 00 at iteration: 40
G1 average-degree lower bound on lambda_max: 47.94
G1 reported-cut Rayleigh upper bound on lambda_min: -10.18
```

---

## 10. 可核查的一手文献

1. Liu et al., “Smoothing Binary Optimization: A Primal-Dual Perspective,” arXiv:2509.21064v2, 2026. [arXiv](https://arxiv.org/abs/2509.21064)
2. E. Boros and P. L. Hammer, “Pseudo-Boolean Optimization,” *Discrete Applied Mathematics* 123:155--225, 2002. [DOI](https://doi.org/10.1016/S0166-218X(01)00341-9). 用于唯一 multilinear 表示与伪布尔基础。
3. D. S. Johnson, C. H. Papadimitriou, M. Yannakakis, “How Easy Is Local Search?”, *JCSS* 37(1):79--100, 1988. [DOI](https://doi.org/10.1016/0022-0000(88)90046-3). 提出 PLS 框架。
4. A. A. Schäffer and M. Yannakakis, “Simple Local Search Problems that are Hard to Solve,” *SIAM Journal on Computing* 20(1):56--87, 1991. [DOI](https://doi.org/10.1137/0220004). 其 weighted MAX-CUT/FLIP 结果使用二进制编码整数边权和单顶点移动邻域，并证明寻找局部最优 cut 是 PLS-complete；`refinement.py` 的 one-flip 邻域与之对应，但小整数权的伪多项式界不与 PLS-completeness 矛盾。
5. M. Etscheid and H. Röglin, “Smoothed Analysis of Local Search for the Maximum-Cut Problem,” *ACM Transactions on Algorithms* 13(2), Article 25, 2017. [DOI](https://doi.org/10.1145/3011873). 结论针对独立有界密度边权扰动下的 FLIP，不直接覆盖确定性 Gset 或 PDBO。
6. O. Angel, S. Bubeck, Y. Peres, F. Wei, “Local Max-Cut in Smoothed Polynomial Time,” STOC 2017. [arXiv:1610.04807](https://arxiv.org/abs/1610.04807). 同样是明确 smoothed model 下的离散 local search 结果。
7. M. X. Goemans and D. P. Williamson, “Improved Approximation Algorithms for Maximum Cut and Satisfiability Problems Using Semidefinite Programming,” *JACM* 42(6):1115--1145, 1995. [DOI](https://doi.org/10.1145/227683.227684). 非负加权 Max-Cut 的 $0.878\ldots$ 基准；不能映射成 PDBO 保证。
8. L. Trevisan, “Max Cut and the Smallest Eigenvalue,” *SIAM Journal on Computing* 41(6):1769--1786, 2012. [arXiv:0806.1978](https://arxiv.org/abs/0806.1978). 其保证使用归一化谱；只在规则图等情形与本文未归一化 $W$ 直接对应。
9. E. Hazan, K. Y. Levy, S. Shalev-Shwartz, “On Graduated Optimization for Stochastic Non-Convex Problems,” ICML 2016. [PMLR](https://proceedings.mlr.press/v48/hazanb16.html). 需要相邻平滑尺度极小点邻近及局部强凸，PDBO 尚未验证。
10. N. Fenichel, “Geometric Singular Perturbation Theory for Ordinary Differential Equations,” *JDE* 31:53--98, 1979. [DOI](https://doi.org/10.1016/0022-0396(79)90152-9). 可用于强凸阶段的 slow--fast tracking；在 Hessian 特征值过零处 normal hyperbolicity 失效。
11. C. Jin, P. Netrapalli, M. I. Jordan, “What Is Local Optimality in Nonconvex-Nonconcave Minimax Optimization?”, ICML 2020. [arXiv:1902.00618](https://arxiv.org/abs/1902.00618). 其非退化 local-minimax 二阶刻画要求严格的 $yy$ 曲率（例如相应 Hessian block 可逆/负定），不能直接用于对 $y$ 线性的 PDBO。

这些外部结果只在各自条件内使用；文献 5--11 主要用于比较与后续方向，不参与定理 1--4 的证明。本报告的定理与反例均为自包含推导。
