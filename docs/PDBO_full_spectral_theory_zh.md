# PDBO 在 Max-Cut 上的完整谱动力学理论

## 中心随机初始化、移动谱前沿、非线性谱修复与一般图质量证书

本文研究固定步长、同步 primal-dual 更新、quadratic binary constraint

$$
g_i(x)=x_i^2-x_i
$$

下的 PDBO。令 $W=W^\top$、$W_{ii}=0$，边权暂取非负，总边权为

$$
M=\sum_{i<j}w_{ij},
$$

并用 sign vector $s\in\{\pm1\}^n$ 表示 cut：

$$
C(s)=\frac M2-\frac14s^\top Ws.
$$

中心化变量、谱分解和随机初始化分别记为

$$
x=\frac12\mathbf1+z,
\qquad
Wv_k=\lambda_kv_k,
\qquad
\lambda_1\le\cdots\le\lambda_n,
$$

$$
z^0=\rho\xi,\qquad
\xi_i\overset{\rm iid}{\sim}{\rm Unif}[-1,1],\qquad
0<\rho<\frac12.
$$

结论按证据强度分为四类：

- **[U] 无条件恒等式或确定性定理**：在列出的代数条件下对任意图、任意轨迹成立；
- **[C] 可认证的轨迹条件定理**：条件可由一次实际运行中的量直接核验；
- **[S] 结构化图定理**：对二部、近二部、flat ground-state 或 uniform-leverage 图成立；
- **[I] 不可能性边界**：说明哪些一般图结论不能从现有机制推出。

本文最重要的结论不是“PDBO 对所有图自动近优”，而是下面这条严格闭合的链：

$$
\boxed{
\begin{array}{c}
\text{SDP-dual 可行凸化}\;\longrightarrow\;
\text{全谱多项式低通}\;\longrightarrow\;
\text{移动前沿逐级激活}\\[2mm]
\longrightarrow\;
\text{anisotropy/projection 受迫谱修复}\;\longrightarrow\;
\text{饱和与舍入}\;\longrightarrow\;
\text{实例可认证的 cut 质量}.
\end{array}}
$$

前四个箭头有精确动力学定理；最后的质量由显式 gap potential 控制。在一般图上，这个 potential 是否变得足够小是轨迹条件，而不是自动事实。

---

## 1. 为什么从 convex stage 开始有严格意义

### 定理 1 [U]：diagonal convexifier 正好是 Max-Cut SDP dual

标准 Max-Cut SDP 写为

$$
\operatorname{SDP}(W)
=\max_{X\succeq0,\ \operatorname{diag}X=\mathbf1}
\left\{
\frac M2-\frac14\operatorname{tr}(WX)
\right\}.
$$

对任意 $q\in\mathbb R^n$，令

$$
H_q=W+\operatorname{Diag}q.
$$

若 $H_q\succeq0$，则自动有 $q_i=(H_q)_{ii}\ge0$，并且

$$
\boxed{
U_q:=\frac M2+\frac14\mathbf1^\top q
}
\tag{1}
$$

是 Max-Cut 的 SDP-dual 可行上界。事实上，

$$
C(s)=U_q-\frac14s^\top H_qs\le U_q,
\tag{2}
$$

且

$$
\min_{q:H_q\succeq0}U_q=\operatorname{SDP}(W)\ge\operatorname{OPT}.
\tag{3}
$$

取均匀 shift $q=-\lambda_1\mathbf1$，得到谱上界

$$
\boxed{
U_{\rm spec}=\frac M2-\frac{n\lambda_1}{4}.
}
\tag{4}
$$

**证明。** 对任意 sign vector，$s^\top\operatorname{Diag}(q)s=\mathbf1^\top q$。将其加到 $s^\top Ws$ 中即得 (2)。式 (3) 是 SDP 的标准对偶。由于 $W_{ii}=0$，$H_q\succeq0$ 还强制其对角元 $q_i$ 非负。证毕。

这给出 convex initialization 的第一个严格解释：$W+\operatorname{Diag}y^t\succeq0$ 不仅表示 primal landscape 凸，还表示当前 $y^t$ 是一个合法的 Max-Cut SDP-dual 证书。

### 定理 2 [U]：凸阶段单调收紧 dual 上界

定义

$$
m_t=\frac1n\mathbf1^\top y^t,
\qquad
\theta_t=-m_t,
\qquad
F_t=1-\frac{4\|z^t\|^2}{n}.
\tag{5}
$$

因为 $z^t\in[-1/2,1/2]^n$，有 $0\le F_t\le1$。同步 dual 更新精确给出

$$
\boxed{
\theta_{t+1}-\theta_t=\frac\beta4F_t,
\qquad
\theta_t=\theta_0+\frac\beta4\sum_{r<t}F_r.
}
\tag{6}
$$

另一方面，raw dual objective 为

$$
U(y^t)=\frac M2+\frac14\mathbf1^\top y^t
=\frac M2-\frac{n\theta_t}{4},
$$

故

$$
\boxed{
U(y^{t+1})-U(y^t)=-\frac{\beta n}{16}F_t.
}
\tag{7}
$$

只要 $H_{y^t}=W+\operatorname{Diag}y^t\succeq0$，这个不断下降的量就是有效 SDP-dual 上界。

并且

$$
H_{y^{t+1}}-H_{y^t}
=\beta\operatorname{Diag}\left(z^{t\odot2}-\frac14\mathbf1\right)
\preceq0.
\tag{8}
$$

因此 convex stage 是一段连续的初始时间区间：一旦 $H_{y^t}$ 失去半正定性，之后不会重新进入 PSD cone。

式 (6) 和 (7) 是同一个事实的动力学与优化解释：fractionality 推动谱前沿前进，同时以完全相同的速度收紧 dual 上界；接近二值时 $F_t\downarrow0$，两者都自动减速。

### 任意时刻都有效的 diagonal-shift 上界

即使 $H_y$ 已非凸，令

$$
\ell(y)=\lambda_{\min}(W+\operatorname{Diag}y),
\qquad
\widehat y=y-\ell(y)\mathbf1,
$$

则 $H_{\widehat y}\succeq0$，从而

$$
\boxed{
\operatorname{UB}(y)
=\frac M2+
\frac{\mathbf1^\top y-n\ell(y)}4
}
\tag{9}
$$

永远是合法 Max-Cut 上界。均匀 $y=m\mathbf1$ 时，(9) 恒等于 $U_{\rm spec}$；所以均匀中心轨道上的 raw bound 下降只是在消去多余 uniform shift，不能单独证明 SDP 最优或 cut 近优。真正可能优于谱上界的是后期形成的非均匀 diagonal shift。

---

## 2. 精确中心动力学与受迫分解

因为 $W\mathbf1=d$，Max-Cut Lagrangian 的中心化梯度为

$$
\nabla_xL(x,y)=2(W+\operatorname{Diag}y)z.
$$

将 dual 分解为

$$
y^t=m_t\mathbf1+\eta^t,
\qquad
\mathbf1^\top\eta^t=0.
\tag{10}
$$

定义 $h^t$ 为 box projection 与 Algorithm 1 中心 kick 相对于完整 raw primal step 的总残差。则完整算法精确满足

$$
\boxed{
z^{t+1}
=\underbrace{[I+2\alpha(\theta_tI-W)]z^t}_{\text{mean-dual 标量比较系统}}
+\underbrace{f^t}_{\text{非线性 forcing}},
}
\tag{11}
$$

其中

$$
f^t=-2\alpha\operatorname{Diag}(\eta^t)z^t+h^t.
\tag{12}
$$

同时 dual anisotropy 的生成律为

$$
\boxed{
\eta^{t+1}-\eta^t
=\beta\left(
z^{t\odot2}-\frac{\|z^t\|^2}{n}\mathbf1
\right).
}
\tag{13}
$$

注意 (13) 是增量恒等式；不同时间的增量可以抵消，不能据此声称 $\|\eta^t\|$ 必然单调增加。

### 定理 3 [U]：逐模态 Duhamel 公式

令

$$
a_k^t=v_k^\top z^t,
\qquad
\mu_k(t)=1+2\alpha(\theta_t-\lambda_k),
$$

$$
G_k(T,s)=\prod_{r=s}^{T-1}\mu_k(r).
\tag{14}
$$

则任意时刻、任意模态精确满足

$$
\boxed{
a_k^T
=G_k(T,0)a_k^0
+\sum_{t=0}^{T-1}G_k(T,t+1)\,v_k^\top f^t.
}
\tag{15}
$$

第一项是 convex/activation 阶段留下的多项式谱记忆；第二项是 dual anisotropy、投影和中心 kick 对该模态的全部再注入。没有任何未列出的非线性来源。

定义

$$
R_k(T)=\sum_{t<T}|G_k(T,t+1)|\,|v_k^\top f^t|.
\tag{16}
$$

则

$$
|a_k^T-G_k(T,0)a_k^0|\le R_k(T).
\tag{17}
$$

若

$$
\varepsilon_k(T)
=\frac{R_k(T)}{|G_k(T,0)a_k^0|}<1,
\tag{18}
$$

便有可直接实验核验的 shadowing 界

$$
(1-\varepsilon_k)|G_ka_k^0|
\le |a_k^T|
\le(1+\varepsilon_k)|G_ka_k^0|.
\tag{19}
$$

这条余项条件非常重要：一个高模态可以被齐次滤波压到极小，以至于绝对值很小的 forcing 也会使其相对预测误差很大；这不妨碍整个状态的能量仍与齐次低通预测高度一致。

---

## 3. 有限随机种子的完整 convex-stage 低通理论

先固定一条与随机种子无关的标量 schedule $\theta_0,\ldots,\theta_{T-1}$，并令 $f^t=0$。则

$$
z^T=p_T(W)z^0,
\qquad
p_T(\lambda)=\prod_{t<T}[1+2\alpha(\theta_t-\lambda)].
\tag{20}
$$

假设全部 multiplier 为正。一个简单充分条件是

$$
2\alpha(\lambda_n-\theta_0)<1,
\tag{21}
$$

因为 $\theta_t$ 单调不减。

### 定理 4 [U]：任意两个特征值之间的全谱过滤

对任意 $\lambda_i<\lambda_j$、$T\ge1$，若 $a_i^0a_j^0\ne0$，则

$$
\boxed{
\frac{|a_i^T|/|a_i^0|}{|a_j^T|/|a_j^0|}
=\prod_{t<T}
\frac{1+2\alpha(\theta_t-\lambda_i)}
     {1+2\alpha(\theta_t-\lambda_j)}
>1.
}
\tag{22}
$$

因此低特征值相对高特征值的优势在每一步都严格增加。这个结论覆盖任意两个单独特征值、任意两个分离谱带；原来“最小特征空间相对其补空间”的结论只是第一个谱隙的特例。

若低谱带 $L$ 和高谱带 $H$ 满足

$$
b=\max_{i\in L}\lambda_i
<c=\min_{j\in H}\lambda_j,
$$

则

$$
\frac{\|P_Lz^T\|/\|P_Hz^T\|}
     {\|P_Lz^0\|/\|P_Hz^0\|}
\ge
\prod_{t<T}
\frac{1+2\alpha(\theta_t-b)}
     {1+2\alpha(\theta_t-c)}.
\tag{23}
$$

### 定理 5 [U/C]：有限随机种子的谱带能量

设 $P_B$ 是任意固定谱带投影。对 iid center-uniform seed，

$$
\mathbb E\|P_Bp_T(W)z^0\|^2
=\frac{\rho^2}{3}
\sum_{k\in B}p_T(\lambda_k)^2.
\tag{24}
$$

Hanson-Wright 不等式进一步给出：存在普适常数 $c,C>0$，对任意 $u>0$，以至少 $1-2e^{-u}$ 的概率，

$$
\left|
\|P_Bp_T(W)z^0\|^2
-\frac{\rho^2}{3}\operatorname{tr}A_B
\right|
\le
C\rho^2\left(
\|A_B\|_F\sqrt u+\|A_B\|u
\right),
\tag{25}
$$

其中

$$
A_B=p_T(W)P_Bp_T(W).
$$

对有限个时刻和谱带取 union bound 即得到有限随机种子版本的完整全谱直方图定理。若使用 $B_0$ 个独立 batch 并平均能量，浓度还会随 $B_0$ 改善。

对单独模态，连续随机初始化只保证 $a_k^0\ne0$ 几乎处处成立。任何定量有限时间恢复结论都必须显式条件于 small-ball 事件 $|a_k^0|\ge\gamma$；不能把“概率一非零”替换成不存在的确定性下界。

实际 PDBO 的 $\theta_t$ 由同一个随机 seed 决定，不能条件于随机 $p_T$ 后仍把 $z^0$ 当作 iid uniform。严格的转移方式是使用与 seed 无关的中心参考 schedule

$$
\overline\theta_t=\theta_0+\frac{\beta t}{4}.
$$

由 (6) 有精确偏差

$$
\theta_t-\overline\theta_t
=-\frac\beta n\sum_{r<t}\|z^r\|^2.
\tag{25a}
$$

把真实递推改写为

$$
z^{t+1}
=[I+2\alpha(\overline\theta_tI-W)]z^t
+\widetilde f^t,
$$

$$
\widetilde f^t
=f^t+2\alpha(\theta_t-\overline\theta_t)z^t.
\tag{25b}
$$

显式定义参考传播子

$$
\overline\mu_k(t)=1+2\alpha(\overline\theta_t-\lambda_k),
\qquad
\overline G_k(T,s)=\prod_{r=s}^{T-1}\overline\mu_k(r),
$$

以及

$$
\widetilde R_k(T)
=\sum_{t<T}|\overline G_k(T,t+1)|
\,|v_k^\top\widetilde f^t|.
\tag{25c}
$$

此时参考多项式与 seed 独立，(24)--(25) 可直接应用；真实模态满足

$$
|a_k^T-\overline G_k(T,0)a_k^0|
\le\widetilde R_k(T),
\tag{25d}
$$

并有与 (18)--(19) 相同的相对 shadowing 界。schedule 偏差项在小中心种子阶段同样是三阶量。这才是有限随机种子、有限时间、全谱且带非线性余项的完整过滤定理。

---

## 4. 移动谱前沿与逐级激活

式 (6) 说明

$$
q_t:=\frac\beta4\sum_{r<t}F_r=\theta_t-\theta_0
\tag{26}
$$

是精确的 fractionality 时钟。在裸迭代时间中前沿会减速；在 $q_t$ 中它以单位速度移动。

### 定理 6 [U]：每个特征向量的 crossing、trough 和谱面积

考虑齐次标量比较系统。若下述 crossing 有限，定义

$$
\tau_k=\min\{t:\theta_t\ge\lambda_k\}.
\tag{27}
$$

若 $a_k^0\ne0$ 且 multiplier 为正，则

$$
0<\mu_k(t)<1\quad(t<\tau_k),
\qquad
\mu_k(t)\ge1\quad(t\ge\tau_k).
$$

因此 $|a_k^t|$ 在 crossing 前严格下降，在 crossing 后非减，谷底位于 $t=\tau_k$；若某一步恰有 $\theta_t=\lambda_k$，则谷底可以形成相邻两时刻的平台。

令

$$
A_k^-(s)=\sum_{t=s}^{\tau_k-1}(\lambda_k-\theta_t),
\qquad
A_k^+(T)=\sum_{t=\tau_k}^{T-1}(\theta_t-\lambda_k).
\tag{28}
$$

若 crossing 前 $2\alpha(\lambda_k-\theta_t)\le\zeta_-<1$，则

$$
e^{-\frac{2\alpha}{1-\zeta_-}A_k^-(s)}
\le
\frac{|a_k^{\tau_k}|}{|a_k^s|}
\le
e^{-2\alpha A_k^-(s)}.
\tag{29}
$$

若 crossing 后 $2\alpha(\theta_t-\lambda_k)\le\zeta_+$，则

$$
e^{\frac{2\alpha}{1+\zeta_+}A_k^+(T)}
\le
\frac{|a_k^T|}{|a_k^{\tau_k}|}
\le
e^{2\alpha A_k^+(T)}.
\tag{30}
$$

所以一个模态何时恢复到初始幅度，不由“越过阈值多少步”单独决定，而由 crossing 前后的累计谱面积平衡决定。

### 定理 7 [U]：任意两模态的 exclusive-window 增益

取 $\lambda_i<\lambda_j$，假设两次 crossing 都有限、两模态初始投影非零，并且 exclusive window 内所有相关 multiplier 为正，记

$$
\Delta_{ij}=\lambda_j-\lambda_i,
\qquad
\mathcal I_{ij}=\{t:\lambda_i\le\theta_t<\lambda_j\},
$$

$$
N_{ij}=|\mathcal I_{ij}|.
$$

在该窗口中，低模态已经增长而高模态仍在收缩，并且逐步有

$$
\frac{\mu_i(t)}{\mu_j(t)}
=1+\frac{2\alpha\Delta_{ij}}{\mu_j(t)}
\ge1+2\alpha\Delta_{ij}.
$$

故窗口内精确选择下界为

$$
\boxed{
\frac{|a_i^{\tau_j}|/|a_j^{\tau_j}|}
     {|a_i^{\tau_i}|/|a_j^{\tau_i}|}
\ge
(1+2\alpha\Delta_{ij})^{N_{ij}}.
}
\tag{31}
$$

若某个 $0<\overline F\le1$ 使得从进入窗口前的最后一步起直到离开窗口均有 $F_t\le\overline F$，且前沿从窗口下方完整穿越，则离散端点修正后的步数界是

$$
\boxed{
N_{ij}\ge
\left\lfloor\frac{4\Delta_{ij}}{\beta\overline F}\right\rfloor.
}
\tag{32}
$$

这里使用了首次 crossing 前严格位于窗口下方这一事实；若只知道某个任意起点已经落在窗口内，则应改用实际观测的 $N_{ij}$。一个保证窗口不被整个跳过的充分条件是

$$
\boxed{
\frac{\beta\overline F}{4}<\Delta_{ij}.
}
\tag{33}
$$

小 $\alpha\Delta$ 时，(31)--(32) 的主阶为

$$
\boxed{
\log\operatorname{Gain}_{ij}
\simeq
\frac{8\alpha\Delta_{ij}^2}{\beta F}.
}
\tag{34}
$$

这就是逐级激活的核心定量律。减小 $\beta$、增大 $\alpha$、增大谱隙，或因二值化使 $F$ 下降，都会增加窗口选择强度。

### 谱带 trough 为什么早于 histogram bin center

对谱带 $B$，令

$$
E_B^t=\sum_{k\in B}(a_k^t)^2,
\qquad
\pi_k^t=\frac{(a_k^t)^2}{E_B^t}.
$$

齐次系统满足精确能量递推

$$
\boxed{
\frac{E_B^{t+1}-E_B^t}{4\alpha E_B^t}
=
\theta_t-\sum_{k\in B}\pi_k^t\lambda_k
+\alpha\sum_{k\in B}\pi_k^t(\theta_t-\lambda_k)^2.
}
\tag{35}
$$

因此一个宽谱带的 trough 由当前能量加权特征值决定，而不是由绘图时人为选取的 bin center 决定。低通会先在 bin 内把权重推向较低特征值，故观测到的 band trough 通常早于 $\theta_t$ 穿过几何中心；这不违反逐模态 crossing 定理。

### 真实前沿是一个带，而不是一条线

实际 Hessian 为

$$
H_t=W-\theta_tI+\operatorname{Diag}\eta^t.
$$

Weyl 不等式给出

$$
\left|\lambda_k(H_t)-(\lambda_k-\theta_t)\right|
\le\|\eta^t\|_\infty.
\tag{36}
$$

所以第 $k$ 个实际曲率方向：

$$
\theta_t<\lambda_k-\|\eta^t\|_\infty
\Longrightarrow \lambda_k(H_t)>0,
$$

$$
\theta_t>\lambda_k+\|\eta^t\|_\infty
\Longrightarrow \lambda_k(H_t)<0.
\tag{37}
$$

dual anisotropy 把理想的 crossing 拓宽为宽度至多 $2\|\eta^t\|_\infty$ 的过渡带。若目标谱簇与其余谱相隔 $\gamma$ 且 $\|\eta^t\|_\infty<\gamma/2$，Davis-Kahan 还给出子空间旋转量 $O(\|\eta^t\|_\infty/\gamma)$。

最后，对真实固定-$W$ 模态，forcing 小于齐次 margin 给出逐步单调性的充分条件。当 $\theta_t>\lambda_k$ 且 $\mu_k(t)>0$ 时，

$$
|v_k^\top f^t|
<2\alpha|\theta_t-\lambda_k|\,|a_k^t|.
\tag{38}
$$

保证 $|a_k^{t+1}|>|a_k^t|$；当 $\theta_t<\lambda_k$ 且 $\mu_k(t)>0$ 时，同一不等式保证严格收缩。超过这一边界后，正确理论是 Duhamel 公式 (15)，而不是继续声称每个模态独立、单调地按前沿增长。

---

## 5. 非线性离散化：为什么谱会重新变宽

### 5.1 幅值不平坦精确生成 dual anisotropy

对 $z^t\ne0$，令

$$
c_t=\frac{\|z^t\|}{\sqrt n},
\qquad
r^t=\frac{|z^t|}{c_t},
\qquad
\|r^t\|=\sqrt n.
$$

由 (13)，

$$
\|\eta^{t+1}-\eta^t\|
=\beta c_t^2\|(r^t)^{\odot2}-\mathbf1\|.
$$

逐坐标使用 $|r_i^2-1|=|r_i-1|(r_i+1)$，得到

$$
\boxed{
\beta c_t^2\|r^t-\mathbf1\|
\le
\|\eta^{t+1}-\eta^t\|
\le
\beta c_t^2(\|r^t\|_\infty+1)\|r^t-\mathbf1\|.
}
\tag{39}
$$

所以新的 anisotropy 当且仅当当前坐标绝对值不全相等时产生。早期低谱特征向量通常并不 flat，因而会自动产生非均匀 dual；已有 $\eta^t$ 不会因为某一步幅值变平而自动消失。

若 $\eta^0=0$ 且 $\|z^r\|_\infty\le R$，则粗略但显式地

$$
\|\eta^t\|_\infty\le\beta tR^2,
\qquad
\|f_\eta^t\|
\le2\alpha\beta tR^2\|z^t\|.
\tag{40}
$$

若整个考察区间而不只是初始时刻都满足 $\|z^r\|_\infty=O(\rho)$，则 anisotropic forcing 的逐坐标尺度为 $O(\alpha\beta t\rho^3)$，其二范数上界相应带有 $O(\sqrt n)$ 因子。这解释了为什么在实际轨迹先收缩的小中心种子阶段，标量多项式滤波器可以保持很长时间的准确性。

### 5.2 精确模态混合

dual anisotropy 在 $W$ 特征基中的 forcing 为

$$
v_k^\top f_\eta^t
=-2\alpha\sum_\ell a_\ell^t
\left\langle v_k,\eta^t\odot v_\ell\right\rangle.
\tag{41}
$$

若再写 $\eta^t=\sum_jb_j^tv_j$ 及三阶张量

$$
T_{kj\ell}=\sum_{r=1}^nv_{k,r}v_{j,r}v_{\ell,r},
$$

则 (41) 变成

$$
v_k^\top f_\eta^t
=-2\alpha\sum_{j,\ell}T_{kj\ell}b_j^ta_\ell^t.
\tag{42}
$$

这是真正的全谱耦合。box projection 和中心 kick 产生的 $h^t$ 也是坐标非线性，通常同样具有全谱投影。因此，后期正特征值模态重新出现，并不意味着标量前沿已经穿过它们；它们可以在仍处于齐次收缩区时被 forcing 持续制造。

特别地，若直到 $T$ 都有 $\theta_t<\lambda_k$ 且全部相关 multiplier 为正，则每个传播因子满足 $0<G_k(T,s)\le1$，于是

$$
|a_k^T|
\le |a_k^0|+\sum_{t<T}|v_k^\top f^t|,
\tag{43}
$$

并且观测到的超出齐次残余的部分至少需要

$$
R_k(T)\ge
\left[|a_k^T|-G_k(T,0)|a_k^0|\right]_+.
\tag{44}
$$

这给出“未被前沿访问的高模态来自非线性再注入”的严格可检验判据。

### 5.3 一个条件性的有限步饱和 basin

固定 sign pattern $s$，令

$$
a_i^t=s_iz_i^t>0,
$$

并定义该 cut 在顶点 $i$ 的 crossing 与 uncut 权重

$$
c_i=\sum_{j:s_j\ne s_i}w_{ij},
\qquad
u_i=\sum_{j:s_j=s_i}w_{ij}.
$$

### 定理 8 [C]：good orthant 内的有限步二值化

若某时刻 $T$ 存在 $m\in(\delta,1/2)$，使

$$
m\le a_i^T\le\frac12,
\qquad
y_i^T\le0,
\qquad
\gamma_i:=mc_i-\frac{u_i}{2}>0
\quad(\forall i),
\tag{45}
$$

则该 orthant 与下界 $m$ 正向不变，且所有坐标在有限步内达到

$$
z=\frac s2.
$$

若从 $T$ 起中心 kick 被 $m>\delta$ 排除，则步数满足

$$
T_{\rm sat}-T
\le
\max_i
\left\lceil
\frac{1/2-a_i^T}{2\alpha\gamma_i}
\right\rceil.
\tag{46}
$$

**证明。** raw step 精确满足

$$
a_i^{t+1,\rm raw}
=a_i^t+2\alpha\left[
\sum_{j:s_j\ne s_i}w_{ij}a_j^t
-\sum_{j:s_j=s_i}w_{ij}a_j^t
-y_i^ta_i^t
\right].
$$

由 (45) 括号至少为 $\gamma_i>0$；投影只会把坐标截到 $1/2$。又因 $a_i^2-1/4\le0$，dual 保持非正。归纳即得不变性和有限步界。证毕。

条件 $\gamma_i>0$ 自动推出 $c_i>u_i$，所以终点是 strict one-flip stable cut，并满足 $C(s)>M/2$。反过来，任意 strict one-flip cut 都可选择某个 $m<1/2$ 使不等式 $\gamma_i>0$ 成立；定理仍需另外假设实际轨迹已满足 $a_i\ge m$ 与 $y_i\le0$。

这一定理只证明“轨迹已经进入一个 good orthant 的足够深区域后会二值化”，不证明一般轨迹必然进入这样的 orthant。

---

## 6. 二值化为什么通常必须重新注入高谱

令

$$
A=W-\lambda_1I\succeq0,
\qquad
\Lambda=\lambda_n-\lambda_1.
$$

### 定理 9 [U]：binary incompatibility 与最小谱修复

定义

$$
\kappa_A
=\min_{s\in\{\pm1\}^n}\|A^{1/2}s\|.
\tag{47}
$$

则

$$
\boxed{
\kappa_A^2=4(U_{\rm spec}-\operatorname{OPT}).
}
\tag{48}
$$

并且对任意 binary cut，

$$
\boxed{
\operatorname{OPT}-C(s)
=\frac14\left(s^\top As-\kappa_A^2\right).
}
\tag{49}
$$

因此最优 cut 正好是在所有 sign vector 中完成最小 $A$-能量谱修复的点；一个一般 cut 的质量损失，精确等于它比最小必要修复多支付的能量。

更细地，对 $r<n$ 的低谱子空间

$$
E_r=\operatorname{span}\{v_1,\ldots,v_r\},
\qquad
d_r=\min_{s\in\{\pm1\}^n}\|P_{E_r^\perp}s\|,
\tag{50}
$$

$d_r>0$ 当且仅当 $E_r$ 不含 sign vector。任意 binary 状态 $z=s/2$ 都满足

$$
\sum_{k>r}(a_k)^2\ge\frac{d_r^2}{4},
$$

$$
\boxed{
\sum_k(\lambda_k-\lambda_1)a_k^2
\ge
\frac{(\lambda_{r+1}-\lambda_1)d_r^2}{4}.
}
\tag{51}
$$

所以只要早期低谱空间与 hypercube 不兼容，后期趋于 binary 就必然重新产生该空间之外的谱质量。这是必要性定理，不是“所有高模态都应单调增长”的结论。

### 连续方向、flatness 与最终 sign cut

对 $z\ne0$ 定义

$$
u=\sqrt n\frac{z}{\|z\|},
\qquad
s=\operatorname{sign}(z),
\qquad
\phi(z)=\|u-s\|.
$$

三角不等式给出

$$
\sqrt{u^\top Au}+\sqrt\Lambda\,\phi(z)\ge\kappa_A.
\tag{52}
$$

进而

$$
\boxed{
\operatorname{OPT}-C(\operatorname{sign}z)
\le
\frac14\left[
\left(
\sqrt{u^\top Au}+\sqrt\Lambda\,\phi(z)
\right)^2-\kappa_A^2
\right].
}
\tag{53}
$$

式 (53) 把两个阶段第一次严格接起来：前期降低归一化方向的低谱能量，后期降低幅值不平坦度；只有二者的组合接近最小 binary repair 时，才能推出接近最优。

### 非线性 forcing 的必要预算

从过滤结束时刻 $t_0$ 起使用 (15)。若 $T$ 时

$$
D_T=\left\|z^T-\frac{\operatorname{sign}(z^T)}2\right\|,
$$

则

$$
\boxed{
\sum_{t=t_0}^{T-1}
\|A^{1/2}\Phi(T,t+1)f^t\|
\ge
\left[
\frac{\kappa_A}{2}
-\sqrt\Lambda D_T
-\|A^{1/2}\Phi(T,t_0)z^{t_0}\|
\right]_+,
}
\tag{54}
$$

其中 $\Phi(T,s)=\prod_{r=s}^{T-1}[I+2\alpha(\theta_rI-W)]$。若 (54) 方括号中的可计算下界严格为正，则完成相应精度的二值化必然需要正的累计 forcing budget。若只知道 $\kappa_A>0$，但尚未控制 $D_T$ 和齐次传播项，则还不能断言某次有限轨迹的 forcing 已被 integrality gap 强制。G3 中观测到的后期谱展宽与这一必要修复机制一致，但本实验没有计算 (54) 的正下界，故不把它称为已认证的必要 forcing。

---

## 7. 一般图上的完整质量定理

### 定理 10 [U]：任意 diagonal convexifier 的 Bernoulli gap

从连续状态独立采样

$$
B_i\sim\operatorname{Bernoulli}\left(\frac12+z_i\right).
$$

对 bit vector 定义

$$
C(B)=\sum_{i<j}w_{ij}\mathbf1\{B_i\ne B_j\}
=C(2B-\mathbf1).
$$

则

$$
\boxed{
\mathbb EC(B)=\frac M2-z^\top Wz.
}
\tag{55}
$$

对任意 $q$ 满足 $H_q\succeq0$，定义

$$
\delta_q(z)
=z^\top H_qz
+\sum_iq_i\left(\frac14-z_i^2\right).
\tag{56}
$$

有精确 primal-dual gap 分解

$$
\boxed{
U_q-\mathbb EC(B)=\delta_q(z).
}
\tag{57}
$$

两项都非负：第一项是 convexifier 度量下的谱能量，第二项是带坐标权的 saturation deficit。若

$$
\kappa_q^2=\min_{s\in\{\pm1\}^n}s^\top H_qs
=4(U_q-\operatorname{OPT}),
$$

则还有精确式

$$
\boxed{
\operatorname{OPT}-\mathbb EC(B)
=\delta_q(z)-\frac{\kappa_q^2}{4}
\le\delta_q(z).
}
\tag{58}
$$

均匀 shift $q=-\lambda_1\mathbf1$ 给出最简单、无需求 SDP 的势函数

$$
\boxed{
\delta_{\rm spec}(z)
=\sum_k(\lambda_k-\lambda_1)a_k^2
+(-\lambda_1)\left(\frac n4-\|z\|^2\right).
}
\tag{59}
$$

第一项是 weighted spectral tail，第二项是 saturation deficit。

### 定理 11 [U]：多次 Bernoulli rounding 的高概率保证

固定 $z$，独立进行 $R$ 次舍入，则对任意 $\tau>0$，

$$
\boxed{
\Pr\left(
\operatorname{OPT}-\max_{r\le R}C(B^{(r)})\ge\tau
\right)
\le
\min\left\{1,
\left(
\frac{\delta_q(z)-\kappa_q^2/4}{\tau}
\right)^R
\right\}
\le
\min\{1,(\delta_q(z)/\tau)^R\}.
}
\tag{60}
$$

证明只用非负随机变量 $\operatorname{OPT}-C(B)$ 的 Markov 不等式和 $R$ 次独立性。指数 $R$ 不能用于相关样本。

条件期望法还保证存在一个可确定性构造的 cut，其 gap 不超过 (58) 的期望 gap；当前代码使用随机采样而不是该 derandomization。

### 定理 12 [U]：threshold rounding 与 archive

对非负权图，令 $d_i=\sum_jw_{ij}$，则

$$
\boxed{
\operatorname{OPT}-C(\operatorname{sign}z)
\le
\delta_q(z)-\frac{\kappa_q^2}{4}
+\sum_i d_i\left(\frac12-|z_i|\right).
}
\tag{61}
$$

若允许 signed weights，应将 $d_i$ 替换为 $\sum_j|w_{ij}|$。此外

$$
\sum_i d_i\left(\frac12-|z_i|\right)
\le
2d_{\max}\left(\frac n4-\|z\|^2\right).
\tag{62}
$$

若算法保存所有时刻的 threshold incumbent，则最终 archive 对 (61) 右端取历史最好值；它不会被后期可能过量的谱注入破坏。

还有一个不依赖任何谱结构的基线。center-uniform seed 的初始 signs 是独立 Rademacher；若 archive 明确包含 $t=0$ 的 threshold cut，则

$$
\mathbb E C(\operatorname{sign}z^0)=\frac M2
\ge\frac{\operatorname{OPT}}2.
\tag{62a}
$$

因此非负权一般图至少有期望 $1/2$ 的初始 archive 保底。它不能解释接近最优的部分；后续相对提升才由低谱过滤和非线性修复机制解释。

### 非线性修复何时改善质量

式 (59) 还可化简为

$$
\delta_{\rm spec}(z)=z^\top Wz-\frac{n\lambda_1}{4}.
\tag{63}
$$

因此相邻时刻精确满足

$$
\boxed{
\delta_{\rm spec}(z^{t+1})-\delta_{\rm spec}(z^t)
=\sum_k\lambda_k\left[(a_k^{t+1})^2-(a_k^t)^2\right].
}
\tag{64}
$$

这给出非线性离散化的定量判据：

- 负特征模态增长严格改善 Bernoulli quality potential；
- 正特征模态增长付出代价；
- 零附近模态近似是免费的幅值修正；
- 用 (59) 的分解看，修复有益当且仅当 spectral-tail 增量小于 saturation-deficit 的下降量。

所以后期“谱变宽”本身既不是好事也不是坏事；要看它为降低 saturation deficit 支付了多少加权谱代价。

### 定理 13 [C]：完整轨迹的可认证高质量结论

对任一实际时刻 $T$，由 Duhamel 定义

$$
\overline a_k(T)
=|G_k(T,0)a_k^0|+R_k(T).
$$

则

$$
\delta_{\rm spec}(z^T)
\le
\sum_k(\lambda_k-\lambda_1)\overline a_k(T)^2
+(-\lambda_1)\left(\frac n4-\|z^T\|^2\right)
=:\overline\delta_T.
\tag{65}
$$

于是 $R$ 次独立 Bernoulli rounding 满足

$$
\Pr(\operatorname{OPT}-C_{\rm out}\ge\tau)
\le
\min\{1,(\overline\delta_T/\tau)^R\}.
\tag{66}
$$

若“forcing/saturation 使 $\overline\delta_T\le\varepsilon$”这一轨迹事件对随机初始化以概率至少 $1-\zeta$ 成立，则总失败概率至多

$$
\zeta+(\varepsilon/\tau)^R.
\tag{67}
$$

这是一般图上可以诚实声称的完整 end-to-end 定理：convex filtering 控制齐次传播，逐级激活控制相对选择，所有非线性进入 $R_k$，saturation deficit 衡量离散化，最后由舍入转成 cut 质量。它是可认证定理，不是“所有图、所有参数自动使 $\overline\delta_T$ 很小”的定理。

### 输出 cut 的直接 a posteriori 证书

对任意输出 sign vector $s$ 和任意 $y$，(9) 给出

$$
\operatorname{OPT}-C(s)
\le\operatorname{UB}(y)-C(s).
\tag{68}
$$

特别地令

$$
\gamma_i=-s_i(Ws)_i,
\qquad
H_s=W+\operatorname{Diag}\gamma,
$$

则 $H_ss=0$，从而

$$
\boxed{
\operatorname{OPT}-C(s)
\le-\frac n4\lambda_{\min}(H_s).
}
\tag{69}
$$

若 $H_s\succeq0$，该输出就是全局最优。这个证书不要求运行结束时 $y\ge0$。

---

## 8. Gaussian 多项式核的隐式 SDP 解释

这一节给出一般图上早期 threshold cut 为什么可能迅速改善的另一条严格路径，但其随机性条件必须单独说明。

### 定理 14 [U/C]：确定滤波器的 Gaussian hyperplane 等价

设 $p$ 是与随机种子独立的确定性多项式，

$$
g\sim N(0,I),
\qquad
z=p(W)g,
\qquad
K=p(W)^2.
$$

若 $K_{ii}>0$，令

$$
D=\operatorname{Diag}(K_{11},\ldots,K_{nn}),
\qquad
X=D^{-1/2}KD^{-1/2}.
$$

则 $X\succeq0$、$\operatorname{diag}X=\mathbf1$，且 $\operatorname{sign}(z)$ 精确等价于对 SDP 点 $X$ 做一次 Gaussian hyperplane rounding。因此

$$
\mathbb EC(\operatorname{sign}z)
=\sum_{i<j}w_{ij}\frac{\arccos X_{ij}}\pi
\ge
\alpha_{\rm GW}\Phi(X),
\tag{70}
$$

其中

$$
\Phi(X)=\frac M2-\frac14\operatorname{tr}(WX),
\qquad
\alpha_{\rm GW}\simeq0.87856.
$$

当 $\operatorname{OPT}>0$ 时，因此有图依赖的比值下界

$$
\frac{\mathbb EC}{\operatorname{OPT}}
\ge
\alpha_{\rm GW}\frac{\Phi(X)}{U_{\rm spec}}.
\tag{71}
$$

不过 $\Phi(X)$ 可能远低于 SDP optimum，故不能省略该因子。

实际 PDBO 的 $\theta_t$ 依赖同一个随机种子，因而 $p_t$ 也与 $g$ 耦合；条件于随机 $p_t$ 后，$g$ 不再保持 Gaussian。定理 14 对以下情形精确成立：

1. 外生确定的 scalar schedule；
2. 用 pilot trajectory 确定 $p_t$，再用独立 Gaussian 做 kernel rounding；
3. 先另证 schedule concentration，使随机 $p_t$ 可由确定极限替代。

仓库的 iid uniform-cube seed 也不具有精确 arcsine law；它具有第 3 节的协方差和能量过滤定理，但要获得 (70) 还需 universality 条件。

### 推论 14.1 [S]：uniform-leverage ground space

设最小特征空间投影 $P_{\min}$ 的秩为 $r$，且

$$
(P_{\min})_{ii}=\frac rn
\qquad(\forall i).
\tag{72}
$$

若滤波器极限集中到该空间，则

$$
X_\infty=\frac nrP_{\min},
$$

并且

$$
\Phi(X_\infty)=U_{\rm spec}=\operatorname{SDP}(W).
$$

故

$$
\mathbb EC\ge\alpha_{\rm GW}\operatorname{OPT}.
\tag{73}
$$

若 $r=1$ 且 ground eigenvector 是 flat sign vector $s/\sqrt n$，则 rounding 只产生 $s$ 或 $-s$，从而精确恢复全局最优。

这也是 convex stage 与 SDP complementary slackness 的最干净情形：uniform shift $W-\lambda_1I$ 是 dual optimal，$X_\infty$ 是 primal optimal，且二者乘积为零。

---

## 9. 二部、近二部与 planted 图上的恢复

### 定理 15 [S]：谱方向误差到 cut 误差

设目标 cut 为 $s^\star$，并令其正归一化幅值向量为

$$
v_\star=s^\star\odot q,
\qquad
q_i>0,
\qquad
\|q\|=1.
$$

对任意单位方向 $u=z/\|z\|$，记

$$
\varepsilon=\min_{\pm}\|u\mp v_\star\|.
$$

令

$$
\chi=\max_i\frac{d_i}{q_i^2}.
$$

则 threshold 相对目标 cut 的损失满足

$$
\boxed{
C(\operatorname{sign}z)
\ge C(s^\star)-\chi\varepsilon^2.
}
\tag{74}
$$

**证明。** 若顶点 $i$ 被误分类，则 $|u_i-(v_\star)_i|\ge q_i$，故误分类点的总 incident weight 至多

$$
\sum_i d_i\frac{|u_i-(v_\star)_i|^2}{q_i^2}
\le\chi\varepsilon^2.
$$

改变这些顶点最多损失该权重。证毕。

### 推论 15.1 [S]：连通二部图全局最优恢复

对连通非负权二部图，Perron-Frobenius 给出单重最小特征向量

$$
v_1=s^\star\odot q,
\qquad q_i>0,
$$

其中 $s^\star$ 是真实二分。若某时刻

$$
\|P_{v_1^\perp}z^T\|_\infty
<|a_1^T|q_{\min},
\tag{75}
$$

则 threshold 精确恢复 $\pm s^\star$，archive 从此保存全局最优 cut $M$。

对随机连续中心种子，$a_1^0\ne0$ 几乎必然。有限时间、高概率版本需要同时满足：

1. small-ball 事件 $|a_1^0|\ge\gamma$；
2. 逐级激活给出的谱隙增益足以使齐次 residual 满足 (75)；
3. Duhamel forcing 余项小于剩余 sign margin。

用第 2 节的记号，一个完全显式的充分条件是

$$
\left(|G_1(T,0)a_1^0|-R_1(T)\right)q_{\min}
>
\|P_{v_1^\perp}\Phi(T,0)z^0\|_\infty
+\sum_{t<T}
\|P_{v_1^\perp}\Phi(T,t+1)f^t\|_\infty,
\tag{75a}
$$

并要求左侧为正。它同时控制 ground-mode small ball、其 forcing cancellation、其他齐次模态和正交 forcing。

若谱隙为 $\Delta=\lambda_2-\lambda_1$，并且 $q_{\min}^{-1}$、small-ball 代价和相对 forcing budget 都至多按 $n$ 的多项式增长，则一个典型的充分尺度是

$$
\frac{\alpha\Delta^2}{\beta\overline F}
\gtrsim\log\frac{n}{\zeta},
\tag{76}
$$

其中 $\zeta$ 是失败概率；(75a) 才是完整、可核验的恢复条件，(76) 是它在上述非退化条件和常速小步长区间的尺度推论。这是“中心随机初始化 + 足够慢前沿”恢复二部全局最优的条件定理，而不是对任意参数的无条件陈述。

### 推论 15.2 [S]：近二部与 planted 模型

若 planted cut $s^\star$ 的未切边总权至多 $B$，则

$$
C(s^\star)=M-B,
\qquad
\frac{C(s^\star)}{\operatorname{OPT}}\ge1-\frac BM.
\tag{77}
$$

再由 (74)，

$$
\boxed{
\operatorname{OPT}-C(\operatorname{sign}z)
\le B+\chi\varepsilon^2.
}
\tag{78}
$$

Davis-Kahan 只控制 planted 矩阵扰动造成的 eigenspace rotation $O(\|E\|/\operatorname{gap})$；多项式过滤 residual 由传播子控制，Duhamel forcing residual 由 (15)--(17) 控制，最后用三角不等式合并成 $\varepsilon$。任何 planted 随机模型只需另外证明其以高概率具有谱隙与 eigenvector alignment；动力学到 cut 的部分由 (74)--(78) 统一完成。

---

## 10. 一般图上的不可能性边界

### 定理 16 [I]：最小特征向量 cut 可任意差

对 $N\ge2$，令 $H_N$ 为 $K_{N+1}$ 在一个 distinguished clique vertex 上再接一个 leaf。按 distinguished vertex、其余 $N$ 个 clique 顶点的归一化常数方向、leaf 排列，对称商矩阵为

$$
Q_N=
\begin{pmatrix}
0&\sqrt N&1\\
\sqrt N&N-1&0\\
1&0&0
\end{pmatrix}.
$$

其余普通 clique 顶点的零和子空间给出重数 $N-1$ 的特征值 $-1$。删除 leaf 后用 Cauchy interlacing，并结合

$$
\det(\lambda I-Q_N)
=(\lambda-N+1)(\lambda^2-1)-N\lambda
$$

在 $(-\infty,-1)$ 中有根，可知最小特征值是该区间内的单根。若对应商特征向量三分量为 $(a,b,c)$，则

$$
b=\frac{\sqrt N\,a}{\lambda-(N-1)},
\qquad
c=\frac a\lambda.
$$

因为 $\lambda<-1$，$b,c$ 与 $a$ 异号。因此其 sign 把 distinguished vertex 放在一侧，其余 $N$ 个 clique 顶点和 leaf 放在另一侧，故

$$
C(\operatorname{sign}v_1)=N+1,
$$

而

$$
\operatorname{OPT}
=\left\lfloor\frac{(N+1)^2}{4}\right\rfloor+1.
$$

故

$$
\frac{C(\operatorname{sign}v_1)}{\operatorname{OPT}}
\longrightarrow0.
\tag{79}
$$

这严格否定“充分低通或最小特征方向占主导就自动产生一般图好 cut”。它不单独证明带 archive 和非线性 forcing 的完整 PDBO 必然在该图失败；它证明的是低谱对齐本身不足以作为一般图质量定理。

### 任意 binary cut 都可被足够负的 dual 稳定

在 $z=s/2$ 处，若

$$
y_i\le-s_i(Ws)_i
\qquad(\forall i),
\tag{80}
$$

则 raw primal step 向相同边界外推，box projection 返回 $s/2$；又因 $z_i^2=1/4$，dual 不再更新。因此任意 cut，包括很差的 cut，都可成为完整 projected primal-dual fixed point。

同样，非负权 Max-Cut 的 strict one-flip local optimum 最坏只能保证紧的 $1/2$ approximation；这也不足以解释接近 best-known 的结果。

因此以下命题均不成立：

- 最小特征空间占主导 $\Rightarrow$ 一般图高质量；
- 收敛到 binary $\Rightarrow$ 高质量；
- projected fixed point $\Rightarrow$ one-flip stable；
- 后期全谱增长 $\Rightarrow$ 接近最优；
- center-random PDBO $\Rightarrow$ 所有一般图无条件近最优。

合法的最强结论是：一般图上有 trajectory-dependent gap theorem、实例上界证书和结构化图恢复定理。

---

## 11. Gset G3 实验：理论的定量检验

实验使用

$$
n=800,
\quad M=19176,
\quad\lambda_1=-13.4732265,
\quad\lambda_n=48.8823111,
$$

以及 center-uniform 初始化 $\rho=0.05$、batch $=10$、$y^0=15\mathbf1$。基准参数为

$$
\alpha=0.002,
\qquad
\beta=0.02.
$$

条件 (21) 成立。研究脚本为 `research/g3_spectral_dynamics.py`。

### 11.1 三阶段与关键时刻

| 时刻 | $\theta=-\bar y$ | fractionality | Rayleigh | 前 15 模能量 | spectral tail | saturation deficit | $\delta_{\rm spec}$ | archive cut |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -15.000 | 0.9966 | -0.0017 | 1.7484% | 9.05 | 2685.60 | 2694.64 | 9643 |
| 280 | -13.600 | 1.0000 | -12.9187 | 90.0603% | 0.0003 | 2694.64 | 2694.64 | 11296 |
| 600 | -12.000 | 1.0000 | -13.3055 | 99.6385% | 0.0002 | 2694.63 | 2694.63 | 11330 |
| 1030 | -9.856 | 0.9703 | **-13.4250** | **99.9911%** | 0.286 | 2614.67 | 2614.95 | 11338 |
| 1500 | -8.605 | 0.3270 | -11.9263 | 74.3738% | 208.21 | 881.23 | 1089.43 | 11503 |
| 3000 | -7.103 | 0.1263 | -10.8174 | 57.7028% | 464.07 | 340.37 | **804.44** | **11598** |

表中连续谱量均为 10 个 batch 的均值；archive cut 是求解器逐步更新的完整 incumbent，而不是仅在采样时刻比较的局部最好值。

解释如下：

1. **convex contraction**：所有模态先缩小，但任意低模相对高模的比值按 (22) 增加；threshold archive 已在 $t=280$ 从随机 cut 9643 提升到 11296。
2. **sequential activation**：前沿先穿过最低特征值，低模态开始增长而高模态仍收缩；Rayleigh 在标量理论整体仍准确、首次采样观测到 clipping 的同一时刻降到 -13.4250。
3. **nonlinear discretization**：以每 10 步采样的诊断分辨率，$t=1030$ 首次观测到 clipping，$t=1070$ 首次观测到标量预测误差超过 10%，$t=1110$ 首次观测到 forcing 超过 $\|z\|$ 的 1%。真实首次发生时刻最多可早 9 步，瞬时事件也可能被采样遗漏。此后谱重新变宽，但 saturation deficit 的下降远大于 spectral-tail 的增加，故总质量势从 2614.95 降到 804.44。

最终前沿只到 $\theta=-7.10$，所以 $\lambda>-7.10$ 的大量模态从未被标量前沿激活，却在后期明显增长。这正是 (43)--(44) 所描述的 nonlinear forcing，而不是前沿逐个穿过全谱。

### 11.2 $\beta$ 与 $\alpha$ 对照

| 设置 | 最低三谱带 crossing/trough | 首次采样 clip | 首次采样预测误差 $>10\%$ | 首次采样 forcing $>1\%$ | 最低 Rayleigh | 最好 cut | 最小 $\delta_{\rm spec}$ |
|---|---|---:|---:|---:|---|---:|---:|
| $\alpha=.002,\beta=.02,T=3000$ | 440/370, 680/600, 930/830 | 1030 | 1070 | 1110 | -13.42504 @1030 | 11598 | 804.44 |
| $\alpha=.002,\beta=.01,T=6000$ | 870/660, 1360/1160, 1890/1380 | 1720 | 1770 | 1860 | -13.46477 @1710 | 11579 | 895.68 |
| $\alpha=.004,\beta=.02,T=3000$ | 440/330, 680/580, 940/630 | 860 | 890 | 900 | -13.46486 @860 | 11582 | 893.55 |

$\beta$ 减半使 crossing 时间近乎乘 2；$\alpha$ 加倍几乎不改变前沿 crossing，却使增长和 clipping 提前。这分别验证了 (6) 与 (29)--(34)。

$\beta/2$ 的 6000 步与 $2\alpha$ 的 3000 步还满足相同的 $\alpha/\beta$、$\alpha T$ 和 $\beta T$。两者在 clipping 时的前沿分别为 $-10.7037$ 与 $-10.7028$，终点前沿为 $-7.4596$ 与 $-7.4596$，终点前 15 模能量占比为 59.176% 与 59.224%，$\delta_{\rm spec}$ 为 895.68 与 893.55。这个近乎重合的 collapse 说明定理应按累计 primal-dual 时间和谱前沿陈述，而不是按裸迭代次数陈述。

宽 bin 的 trough 不能直接与 bin center crossing 比较。对单独模态 $k=0,5,14$，基准组的 crossing/trough 分别为

$$
310/310,\qquad450/450,\qquad560/550,
$$

都在 10 步采样误差内吻合定理 6。

### 11.3 $\alpha\Delta^2/(\beta F)$ 尺度的直接检验

取 $k=0$ 与 $k=14$，

$$
\Delta=1.224834,
\qquad F\simeq1.
$$

| 设置 | $q=\alpha\Delta^2/(\beta F)$ | 理论主阶 $8q$ | 实测 log-gain | 齐次精确预测 |
|---|---:|---:|---:|---:|
| baseline | 0.1500 | 1.2002 | 1.2248 | 1.2247 |
| $\beta/2$ | 0.3000 | 2.4003 | 2.3879 | 2.4005 |
| $2\alpha$ | 0.3000 | 2.4003 | 2.4449 | 2.4491 |

控制参数使 $q$ 翻倍时，log 相对选择增益也近乎翻倍。这是 (34) 的强定量验证，而不只是“低谱看起来更大”的定性图像。

对更高、已被压到极小的模态，forcing 条件不可省略。例如 $k=36$ 在 $\beta/2$ 与 $2\alpha$ 两个 $q$ 相同的控制组中，实测 `residual RMS / predicted RMS` 分别约为 88.70 和 91.46，齐次 pairwise 预测因而失效；这里保存的是带符号实际 residual，不是 (16) 的绝对值累计上界 $R_k$。但这些高模态在 clipping 前的总实际能量低于 $2.3\times10^{-6}$，所以全状态齐次预测误差仍低于 0.45%。这直接说明逐模态结论必须带 (17)--(19) 一类 residual 条件。

### 11.4 G3 的解析上界与数值名义证书

谱上界公式 (4) 是严格的。用双精度 eigensolver 得到的 $\widehat\lambda_1=-13.4732265$ 代入后，名义数值为

$$
\widehat U_{\rm spec}=12282.6453.
$$

输出 cut 为 11598，因此不使用 best-known 值可得到名义比值

$$
\boxed{
\frac{11598}{\widehat U_{\rm spec}}
=94.426\%.
}
\tag{81}
$$

若用严格下界 $\underline\lambda_1\le\lambda_{\min}(W)$ 替换 $\widehat\lambda_1$，则

$$
\frac{11598}{\operatorname{OPT}}
\ge
\frac{11598}{M/2-n\underline\lambda_1/4}
$$

才是机器严格的比值证书。

最终连续状态的 Bernoulli 期望为

$$
\widehat U_{\rm spec}-\delta_{\rm spec}
=12282.6453-804.4383
=11478.2070.
$$

最终 PDBO dual 经 (9) 修正后的最好名义数值上界是 12342.1242，比均匀谱值更差；candidate-specific shift (69) 给出 13088.6870，也更差。因此这次运行最强的名义数值证书仍来自均匀谱上界。若要求机器可验证的区间证书，应对所有最小特征值计算加入 validated residual/interval 安全裕量。这个负结果同样重要：它表明 PDBO dual 没有自动给出比 uniform shift 更好的 SDP diagonal shift，不能把 convex-stage 机制夸大为“已经求解 SDP”。

---

## 12. 最终结论：已经证明了什么

### 无条件成立

1. convex stage 中的 $y^t$ 是 SDP-dual feasible diagonal shift，raw upper bound 以 $\beta nF_t/16$ 的速度下降；
2. mean-dual 前沿满足精确 fractionality 时钟；
3. 完整轨迹满足逐模态 Duhamel 公式；
4. 齐次比较系统对任意两特征值实施严格低通，并在各自 crossing 处产生 trough；
5. anisotropy 由幅值不平坦精确驱动，dual anisotropy、box projection 与中心 kick 是后期全谱 forcing 的全部来源；
6. 与低谱空间不兼容的 binary 化必须支付正的谱修复能量；
7. Bernoulli、threshold、diagonal-shift 和 spectral upper bound 给出一般图实例证书。

### 需要轨迹条件

1. 要认证实际逐模态激活由齐次谱前沿驱动，一个充分条件是 forcing 小于齐次信号或单步 growth margin；
2. PDBO 得到 $\varepsilon$-近优解，需要某时刻的 $\delta_q$、Duhamel budget 或 a posteriori upper-bound gap 足够小；
3. 有限步饱和需要轨迹已经进入满足 cut-dominance 的 good orthant；
4. Gaussian/GW 解释需要与种子独立的确定滤波器，或额外的 schedule concentration/universality 证明。

### 结构化图上可升级

1. 连通二部图在谱选择、small-ball 和 forcing margin 条件下恢复全局最优；
2. 近二部与 planted 图得到 $B+\chi\varepsilon^2$ 加性 gap；
3. uniform-leverage ground space 在确定滤波器配合独立 Gaussian/spherical seed 的条件下得到 $0.87856$ 期望保证；
4. rank-one flat ground state 在相应 scalar-filter shadowing 与 sign-margin 条件下得到精确全局恢复。

### 不能声称

plain PDBO 配合中心随机初始化在每个一般图上都无条件接近最优。反例说明低谱对齐、二值收敛和 fixed-point stability 分别都不足以推出该结论。

因此，当前理论已经能够**部分且定量地解释** PDBO 为什么在 G3 一类实例上表现好：凸阶段提供全谱筛选与 SDP-dual 边界，逐级激活提供按 $\alpha\Delta^2/(\beta F)$ 缩放的低谱优势，非线性阶段产生与 binary incompatibility 所要求的谱修复一致的再填充，archive 与 gap potential 保留修复代价和饱和收益平衡最好的 cut。它还不是任意图的统一近优理论，但已经从“现象解释”推进到了可逐项实验检验、可给实例质量证书的完整机制理论。
