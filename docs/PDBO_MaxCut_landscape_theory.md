# PDBO 在 Max-Cut 上的动态景观与谱机制

## 一套分层、可核查的理论草稿

本文研究 PDBO 在 Max-Cut 上的求解机制。重点不是再次证明二进制可行性，而是回答三个问题：

1. dual 如何使 primal landscape 从凸到非凸，并让谱模态依次失稳；
2. 在何种条件下轨迹会对齐原始矩阵 $W$ 的最小特征向量，以及离开中心时 landscape 已有多深的负曲率；
3. 进入盒子边界后，完整对角 dual 如何产生 active-set/residual spectral dynamics，从而超越一次性的 $\operatorname{sign}(v_1)$ 舍入。

文中把结论分为三类：

- **无条件结论**：由代数、谱理论或投影 KKT 条件直接得到；
- **条件定理**：在明确的标量 dual、谱隙、慢时间尺度或固定 active set 假设下成立；
- **尚待闭合的环节**：从首次谱逃逸到一般图上最终高质量二进制捕获的全局概率保证。

---

## 1. 基本形式

设 $W=W^\top\in\mathbb R^{n\times n}$、$W_{ii}=0$，边权暂取非负，总边权

$$
M=\sum_{i<j}w_{ij}.
$$

用 $s\in\{\pm1\}^n$ 表示 cut，则

$$
C(s)=\frac M2-\frac14s^\top Ws.
$$

PDBO 使用 $x\in[0,1]^n$、$g(x_i)=x_i^2-x_i$，其 Lagrangian 为

$$
L(x,y)=x^\top Wx-\mathbf1^\top Wx+\sum_i y_i(x_i^2-x_i).
$$

令中心变量

$$
z=2x-\mathbf1\in[-1,1]^n.
$$

则

$$
L(z,y)
=-\frac M2+
\frac14\left[z^\top A(y)z-\mathbf1^\top y\right],
\qquad
A(y):=W+\operatorname{Diag}(y).
$$

因此

$$
\nabla_x^2L(x,y)=2A(y),
\qquad
\nabla_z^2L(z,y)=\frac12A(y).
$$

两者仅相差一个正常数；以下把 $A(y)$ 称为 primal Hessian 的等价曲率矩阵。

记

$$
\eta:=\frac\beta4.
$$

在不触发论文中特殊中心扰动时，离散更新精确等价于

$$
z^{t+1}
=\Pi_{[-1,1]^n}\bigl(z^t-2\alpha A_tz^t\bigr),
\qquad
A_t:=A(y^t),
\tag{1}
$$

$$
y^{t+1}
=y^t-\eta\bigl(\mathbf1-z^t\odot z^t\bigr).
\tag{2}
$$

由此得到精确的“分数性时钟”恒等式

$$
y_i^t
=y_i^0-\eta\sum_{\tau<t}\left(1-(z_i^\tau)^2\right)
=y_i^0-\beta\sum_{\tau<t}x_i^\tau(1-x_i^\tau).
\tag{3}
$$

它说明 dual 不是外生 annealing schedule，而是每个坐标累计分数性的积分反馈。

### 与 Max-Cut SDP dual 的精确关系

标准 Max-Cut SDP 可写为

$$
\max_{X\succeq0,\ \operatorname{diag}(X)=\mathbf1}
\left\{
\frac M2-\frac14\langle W,X\rangle
\right\}.
$$

其 dual 为

$$
\boxed{
\min_y
\left\{
B(y):=\frac M2+\frac14\mathbf1^\top y:
W+\operatorname{Diag}(y)\succeq0
\right\}.
}
\tag{3a}
$$

所以 $A(y)\succeq0$ 不只是“使目标凸”的条件，同时正是 SDP dual feasibility。PDBO 从这个 cone 内部出发，沿 dual 下降改善 $B(y)$，然后主动穿出该 cone 并发生 rank-one 谱对称破缺。PDBO 并不等价于求解 SDP，但早期 landscape 与 SDP dual slack matrix 完全相同。

---

## 2. 谱序贯失稳

### 定理 2.1：Loewner 单调非凸化（无条件）

对任意 PDBO 轨迹，

$$
A_{t+1}
=A_t-\eta\operatorname{Diag}\left(\mathbf1-z^t\odot z^t\right)
\preceq A_t.
\tag{4}
$$

因此，对按升序排列的全部特征值，

$$
\lambda_k(A_{t+1})\le \lambda_k(A_t),
\qquad k=1,\ldots,n.
\tag{5}
$$

特别地，完整空间中的负特征值数量不减。

#### 证明

因为 $z^t\in[-1,1]^n$，

$$
\operatorname{Diag}(\mathbf1-z^t\odot z^t)\succeq0.
$$

式 (4) 随即成立；式 (5) 由 Weyl 单调性或 Courant--Fischer 极小极大表征得到。$\square$

这个定理比“dual 逐渐减小”更强：它说明当前 primal landscape 的每一个有序主曲率都只会下降。不过特征向量可以旋转，所以“第 $k$ 条特征值曲线”不等于始终跟踪同一个固定空间方向。

### 定理 2.2：标量 dual 下的精确模态分离（条件定理）

设在时间区间 $0\le t<T$ 内：

1. 投影未激活；
2. $y^t=\mu_t\mathbf1$；
3. $Wv_k=\lambda_kv_k$，且 $\lambda_1<\lambda_2\le\cdots\le\lambda_n$；
4. 步长足够小，使
   $$
   0\le 1-2\alpha(\lambda_n+\mu_t).
   \tag{6}
   $$

将

$$
z^t=\sum_{k=1}^na_k^tv_k
$$

代入更新，则

$$
a_k^{t+1}
=\left(1-2\alpha(\lambda_k+\mu_t)\right)a_k^t.
\tag{7}
$$

所以第 $k$ 个模态在

$$
\mu_t=-\lambda_k
$$

处由收缩转为扩张。若 $\mu_t$ 单调下降且特征值简单，则失稳阈值严格按

$$
-\lambda_1>-\lambda_2>\cdots
$$

的顺序被穿过。

更进一步，设谱隙

$$
\Delta:=\lambda_2-\lambda_1>0,
$$

并且在整个区间内 $\mu_t\ge-\lambda_2$。定义

$$
D_T:=\max_{0\le t<T}\left[1-2\alpha(\lambda_1+\mu_t)\right].
$$

只要 $a_1^0\ne0$，就有

$$
\frac{\|P_{v_1^\perp}z^T\|_2}{|v_1^\top z^T|}
\le
\frac{\|P_{v_1^\perp}z^0\|_2}{|v_1^\top z^0|}
\exp\left(-\frac{2\alpha\Delta}{D_T}T\right).
\tag{8}
$$

#### 证明

式 (7) 由 $W$ 的正交特征分解直接得到。对任意 $k\ge2$，令

$$
m_{k,t}:=1-2\alpha(\lambda_k+\mu_t).
$$

由假设，$0\le m_{k,t}\le m_{1,t}$，且

$$
\frac{m_{k,t}}{m_{1,t}}
=1-\frac{2\alpha(\lambda_k-\lambda_1)}{m_{1,t}}
\le
\exp\left(-\frac{2\alpha\Delta}{D_T}\right).
$$

逐时刻相乘，再对 $k\ge2$ 的正交分量求二范数，即得式 (8)。$\square$

式 (8) 给出了真正的“谱过滤”定理：在第二模态失稳前，轨迹相对 $v_1$ 的夹角以指数速度缩小。

### 推论 2.3：谱符号解的 flatness 质量界

令

$$
s^{\rm spec}:=\operatorname{sign}(v_1),
$$

暂假设 $v_{1i}\ne0$。因为

$$
\cos\angle\left(\frac{s^{\rm spec}}{\sqrt n},v_1\right)
=\frac{\|v_1\|_1}{\sqrt n},
$$

Rayleigh 商分解给出

$$
\frac1n(s^{\rm spec})^\top Ws^{\rm spec}
\le
\lambda_1
+(\lambda_n-\lambda_1)
\left(
1-\frac{\|v_1\|_1^2}{n}
\right).
$$

结合特征值上界

$$
C^*\le\frac M2-\frac n4\lambda_1,
$$

得到

$$
\boxed{
C^*-C(s^{\rm spec})
\le
\frac n4(\lambda_n-\lambda_1)
\left(
1-\frac{\|v_1\|_1^2}{n}
\right).
}
\tag{8a}
$$

特别地，若最小特征向量完全平坦，

$$
|v_{1i}|=\frac1{\sqrt n},
\qquad\forall i,
$$

那么 $s^{\rm spec}=\sqrt n\,v_1$ 达到最小特征值 Rayleigh 下界，因而是全局最优 cut。这个特例说明：第一谱模态能否直接给出好 cut，不只取决于谱隙，也取决于特征向量的 flatness/delocalization。

还要注意，二范数对齐不自动推出所有坐标符号一致。若

$$
\left\|
\frac{z}{\|z\|}
-\sigma v_1
\right\|_2\le\varepsilon,
\qquad \sigma\in\{\pm1\},
$$

那么只能保证所有满足

$$
|v_{1i}|>\varepsilon
$$

的高置信度坐标具有

$$
\operatorname{sign}(z_i)=\sigma\operatorname{sign}(v_{1i}).
$$

$v_{1i}$ 很小的坐标恰好是后续 residual dynamics 最有可能重新决定的部分。

### 高概率推论

设初始方向来自球面均匀分布，等价地

$$
z^0=r_0\frac g{\|g\|_2},
\qquad
g\sim N(0,I_n),
\qquad 0<r_0<1.
$$

由高斯范数集中与一维高斯小球概率，对任意 $\delta\in(0,1/2)$，以至少 $1-2\delta$ 的概率，

$$
\frac{\|P_{v_1^\perp}z^0\|_2}{|v_1^\top z^0|}
\le
\frac{\sqrt{n-1}+\sqrt{2\log(1/\delta)}}{\sqrt{\pi/2}\,\delta}.
\tag{9}
$$

结合式 (8)，若希望

$$
\frac{\|P_{v_1^\perp}z^T\|_2}{|v_1^\top z^T|}\le\varepsilon_{\rm align},
$$

充分条件为

$$
T\ge
\frac{D_T}{2\alpha\Delta}
\log\left(
\frac{\sqrt{n-1}+\sqrt{2\log(1/\delta)}}
{\sqrt{\pi/2}\,\delta\,\varepsilon_{\rm align}}
\right).
\tag{10}
$$

对一般连续随机初始化，只需把式 (9) 替换为相应的 anti-concentration 条件。并行 $B$ 个独立初值会把“所有轨迹的 $v_1$ 投影都过小”的概率以 $B$ 次方衰减。

### 谱隙与 dual 速度条件

在中心附近，标量 dual 每步约下降

$$
\eta=\frac\beta4.
$$

从第一模态失稳到第二模态失稳大约有

$$
T_{1\to2}\approx\frac\Delta\eta=\frac{4\Delta}{\beta}
$$

步。要在这段一维不稳定窗口内完成式 (10) 所需的对齐，保守地需要

$$
\frac{8\alpha\Delta^2}{D_T\beta}
\gtrsim
\log\left(
\frac{R_0}{\varepsilon_{\rm align}}
\right),
$$

即

$$
\boxed{
\frac\beta\alpha
\lesssim
\frac{8\Delta^2}
{D_T\log(R_0/\varepsilon_{\rm align})}
}
\tag{11}
$$

其中 $R_0=\|P_{v_1^\perp}z^0\|/|v_1^\top z^0|$。这给出了“谱隙越小，dual 必须越慢”的定量解释。

---

## 3. 早期 near-scalar dual 的可验证条件

完整 PDBO 并不保持 $y_i^t$ 完全相同。定义

$$
\mu_t:=\frac1n\mathbf1^\top y^t,
\qquad
e^t:=y^t-\mu_t\mathbf1,
\qquad
E_t:=\operatorname{Diag}(e^t).
$$

则

$$
A_t=W+\mu_tI+E_t.
$$

由 dual 更新可精确得到

$$
e_i^{t+1}-e_i^t
=\eta\left((z_i^t)^2-\frac1n\|z^t\|_2^2\right).
$$

若 $e^0=0$，则

$$
\boxed{
\|E_t\|_2
=\|e^t\|_\infty
\le
\eta\sum_{\tau<t}\|z^\tau\|_\infty^2.
}
\tag{12}
$$

因此，“早期 dual 接近标量”不必作为完全不可观测的假设；它可由累计的中心偏离量直接验证。

若

$$
\sup_{t\le T}\|E_t\|_2\le\varepsilon_y<\frac\Delta2,
$$

则 Davis--Kahan 扰动界给出当前最小特征向量 $u_1(A_t)$ 与 $v_1(W)$ 的夹角满足

$$
\sin\angle(u_1(A_t),v_1)
\le
\frac{2\varepsilon_y}{\Delta}.
\tag{13}
$$

这说明只要式 (12) 的右端远小于谱隙，最先出现的负曲率方向仍然接近原始 $v_1$。

### 连续时间鲁棒对齐引理

考虑投影未激活时的连续近似

$$
\dot z=-2(W+\mu(t)I+E(t))z,
\qquad
\|E(t)\|_2\le\varepsilon_y.
$$

去除对所有方向相同的标量因子 $\mu(t)I$，并写

$$
z=a v_1+r,
\qquad r\perp v_1,
\qquad R=\frac{\|r\|}{|a|}.
$$

若在某个时刻 $R\le1$，且 $\varepsilon_y\le\Delta/8$，则只要 $a$ 不穿零，

$$
\dot R
\le-2\Delta R+2\varepsilon_y(1+R)^2
\le-2\Delta R+8\varepsilon_y.
$$

从而

$$
\boxed{
R(t)
\le
R(t_0)e^{-2\Delta(t-t_0)}
+\frac{4\varepsilon_y}{\Delta}
\left(1-e^{-2\Delta(t-t_0)}\right).
}
\tag{14}
$$

证明要点如下。去除公共标量缩放并把 $\lambda_1I$ 一并平移掉后，

$$
\dot a=-2v_1^\top E(av_1+r),
$$

$$
\frac d{dt}\|r\|
\le
-2\Delta\|r\|
+2\varepsilon_y(|a|+\|r\|).
$$

在 $a$ 不穿零的区间内对 $R=\|r\|/|a|$ 求导，得到

$$
\dot R
\le
-2\Delta R
+2\varepsilon_y(1+R)^2.
$$

当 $R\le1$ 时应用 Grönwall 不等式即得式 (14)；若初始 $R\le1$ 且 $4\varepsilon_y/\Delta<1$，这个锥形区域保持不变。

因此，少量对角异质性不会摧毁谱对齐，只会产生 $O(\varepsilon_y/\Delta)$ 的对齐误差地板。式 (12)--(14) 联合构成了从“完整 dual”退化到“早期近标量谱动力学”的一个可证明桥梁。

---

## 4. 曲率--质量夹逼

对任意二进制 cut $s\in\{\pm1\}^n$，定义翻转第 $i$ 个顶点的 cut 增益

$$
G_i(s)
:=C(s^{(i)})-C(s)
=s_i(Ws)_i.
\tag{15}
$$

给定任意 $y$，定义

$$
A=W+\operatorname{Diag}(y),
\qquad
\kappa(y):=[-\lambda_{\min}(A)]_+,
\tag{16}
$$

以及顶点 $s$ 的投影固定点残差

$$
r_i(s,y):=[G_i(s)+y_i]_+,
\qquad
R(s,y):=\sum_i r_i(s,y).
\tag{17}
$$

### 引理 4.1：顶点固定点与 one-flip 条件

对任意 $\alpha>0$，顶点 $s$ 是映射

$$
T_y(z)=\Pi_{[-1,1]^n}(z-2\alpha Az)
$$

的固定点，当且仅当

$$
G_i(s)+y_i\le0,
\qquad\forall i.
\tag{18}
$$

若所有不等式严格成立，则在非退化条件下 $s$ 是局部吸引顶点。

特别地，若 $y_i\ge0$，则任何固定顶点都满足

$$
G_i(s)\le-y_i\le0,
$$

因而是具有翻转 margin 的 one-flip local optimum。

### 定理 4.2：曲率--质量夹逼（无条件）

令 $C^*$ 为 Max-Cut 最优值，并定义

$$
B(y):=\frac M2+\frac14\mathbf1^\top y.
$$

则对任意二进制 $s$ 和任意 $y$，

$$
\boxed{
B(y)-\frac14R(s,y)
\le C(s)\le C^*
\le B(y)+\frac n4\kappa(y).
}
\tag{19}
$$

因此

$$
\boxed{
C^*-C(s)
\le
\frac14\left[n\kappa(y)+R(s,y)\right].
}
\tag{20}
$$

若 $s$ 是精确投影固定点，则 $R(s,y)=0$，从而

$$
\boxed{
C^*-C(s)\le\frac n4\kappa(y).
}
\tag{21}
$$

#### 证明

由 $G_i(s)+y_i\le r_i(s,y)$，求和得到

$$
s^\top Ws+\mathbf1^\top y\le R(s,y).
$$

所以

$$
C(s)=\frac M2-\frac14s^\top Ws
\ge B(y)-\frac14R(s,y).
$$

另一方面，

$$
A+\kappa(y)I\succeq0.
$$

因而对任意二进制 $q$，

$$
q^\top Wq+\mathbf1^\top y+n\kappa(y)
=q^\top[A+\kappa(y)I]q\ge0.
$$

即

$$
C(q)\le B(y)+\frac n4\kappa(y).
$$

对所有 $q$ 取最大即得上界。$\square$

### 更紧的对角修复版本

定义把当前 Hessian 修复到 PSD cone 所需的最小非负对角预算

$$
\rho(A)
:=\min_{d\ge0}
\left\{\mathbf1^\top d:
A+\operatorname{Diag}(d)\succeq0
\right\}.
\tag{22}
$$

因为 $d=\kappa(y)\mathbf1$ 总是可行，

$$
\rho(A)\le n\kappa(y).
$$

重复定理 4.2 的证明可得更强界

$$
\boxed{
C^*-C(s)
\le\frac14\left[\rho(A)+R(s,y)\right].
}
\tag{23}
$$

式 (23) 比单一最小特征值更接近完整 SDP dual geometry。

### “坏 basin 晚出生”的严格含义

若 $s$ 是精确固定点且其最优差距为

$$
\Delta_C=C^*-C(s),
$$

那么式 (21) 反向给出必要条件

$$
\boxed{
\kappa(y)\ge\frac{4\Delta_C}{n}.
}
\tag{24}
$$

因此，差距较大的稳定二进制吸引子不可能在浅非凸阶段出现。这是“PDBO 不是逃离已经存在的所有坏 basin，而是让坏 basin 晚出现”的严格版本。

### 分数性暴露与错误锁定

由式 (3) 和固定点条件，若最终顶点 $s$ 在坐标 $i$ 上仍有正 one-flip 增益 $G_i(s)>0$，则必有

$$
\sum_{t<T}x_i^t(1-x_i^t)
\ge
\frac{y_i^0+G_i(s)}{\beta}.
\tag{25}
$$

又因为 $x_i(1-x_i)\le1/4$，所以这种错误锁定至少需要

$$
T\ge\frac{4(y_i^0+G_i(s))}{\beta}
\tag{26}
$$

步。明显违反 one-flip optimality 的坐标只有在长时间保持分数并使 $y_i$ 足够负后才可能稳定。

---

## 5. 从高概率对齐到逃逸曲率

需要区分两个时刻：

- **谱逃逸时刻**：轨迹离开中心邻域，或首次触碰盒子边界；
- **二进制捕获时刻**：轨迹到达一个稳定二进制顶点。

第二个时刻必然不早于第一个时刻。曲率--质量定理直接适用于第二个时刻，而早期谱分析首先控制的是第一个时刻。把两者混为一谈会产生错误的全局质量结论。

### 定理 5.1：理想标量动力学的精确解

考虑投影尚未激活、dual 近似匀速下降的连续模型

$$
\dot z=-2(W+\mu(t)I)z,
\qquad
\dot\mu=-\varepsilon,
\qquad
\varepsilon:=\frac{\beta}{4\alpha}.
\tag{27}
$$

设

$$
\mu(0)=-\lambda_1+d_0,
\qquad d_0\ge0,
$$

其中 $d_0$ 是初始凸化余量。定义首次穿过凸性边界后的负曲率参数

$$
\kappa:=-(\lambda_1+\mu)=\varepsilon t-d_0.
$$

于是 $\kappa=0$ 对应 $\mu=-\lambda_1$。若

$$
z(0)=\sum_ka_k(0)v_k,
\qquad
\Delta_k:=\lambda_k-\lambda_1,
$$

则对 $\kappa\ge0$，有精确公式

$$
\boxed{
a_1(\kappa)
=a_1(0)
\exp\left(\frac{\kappa^2-d_0^2}{\varepsilon}\right),
}
\tag{28}
$$

$$
\boxed{
a_k(\kappa)
=a_k(0)
\exp\left(
\frac{\kappa^2-d_0^2
-2\Delta_k(d_0+\kappa)}
{\varepsilon}
\right),
\quad k\ge2.
}
\tag{29}
$$

因此

$$
\boxed{
\frac{|a_k(\kappa)|}{|a_1(\kappa)|}
=
\frac{|a_k(0)|}{|a_1(0)|}
\exp\left(
-\frac{2\Delta_k(d_0+\kappa)}{\varepsilon}
\right).
}
\tag{30}
$$

#### 证明

每个模态满足

$$
\dot a_k
=-2(\lambda_k+\mu(t))a_k
=-2(\Delta_k+d_0-\varepsilon t)a_k.
$$

直接积分，并使用 $t=(d_0+\kappa)/\varepsilon$，即可得到式 (28)--(30)。$\square$

式 (30) 是高概率对齐的核心：谱隙 $\Delta$ 越大、dual 越慢，非主模态相对 $v_1$ 衰减越强。

### 逃逸曲率的精确 leading-order 公式

假设到首次触碰边界前，轨迹已经充分对齐 $v_1$。若把“逃逸”定义为

$$
|a_1|=a_{\rm tar},
$$

例如首次达到 $\|z\|_\infty=r<1$ 时可取

$$
a_{\rm tar}=\frac{r}{\|v_1\|_\infty},
$$

那么由式 (28)，

$$
\boxed{
\kappa_{\rm esc}^2
=d_0^2
+\varepsilon
\log\frac{a_{\rm tar}}{|a_1(0)|}.
}
\tag{31}
$$

若改用穿过凸性边界时的有效 seed

$$
\xi_c:=|a_1(\kappa=0)|
=|a_1(0)|e^{-d_0^2/\varepsilon},
$$

则

$$
\boxed{
\kappa_{\rm esc}
=
\sqrt{
\varepsilon\log\frac{a_{\rm tar}}{\xi_c}
}.
}
\tag{32}
$$

这里出现了一个重要但容易忽略的现象：

> 慢 dual 一方面提供更长的谱过滤时间，另一方面也会在凸阶段把绝对 seed 压得更小。

若没有在临界点附近重新注入扰动，则令 $\varepsilon\to0$ 并不必然使 $\kappa_{\rm esc}\to0$。由式 (31)，

$$
\kappa_{\rm esc}\to d_0.
$$

因此，过度凸化 $d_0\gg0$ 会导致明显的 dynamic bifurcation delay：轨迹在凸性边界前被压缩多少，往往要在边界后获得相近的负曲率深度才能恢复。

### 第一模态在第二模态失稳前逃逸的条件

第二模态在

$$
\kappa=\Delta=\lambda_2-\lambda_1
$$

时失稳。因此，纯第一模态逃逸的充分条件是

$$
\boxed{
d_0^2
+\varepsilon
\log\frac{a_{\rm tar}}{|a_1(0)|}
<\Delta^2.
}
\tag{33}
$$

这比单独要求“$\beta/\alpha$ 小”更精确：

- 小 $\varepsilon=\beta/(4\alpha)$ 有利；
- 大谱隙 $\Delta$ 有利；
- 初始 $v_1$ seed 大有利；
- 但初始凸化余量 $d_0$ 过大不利。

特别地，若 $d_0\ge\Delta$，那么在没有临界点新扰动的理想标量模型中，第一模态不可能在第二模态失稳前恢复到原始量级。

### 高概率版本

假设随机初始化满足 anti-concentration 条件：存在尺度 $\sigma$ 和常数 $C_{\rm ac}$，使

$$
\mathbb P\left(
|v_1^\top z^0|\le \sigma u
\right)
\le C_{\rm ac}u.
\tag{34}
$$

则以至少 $1-\delta$ 的概率，

$$
|a_1(0)|
\ge
\frac{\sigma\delta}{C_{\rm ac}}.
$$

代入式 (31) 得到

$$
\boxed{
\kappa_{\rm esc}^2
\le
d_0^2
+\varepsilon
\log\left(
\frac{C_{\rm ac}a_{\rm tar}}
{\sigma\delta}
\right).
}
\tag{35}
$$

若有 $B$ 个独立 batch，则“所有 seed 都小于 $\sigma u$”的概率至多为

$$
(C_{\rm ac}u)^B.
$$

因此 batch 的理论作用之一，是降低所有轨迹在第一模态上都几乎无 seed 的概率。

### 一个直接的算法设计推论

式 (31) 暗示：扰动最好在

$$
\lambda_{\min}(A_t)\approx0
$$

时触发，而不是只在 $y_i\le0$ 且 $z_i\approx0$ 时触发。若在首次谱穿零时向近零特征空间注入幅度 $\xi_{\rm new}$，那么后续逃逸满足

$$
\kappa_{\rm esc}
\approx
\sqrt{
\varepsilon
\log\frac{a_{\rm tar}}{\xi_{\rm new}}
},
$$

初始过度凸化余量 $d_0$ 不再出现在 leading term 中。这给出了一个可实验检验的 **spectral-triggered perturbation** 变体。

### 离散时间夹逼

在第一模态失稳以后，若

$$
\kappa_j\simeq \eta j,
\qquad
a_1^{j+1}=(1+2\alpha\kappa_j)a_1^j,
$$

并且 $2\alpha\kappa_j\le1$，利用

$$
\frac u2\le\log(1+u)\le u,
\qquad 0\le u\le1,
$$

可得从 seed $\xi_c$ 境长到目标 $a_{\rm tar}$ 所需步数 $T_{\rm esc}$ 满足同阶界

$$
\sqrt{
\frac{\log(a_{\rm tar}/\xi_c)}
{\alpha\eta}
}
\ \lesssim\
T_{\rm esc}
\ \lesssim\
1+
\sqrt{
\frac{2\log(a_{\rm tar}/\xi_c)}
{\alpha\eta}
}.
\tag{36}
$$

所以

$$
\kappa_{\rm esc}
=O\left(
\sqrt{
\frac{\beta}{\alpha}
\log\frac{a_{\rm tar}}{\xi_c}
}
\right),
\tag{37}
$$

与连续时间公式一致。

### 尚不能从式 (31) 直接推出最终质量

$\kappa_{\rm esc}$ 控制的是离开中心或首次触碰边界时的曲率深度。曲率--质量界要求一个稳定二进制顶点。因此还需要控制首次逃逸到最终捕获之间的额外曲率增长。

若定义

$$
K_{\rm post}
:=T_{\rm binary}-T_{\rm esc},
$$

由每一步

$$
\|A_{t+1}-A_t\|_2\le\eta
$$

可得粗略条件界

$$
\kappa_{\rm binary}
\le
\kappa_{\rm esc}+\eta K_{\rm post}.
\tag{38}
$$

于是对最终近似固定点 $s$，

$$
C^*-C(s)
\le
\frac14
\left[
n(\kappa_{\rm esc}+\eta K_{\rm post})
+R(s,y)
\right].
\tag{39}
$$

一般图上真正困难的环节正是控制 $K_{\rm post}$。下一节的 active-set 理论描述这段后期动力学。

---

## 6. Active-set 与 residual spectral dynamics

### 6.1 真正相关的后期 Hessian

当部分坐标位于边界时，完整 $A_t$ 的负特征向量可能指向盒子外部，不再是可行移动方向。

设在某一时间区间内，活动坐标集合为 $B$，其符号固定为

$$
z_B=s_B\in\{\pm1\}^{|B|},
$$

其余自由坐标为 $F=[n]\setminus B$。可行切空间中的二阶曲率由主子矩阵

$$
A_F(y_F)
:=W_{FF}+\operatorname{Diag}(y_F)
\tag{40}
$$

决定，而不是完整 $A_t$。

在 active set 和符号保持不变的区间内，令

$$
b_F:=W_{FB}s_B.
\tag{41}
$$

则自由变量满足精确的 reduced dynamics

$$
z_F^{t+1}
=z_F^t
-2\alpha\left[
A_F(y_F^t)z_F^t+b_F
\right],
\tag{42}
$$

$$
y_F^{t+1}
=y_F^t-\eta(\mathbf1-z_F^t\odot z_F^t).
\tag{43}
$$

活动变量在保持 $|z_i|=1$ 时满足 $y_i^{t+1}=y_i^t$。

### 6.2 准静态分数支路

固定 $y_F$。若

$$
A_F(y_F)\succ0,
$$

则不考虑盒边界时，唯一稳定平衡为

$$
\boxed{
z_F^*(y_F)
=-A_F(y_F)^{-1}b_F.
}
\tag{44}
$$

它说明后期轨迹不是再次回到中心，而是围绕由已固定变量产生的条件场 $b_F$ 形成一个移动的分数支路。

### 定理 6.1：稳定支路的慢追踪

考虑连续 reduced dynamics

$$
\dot z_F=-2(A_F(t)z_F+b_F).
$$

假设在时间区间内：

$$
A_F(t)\succeq mI,\qquad m>0,
$$

$$
\|z_F^*(t)\|_2\le R,
\qquad
\|\dot A_F(t)\|_2\le\nu,
$$

并且 $z_F^*(t)$ 与盒边界保持正距离。则

$$
\boxed{
\|z_F(t)-z_F^*(t)\|_2
\le
e^{-2m(t-t_0)}
\|z_F(t_0)-z_F^*(t_0)\|_2
+\frac{\nu R}{2m^2}.
}
\tag{45}
$$

#### 证明

由

$$
A_Fz_F^*+b_F=0
$$

求导得到

$$
\dot z_F^*
=-A_F^{-1}\dot A_Fz_F^*,
\qquad
\|\dot z_F^*\|
\le\frac{\nu R}{m}.
$$

令 $e=z_F-z_F^*$，则

$$
\dot e=-2A_Fe-\dot z_F^*.
$$

所以

$$
\frac d{dt}\|e\|
\le-2m\|e\|+\frac{\nu R}{m}.
$$

积分即得式 (45)。$\square$

这给出了 active-set 阶段使用慢--快系统理论的精确入口：只要 residual Hessian 与奇异性保持距离，primal 会以 $O(\nu/m^2)$ 误差追踪移动平衡。

### 定理 6.2：Residual spectral blow-up

设沿某个固定 active-set 阶段，

$$
A_F(\tau)\succ0,\qquad \tau<\tau_c,
$$

并且最小特征值

$$
\theta_1(\tau)\downarrow0
$$

是简单的，其他特征值满足

$$
\theta_2(\tau)\ge\gamma>0.
$$

设归一化最小特征向量 $u_1(\tau)\to u_1^c$，且

$$
c:=(u_1^c)^\top b_F\ne0.
$$

则准静态支路满足

$$
\boxed{
z_F^*(\tau)
=
-\frac{c}{\theta_1(\tau)}u_1^c
+O(1).
}
\tag{46}
$$

因此

$$
\frac{z_F^*(\tau)}{\|z_F^*(\tau)\|}
\longrightarrow
-\operatorname{sign}(c)u_1^c.
\tag{47}
$$

由于右端范数发散，**准静态平衡支路**必在 $\theta_1=0$ 以前触碰至少一个新边界坐标。若定理 6.1 的 slow-tracking 误差在该碰撞前始终小于支路到边界的 transversal margin，则真实轨迹也会在零特征值以前触碰边界；否则真实轨迹可能像第一阶段一样出现 dynamic bifurcation delay。

#### 证明

对 $A_F$ 做谱分解：

$$
-A_F^{-1}b_F
=-\sum_k
\frac{u_k^\top b_F}{\theta_k}u_k.
$$

第一项按 $1/\theta_1$ 发散，其余项由 $\theta_k\ge\gamma$ 一致有界。由 $c\ne0$ 即得结论。 $\square$

这个定理给出比“一次性 $\operatorname{sign}(v_1)$”更丰富的机制：

- 第一个阶段 $B=\varnothing$、$b_F=0$，需要随机 seed 选择 $\pm v_1$；
- 一旦部分坐标固定，$b_F\ne0$，后续 residual mode 的方向通常由已固定变量产生的场确定，而不再是完全随机的正负对称；
- 每个阶段使用的是新的 $A_F$ 最小特征向量，而不是原始 $W$ 的 $v_1$。

### 6.3 Active-set 改变产生谱“重置”

在固定 $F$ 的阶段内，

$$
A_F^{t+1}\preceq A_F^t,
$$

所以 residual 特征值仍然单调下降。

但若某个自由变量 $i$ 到达边界并从 $F$ 中删除，新 Hessian

$$
A_{F\setminus\{i\}}
$$

是原 Hessian 的主子矩阵。由 Cauchy interlacing，

$$
\lambda_k(A_F)
\le
\lambda_k(A_{F\setminus\{i\}})
\le
\lambda_{k+1}(A_F).
\tag{48}
$$

特别地，删除当前负模态的重要坐标后，新的 residual 最小特征值可能重新变正。于是后期可行 landscape 可以经历

$$
\text{residual convex}
\longrightarrow
\text{residual singular}
\longrightarrow
\text{新坐标固定}
\longrightarrow
\text{residual 再凸化}
$$

的多阶段循环，即使完整 $A_t$ 的特征值始终单调下降。

这是一种由盒投影自动实现的谱 deflation。

### 6.4 Active set 不是不可逆的

若某个活动坐标 $i$ 当前取值为 $s_i$，则它保持在边界的条件是

$$
s_i(A_tz^t)_i\le0.
\tag{49}
$$

如果其他变量变化使

$$
s_i(A_tz^t)_i>0,
$$

下一次投影梯度更新会把它拉回盒子内部。因而早期决定可以被纠正；只有具有严格负 margin 的活动变量才会稳定保持。

---

## 7. 为什么 residual dynamics 能超越 $\operatorname{sign}(v_1)$

不能无条件证明 PDBO 总是优于 $\operatorname{sign}(v_1)$：错误的早期 active set、过快的 dual 或过负的 $y_i$ 都可能把次优符号锁住。一般图上的统一 dominance 结论也会与问题的 NP-hard 性冲突。

但可以给出严格的条件性改进定理。

固定已经决定的集合 $B$ 和符号 $s_B$。剩余二进制 completion 的二次目标为

$$
q_F(s_F)
=s_F^\top W_{FF}s_F
+2b_F^\top s_F
+\text{constant},
\qquad
b_F=W_{FB}s_B.
\tag{50}
$$

### 定理 7.1：独立 residual 集合上的最优 completion

若

$$
W_{FF}=0,
$$

则给定 $s_B$ 的全局最优 completion 为

$$
\boxed{
s_i^*=-\operatorname{sign}(b_i),
\qquad i\in F,\ b_i\ne0.
}
\tag{51}
$$

并且若谱舍入 completion $s_F^{\rm spec}$ 在集合

$$
J=\{i\in F:s_i^{\rm spec}=\operatorname{sign}(b_i)\}
$$

上与条件最优方向相反，则

$$
\boxed{
C(s_B,s_F^*)-C(s_B,s_F^{\rm spec})
=\sum_{i\in J}|b_i|.
}
\tag{52}
$$

#### 证明

当 $W_{FF}=0$ 时，式 (50) 化为 $2\sum_i b_is_i+\text{constant}$，逐坐标最小化即得式 (51)。把错误符号从 $\operatorname{sign}(b_i)$ 翻为其相反数，使 $q_F$ 减少 $4|b_i|$，所以 cut 增加 $|b_i|$。求和即得式 (52)。$\square$

在连续动力学中，当 $z_i=0$ 时，

$$
\dot z_i=-2b_i,
$$

dual 项 $y_iz_i$ 消失。因此 PDBO 在中心穿越处会自动选择式 (51) 的方向；dual 只负责随后极化。

### 定理 7.2：边界场占优时的唯一最优 completion

更一般地，定义边界场占优 margin

$$
m_i
:=
|b_i|
-\sum_{j\in F}|W_{ij}|.
\tag{53}
$$

若对所有 $i\in F$，

$$
m_i>0,
\tag{54}
$$

则式 (51) 仍然是给定 $s_B$ 的唯一全局最优 completion。

此外，对任意错误取值

$$
s_i=\operatorname{sign}(b_i),
$$

其 one-flip 改善量至少为 $m_i$。因此，只要在符号决定阶段

$$
y_i>-m_i,
\tag{55}
$$

这个错误符号不可能成为 PDBO 的稳定边界坐标。

若从任意 completion 逐个修正与式 (51) 不同的坐标，每次 cut 至少改善 $m_i$，所以相对于谱舍入，

$$
C(s_B,s_F^*)-C(s_B,s_F^{\rm spec})
\ge
\sum_{i:s_i^{\rm spec}\ne s_i^*}m_i.
\tag{56}
$$

#### 证明

对任意其他自由变量赋值，

$$
\left|(W_{FF}s_F)_i\right|
\le\sum_{j\in F}|W_{ij}|
<|b_i|.
$$

所以局部场

$$
(W_{FF}s_F)_i+b_i
$$

始终与 $b_i$ 同号。若 $s_i=\operatorname{sign}(b_i)$，翻转 $i$ 的 cut 增益至少为 $m_i>0$，因此任何包含错误符号的 completion 都不是最优；反复修正最终唯一到达式 (51)。PDBO 稳定条件为 $G_i+y_i\le0$，而错误符号满足 $G_i\ge m_i$，所以式 (55) 排除错误锁定。式 (56) 由逐次翻转增益求和得到。 $\square$

这些定理揭示了 full PDBO 超越一次谱舍入的一个严格机制：

1. $v_1$ 首先决定高置信度核心变量；
2. 这些变量产生条件边界场 $b_F$；
3. 原始 $v_1$ 上幅度很小、符号不可靠的变量，可以根据真实条件 cut 增益重新决定；
4. 只要 dual 尚未负到掩盖该改善 margin，错误谱符号不能被锁住。

---

## 8. 条件性“递归谱舍入”定理

把前述结论组合起来，可以得到以下研究级条件定理。

### 定理 8.1：Adiabatic recursive spectral rounding（条件性）

考虑 PDBO 的连续极限。假设：

1. 初始 $A_0\succ0$，且 $W$ 的 $\lambda_1$ 简单、有谱隙 $\Delta>0$；
2. 首次边界事件前，式 (12) 给出的 dual 异质性满足
   $$
   \sup_t\|E_t\|\ll\Delta;
   $$
3. 式 (33) 成立，因此第一模态在第二模态失稳前触碰边界；
4. 每个 active-set 阶段中，当前 residual Hessian 的最小特征值简单，其他特征值与零保持统一间隔；
5. 准静态分数支路与盒边界的碰撞是 transversal 的；
6. 新固定坐标具有足够 margin，使 active set 在该阶段内保持稳定；
7. dual 时间尺度足够慢，使定理 6.1 的追踪误差小于谱与边界 margin。

则轨迹具有如下结构：

- 第一个 active-set 事件的方向在 $O(\|E\|/\Delta)$ 误差内由 $\pm v_1(W)$ 决定；
- 在第 $\ell$ 个固定 active-set 阶段，轨迹追踪
  $$
  z_{F_\ell}^*
  =-A_{F_\ell}^{-1}b_{F_\ell};
  $$
- 当 residual 最小特征值趋近于零时，下一次边界事件沿
  $$
  -\operatorname{sign}
  \bigl((u_1^{(\ell)})^\top b_{F_\ell}\bigr)
  u_1^{(\ell)}
  $$
  发生；
- active-set 更新后，主子矩阵 interlacing 产生新的 residual spectrum，过程递归继续，直至二进制化。

因此，理想慢时间尺度下的 full PDBO 不是一次谱舍入，而是一个由边界条件驱动的 **recursive conditional spectral rounding**。

#### 证明思路

第一阶段由定理 2.2、式 (12)--(14) 和定理 5.1 控制。固定 active set 后，定理 6.1 保证对稳定支路的追踪；定理 6.2 决定下一次谱软化与边界碰撞方向；式 (48) 描述 active-set 更新后的谱变化。对 active-set 阶段归纳即可。完整严格证明需要在每次失去 normal hyperbolicity 的邻域使用动态分岔匹配，并利用 transversal/margin 假设排除滑动、同时多坐标退化和 active-set 抖动。

这个定理的假设较强，但每一项都是可测量、可通过实验诊断的，不是不可证伪的叙事。

---

## 9. 当前理论已经完成与尚未完成的边界

### 已经可以正式写入论文的结果

1. centered-coordinate 精确动力学与分数性时钟，式 (1)--(3)；
2. Hessian 的 Loewner 单调下降与 Morse index 单调性，定理 2.1；
3. 标量 dual 下的模态阈值、指数对齐与谱隙--速度条件，定理 2.2；
4. early near-scalar 的轨迹可验证条件和 $O(\varepsilon_y/\Delta)$ 鲁棒误差，式 (12)--(14)；
5. 曲率--质量夹逼及对角 PSD repair 版本，式 (19)--(23)；
6. 错误 one-flip 锁定所需的累计分数性预算，式 (25)--(26)；
7. 理想标量模型下的精确逃逸曲率和过度凸化 trade-off，式 (28)--(35)；
8. 固定 active set 下的 reduced dynamics、慢支路追踪、residual spectral blow-up 与 interlacing，定理 6.1--6.2；
9. residual 独立或边界场占优条件下，严格优于错误谱 completion 的定理 7.1--7.2。

### 尚未无条件闭合的部分

最困难的目标仍然是：对一般 Gset 类型图，在不过强的结构假设下，高概率控制

$$
K_{\rm post}
=T_{\rm binary}-T_{\rm esc}
$$

以及最终

$$
\kappa_{\rm binary}.
$$

这是从“第一模态选择”走到“全局近似质量保证”的真正缺口。一般情况下不能仅由 $\lambda_1$ 简单和有谱隙推出，因为：

- $v_1$ 可能高度局部化或含有极小坐标；
- 多个变量可能同时触碰边界；
- active set 可能释放再进入；
- residual 谱隙可能关闭；
- $y_i$ 可能在 primal 完成条件调整前变得过负；
- 最终稳定顶点对应的 full-space $\kappa$ 可能已经较大。

可行的下一步不是声称已经得到一般高概率全局近似比，而是证明以下更窄的结果之一：

1. **delocalized eigenvector 类图**：假设
   $$
   c/\sqrt n\le |v_{1i}|\le C/\sqrt n,
   $$
   控制第一阶段同时极化比例和 $K_{\rm post}$；
2. **边界场占优 residual 类图**：用定理 7.2 保证剩余变量快速完成最优条件 completion；
3. **随机图或 planted model**：利用随机矩阵谱隙、特征向量 delocalization 和局部场集中证明高概率捕获；
4. **a posteriori certificate**：不预先保证所有实例，而是在每条实际轨迹上计算
   $$
   \rho(A_t)+R(s_t,y_t)
   $$
   给出实例级最优差距证书。

---

## 10. 最关键的实验判据

理论应通过以下轨迹量检验：

1. 首次穿零时间
   $$
   t_c=\min\{t:\lambda_{\min}(A_t)\le0\};
   $$
2. dual 异质性
   $$
   \|y^t-\bar y^t\mathbf1\|_\infty
   \quad\text{与}\quad
   \eta\sum_{\tau<t}\|z^\tau\|_\infty^2;
   $$
3. 原始谱对齐
   $$
   \frac{|v_1^\top z^t|}{\|z^t\|};
   $$
4. 当前负子空间对齐
   $$
   \frac{\|P_-(A_t)z^t\|}{\|z^t\|};
   $$
5. 逃逸曲率 $\kappa_{\rm esc}$、二进制捕获曲率 $\kappa_{\rm binary}$ 与二者间隔；
6. 曲率--质量证书
   $$
   \frac14[\rho(A_t)+R(s_t,y_t)];
   $$
7. 每次 active-set 更新前后的 residual 最小特征值，验证 interlacing/reset；
8. residual 最小特征向量与下一批取整坐标的重合度；
9. 比较 direct GD、PDBO-s、full PDBO 和 spectral-triggered perturbation；
10. 对 incumbent 必须使用产生它的同一 batch 的 $y$，不能使用 batch 均值。

---

## 11. 一句话理论总结

PDBO 在 Max-Cut 上可被理解为：

> 从 SDP-dual-feasible 的凸二次景观出发，以累计分数性驱动对角曲率下降；早期通过谱隙完成最小特征模态过滤，随后经由盒边界触发 active-set deflation，并在一系列受已固定变量条件场驱动的 residual spectral problems 上递归舍入。浅非凸阶段可稳定的二进制点由曲率--质量夹逼保证必须高质量，而明显错误的局部决定只有在对应 dual 已积累足够负值后才可能被锁定。
