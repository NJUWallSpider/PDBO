# 研究报告：PDBO 在 Max-Cut 上的谱动力学

## 问题定义

Max-Cut 的问题定义如下：
$$
    \min_{x \in \{0, 1\}^n} \quad f(x) = x^\top W x - \mathbf{1}^\top W x.
$$

其对应的 Langrangian Relaxation 为
$$
    L(x, y) = x^\top W x - \mathbf{1}^\top W x + y^\top(\text{Diag}(x) - I)x.
$$

所以
$$
    \begin{aligned}

    \nabla_x L(x, y)
    &= 2Wx - W\mathbf 1 + 2\text{Diag}(y)x - y \\
    &= 2W(x - \frac{1}{2} \mathbf 1) + 2 \text{Diag}(y)(x - \frac{1}{2} \mathbf 1).

    \end{aligned}
$$

记 $z = x - \frac{1}{2} \mathbf 1$，直观上 $z$ 是 $x$ 偏离超立方体中心的距离。用 $z$ 替换 $x$，上式可以简化为
$$
    \nabla_x L(x, y) = 2(W - \text{Diag}(y)) z.
$$

所以在 PDBO 中 Max-Cut 的精确迭代公式为

$$

    z^{t+1} = \Pi_{[-\frac{1}{2}, \frac{1}{2}]}\left[I - 2\alpha(W - \text{Diag} (y^t)) \right]z^t\\
    y_i^{t+1} = y_i^t + \beta\left[(z_i^t)^2 - \frac{1}{4} \right].

$$

为了方便分析，对 $z$ 进行随机中心初始化
$$
    z^0=\rho\xi,\qquad
    \xi_i\overset{\rm iid}{\sim}{\rm Unif}[-1,1],\qquad
    0<\rho<\frac12.
$$

### Max-Cut 中的特征值

在 Max-Cut 中，$W$ 满足 $W = W^\top \ne 0, W_{ii} = 0$，设 $W$ 的特征值为 $\lambda_1 \le \cdots \le \lambda_n$。

因为 $W$ 是实对称矩阵，所以它的特征值都是实数，并且 $\text{tr}(W) = \sum_{i=1}^n \lambda_i = \sum_{i=1}^n W_ii = 0$，因此 $\sum_{i=1}^n \lambda_i = 0$。

$W$ 可以对角化为
$$
    W = Q\text{diag}(\lambda_1, \ldots, \lambda_n) Q^\top ,
$$
因为 $W \ne 0$，所以至少有一个特征值非零，但它们的和又是零，所以必然有
$$
    \lambda_1 < 0 , \qquad \lambda_n > 0.
$$

## 中心动力学

我们把 dual 拆成均匀部分和非均匀部分，定义
$$
    y^t = m_t \mathbf 1 + \eta^t, \qquad m_t = \frac 1n \mathbf 1^\top y^t, \qquad \mathbf 1^\top \eta^t = 0,
$$
其中 $m_t \mathbf 1$ 是所有坐标共有的 uniform shift，$\eta^t$描述各坐标相对于平均值的偏差。

暂时忽略 $z$ 的 box projection，gradient step 可以写成

$$
    z^{t+1} = [I - 2\alpha(W + m_t \mathbf 1)]z^t - 2\alpha \text{Diag}(\eta^t)z^t.
$$

再把 box projection 和 center kick 相对于 raw step 的差统一记为 $h^t$，就得到
$$
\boxed{z^{t+1}=[I - 2\alpha(W + m_t \mathbf 1)]z^t+f^t,}
$$
其中
$$
\boxed{f^t=-2\alpha\operatorname{Diag}(\eta^t)z^t+h^t.}
$$

### mean-dual 部分

我们先假设 $f^t = 0$，设 $W$ 的特征分解为 $Wu_k=\lambda_ku_k$, 并将 $z^t=\sum_k a_k^tu_k$。
那么每个谱分量独立演化：
$$
\boxed{
a_k^{t+1}
=
\left[1 - 2\alpha(m_t + \lambda_k)\right]a_k^t.
}
$$
因此 $m_t$ 像一条随时间移动的谱阈值：
- $|1 - 2\alpha(m_t + \lambda_k)| > 1$ 的模态倾向于被放大；
- $|1 - 2\alpha(m_t + \lambda_k)| < 1$ 的模态倾向于被抑制。

我们希望高特征值分量在初始迭代时被抑制，这指导 $\alpha$ 的选择应当满足

$$
    \alpha < \frac{1}{\bar y^0 + \lambda_n}
$$

如果 $\alpha$ 大于此阈值，算法会实效，并且根据实验结论，当 $\alpha$ 接近此阈值时，算法得到的解是最好的。


