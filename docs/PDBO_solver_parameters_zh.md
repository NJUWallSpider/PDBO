# PDBO Solver 运行参数手册

本文档对应 `src/main.py` 的命令行入口，默认从仓库根目录运行：

```bash
python src/main.py [参数]
```

安装运行依赖：

```bash
pip install -e .
```

查看当前代码实际支持的参数：

```bash
python src/main.py --help
```

## 1. 最常用命令

随机正则图 MIS：

```bash
python src/main.py \
  --task mis --graph reg --n 10000 --d 3 --seed 0 \
  --batch 10 --max_iters 5000 \
  --lr_x 0.02 --lr_y 0.02 --dual_init 5 \
  --timelimit 180
```

Gset Max-Cut，手工设置步长：

```bash
python src/main.py \
  --task mc --graph Gset --Gset_id 1 \
  --batch 100 --max_iters 5000 \
  --lr_x 0.025 --lr_y 0.025 --dual_init 6 \
  --conditional_rounding
```

Gset Max-Cut，谱自适应初始化与步长：

```bash
python src/main.py \
  --task mc --graph Gset --Gset_id 70 \
  --batch 10 --max_iters 3000 --rho 0.05 \
  --lr_y 0.02 \
  --dual_init_mode spectral --dual_burn_in 0 --dual_psd_margin 0 \
  --primal_lr_mode spectral --spectral_step_fraction 0.5 \
  --conditional_rounding
```

LABS：

```bash
python src/main.py \
  --task labs --labs_n 47 --labs_penalty 10000 \
  --batch 100 --max_iters 5000 \
  --lr_x 0.02 --lr_y 0.02 --dual_init 5
```

布尔参数使用 `argparse.BooleanOptionalAction`，例如可用
`--verbose/--no-verbose`、`--save/--no-save` 等形式。

## 2. 参数总览

### 2.1 问题与数据

| 参数 | 类型/可选值 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--task` | `mis`, `mc`, `labs` | `mc` | 选择最大独立集、Max-Cut 或低自相关二进制序列问题。 |
| `--graph` | `reg`, `Gset` | `Gset` | `mis`/`mc` 的图来源；对 `labs` 无效。 |
| `--Gset_id` | 整数 | `3` | 读取 `instance/Gset/G<ID>.txt`；仅在 `graph=Gset` 时使用。 |
| `--n` | 整数 | `50000` | 随机正则图的顶点数；仅在 `graph=reg` 时使用。 |
| `--d` | 整数 | `100` | 随机正则图的度；仅在 `graph=reg` 时使用。必须满足 NetworkX 正则图生成条件，例如 `0 <= d < n` 且 `n*d` 为偶数。 |
| `--seed` | 整数 | `0` | 随机图、PDBO 初始化、随机舍入及部分谱估计的随机种子。 |
| `--penalty` | 浮点数 | `4` | MIS 中违反独立集约束的边惩罚；只对 `task=mis` 有效。 |
| `--labs_n` | 整数 | `47` | LABS 序列长度。 |
| `--labs_penalty` | 浮点数 | `10000` | LABS QUBO 辅助变量约束的惩罚系数。 |

三类问题最终都转为如下最小化 QUBO：

```text
minimize  x^T Q x + c^T x,    x in {0, 1}^n
```

因此 Max-Cut 的内部目标值通常为负数：终端的 `best=-C` 对应 cut
值 `C`。MIS 在解可行时通常有 `best=-|S|`。LABS 的 QUBO 还含一个
常数 `objective_offset`，终端 `best` 是未加常数项的内部目标；保存文件
中的 `incumbents` 会加回该常数，`labs energy` 才是原始 LABS 能量。

### 2.2 规模、迭代和停止

| 参数 | 类型 | 默认值 | 约束/语义 |
| --- | --- | --- | --- |
| `--batch` | 整数 | `10` | 并行 primal-dual 轨迹数，必须 `>= 1`。更大通常增加解的多样性，也线性增加主要内存与计算量。 |
| `--max_iters` | 整数 | `5000` | 每个阶段的最大迭代数，必须 `>= 0`。启用 `--restart` 后最多执行两个阶段，即最多 `2 * max_iters` 步。 |
| `--timelimit` | 浮点数或不填 | `None` | 总求解时间上限，单位秒。每次完整迭代后检查；动画中等待人工操作的时间不计入。精修时间也不计入。 |
| `--patience` | 整数或不填 | `None` | 连续多少次迭代没有达到 `min_delta` 的 incumbent 改进后停止，必须 `>= 1`。重启后重新计数。 |
| `--min_delta` | 浮点数 | `0.0` | 判定“有效改进”的最小目标下降量，用于 history、patience 和最终随机舍入接受判断。通常保持 `0`。 |
| `--check_every` | 整数 | `100` | 日志和 Max-Cut 动态上界的检查间隔，必须 `>= 1`。它不控制 patience 的检查频率。 |
| `--dual_patience_threshold` | 非负浮点数或不填 | `None` | 启用基于 dual 均值的提前停止；每 `--dual_patience_every` 次迭代计算一次下降量，若小于该阈值则停止。 |
| `--dual_patience_every` | 正整数 | `100` | dual patience 的检查间隔；与日志 `--check_every` 独立。 |
| `--restart` | 布尔 | `False` | 第一阶段结束后，以当前最佳二进制解填充整个 primal batch，并把 dual 重置为有效 `dual_init`，再运行一个阶段。全局 `timelimit` 仍跨越两个阶段。 |

停止原因会显示为：

- `max_iters`：正常耗尽迭代预算；
- `timelimit`：达到总时间限制；
- `patience`：incumbent 长时间没有足够大的改进。

### 2.3 Primal 参数

| 参数 | 类型/可选值 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--lr_x` | 正浮点数 | `0.02` | 手工 primal 步长；仅在 `primal_lr_mode=configured` 时作为实际步长。 |
| `--primal_lr_mode` | `configured`, `spectral` | `configured` | 使用 `lr_x`，或根据 `W` 的最大特征值自动算步长。 |
| `--spectral_step_fraction` | 浮点数 | `0.5` | 谱步长系数，必须严格满足 `0 < value < 1`；仅在 `primal_lr_mode=spectral` 时使用。 |
| `--primal_init` | `center_uniform`, `uniform`, `half`, `binary` | `center_uniform` | primal batch 的初始化方式，见下表。 |
| `--rho` | 浮点数 | `0.05` | `center_uniform` 的中心半径，必须满足 `0 < rho <= 0.5`；其他初始化模式忽略它。 |

初始化模式：

| `primal_init` | 初始化 |
| --- | --- |
| `center_uniform` | `x ~ Uniform(0.5-rho, 0.5+rho)`，当前 CLI 默认。 |
| `uniform` | `x ~ Uniform(0, 1)`。 |
| `half` | 所有坐标精确设为 `0.5`。 |
| `binary` | 每个坐标独立、等概率初始化为 `0` 或 `1`。 |

谱模式的实际步长为：

```text
lr_x_effective = spectral_step_fraction /
                 (lambda_max(W) + dual_init_effective)
W = (Q + Q.T) / 2
```

使用谱模式时，命令行的 `--lr_x` 仍需为正数以通过参数校验，但不会作为
实际步长。实际值会在 verbose 输出的 `effective_parameters` 和保存文件的
`effective primal lr` 中给出。

### 2.4 Dual 与二进制约束参数

| 参数 | 类型/可选值 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--lr_y` | 正浮点数 | `0.02` | dual 固定步长。 |
| `--dual_init` | 正浮点数 | `5` | 手工 dual 初值；仅在 `dual_init_mode=configured` 时作为实际初值。 |
| `--dual_init_mode` | `configured`, `spectral` | `configured` | 使用 `dual_init`，或根据 `W` 的最小特征值自动初始化。 |
| `--dual_burn_in` | 非负整数 | `0` | 谱初始化中的理想化中心 burn-in 步数；只在 `dual_init_mode=spectral` 时使用。 |
| `--dual_psd_margin` | 非负浮点数 | `0.0` | 谱初始化相对 PSD 边界增加的非负裕量。 |
| `--delta` | 浮点数 | `1e-8` | 论文 Algorithm 1 的中心扰动容差，要求 `0 < delta < 0.5`。 |
| `--g_type` | `quadratic`, `absolute` | `quadratic` | 二进制等式函数：原始算法 `x^2-x`，或实验性 `abs(x-1/2)-1/2`。 |

谱 dual 初值为：

```text
dual_init_effective = -lambda_min(W)
                      + dual_psd_margin
                      + lr_y * dual_burn_in / 4
```

实现会向上取一个可表示的 `float32` 数。使用谱模式时，命令行的
`--dual_init` 仍需为正数以通过校验，但实际值由上式决定。`delta` 也会被
提升到不小于 `float32` 在 `0.5` 附近的间距，因此默认请求值 `1e-8` 的
实际值约为 `5.96e-8`；verbose 输出会显示实际值。

`g_type=absolute` 时，只有坐标精确等于 `0.5` 才可能触发中心扰动；结果
目录会从 `result/pdbo/` 改为 `result/pdbo_absolute/`。

### 2.5 舍入

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--rounding` | `nearest` / `bernoulli` | `nearest` | 控制初始状态及每次迭代生成 incumbent 候选的舍入方式。`nearest` 使用就近舍入；`bernoulli` 独立地以当前 `x_i` 为取 1 概率采样，并使用 solver 的随机种子。 |
| `--rounding_samples` | 非负整数 | `0` | 求解结束后，对最终 relaxed batch 的每条轨迹额外生成这么多组 Bernoulli 样本。候选总数为 `rounding_samples * batch`。 |
| `--conditional_rounding` | 布尔 | `False` | 跟踪全轨迹中最好的 Bernoulli 期望目标，并在结束时用条件期望法确定性地二值化。 |

推荐先启用 `--conditional_rounding`。`rounding_samples` 会增加
`O(rounding_samples * batch * n)` 的随机样本存储与评分开销，大实例应从
较小值开始。

### 2.6 输出、保存和可视化

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--verbose` | 布尔 | `True` | 打印有效参数、谱分布、谱窗口和 checkpoint；checkpoint 包含当前最优目标值以及 dual 的 `dual_mean`、`dual_min`、`dual_max`。使用 `--no-verbose` 可关闭这些日志。 |
| `--save` | 布尔 | `False` | 将结果写入 `result/`。若目标文件已存在，程序打印 `PASS <path>` 并直接跳过本次求解，不覆盖文件。 |
| `--spectral_animation` | 布尔 | `False` | 打开谱模态投影、dual 变量 `y`、中心化变量 `z=x-0.5` 的分布动画，以及 batch 均值 `L(x,y)` 和 `f(x)` 随全局迭代数 `t` 变化的折线图，需要 Matplotlib。`n <= 2000` 使用完整特征分解，大图使用 Lanczos-Ritz 近似模态。 |
| `--spectral_animation_bins` | 正整数 | `50` | 动画的特征值、`y` 和 `z` 直方图分箱数。 |
| `--spectral_animation_modes` | 正整数 | `128` | 大图动画使用的 Lanczos-Ritz 模态数；越大则谱分辨率、初始化时间和刷新开销越高。 |
| `--spectral_animation_every` | 正整数 | `100` | 连续运行时每隔多少次迭代刷新四个动画面板；折线图也按这些时刻采样。手动 `Next step` 模式仍然每步刷新。 |

可视化依赖：

```bash
pip install -e ".[visualization]"
```

G64 等大图可直接使用近似谱动画：

```bash
python src/main.py --task mc --graph Gset --Gset_id 64 \
  --spectral_animation --spectral_animation_modes 128
```

动画初始为暂停状态：`Next step` 前进一步，`Run` 连续运行，运行中按钮变为
`Pause`。最终窗口会保持打开直到手动关闭。

保存路径规则：

```text
quadratic + Gset: result/pdbo/<task>/Gset/G<ID>.txt
quadratic + reg:  result/pdbo/<task>/reg/n=<n>d=<d>s=<seed>.txt
absolute + ...:   result/pdbo_absolute/...
LABS:             result/<solver_dir>/labs/n=<labs_n>_p=<labs_penalty>_s=<seed>.txt
```

注意：Gset 文件名不包含 seed、batch、步长等超参数；随机图文件名也不包含
大多数 solver 参数。因此使用 `--save` 做参数扫描时，不同配置可能指向同一
文件，后续运行会被 `PASS` 跳过。需要保留多组实验时，应在运行间重命名结果、
使用独立输出目录，或扩展 `result_path()` 的命名规则。

保存内容包括 incumbent 改进记录、时间记录、总求解时间、停止原因、扰动次数、
实际 primal 步长和 dual 初值。启用条件舍入时也会保存对应期望目标。MC/MIS
当前不在结果文件中写出最终 bitstring；LABS 始终写出原序列 bitstring。

## 3. 如何选参数

### 稳妥起点

```text
batch=10
max_iters=3000~5000
primal_init=center_uniform
rho=0.05
lr_y=0.02
dual_init_mode=spectral
primal_lr_mode=spectral
spectral_step_fraction=0.5
conditional_rounding=true
```

谱模式适合先建立无需手工缩放的基线。若需要复现已有脚本或已有实验，再使用
`configured` 模式明确指定 `lr_x` 和 `dual_init`。

### 速度与质量的主要旋钮

| 目标 | 优先调整 | 代价/风险 |
| --- | --- | --- |
| 增加解的多样性 | 增大 `batch` | 内存和每步耗时近似线性增加；Max-Cut bound 更贵。 |
| 给搜索更多时间 | 增大 `max_iters` 或 `timelimit` | 运行时间增加。 |
| 减少日志/bound 开销 | 增大 `check_every`，必要时 `--no-verbose` | bound 轨迹更稀；`--no-verbose` 本身不会停算 bound。 |
| 改善最终二进制解 | `--conditional_rounding` | 增加结束阶段的确定性舍入耗时。 |
| 增加随机舍入探索 | 增大 `rounding_samples` | 增加内存与评分开销。 |
| 提前停止 | 设置 `timelimit` 或 `patience` | 可能过早结束尚未改善的轨迹。 |
| 两阶段开发 | `--restart` | 最大迭代量翻倍；仍受同一个总 `timelimit` 限制。 |

手工调步长时，一次只改变一项并记录 seed。若目标发散、频繁撞边界或不同 seed
波动很大，优先减小 `lr_x`；dual 演化过快时减小 `lr_y`。`labs_penalty` 和 MIS
`penalty` 不是通用“越大越好”的质量参数：过小可能产生不可行偏好，过大会放大
QUBO 系数尺度，此时通常也要相应减小 primal 步长或改用谱步长。

## 4. 常见组合

限时运行并关闭详细日志：

```bash
python src/main.py \
  --task mc --graph Gset --Gset_id 70 \
  --batch 10 --max_iters 100000 --timelimit 300 \
  --dual_init_mode spectral --primal_lr_mode spectral \
  --conditional_rounding \
  --check_every 500 --no-verbose
```

基于 incumbent 停滞提前停止：

```bash
python src/main.py \
  --task mc --graph Gset --Gset_id 1 \
  --patience 1000 --min_delta 0 --check_every 100
```

保存结果：

```bash
python src/main.py \
  --task mis --graph reg --n 10000 --d 3 --seed 0 \
  --conditional_rounding --save
```

绝对值 integrality function 实验：

```bash
python src/main.py \
  --task mc --graph Gset --Gset_id 1 \
  --g_type absolute --primal_init center_uniform --rho 0.05
```

## 5. 参数适用性速查

| 参数组 | MIS | Max-Cut | LABS |
| --- | --- | --- | --- |
| `graph`, `Gset_id`, `n`, `d` | 是 | 是 | 否 |
| `penalty` | 是 | 否 | 否 |
| `labs_n`, `labs_penalty` | 否 | 否 | 是 |
| 通用迭代、步长、初始化、舍入参数 | 是 | 是 | 是 |

## 6. 当前接口注意事项

- 参数名区分大小写，Gset 使用 `--Gset_id`，不是 `--gset_id`。
- 必须从仓库根目录运行；Gset 路径按 `./instance/Gset/` 相对解析。
- `--save` 不是“覆盖保存”，而是“目标不存在时保存”。
- `--max_iters 0` 合法，可用于只评估初始化/结束舍入流程。
