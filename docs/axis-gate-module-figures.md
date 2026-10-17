# 轴向与门控上采样模块结构图

结构依据：`ultralytics/nn/modules/block.py` 中的 `C2fAxisResidual`、`C2fAxisBounded` 和 `GatedInterpolationUpsample`。复现命令：

```bash
python paper/draw_axis_gate_modules.py
```

每张图均提供可编辑文本的 SVG 矢量版与 PNG 预览，保存于 `paper/figures/`。图中的张量尺寸省略 batch 维，默认通道比例与参数取自当前配置。图未替换论文里旧 DME/CAFR 的配图。

| 图文件（均有 `.svg` 与 `.png`） | 对应实现与配置 |
| --- | --- |
| `axis_residual_module` | `C2fAxisResidual`，当前优先测试的 `yolov8n-fabric-axis-interp.yaml` 使用的轴向模块 |
| `gated_interpolation_module` | `GatedInterpolationUpsample`，上述配置中的两处上采样 |
| `axis_bounded_module` | `C2fAxisBounded`，`yolov8n-fabric-axis-bounded-interp.yaml` 中的有界轴向候选 |

## 轴向模块

上部为 C2f 内的位置，按 nano 配置的两个 Bottleneck 展示；下部展开最后一个隐藏分支的处理。`c=C_out/2` 是 C2f 的隐藏通道数，`q=max(1,floor(c/4))` 是参与轴向调制的通道数。因此，25% 指**最后隐藏分支**的通道比例，并不是整个输入特征的 25%。其余通道和更早的 C2f 分支保留。

原轴向模块先沿宽/高方向求均值，再分别经过 `3×1`/`1×3` 深度卷积，广播相加后生成调制响应：

```text
R = Mean_W(Fa), C = Mean_H(Fa)
A = 2 * Hardsigmoid(DW3x1(R) + DW1x3(C)) - 1
Fa' = Fa + gamma * Fa * A
```

`gamma` 是每个活动通道一个的可学习参数，初始化为零，后续没有幅度约束。两个方向卷积包含偏置。

有界版本额外计算 `M=Mean_HW(Fa)`，在方向滤波前减去 M，方向卷积不带偏置并初始化为零：

```text
S = 1 + 0.25 * (2 * Hardsigmoid(DW3x1(R-M) + DW1x3(C-M)) - 1)
Fa' = Fa * S
```

`S∈[0.75,1.25]`，该版本没有可学习的 gamma。两者初始都保留原 C2f 的映射，但初始等价及幅度限制不等于已经证明训练稳定性或精度提升。

## 新门控上采样模块

两个全通道特征路径分别采用最近邻和双线性插值。第三条路径从低分辨率输入通过 `3×3, C→1` 卷积、Hardsigmoid 和最近邻上采样产生一个空间门控 G；**G 在通道间共享**。双线性插值使用 `align_corners=False`。

```text
N = Nearest_x2(X)
B = Bilinear_x2(X)
G = Nearest_x2(Hardsigmoid(Conv3x3_C_to_1(X)))
Y = N + G * (B-N)
```

图中的减法节点表示 `B−N`，顶部输入为被减去的 N。`G∈[0,1]`，输出是两种插值的逐元素凸组合；默认初始为 `0.95N+0.05B`。该图对应双插值门控，不是原 CAFR/CAFR-Lite 的动态邻域重组。

可用于图注的英文说明：

> **Axial module:** Structure of C2fAxisResidual. Axial mean descriptors modulate a subset of the final hidden branch, while the remaining channels and preceding C2f branches are retained. Batch dimensions are omitted.

> **Gated upsampling:** Structure of GatedInterpolationUpsample. A spatial gate predicted at the input resolution blends nearest-neighbor and bilinear upsampling results. The gate is shared across channels and upsampled using nearest-neighbor interpolation.

> **Bounded variant:** Structure of C2fAxisBounded. Centered axial descriptors generate a bounded multiplier for partial-channel modulation, with the unchanged C2f branches retained.
