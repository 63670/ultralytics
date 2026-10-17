# 回到结构本身：轴向调制与上采样的受约束候选

日期：2026-10-17。此轮只改变模型，不改变现有训练参数、数据划分、检测头或标注形式。旧 axis-cafr-lite 定义和 checkpoint 类保持原样。

## 1. 修正优化优先级

相同训练协议下，新模型的波动和平均表现更差，需要优先处理模型自身的敏感性。更换训练配方可以另作实验，但不能替代结构改进的验证。本轮暂停把训练参数调整作为主要优化方向。

已有结果只证明结构的收益不稳定，不能确定哪个内部机制导致波动。下面两项基于代码中可确认的自由度提出假设，再通过单因素结构对照检验，不把数学约束或单次高分称为训练稳定性已经改善。

## 2. 两个值得检验的结构问题

### 原 CAFR-Lite 注入的是整幅投影特征

现有计算可简写为：

`Y = nearest(X) + gain * project(reassemble(compress(X)))`

它并未将附加分支定义为相对固定插值的差分，而是把动态重组后的潜在特征投影回所有通道，再直接相加。`gain` 是一个无界的全局标量。核预测、压缩、投影都包含随机初始化参数；初始 gain 虽为零，打开后仍会引入随机分支学习出的特征。代码允许这样的自由度，但没有证据说明实际训练中 gain 一定过大，或它就是低分 seed 的原因。

### 原轴向分支的调制幅度无结构上限

现有计算为：

`X_active * (1 + gamma * (2 * Hardsigmoid(row + column) - 1))`

`gamma` 虽从零开始，却没有幅度限制；row/column 描述来自原始轴向均值。对于弱瑕疵与重复纹理，是否会学到不合适的调制值得检验。全轴汇聚可以提供长程信息，但本身不等于准确的边界定位或旋转不变性。

## 3. 改动 A：内容门控的双插值上采样

实现：`GatedInterpolationUpsample`。保留 P5→P4、P4→P3 两个位置，替换这两处 CAFR-Lite。

```text
N = nearest(X)
B = bilinear(X, align_corners=False)
G = nearest(Hardsigmoid(Conv3x3_C_to_1(X)))
Y = N + G * (B - N)
```

设计目的：

- 用来自 P4/P5 输入的内容预测一个空间门控，共享给同一位置的通道，不生成九个独立邻域权重，也不引入新的通道投影分支。
- `G∈[0,1]`，因此对给定输入，输出逐元素位于两种插值结果之间。它不作超出这两种值的外推；这不等同于限制整个神经网络的全局增益。
- 插值有确定的几何对应关系，常量特征在边界处也保持常量。上采样本身没有固定的横/纵方向选择，但这不构成模型旋转不变性的证明。
- 新门控卷积权重初始化为零，偏置设为 −2.7，因此初始约为 `95% nearest + 5% bilinear`。对于同一个输入，该新增模块的初始映射不依赖随机种子；整模型的初始化、训练过程和结果仍然依赖种子。
- 不使用零标量 gain，门控参数可在首次检测损失反传时学习。

这是缩小可学习上采样空间的候选。固定插值融合及门控都是通用构件，本轮不声称基础操作原创。是否能在细长瑕疵定位上形成有效任务适配，由数据对照决定。

## 4. 改动 B：有界、中心化的轴向调制

实现：`C2fAxisBounded`。保留原 C2f 的 bottleneck、通道数及聚合，仅处理最后隐藏分支的四分之一通道。

```text
R = mean_width(X_active)
C = mean_height(X_active)
M = mean_height(R)
T = DW3x1(R - M) + DW1x3(C - M)
Y_active = X_active * (1 + 0.25 * (2 * Hardsigmoid(T) - 1))
```

设计目的：

- 轴向描述减去通道的空间均值，建模相对变化；方向滤波器不使用偏置。空间常量特征的调制保持中性。这只是特征级性质，不是对原始图像亮度变化的完整不变性保证。
- 调制系数限定为 `[0.75,1.25]`，不会由本分支把特征反号或全部压成零；强度为固定设计值，未通过本数据集调参证明最优。
- 两个方向滤波器零初始化，使加载同一 C2f 参数后，初始映射与 C2f 一致；取消可学习的零 gamma，滤波器第一步即有梯度。
- 保留长程轴向汇聚与原有局部卷积，未压缩或替换预训练骨干。

该改动增加限制而非增加表达容量。约束可能减少不必要的调制，也可能限制有益的修正，需要实测。

## 5. 三个配置组成结构对照

| 文件，均位于 `ultralytics/cfg/models/v8/` | 轴向模块 | 两处上采样 | 用途 |
| --- | --- | --- | --- |
| `yolov8n-fabric-axis-cafr-lite.yaml` | 原始 | 原 CAFR-Lite | 现有对照 |
| `yolov8n-fabric-axis-interp.yaml` | 原始 | 新插值门控 | **优先测试，只替换上采样** |
| `yolov8n-fabric-axis-bounded-cafr-lite.yaml` | 新有界模块 | 原 CAFR-Lite | 只检查轴向调制 |
| `yolov8n-fabric-axis-bounded-interp.yaml` | 新有界模块 | 新插值门控 | 组合候选 |

不是先认定两个改动都好，再只训练组合。先比较单项，保留均值和波动均有依据的改动。训练设置沿用旧实验；不同时切换 `fabric-stability-geometry.yaml` 等新配方。

CPU、640×640、未融合模型的统计如下：

| 配置 | 参数量 | THOP GFLOPs |
| --- | ---: | ---: |
| 原 axis-cafr-lite | 3,030,555 | 8.24934 |
| 只改上采样 | 3,015,107 | 8.19968 |
| 只改轴向 | 3,030,483 | 8.24934 |
| 两者组合 | 3,015,035 | 8.19968 |

THOP 未完整覆盖函数式插值、乘加和访存。参数变少不等于已证明更快，更不等于精度或跨 seed 稳定性已提高。

## 6. 第一轮训练方式

在同步新文件后的仓库中，优先训练只替换上采样的版本。示例保持原参数：

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yolov8n-fabric-axis-interp.yaml \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml pretrained=yolov8n.pt \
  imgsz=640 epochs=300 patience=80 batch=16 workers=8 device=1 seed=0 \
  optimizer=auto deterministic=True mosaic=1.0 scale=0.5 close_mosaic=10 max_det=16 \
  project="$PWD/runs/detect" name=axis-interp_0
```

使用预先固定的同一组 seed，保留完整结果；以配对均值、标准差及分类别 AP 评价，不增加种子寻找最高分。当前测试集已经多次被开发过程查看，后续仍应把验证集作为 checkpoint 和方案选择依据，测试结果保留其探索性边界。

所有可迁移的基线参数形状保留。需要区分：可从同一三类别基线完整继承共同参数，不等于由 COCO 权重分别构建不同结构时、仅设置同 seed 就保证新分类头初值逐元素一致。不同结构会消耗不同数量的初始化随机数；严格对照可进一步统一三类别初始 checkpoint。

## 7. 导出与已完成检查

继续使用现有 RKOPT 九输出流程：

```bash
python tools/export_rkopt_onnx.py \
  --weights runs/detect/axis-interp_0/weights/best.pt \
  --output exports/axis-interp.onnx --imgsz 640
```

插值候选日志应包含 `Gated interpolation modules: ['model.10', 'model.13']`。该模型的重组路径无 Softmax、Unfold、GridSample 或五维特征展开，但新增双线性 Resize 的板端支持、调度和数值仍需用实际 RKNN 版本核查，不能根据算子名称保证 NPU 性能。

未训练模型的方向滤波器为零、门控为常量，编译器可能裁剪部分路径，所以不能把这种图的测速直接当作训练后速度。

完成 8 项新测试：确定初始混合与种子无关、凸组合范围、常量保持、加载 C2f 后的初始等价、调制上下界、首次检测损失反传梯度、非零修正下 split/fuse/ONNX Runtime 数值一致性，以及新 checkpoint 经正式导出 CLI 的保存/加载。另有 20 项旧模块和 RKOPT 测试通过，总计 28 项。

本地测试环境：CPU PyTorch 2.9.1、ONNX 1.18.0、ONNX Runtime 1.23.2。尚无新模型的真实训练结果、跨 seed 精度结论或 RK3566 测速；本轮没有上传源码或修改服务器任务。
