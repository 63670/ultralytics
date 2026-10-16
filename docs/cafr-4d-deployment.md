# CAFR 四维等价导出实验

用户提供的 RKNN profiling 确认两处 CAFR 占逐层耗时约 90.9%，其中 CPU Transpose/Reshape 合计约 434 ms。
本次先保留训练好的核预测器和原计算定义，改变 CAFR 的部署表示，供同权重对照测试。
该改写仅在显式指定导出选项时生效，不改变原 checkpoint、训练模块或默认导出行为。

## 使用现有 DMFR 权重导出

更新代码后，在原训练环境、仓库根目录执行；替换实际包含 CAFR 的 checkpoint 路径：

```bash
python tools/export_rkopt_onnx.py --weights /path/to/dmfr/best.pt --output exports/dmfr-cafr4d.onnx --imgsz 640 --cafr-mode streaming
```

命令行应出现：

```text
CAFR mode: streaming; rewritten modules: ['model.10', 'model.13']
```

若列表为空，说明加载的模型没有匹配的 `ContentAwareUpsample`，没有发生 CAFR 改写。
ONNX 仍为九输出格式，输入静态 batch=1、640×640，默认 opset=12。
输出旁的 `.rkopt.json` 记录模式、改写模块、输出形状和 checkpoint SHA256。
如果没有指定 `--output`，该模式默认产生 `best-rkopt-cafr4d.onnx`，与原版导出区分。

保留原版图用于对照：

```bash
python tools/export_rkopt_onnx.py --weights /path/to/dmfr/best.pt --output exports/dmfr-original.onnx --imgsz 640 --cafr-mode original
```

## 具体改变

核预测路径仍是原来的 `compress → kernel_encoder → PixelShuffle → Softmax`，直接复用训练权重。
改写的是特征邻域提取和加权求和：

| 原实现 | 四维实现 |
| --- | --- |
| unfold 显式展开 25 个邻域 | 原特征只做一次最近邻上采样和零填充 |
| 邻域展开后再上采样 | 逐个取固定二维偏移的四维切片 |
| 构造 `[B,C,25,2H,2W]` 参与乘法和归约 | 每项 `[B,C,2H,2W] × [B,1,2H,2W]`，再相加 |

对于核索引 `(i,j)`、输出位置 `(u,v)`，两个实现读取的原特征位置都是
`(floor(u/2)+i-p, floor(v/2)+j-p)`，其中 `p=floor(k/2)`，越界位置补零。
每项仍乘以原核预测器在该输出位置的对应权重，因此数学表达相同。
边界处也不重新归一化剩余权重，与原 CAFR 一致。

这避免显式构造包含全部邻域的单个大张量，主特征张量维度保持 BCHW，且没有将通道挪入 batch 的计算要求。
浮点加法次序发生变化，数值可在容差内一致，不保证逐位一致。

该方案仍有 25 次逐项乘法和 24 次加法，以及相应切片；RKNN 是否融合、是否仍有 CPU 操作和最终调度均未知。
它并没有减少动态重组的全部算术量，也不保证整体峰值内存或延迟按 25 倍下降。
Softmax 保留，原报告中这两处合计约 8.2 ms，应在新的 profiling 中继续检查。

## 转换和连板重测

用同一个 Toolkit、同一份训练集校准图片列表和相同 INT8 配置转换两个 ONNX。示例：

```python
from ultralytics.utils.export.rknn import onnx2rknn

onnx2rknn(
    onnx_file="exports/dmfr-cafr4d.onnx",
    output_dir="exports/dmfr-cafr4d_rknn_model",
    name="rk3566",
    quantize=8,
    dataset="/path/to/fabric-calibration.txt",
    batch=1,
)
```

```bash
python tools/benchmark_rknn.py --model exports/dmfr-cafr4d_rknn_model/dmfr-cafr4d-rk3566.rknn --output results/cafr4d-perf.txt
python tools/benchmark_rknn.py --model exports/dmfr-cafr4d_rknn_model/dmfr-cafr4d-rk3566.rknn --profile --warmup 0 --repeat 1 --output results/cafr4d-profile.txt
```

用不带 `--profile` 的结果比较总体耗时，带 profile 的结果用于检查 `model.10`、`model.13` 的 CPU 布局转换是否消失，
以及新的 Slice/Mul/Add 是否成为主要开销。不要把旧逐层调试总时间直接与新非调试时间当作加速比。

原训练权重无需重新训练即可试这个导出实验。但 INT8 图的算子划分和量化误差会改变，必须重新校准，并复核量化后精度。
九输出仍需瑞芯微风格 CPU 后处理；系统总 FPS 应包含这些后处理。

## 已完成验证

- k=1、3、5，含小尺寸、矩形、batch=2 和补零边界的模块输出及输入/权重梯度一致性。
- 完整自定义模型的矩形输入，原实现与四维实现的输出一致性。
- 四维 ONNX 通过 checker 和 ONNX Runtime，逐输出与原实现比较；CAFR 已知中间张量 rank 不超过 4。
- 从包含 DME+CAFR 的合成 checkpoint 经 CLI 导出 640×640 九输出，并与原实现进行完整尺寸数值比对。
- 原有导出、检测头外置 DFL 和 DME 保留检查继续通过，共 7 项测试。

```bash
python -m unittest discover -s tools/tests -p test_rkopt_export.py -v
```

尚未在本机完成 RKNN 编译或 RK3566 实测。达到 10 FPS 与否必须由实际转换和连板结果确认。
若这版仍明显超出预算，再依据新的逐层报告决定继续改写或替换动态重组机制；改变计算机制通常需要重训。
