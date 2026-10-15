# Fabric Axis-Detail v1：训练入口与结构说明

这是一个待训练验证的实验模型，配置文件为
`ultralytics/cfg/models/v8/yolov8n-fabric-axis-detail.yaml`。
默认三类仍为 row、col、hole，实际类别名称和顺序以训练数据 YAML 为准。
本版本不使用原 DME、CAFR，不改检测损失、标签分配和三个检测尺度。

## 先跑这一条训练命令

在包含本次代码的仓库根目录、原有训练环境中执行。将 `your-data.yaml` 换成原实验使用的数据配置，
`yolov8n.pt` 使用原始 YOLOv8n 预训练权重。其他参数最好与原基线实验一致。

```bash
yolo detect train model=ultralytics/cfg/models/v8/yolov8n-fabric-axis-detail.yaml data=your-data.yaml pretrained=yolov8n.pt imgsz=640 epochs=300 patience=80 batch=16 device=0 workers=4 seed=0 project=runs/fabric name=axis-detail-v1
```

如现有训练脚本通过 Python 调用，只需替换模型构建部分：

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v8/yolov8n-fabric-axis-detail.yaml")
    model.load("yolov8n.pt")
    model.train(
        data="your-data.yaml",
        imgsz=640,
        epochs=300,
        patience=80,
        batch=16,
        device=0,
        workers=4,
        seed=0,
        project="runs/fabric",
        name="axis-detail-v1",
    )
```

训练机器必须使用本仓库的新代码，仅复制 YAML 到原版 pip 包会缺少新模块。
可用 `python -c "import ultralytics; print(ultralytics.__file__)"` 确认加载的是本仓库。
新增层随机初始化；形状和名称匹配的原层按框架机制迁移，不能完整继承原模型的所有权重。
主干和检测头的阶段通道、顶层索引与基线一致，保留 C2f 的 `cv1`、`cv2` 命名以便迁移兼容参数。

## 两项结构改动

### C2f 内部的部分通道计算和轴向上下文

四个主干 C2f 改为 `C2fPartialAxis`。外部的分支聚合、输入输出通道和重复次数沿用基线，
内部每个 bottleneck 的隐藏特征只取 1/4 通道进行 3×3 空间卷积，然后使用 1×1 卷积混合全部通道。
剩余通道绕过当前单元的空间卷积，仍参与后续通道混合和 C2f 聚合。

仅 P3、P4 开启轴向上下文：对参与空间计算的通道分别沿宽、高求均值，形成 H×1 和 1×W 描述；
经过短的逐通道一维卷积和广播相加，再用 HardSigmoid 调节局部特征。
描述子聚合覆盖整条轴线，但会压缩该轴的位置细节，因此保留局部空间路径。
本设计不是完整的两两注意力，是否能减少断裂检测和端点遗漏需要实验验证。
P2、P5 只采用部分通道的局部计算，控制模块数量和开销。

### P3 特征金字塔的浅层细节补偿

第 15 层由普通 C2f 改为 `C2fDetailFusion`：保留原主融合路径，另外读取第 2 层 P2 特征。
P2 从 32 通道先压缩到 8 通道，经 3×3 深度卷积下采样，再投影到 P3 的 64 通道。
补偿特征乘以初始化为 0.1 的可学习标量后，与主融合输出相加。
8 是实际通道数，不再乘模型的宽度系数。

这条路径旨在补充细长目标的短轴及端点定位信息，未增加 P2 检测头。
后续仍使用 P3/P4/P5 检测，640×640 时步长为 8/16/32。

## 当前已验证的内容

同环境下，三类、640×640、batch=1 的静态统计如下：

| 配置 | 参数量 | GFLOPs |
| --- | ---: | ---: |
| 原 YOLOv8n fabric baseline | 3,011,433 | 8.1938 |
| Axis-Detail v1 | 2,570,474 | 6.9206 |

参数减少约 14.6%，计算量估计减少约 15.5%。GFLOPs 使用 THOP 对完整 640×640 前向统计，
按 2×MACs 计；其默认规则不完整统计函数式均值、广播和逐元素运算，因此是估算值，不能代替板端延迟。
这些数值不代表已经获得精度提升或 RK3566 加速。

已完成以下检查：

- 模型解析、三个检测步长，以及参数量低于基线。
- 轴向上下文使远距离同轴像素能够影响输出；关闭该上下文时此单元没有该远距连接。
- 使用横向细长、纵向细长、局部框的合成批次计算真实检测损失并反向传播；两个新路径均有有限、非零梯度且参数可更新。
- 640×640 和 256×320 前向，导出前 C2f 的 `forward_split` 切换一致性，Conv-BN 融合一致性。
- 兼容层权重迁移，以及 `YOLO(yaml)` 公共接口构建。

复现检查：

```bash
python -m unittest discover -s tools/tests -p test_fabric_axis_detail.py -v
```

尚未完成真实数据训练、完整 ONNX 导出、RKNN 转换和板端测试。
新模块没有展开邻域或动态采样，但均值归约、广播和其他算子的编译、执行位置仍需实际 RKNN 验证。

## 训练后优先保留的结果

先观察验证集 AP、AP75、各类别 AP 和框的可视化，重点检查细长框的宽度、短轴偏移及端点。
保持原数据划分与训练设置，用验证集选择结构，最终测试集用于报告结果。
随后验证 INT8 后精度，并用 `tools/benchmark_rknn.py` 测量实际 `.rknn` 推理耗时。
如第一轮结果有价值，再分别去掉轴向上下文和浅层细节补偿做消融，区分两项结构的贡献。
