# Axis-Detail 后续对照：先验证方向上下文本身

v1 的测试 AP 为 57.9%，且用户反馈训练至 279 轮、验证集也表现不佳。
目前没有证据证明延长训练可以解决，也不能从单次汇总指标确定是哪一个模块造成下降。

代码审查确认 v1 同时更换了四个主干阶段的全部 bottleneck，使这些内部卷积的预训练权重无法迁移，
还加入了新的浅层融合路径。因此先将“主干压缩”与“上下文是否有效”分开。
本次两个配置是诊断对照，不是已经证明精度更高或参数更少的新成果。

## 第一步：只训练 axis-residual

配置：`ultralytics/cfg/models/v8/yolov8n-fabric-axis-residual.yaml`。

- 保留原主干全部 bottleneck、局部空间卷积、通道数、模块索引及原颈部。
- 只在 P3、P4 的 C2f 内，调制最后一条 hidden 分支的 1/4 通道；每个 C2f 只引入一次轴向上下文。
- 行列均值描述经过逐通道短卷积，形成中心值为零的方向调制，再乘以从零初始化的逐通道门控。
- 原始 `cv1`、`cv2`、`m.*` 参数名称和形状均保持，可迁移这些层的原权重。

在 Ubuntu 的更新后仓库中执行，训练参数尽量与原基线一致：

```bash
yolo detect train model=ultralytics/cfg/models/v8/yolov8n-fabric-axis-residual.yaml data=/home/tkz/datasets/pingwen_yolo/data.yaml pretrained=yolov8n.pt imgsz=640 epochs=300 patience=80 batch=16 device=0 workers=4 seed=0 project="$PWD/runs/fabric" name=axis-residual-v2
```

`project` 使用绝对路径，避免当前环境把相对 project 放到 `runs/detect/runs/fabric` 下。
模型选择先看验证集；第一步没有稳定收益时，暂不继续叠加第二项结构。

## 第二步：可选的零门控细节补偿

配置：`ultralytics/cfg/models/v8/yolov8n-fabric-axis-detail-v2.yaml`。
在第一步的基础上，使用 v1 相同的 8 通道 P2→P3 细节路径，但将 `detail_scale` 初始值改为 0。
旧 v1 的默认值仍是 0.1，其配置和结构行为保持原样。

```bash
yolo detect train model=ultralytics/cfg/models/v8/yolov8n-fabric-axis-detail-v2.yaml data=/home/tkz/datasets/pingwen_yolo/data.yaml pretrained=yolov8n.pt imgsz=640 epochs=300 patience=80 batch=16 device=0 workers=4 seed=0 project="$PWD/runs/fabric" name=axis-detail-v2
```

两项结构的门控初始为零，第一步更新门控参数；门控变为非零后，新上下文和细节分支的卷积权重才获得非零梯度。
这样可以从已有映射开始学习增量，但不保证收敛后精度不会下降。

## 规模和验证范围

| 配置 | 参数量（未融合） | GFLOPs（640×640） |
| --- | ---: | ---: |
| YOLOv8n 三类基线 | 3,011,433 | 8.193843 |
| axis-residual | 3,011,649 | 8.193859 |
| axis-detail-v2 | 3,012,650 | 8.219766 |

GFLOPs 为 THOP 的 2×MACs 估计，不完整覆盖均值归约、广播、逐元素运算和访存。
axis-residual 的统计仅说明参数及卷积计算增量很小，RK3566 的实际开销仍须连板测量。

新增检查确认：

- 载入**同一个完整三类别基线权重**后，两个配置都能精确迁移基线全部 state_dict 项，初始推理结果与基线完全一致。
- 原版 COCO `yolov8n.pt` 为 80 类；训练三类模型时分类输出仍会按框架规则重新初始化，不能把以上完整三类权重的等价测试直接套用为 COCO→三类检测输出相同。
- 使用真实检测损失，第一步门控梯度非零，第二步新增卷积分支梯度也非零、有限。
- 新门控设为非零时，640×640 前向、矩形输入、C2f `forward_split` 和 Conv-BN 融合仍一致。
- 原有 v1 检查继续通过，共 7 项测试。

```bash
python -m unittest discover -s tools/tests -p test_fabric_axis_detail.py -v
```

两个新配置尚未完成真实数据训练、完整 ONNX/RKNN 导出、INT8 精度评估或板端测速。
如果方向残差确有稳定增益，再单独研究哪些主干层可以压缩；当前先确定增益来源。
