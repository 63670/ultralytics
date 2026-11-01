# P3/P4 的逐 Bottleneck Sobel 候选

本版按骨干 P3、P4 的原位置实现，P5 不加 Sobel。原有的部分通道 v1、位置调整 v2 及其权重不变。

## 结构

新类 `BottleneckSobel` 保留原 Bottleneck 的两个卷积和 shortcut，在其完整输出后增加 `SobelResidual`。`C2fSobelBottleneck` 用该结构替换 C2f 中每个 progressive bottleneck，保持原来的 `cv1`、`cv2`、`m.i.cv1`、`m.i.cv2` 权重名称和形状。

原先只有最终隐藏分支的 1/4 通道做一次 Sobel；现在每个 bottleneck 的完整输出通道都参与边界编码。按照 YOLOv8n 的深度和宽度缩放：

| 位置 | C2f 输出通道 | Bottleneck 隐藏通道 | Bottleneck 数量 | Sobel 单元数 |
|---|---:|---:|---:|---:|
| 骨干 P3，层 4 | 64 | 32 | 2 | 2 |
| 骨干 P4，层 6 | 128 | 64 | 2 | 2 |
| 骨干 P5，层 8 | 256 | 128 | 1 | 0 |

对 C2f 中的两个 bottleneck，计算顺序为：

```text
两个原始 split 分支：A、B

B → Bottleneck 1 → Sobel残差 → Z1 → Bottleneck 2 → Sobel残差 → Z2

Concat(A, B, Z1, Z2) → 原来的 1×1 汇聚卷积
```

因此，边界修正既保留为 C2f 汇聚输入，也参与下一个 bottleneck 的局部特征提取；它不是仅在整个 C2f 后追加一次梯度计算。

## Sobel 残差的含义

设普通 bottleneck 的输出为 F：

1. 两个固定的归一化 Sobel 深度卷积计算 E = |Sx(F)| + |Sy(F)|。
2. 深度卷积和 Hardsigmoid 从 F 生成选择权重 M。
3. 可学习 1×1 卷积 P 混合边缘通道。
4. 输出 Z = F + M × P(E)。

保留 F，是为了保留原有纹理与语义表征；Sobel 提供梯度先验，选择与融合由检测损失学习。固定核作为 buffer，不会被优化器或训练器的参数解冻修改。每个 bottleneck 有独立的选择器和融合权重，没有跨 bottleneck 共享可学习参数。P 零初始化，使新模型在载入基础权重后从原有特征映射开始训练；首步先更新 P，随后选择器也获得梯度。

这一设计增加了边界编码的通道覆盖和渐进处理次数。Sobel 是经典算子，增加插入次数本身不是创新性证据；需要通过单模块、组合、多种子稳定性、误检与定位结果说明其任务价值。

## 与 Phase 的关系

`yolov8n-fabric-sobel-bottleneck-phase.yaml` 保留两处 Phase 上采样；`yolov8n-fabric-sobel-bottleneck.yaml` 使用原始最近邻上采样，作为 Sobel 单独版本。

这里的 Sobel 位于骨干，修正仍会沿后续骨干传入 Phase，不具备 v2 后融合方案的前向路径分离性质。本版检验的是“逐 bottleneck 边界编码是否比最后分支的局部修正更有效”，不能预先称为已经消除 v1 的负向交互。已有 v2 配置仍可用于位置对照。

## 五种子训练

维持先前的训练协议，只切换模型配置，采用 seed 0–4。先跑组合版：

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics
git switch codex/jiot-sobel-phase
git pull --ff-only

bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-sobel-bottleneck-phase.yaml \
  sobel-bottleneck-phase-v3 pretrained=yolov8n.pt workers=4
```

检查新版 Sobel 的独立收益：

```bash
bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-sobel-bottleneck.yaml \
  sobel-bottleneck-only-v3 pretrained=yolov8n.pt workers=4
```

权重及测试日志分别位于 `runs/detect/<运行名>/<seed>/weights/best.pt` 和 `test_metrics.log`。保持所有种子，不用最好的一次结果代替均值与波动；模型调整先看验证集。

## 实现检查

`tools/tests/test_sobel_bottleneck.py` 检查原残差与权重迁移、每个 Sobel 输出确实进入下一个 bottleneck、P3/P4 的实际数量与通道、P5 不变、各个新单元的检测损失反传，以及非零修正后的 split、融合、checkpoint 和 ONNX 一致性。

已在服务器独立临时目录使用 CPU 完成 5 项新测试、6 项 v1 和 5 项 v2 回归测试，16 项全部通过。未覆盖服务器仓库，也未使用正在运行任务的 GPU。

| 配置 | 未融合参数 | 融合后参数 | 融合后 GFLOPs 估算 |
|---|---:|---:|---:|
| Sobel Bottleneck 单独 | 3,023,593 | 3,018,393 | 8.1705 |
| Sobel Bottleneck + Phase | 3,028,593 | 3,023,393 | 8.1785 |

FLOPs 使用与此前一致的 THOP 口径；实际帧率、训练精度和板端速度由后续实测判断。
