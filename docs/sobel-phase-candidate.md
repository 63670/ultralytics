# Sobel C2f 与轻量上采样候选

此实验对应 DMFR-YOLO 的后续结构优化，不修改已写入论文的模型、数据和图表。三个配置均保留 YOLOv8n 的原始 bottleneck、特征拼接、检测头和基础权重名称。v1 的五种子训练结果与位置调整候选见 [v2 位置分工实验](sobel-phase-v2.md)。P3/P4 每个 bottleneck 后增加 Sobel 的版本见 [v3 逐 Bottleneck 实验](sobel-bottleneck-p3-p4.md)。

| 配置 | P3/P4 的 C2f | 两处上采样 |
|---|---|---|
| `yolov8n-fabric-sobel.yaml` | C2fSobel | 最近邻 |
| `yolov8n-fabric-phase.yaml` | 原始 C2f | PhaseAwareUpsample |
| `yolov8n-fabric-sobel-phase.yaml` | C2fSobel | PhaseAwareUpsample |

## C2fSobel：选择性融合边界先验

在 P3、P4 的 C2f 中，取最后一个 bottleneck 输出的 1/4 通道计算边缘，其余 3/4 通道旁路保留；P2、P5 不增加新模块。这里的 1/4 是最后一条隐藏分支的通道比例，并非整个 C2f 输出的 1/4。

对活动通道 U，使用归一化水平、竖直 Sobel 核获得响应，组合为 E = |Sx(U)| + |Sy(U)|。另一路用深度卷积和 Hardsigmoid 从 U 预测选择权重 M；1×1 卷积 P 学习如何混合边缘通道。修正为 U' = U + M × P(E)，随后与旁路通道重新拼接，进入原有 C2f 汇聚。

Sobel 也会响应正常针织纹理，并不能单独辨别瑕疵；这里让学习到的特征选择与通道融合决定其使用方式。此设计是可验证的建模假设，不预先断言能消除纹理干扰。传统差分先验融入 CNN 可参考 [PiDiNet（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/papers/Su_Pixel_Difference_Networks_for_Efficient_Edge_Detection_ICCV_2021_paper.pdf)，Sobel 本身不作为新算子宣称。

Sobel 核以普通深度卷积模块中的 buffer 保存，不属于优化器参数，避免训练器重新开启 requires_grad 时意外改变固定核；这样 FLOPs 分析仍能计入两次卷积。P 零初始化，M 初始为 0.5，所以载入基线权重后输出与原 C2f 一致。首个训练步骤更新 P，随后选择器可以获得梯度。

## PhaseAwareUpsample：低分辨率修正与子像素内容权重

本候选替换 CAFR 的整个上采样结构，仍为 2 倍上采样：

1. 主路径直接最近邻上采样 X，得到 N。
2. 在原分辨率执行一次 3×3 深度卷积 D(X)，再最近邻放大，得到修正 R。
3. 在原分辨率用 1×1 卷积预测 4 张内容权重，经 Hardsigmoid 和 Pixel Shuffle，得到单通道高分辨率权重 A。
4. 输出 Y = N + A × R，A 在特征通道间共享。

四个权重分别对应每个源位置生成的左上、右上、左下、右下子像素，因此同一源位置的四个输出可以不同。这里重建的是高层特征，随后仍与原有侧向特征拼接。它不使用 GIU 的最近邻/双线性插值混合，也不执行原 CAFR 的任意 25 邻域动态加权。它是表达能力受限、计算更紧凑的重构候选，后续通过精度实验判断这种取舍。

不再创建 B×C×25×2H×2W 的邻域张量，无 Unfold、邻域 Softmax、GridSample。学习操作在低分辨率执行，深度卷积权重数为 9C，四相位卷积为 4C+4；高分辨率只保留常规特征张量和共享权重。D 零初始化、A 初始为 0.5，初始输出等于最近邻上采样；首步更新 D，随后相位预测器获得梯度。实际 NPU 支持和速度仍需 RKNN 转换与板端实测。

## 验证记录

使用服务器已有 PyTorch 2.4.1+cu124 环境，在独立临时目录完成，不覆盖服务器仓库。6 项新测试覆盖方向响应、固定核、基线权重迁移、初始化映射、真实检测损失反传、非零修正后的 checkpoint/卷积融合/ONNX 输出一致性。另有 15 项现有 MCGI 与 RKOPT 测试通过。

2026-11-01 的结构测速使用同一 RTX 3090、FP32、关闭 TF32、batch=1、640×640，启用融合；32 张真实测试图像提前缓存到 GPU，预热 50 次，5 轮各 200 次前向。基础权重来自相同的未训练三分类 YOLOv8n，候选修正被设为非零；本结果验证结构耗时，不替代训练完成后的正式模型测速。

| 结构 | 未融合参数量 | 融合后卷积为主的 GFLOPs 估算 | 结构 FPS |
|---|---:|---:|---:|
| YOLOv8n | 3,011,433 | 8.0849 | 550.3 |
| 原 DME + CAFR | 3,394,705 | 8.9491 | 328.1 |
| Sobel C2f | 3,011,993 | 8.0907 | 535.3 |
| 新上采样 | 3,016,433 | 8.0929 | 525.5 |
| Sobel C2f + 新上采样 | 3,016,993 | 8.0987 | 510.4 |

最后复测基线为 548.0 FPS。FLOPs 为 THOP 的算子统计，不完整计入插值、逐元素操作和访存；本表 FPS 仅代表当时的结构测速，五种子训练精度见上述 v2 说明。原始逐轮数据见 [结构测速记录](benchmarks/sobel-phase-rtx3090-structure.json)。

## 训练与测试

保持此前训练超参数，仅切换模型配置。明确指定预训练 checkpoint，避免 YAML 模型意外从随机权重训练。

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics

# 组合版：训练完自动评估测试集
bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-sobel-phase.yaml \
  sobel-phase-v1 pretrained=yolov8n.pt workers=4

# 单模块消融
bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-sobel.yaml \
  sobel-only-v1 pretrained=yolov8n.pt workers=4

bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-phase.yaml \
  phase-only-v1 pretrained=yolov8n.pt workers=4
```

脚本默认数据集为 `/home/tkz/datasets/pingwen_yolo/data.yaml`，输入 640、batch 16、300 epochs、patience 80。组合版权重保存到 `runs/detect/sobel-phase-v1/0/weights/best.pt`，测试日志为同一运行目录的 `test_metrics.log`。如前一组训练还有额外参数，应在这三组和基线中保持一致。结构选择以验证集为依据，测试集用于最终报告。

```bash
# 完成训练后正式测速
python tools/benchmark_torch_fps.py \
  --weights runs/detect/sobel-phase-v1/0/weights/best.pt \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --name Sobel-Phase-v1 --device 0 --imgsz 640 \
  --warmup 50 --iterations 200 --repeat 5 \
  --output runs/detect/sobel-phase-v1/0/fps_rtx3090.json

# RKOPT 原始九输出导出；ONNX 一致性通过，RKNN/板端仍需另测
python tools/export_rkopt_onnx.py \
  --weights runs/detect/sobel-phase-v1/0/weights/best.pt \
  --output exports/sobel-phase-v1.onnx --imgsz 640
```
