# DME + Phase 组合实验

配置：`ultralytics/cfg/models/v8/yolov8n-fabric-directional-phase.yaml`。

保留原版 DME，即骨干 P2、P3、P4、P5 的 `C2fDirectional`（层 2、4、6、8）。每个模块在原 C2f 输出后融合横向、纵向及局部分支。仅将原 DME+CAFR 配置的两处 CAFR 替换为 `PhaseAwareUpsample`（层 10、13），不加入 Sobel，其余融合路径和三尺度检测头保持一致。这样先单独检验替换 CAFR 的影响，避免同时改变 DME 插入位置。

Phase 以最近邻上采样为基础，通过低分辨率深度卷积及四个子像素位置权重添加修正，不构造 CAFR 的逐通道邻域展开张量。它的零初始化使组合模型可从 DME-only 的相同映射开始；实际训练仍统一加载 `yolov8n.pt`，与此前训练协议一致，而不是加载某个已经训练过的 DME 检查点。

## 五种子训练与测试

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics
git switch codex/jiot-sobel-phase
git pull --ff-only

bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-directional-phase.yaml \
  dme-phase-v1 pretrained=yolov8n.pt workers=4
```

脚本逐种子训练并使用 `best.pt` 测试，日志位于 `runs/detect/dme-phase-v1/<seed>/test_metrics.log`。已有目录不会覆盖。对照包括 DME-only（`yolov8n-fabric-directional.yaml`）、Phase-only（`yolov8n-fabric-phase.yaml`）和原 DME+CAFR（`yolov8n-fabric-directional-carafe.yaml`）；精度应比较同种子、同训练协议下的结果。

## 纯推理 FPS

```bash
python tools/benchmark_torch_fps.py \
  --weights runs/detect/dme-phase-v1/0/weights/best.pt \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --device 0 --imgsz 640 --warmup 50 --iterations 200 --repeat 5 \
  --output runs/detect/dme-phase-v1/0/fps.json
```

沿用 RTX 3090、FP32、batch=1 的纯推理口径；新组合的精度及帧率以实测为准。

## 实现验证

已在服务器隔离临时目录使用 CPU 验证：四处 DME 和两处 Phase 的位置、DME-only 权重完整迁移及初始输出一致性、检测损失反向传播和各模块梯度，以及非零 Phase 修正下的融合和 RKOPT ONNX 数值一致性。未启动训练，也未更改服务器仓库。

三类别配置的未融合参数为 3,292,433，融合后为 3,285,313；640×640 输入按 THOP 口径估算为 8.7927 GFLOPs。上述检查不代表精度或实际速度结果。
