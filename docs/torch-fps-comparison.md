# RTX 3090 纯推理 FPS 对比

论文主对比表使用同一 RTX 3090、PyTorch eager FP32、关闭 TF32、batch=1、固定 640×640 的**模型前向吞吐量**。RK3566 的 22 FPS 保留为单独的部署结果，不能填入 GPU FPS 列。

脚本 `tools/benchmark_torch_fps.py` 支持当前三类实现：Ultralytics、MMDetection、RT-DETRv2 PyTorch。各模型应在具备对应依赖的环境中运行，尽量统一 PyTorch/CUDA/cuDNN 版本。测量前关闭占用同一 GPU 的训练和其他重负载任务，保持相同电源、频率和散热设置。

## 计时口径

- 使用同一个实际测试图像目录，按路径排序，默认缓存前32张；预处理和上传 GPU 均在计时前完成。
- 统一将图像缩放到640×640。Ultralytics 和 RT-DETRv2 使用 RGB 浮点输入，范围0–1；后者匹配本仓库的部署参考。MMDetection 通过模型自身的数据预处理器完成通道转换与归一化。
- eval 和 inference_mode；默认启用各框架的推理融合：Ultralytics `fuse()`、MMDetection Conv–BN 融合、RT-DETRv2 `deploy()`。JSON 保存该设置；如使用 `--no-fuse`，须所有模型统一关闭。
- 丢弃50次预热，随后计时5轮，每轮200次 batch=1 的前向调用。使用当前 CUDA 设备上的事件及同步读取时间。
- 不包含读图、预处理、主机到设备的数据传输、模型加载、绘图以及框架外的解码/NMS/后处理。
- **原生 forward 内部的计算仍计入**：例如 Ultralytics Detect 的框解码及部分端到端头的筛选；Faster R-CNN 的 RPN 候选筛选是 RoI 推理所需，仍计入。这是可复现的模型前向边界，不能描述成所有检测器都完全没有候选处理。
- `FPS = 1000 / mean_ms`；mean_ms 是5轮等长度测试合并后的每张平均 CUDA 事件耗时。`std_of_round_means_ms` 是轮均值间的标准差，不是逐帧抖动或P95延迟。不要把它解释为相机到停机的端到端帧率。

## Ultralytics：PAGI-YOLO 和 YOLO 基线

在训练环境、仓库根目录运行，将权重路径换成对比表对应的实际 checkpoint。不要用另一份最佳 seed 的权重替代，以便结果可追踪。

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics

python tools/benchmark_torch_fps.py \
  --weights runs/detect/axis-interp/1/weights/best.pt \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --name PAGI-YOLO --device 0 --imgsz 640 \
  --output results/fps/pagi-yolo.json

python tools/benchmark_torch_fps.py \
  --weights runs/detect/yolov8n/1/weights/best.pt \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --name YOLOv8n --device 0 --imgsz 640 \
  --output results/fps/yolov8n.json
```

其他 YOLO 模型使用同一条命令，只改 `--weights`、`--name` 和输出文件。默认已经是50次预热、5×200次正式测试，不需要重新计算 AP 或训练模型。推理融合仅作用于内存模型，不覆盖 checkpoint。

## MMDetection：DETR、Deformable DETR、Faster R-CNN

激活这些模型训练时使用的环境，然后运行同一脚本。以下配置与最佳权重路径已于2026-10-18在训练服务器上只读核对，使用实验目录保存的配置。后续重新训练或更换对比权重时，应相应更新路径。

```bash
cd /home/tkz/code/github/mmdetection

# DETR-R50
python /home/tkz/code/github/ultralytics/tools/benchmark_torch_fps.py \
  --backend mmdet \
  --config work_dirs/detr_r50/detr_r50_300e.py \
  --weights work_dirs/detr_r50/best_coco_bbox_mAP_epoch_280.pth \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --name DETR-R50 --device 0 --imgsz 640 \
  --output /home/tkz/code/github/ultralytics/results/fps/detr-r50.json

# Deformable DETR-R50
python /home/tkz/code/github/ultralytics/tools/benchmark_torch_fps.py \
  --backend mmdet \
  --config work_dirs/deformable_detr_r50/deformable_detr_r50_300e.py \
  --weights work_dirs/deformable_detr_r50/best_coco_bbox_mAP_epoch_96.pth \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --name Deformable-DETR-R50 --device 0 --imgsz 640 \
  --output /home/tkz/code/github/ultralytics/results/fps/deformable-detr-r50.json

# Faster R-CNN-R50
python /home/tkz/code/github/ultralytics/tools/benchmark_torch_fps.py \
  --backend mmdet \
  --config work_dirs/faster_rcnn_r50/faster_rcnn_r50_300e.py \
  --weights work_dirs/faster_rcnn_r50/best_coco_bbox_mAP_epoch_49.pth \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --name Faster-R-CNN-R50 --device 0 --imgsz 640 \
  --output /home/tkz/code/github/ultralytics/results/fps/faster-rcnn-r50.json
```

脚本直接调用 MMDetection 的 `mode="tensor"` 路径，不调用 `inference_detector()` 的完整图像预测流程。对另外两个模型更换配置、权重和名称即可。输入分辨率由此协议统一为640×640，不能直接混用原生测试命令默认的800或1333尺寸。

## RT-DETR-R18（RT-DETRv2 实现）

激活 RT-DETRv2 的训练环境，使用训练时的配置及对应 checkpoint。RT-DETR-R18 来自 RT-DETR 仓库的 `rtdetrv2_pytorch` 子目录，因此这里选择 `--backend rtdetrv2`，论文模型名称仍为 RT-DETR-R18。以下 `best.pth` 已在服务器上核对存在。

```bash
cd /home/tkz/code/github/RT-DETR/rtdetrv2_pytorch

python /home/tkz/code/github/ultralytics/tools/benchmark_torch_fps.py \
  --backend rtdetrv2 \
  --repo . \
  --config configs/rtdetrv2/rtdetrv2_r18vd_pingwen.yml \
  --weights output/rtdetrv2_r18vd_pingwen/best.pth \
  --images /home/tkz/datasets/pingwen_yolo/images/test \
  --name RT-DETR-R18 --device 0 --imgsz 640 \
  --output /home/tkz/code/github/ultralytics/results/fps/rtdetr-r18.json
```

按官方部署参考优先加载 checkpoint 中的 EMA 权重，否则加载 `model` 权重；模型严格加载参数，再进行部署转换，不调用 postprocessor。自定义验证配置如果另有归一化操作，应先核对与此处引用的标准部署输入协议是否一致。

## 输出与论文

终端末尾打印：

```text
PAGI-YOLO: ... ms/image, ... FPS
```

JSON 同时保存每轮耗时、平均耗时、FPS、GPU名称、框架版本、输入形状、融合状态，以及 checkpoint、配置和输入图像的 SHA-256。不同模型结果的 GPU、精度、尺寸、计时次数和图像清单应一致。该脚本不生成或推断精度指标，也不自动填入论文数值。

表头建议使用 `FPS`，在表注或实验设置中注明 `RTX 3090, FP32, batch=1, 640×640; model forward only`。正文对 RK3566 的22 FPS明确保留其设备口径。
