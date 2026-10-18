# 论文检测对比图：GT 和全部模型

`tools/export_detection_comparison.py` 对同一批原图导出 GT 和预测框。所有列使用相同的类别颜色、线宽、字体，并保存原图分辨率 PNG 和坐标 JSON。无检出也保存图片。此脚本只绘制检测框，不生成热力图，不修改论文原有图片，也不重新训练模型。

## 权重清单

`tools/detection_comparison_models.json` 包含论文主表的11个检测器。2026-10-18只读核对服务器日志后，以下 YOLO 运行的测试 AP50/AP 与论文表中的数字逐项匹配；使用这些对应 checkpoint，而不是自动选择分数最高的 seed。

| 模型 | runs/detect 下的实验 | AP50 / AP (%) |
| --- | --- | --- |
| YOLOv8n | yolov8n/yolov8n14 | 90.5 / 58.4 |
| YOLOv9t | yolov9t/yolov9t6 | 90.1 / 53.3 |
| YOLOv10n | yolov10n | 88.2 / 58.5 |
| YOLO11n | yolo11n/yolo11n20 | 93.3 / 60.8 |
| YOLO12n | yolo12n/yolo12n4 | 89.0 / 56.8 |
| YOLO26n | yolo26n/2 | 90.5 / 58.2 |
| PAGI-YOLO | axis-interp/1 | 97.2 / 65.5 |

RT-DETR 和 MMDetection 使用前次核对的训练配置与最佳权重路径，尚未通过本出图脚本重算 AP。若论文使用的 checkpoint 变更，先修改清单。不要使用 COCO 预训练模型替代纺织数据集训练后的权重。

## 运行

先将新增脚本与清单同步到服务器，在训练时的各框架环境中分别执行。统一使用同一绝对输出目录，默认遍历 train、val、test，可用 `--splits test` 缩小范围。

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python tools/export_detection_comparison.py --backend gt \
  --output /home/tkz/code/github/ultralytics/runs/detection_comparison

python tools/export_detection_comparison.py --backend ultralytics --device 0 \
  --output /home/tkz/code/github/ultralytics/runs/detection_comparison
```

激活 MMDetection 的训练环境（本项目通常为 `mmdet`）：

```bash
conda activate mmdet
cd /home/tkz/code/github/mmdetection

python /home/tkz/code/github/ultralytics/tools/export_detection_comparison.py \
  --backend mmdet --device 0 \
  --output /home/tkz/code/github/ultralytics/runs/detection_comparison
```

激活 RT-DETRv2 训练环境后：

```bash
cd /home/tkz/code/github/RT-DETR/rtdetrv2_pytorch

python /home/tkz/code/github/ultralytics/tools/export_detection_comparison.py \
  --backend rtdetrv2 --device 0 \
  --output /home/tkz/code/github/ultralytics/runs/detection_comparison
```

全部模型导出后，在具有 Pillow 的环境中拼接。默认每张图片生成一行含 GT 和11个检测器的预览，文件名保持对应关系。

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics

python tools/export_detection_comparison.py --backend compose \
  --output /home/tkz/code/github/ultralytics/runs/detection_comparison
```

若论文只展示部分模型，可指定列顺序并提高每个面板宽度：

```bash
python tools/export_detection_comparison.py --backend compose --splits test \
  --models GT YOLOv8n YOLO11n YOLO26n RT-DETR-R18 Faster-R-CNN-R50 PAGI-YOLO \
  --panel-width 1280 \
  --output /home/tkz/code/github/ultralytics/runs/detection_comparison
```

同一路径的预览会被重写；各模型原尺寸图片不受此拼接操作影响。

## 输出与口径

```text
runs/detection_comparison/
  GT/{train,val,test}/原文件名.jpg.png
  YOLOv8n/{train,val,test}/原文件名.jpg.png
  ...
  PAGI-YOLO/{train,val,test}/原文件名.jpg.png
  comparison/{train,val,test}/原文件名.jpg.png
```

- 每个原尺寸 PNG 配套 `.json`，记录原图路径、像素尺寸、图像哈希和原图坐标。模型目录的 `manifest.json` 记录参数、图像清单、checkpoint/config哈希。类别按名称对齐，不凭框架的默认颜色猜类别。
- 默认显示阈值统一为0.25，最多100框；默认只显示类别名称，可对所有导出命令同时加 `--show-conf` 显示置信度，或 `--hide-labels` 只画框。统一样式须对 GT 和全部模型使用同样的绘图选项。
- YOLO 沿用原生预测接口，640方形 letterbox，NMS IoU默认0.7；RT-DETRv2 沿用其部署参考的640方形缩放、RGB 0–1输入和原生后处理，优先使用 EMA；RT-DETR类别顺序在清单中明确为row/col/hole。
- MMDetection 保留训练配置的测试预处理与原生后处理，仅移除需要标注元数据的 `LoadAnnotations`。已核对的三个配置使用保留宽高比的1333×800 Resize，而非640输入；`--imgsz` 不覆盖这些配置。此处是检测定性结果，不能将它等同于另行统一640输入的纯前向 FPS 测量。
- 不给 DETR 额外添加 YOLO NMS，不人为合并预测框。显示阈值用于定性图，不代表 AP 计算时的置信度阈值。
- GT 从 `images/<split>/` 对应的 `labels/<split>/` 读取 YOLO 检测标注；空文本表示正常样本。缺失标注直接报错，避免把漏文件当成正常样本。
- 所有划分均可用于内部查看，论文主检测对比图建议选择测试集同一批图片，图注明确数据划分和显示阈值。训练集案例如展示，应明确标为训练集。图片选择兼顾类别、困难场景和典型失败案例。

本地4项测试覆盖标注坐标、类别映射、无检出保存、三个数据划分的导出拼接和源图一致性检查；另以临时随机初始化的 PAC+GIU checkpoint 做过一次真实 Ultralytics CPU predict 接口验证。服务器清单中的全部权重和配置已确认存在。实际训练 checkpoint 的 GPU 批量推理，以及 MMDetection/RT-DETRv2 的运行验证，需要在对应服务器环境执行。
