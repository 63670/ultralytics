# Fabric Defect Experiments

This guide records repeatable training and evaluation commands for the fabric-defect dataset.

## Dataset

The dataset configuration is expected at `/home/tkz/datasets/pingwen_paper/data.yaml`:

```yaml
path: /home/tkz/datasets/pingwen_paper
train: images/train
val: images/val
test: images/test

names:
  0: row
  1: col
  2: hole
```

## Custom End-to-End YOLOv8n

Train the single-P3 fabric model with directional P5 context, direct box regression, and no NMS:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yolov8-fabric.yaml \
  data=/home/tkz/datasets/pingwen_paper/data.yaml \
  pretrained=yolov8n.pt \
  imgsz=640 \
  epochs=300 \
  patience=80 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  hsv_h=0 \
  hsv_s=0 \
  hsv_v=0.2 \
  mosaic=0.0 \
  project=/home/tkz/datasets/pingwen_paper/runs/detect \
  name=v8n_fabric
```

Evaluate the best checkpoint on the held-out test split:

```bash
yolo detect val \
  model=/home/tkz/datasets/pingwen_paper/runs/detect/v8n_fabric/weights/best.pt \
  data=/home/tkz/datasets/pingwen_paper/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_paper/runs/detect \
  name=v8n_fabric_test
```

## Native YOLOv8n Baseline

Train the unmodified YOLOv8n baseline with identical data and hyperparameters:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=/home/tkz/datasets/pingwen_paper/data.yaml \
  imgsz=640 \
  epochs=300 \
  patience=80 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  hsv_h=0 \
  hsv_s=0 \
  hsv_v=0.2 \
  mosaic=0.0 \
  project=/home/tkz/datasets/pingwen_paper/runs/detect \
  name=yolov8n_baseline
```

Evaluate the baseline on the same test split:

```bash
yolo detect val \
  model=/home/tkz/datasets/pingwen_paper/runs/detect/yolov8n_baseline/weights/best.pt \
  data=/home/tkz/datasets/pingwen_paper/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_paper/runs/detect \
  name=yolov8n_baseline_test
```

## RT-DETRv2-R18 Baseline

Evaluate the RT-DETRv2-R18 checkpoint on the held-out test set. Run this from the
RT-DETRv2 PyTorch repository with the `rtdetr` Conda environment active:

```bash
conda activate rtdetr
cd ~/code/github/RT-DETR/rtdetrv2_pytorch/
CUDA_VISIBLE_DEVICES=1 \
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_pingwen_test.yml \
  -r output/rtdetrv2_r18vd_pingwen/best.pth \
  --test-only
```

### Test Results

| Metric | Value |
| --- | ---: |
| AP@[IoU=0.50:0.95] | 0.686 |
| AP@0.50 | 0.958 |
| AP@0.75 | 0.837 |
| AP (small) | 0.620 |
| AP (medium) | 0.717 |
| AP (large) | 0.636 |
| AR@[IoU=0.50:0.95], maxDets=1 | 0.688 |
| AR@[IoU=0.50:0.95], maxDets=10 | 0.782 |
| AR@[IoU=0.50:0.95], maxDets=100 | 0.805 |
| AR (small), maxDets=100 | 0.644 |
| AR (medium), maxDets=100 | 0.832 |
| AR (large), maxDets=100 | 0.791 |
| AR@0.50, maxDets=100 | 0.982 |
| AR@0.75, maxDets=100 | 0.954 |

## DETR-R50 Baseline

Evaluate the pretrained DETR-R50 checkpoint on the held-out test set. Run this
from the MMDetection repository with the `mmdet` Conda environment active:

```bash
conda activate mmdet
cd ~/code/github/mmdetection
python tools/test.py \
  configs/pingwen/detr_r50_300e.py \
  work_dirs/detr_r50_pingwen_pretrained_300e/best_coco_bbox_mAP_epoch_280.pth
```

### Test Results

| Metric | Value |
| --- | ---: |
| AP@[IoU=0.50:0.95], maxDets=100 | 0.536 |
| AP@0.50, maxDets=1000 | 0.964 |
| AP@0.75, maxDets=1000 | 0.551 |
| AP (small), maxDets=1000 | 0.260 |
| AP (medium), maxDets=1000 | 0.580 |
| AP (large), maxDets=1000 | 0.497 |
| AR@[IoU=0.50:0.95], maxDets=100 | 0.698 |
| AR@[IoU=0.50:0.95], maxDets=300 | 0.698 |
| AR@[IoU=0.50:0.95], maxDets=1000 | 0.698 |
| AR (small), maxDets=1000 | 0.350 |
| AR (medium), maxDets=1000 | 0.727 |
| AR (large), maxDets=1000 | 0.694 |
