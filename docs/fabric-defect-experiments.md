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
  name=v8n_p3_e2e
```

Evaluate the best checkpoint on the held-out test split:

```bash
yolo detect val \
  model=/home/tkz/datasets/pingwen_paper/runs/detect/v8n_p3_e2e/weights/best.pt \
  data=/home/tkz/datasets/pingwen_paper/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_paper/runs/detect \
  name=v8n_p3_e2e_test
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
