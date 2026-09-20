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

## Deformable DETR-R50 Baseline

Evaluate the pretrained Deformable DETR-R50 checkpoint on the held-out test
set. Run this from the MMDetection repository with the `mmdet` Conda
environment active:

```bash
conda activate mmdet
cd ~/code/github/mmdetection
python tools/test.py \
  configs/pingwen/deformable_detr_r50_300e.py \
  work_dirs/deformable_detr_r50_pingwen_pretrained_300e/best_coco_bbox_mAP_epoch_96.pth
```

### Test Results

| Metric | Value |
| --- | ---: |
| AP@[IoU=0.50:0.95], maxDets=100 | 0.643 |
| AP@0.50, maxDets=1000 | 0.968 |
| AP@0.75, maxDets=1000 | 0.781 |
| AP (small), maxDets=1000 | 0.464 |
| AP (medium), maxDets=1000 | 0.650 |
| AP (large), maxDets=1000 | 0.546 |
| AR@[IoU=0.50:0.95], maxDets=100 | 0.703 |
| AR@[IoU=0.50:0.95], maxDets=300 | 0.703 |
| AR@[IoU=0.50:0.95], maxDets=1000 | 0.703 |
| AR (small), maxDets=1000 | 0.594 |
| AR (medium), maxDets=1000 | 0.694 |
| AR (large), maxDets=1000 | 0.653 |

## Faster R-CNN-R50 Baseline

Evaluate the pretrained Faster R-CNN-R50 checkpoint on the held-out test set.
Run this from the MMDetection repository with the `mmdet` Conda environment
active:

```bash
conda activate mmdet
cd ~/code/github/mmdetection
python tools/test.py \
  configs/pingwen/faster_rcnn_r50_300e.py \
  work_dirs/faster_rcnn_r50_pingwen_pretrained_300e/best_coco_bbox_mAP_epoch_162.pth
```

### Test Results

| Metric | Value |
| --- | ---: |
| AP@[IoU=0.50:0.95], maxDets=100 | 0.662 |
| AP@0.50, maxDets=1000 | 0.969 |
| AP@0.75, maxDets=1000 | 0.779 |
| AP (small), maxDets=1000 | 0.382 |
| AP (medium), maxDets=1000 | 0.744 |
| AP (large), maxDets=1000 | 0.570 |
| AR@[IoU=0.50:0.95], maxDets=100 | 0.722 |
| AR@[IoU=0.50:0.95], maxDets=300 | 0.722 |
| AR@[IoU=0.50:0.95], maxDets=1000 | 0.722 |
| AR (small), maxDets=1000 | 0.394 |
| AR (medium), maxDets=1000 | 0.805 |
| AR (large), maxDets=1000 | 0.592 |

## YOLOv8n Baseline Test Results

Evaluate the YOLOv8n checkpoint on the held-out test set:

```bash
conda activate tkz-yolo
cd ~/datasets/pingwen_yolo
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/yolov8n/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=yolov8n_test
```

The test split contains 81 images and 91 instances.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.901 | 0.939 | 0.938 | 0.597 |
| row | 59 | 60 | 0.967 | 0.970 | 0.990 | 0.633 |
| col | 12 | 12 | 0.919 | 0.952 | 0.931 | 0.561 |
| hole | 18 | 19 | 0.816 | 0.895 | 0.893 | 0.597 |

Per-image latency: 1.4 ms preprocessing, 5.8 ms inference, and 9.7 ms
postprocessing.

## YOLO11n Baseline Test Results

Evaluate the YOLO11n checkpoint on the held-out test set:

```bash
conda activate tkz-yolo
cd ~/datasets/pingwen_yolo
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/yolo11n/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=yolo11n_test
```

The test split contains 81 images and 91 instances.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.919 | 0.952 | 0.947 | 0.608 |
| row | 59 | 60 | 0.983 | 0.962 | 0.993 | 0.611 |
| col | 12 | 12 | 0.908 | 1.000 | 0.925 | 0.602 |
| hole | 18 | 19 | 0.866 | 0.895 | 0.923 | 0.611 |

Per-image latency: 1.6 ms preprocessing, 2.9 ms inference, and 9.5 ms
postprocessing.

## YOLO12n Baseline Test Results

Evaluate the YOLO12n checkpoint on the held-out test set:

```bash
conda activate tkz-yolo
cd ~/datasets/pingwen_yolo
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/yolo12n/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=yolo12n_test
```

The test split contains 81 images and 91 instances.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.902 | 0.914 | 0.943 | 0.609 |
| row | 59 | 60 | 0.939 | 0.983 | 0.989 | 0.611 |
| col | 12 | 12 | 0.894 | 0.917 | 0.969 | 0.652 |
| hole | 18 | 19 | 0.873 | 0.842 | 0.871 | 0.564 |

Per-image latency: 1.5 ms preprocessing, 4.1 ms inference, and 9.6 ms
postprocessing.

## YOLO26n Baseline Test Results

Evaluate the YOLO26n checkpoint on the held-out test set:

```bash
conda activate tkz-yolo
cd ~/datasets/pingwen_yolo
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/yolo26n/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=yolo26n_test
```

The test split contains 81 images and 91 instances.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.948 | 0.903 | 0.952 | 0.632 |
| row | 59 | 60 | 0.921 | 0.950 | 0.975 | 0.658 |
| col | 12 | 12 | 0.984 | 0.917 | 0.971 | 0.660 |
| hole | 18 | 19 | 0.939 | 0.842 | 0.911 | 0.579 |

Per-image latency: 1.6 ms preprocessing, 2.9 ms inference, and 2.6 ms
postprocessing.

## DACP-Net Test Results

Evaluate the proposed Directional-Aware and Content-Aware Pyramid Network
(DACP-Net) checkpoint on the held-out test set:

```bash
conda activate tkz-yolo
cd ~/datasets/pingwen_yolo
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/DACP-Net/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=DACP-Net_test
```

The test split contains 81 images and 91 instances.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.958 | 0.923 | 0.969 | 0.663 |
| row | 59 | 60 | 0.973 | 0.967 | 0.988 | 0.617 |
| col | 12 | 12 | 1.000 | 0.906 | 0.995 | 0.738 |
| hole | 18 | 19 | 0.901 | 0.895 | 0.922 | 0.634 |

Per-image latency: 1.6 ms preprocessing, 5.2 ms inference, and 9.5 ms
postprocessing.

## RT-DETR-L Baseline Test Results

Evaluate the RT-DETR-L checkpoint on the held-out test set using the same
Ultralytics evaluation pipeline:

```bash
conda activate tkz-yolo
cd ~/datasets/pingwen_yolo
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/rtdetr_l/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=rtdetr_l_test
```

The test split contains 81 images and 91 instances.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.932 | 0.968 | 0.966 | 0.660 |
| row | 59 | 60 | 0.974 | 0.983 | 0.976 | 0.610 |
| col | 12 | 12 | 0.921 | 0.975 | 0.983 | 0.804 |
| hole | 18 | 19 | 0.900 | 0.945 | 0.940 | 0.567 |

Per-image latency: 1.5 ms preprocessing, 17.0 ms inference, and 2.5 ms
postprocessing.

## YOLOv5nu Baseline Test Results

Evaluate the YOLOv5nu checkpoint on the held-out test set:

```bash
conda activate tkz-yolo
cd ~/datasets/pingwen_yolo
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/yolov5nu/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=yolov5nu_test
```

The test split contains 81 images and 91 instances.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.937 | 0.944 | 0.967 | 0.602 |
| row | 59 | 60 | 1.000 | 0.996 | 0.995 | 0.612 |
| col | 12 | 12 | 0.921 | 0.971 | 0.989 | 0.591 |
| hole | 18 | 19 | 0.892 | 0.867 | 0.917 | 0.604 |

Per-image latency: 1.8 ms preprocessing, 2.8 ms inference, and 9.6 ms
postprocessing.
