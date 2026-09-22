# Fabric Defect Experiments

This document records the latest retained test-set results for the fabric-defect
experiments. Historical validation results, earlier test runs, and removed run
directories are intentionally excluded.

## Evaluation Scope

The YOLO results below use the held-out test split with 81 images and 91
instances. Metrics are read from the current result directories:

- Ultralytics: `/home/tkz/code/github/ultralytics/runs/detect`
- RT-DETRv2: `/home/tkz/code/github/RT-DETR/rtdetrv2_pytorch/output`
- MMDetection: `/home/tkz/code/github/mmdetection/work_dirs`

Each current directory contains one retained test result only. These are
single-run values, not multi-seed means; do not report a standard deviation
unless the corresponding seed runs and test logs are preserved.

## Latest Overall Test-Set Comparison

| Model | Framework | Parameters | GFLOPs | Precision | Recall | AP50 | AP75 | AP / mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv5n | Ultralytics | 2.50M | 7.1 | 0.917 | 0.858 | 0.931 | — | 0.584 |
| YOLOv8n | Ultralytics | 3.01M | 8.1 | 0.884 | 0.911 | 0.941 | — | 0.593 |
| YOLO11n | Ultralytics | 2.58M | 6.4 | 0.917 | 0.936 | 0.950 | — | 0.596 |
| YOLO12n | Ultralytics | 2.56M | 7.3 | 0.902 | 0.914 | 0.943 | — | 0.609 |
| YOLO26n | Ultralytics | 2.38M | 5.3 | 0.916 | **0.945** | 0.965 | — | 0.614 |
| **DACP-Net** | **Ultralytics** | **3.39M** | **8.9** | 0.958 | 0.923 | **0.969** | — | **0.663** |
| RT-DETRv2-R18 | Native COCO | — | — | — | — | 0.952 | 0.823 | **0.677** |
| DETR-R50 | Native COCO | — | — | — | — | 0.964 | 0.551 | 0.536 |
| Deformable DETR-R50 | Native COCO | — | — | — | — | **0.968** | **0.781** | 0.643 |
| Faster R-CNN-R50 | Native COCO | — | — | — | — | 0.961 | 0.773 | 0.647 |

The native COCO rows use AP@[IoU=0.50:0.95] with `maxDets=100`. Their AP50,
AP75, and scale-specific AP values are reported by MMDetection with
`maxDets=1000`; do not treat those columns as directly interchangeable with
the Ultralytics metrics.

## DACP-Net Ablation on the Latest Test Set

WWTE denotes Warp–Weft Texture Encoding (the current implementation class is
`C2fDirectional`). LTFR denotes Local Texture-Guided Feature Reassembly.
All four variants use the same YOLOv8n training pipeline and the Ultralytics
test evaluator.

| Variant | WWTE | LTFR | Parameters | GFLOPs | Precision | Recall | mAP50 | mAP50-95 | Change vs. YOLOv8n |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n baseline | No | No | 3.01M | 8.1 | 0.884 | 0.911 | 0.941 | 0.593 | — |
| Baseline + WWTE | Yes | No | 3.28M | 8.8 | **0.969** | 0.935 | 0.942 | 0.641 | +0.048 |
| Baseline + LTFR | No | Yes | 3.11M | 8.2 | 0.907 | 0.932 | 0.943 | 0.609 | +0.016 |
| **DACP-Net** | **Yes** | **Yes** | **3.39M** | **8.9** | 0.958 | 0.923 | **0.969** | **0.663** | **+0.070** |

## Native COCO Test Details

| Model | AP-small | AP-medium | AP-large | AR100 |
| --- | ---: | ---: | ---: | ---: |
| RT-DETRv2-R18 | 0.688 | 0.667 | 0.608 | 0.794 |
| DETR-R50 | 0.260 | 0.580 | 0.497 | 0.698 |
| Deformable DETR-R50 | 0.464 | 0.650 | 0.546 | 0.703 |
| Faster R-CNN-R50 | **0.702** | **0.691** | 0.545 | **0.707** |

## Reproducible Commands

### Ultralytics

Train and automatically evaluate a model with the repository runner:

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics
scripts/run_experiment.sh --device 0 MODEL RUN_NAME
```

Evaluate an existing checkpoint on the held-out test split:

```bash
yolo detect val \
  model=/home/tkz/code/github/ultralytics/runs/detect/RUN_NAME/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test imgsz=640 batch=16 device=0 workers=8 max_det=16 \
  project=/home/tkz/code/github/ultralytics/runs/detect/RUN_NAME name=test
```

### RT-DETRv2

```bash
conda activate rtdetr
cd /home/tkz/code/github/RT-DETR/rtdetrv2_pytorch
python tools/run_experiment.py \
  configs/rtdetrv2/rtdetrv2_r18vd_pingwen.yml \
  --pretrained rtdetrv2_r18vd_6x_coco_from_paddle.pth \
  --output-dir output/RUN_NAME --device 0 --seed 0
```

### MMDetection

```bash
conda activate mmdet
cd /home/tkz/code/github/mmdetection
python tools/run_experiment.py configs/pingwen/MODEL.py \
  --work-dir work_dirs/RUN_NAME --device 0 --seed 0
```

Use the repository-specific READMEs for complete options:
`RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/README.md` and
`mmdetection/configs/pingwen/README.md`.
