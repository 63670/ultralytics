# Fabric Defect Experiments

All values are the latest retained single-run results on the held-out test set
(81 images and 91 instances). Native COCO AP is AP@[IoU=0.50:0.95] with
`maxDets=100`. For MMDetection, AP50 is evaluator-reported with
`maxDets=1000`. All AP metrics below are reported as percentages.

## Comparison Experiments

| Model | Framework | Parameters | GFLOPs | AP50 (%) | AP / mAP50-95 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv5n | Ultralytics | 2.50M | 7.1 | 93.1 | 58.4 |
| YOLOv8n | Ultralytics | 3.01M | 8.1 | 94.1 | 59.3 |
| YOLO11n | Ultralytics | 2.58M | 6.4 | 95.0 | 59.6 |
| YOLO12n | Ultralytics | 2.56M | 7.3 | 94.3 | 60.9 |
| YOLO26n | Ultralytics | 2.38M | 5.3 | 96.5 | 61.4 |
| RT-DETRv2-R18 | Native COCO | 20.09M | 60.72 | 95.2 | **67.7** |
| DETR-R50 | Native COCO | 41.56M | 38.82 | 96.4 | 53.6 |
| Deformable DETR-R50 | Native COCO | 40.10M | 79.64 | **96.8** | 64.3 |
| Faster R-CNN-R50 | Native COCO | 41.36M | 90.91 | 96.1 | 64.7 |
| **DACP-Net** | **Ultralytics** | **3.39M** | **8.9** | **96.9** | 66.3 |

## Ablation Experiments

WWTE denotes Warp–Weft Texture Encoding and LTFR denotes Local Texture-Guided
Feature Reassembly.

| Variant | WWTE | LTFR | Parameters | GFLOPs | mAP50 (%) | mAP50-95 (%) | Change vs. YOLOv8n (pp) |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n baseline | No | No | 3.01M | 8.1 | 94.1 | 59.3 | — |
| Baseline + WWTE | Yes | No | 3.28M | 8.8 | 94.2 | 64.1 | +4.8 |
| Baseline + LTFR | No | Yes | 3.11M | 8.2 | 94.3 | 60.9 | +1.6 |
| **DACP-Net** | **Yes** | **Yes** | **3.39M** | **8.9** | **96.9** | **66.3** | **+7.0** |
