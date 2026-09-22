# Fabric Defect Experiments

All values are the latest retained single-run results on the held-out test set
(81 images and 91 instances). Native COCO AP is AP@[IoU=0.50:0.95] with
`maxDets=100`. For MMDetection, AP50 and AP75 are evaluator-reported with
`maxDets=1000`.

## Comparison Experiments

| Model | Framework | Parameters | GFLOPs | Precision | Recall | AP50 | AP75 | AP / mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv5n | Ultralytics | 2.50M | 7.1 | 0.917 | 0.858 | 0.931 | — | 0.584 |
| YOLOv8n | Ultralytics | 3.01M | 8.1 | 0.884 | 0.911 | 0.941 | — | 0.593 |
| YOLO11n | Ultralytics | 2.58M | 6.4 | 0.917 | 0.936 | 0.950 | — | 0.596 |
| YOLO12n | Ultralytics | 2.56M | 7.3 | 0.902 | 0.914 | 0.943 | — | 0.609 |
| YOLO26n | Ultralytics | 2.38M | 5.3 | 0.916 | **0.945** | 0.965 | — | 0.614 |
| RT-DETRv2-R18 | Native COCO | — | — | — | — | 0.952 | 0.823 | **0.677** |
| DETR-R50 | Native COCO | — | — | — | — | 0.964 | 0.551 | 0.536 |
| Deformable DETR-R50 | Native COCO | — | — | — | — | **0.968** | **0.781** | 0.643 |
| Faster R-CNN-R50 | Native COCO | — | — | — | — | 0.961 | 0.773 | 0.647 |
| **DACP-Net** | **Ultralytics** | **3.39M** | **8.9** | **0.958** | 0.923 | **0.969** | — | 0.663 |

## Ablation Experiments

WWTE denotes Warp–Weft Texture Encoding and LTFR denotes Local Texture-Guided
Feature Reassembly.

| Variant | WWTE | LTFR | Parameters | GFLOPs | Precision | Recall | mAP50 | mAP50-95 | Change vs. YOLOv8n |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n baseline | No | No | 3.01M | 8.1 | 0.884 | 0.911 | 0.941 | 0.593 | — |
| Baseline + WWTE | Yes | No | 3.28M | 8.8 | **0.969** | **0.935** | 0.942 | 0.641 | +0.048 |
| Baseline + LTFR | No | Yes | 3.11M | 8.2 | 0.907 | 0.932 | 0.943 | 0.609 | +0.016 |
| **DACP-Net** | **Yes** | **Yes** | **3.39M** | **8.9** | 0.958 | 0.923 | **0.969** | **0.663** | **+0.070** |
