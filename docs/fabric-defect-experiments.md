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

## Overall Test-Set Comparison

All results below use the held-out test split (81 images and 91 instances).
Rows labeled `Ultralytics` use the same Ultralytics evaluator; their comparison
is the most direct. The `Official YOLOv5` row uses the official YOLOv5
evaluator. The remaining rows use their native COCO evaluators, whose P/R,
speed, and `maxDets` settings are not directly interchangeable with the
Ultralytics rows.

| Model | Evaluator | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | Parameters | GFLOPs | Inference / image | Postprocess / image |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv5n | Official YOLOv5 | 0.891 | 0.921 | 0.940 | 0.611 | 1.76M | 4.1 | 1.7 ms | 1.4 ms NMS |
| YOLOv8n | Ultralytics | 0.901 | 0.939 | 0.938 | 0.597 | — | — | 5.8 ms | 9.7 ms |
| YOLO11n | Ultralytics | 0.919 | 0.952 | 0.947 | 0.608 | — | — | 2.9 ms | 9.5 ms |
| YOLO12n | Ultralytics | 0.902 | 0.914 | 0.943 | 0.609 | — | — | 4.1 ms | 9.6 ms |
| YOLO26n | Ultralytics | 0.948 | 0.903 | 0.952 | 0.632 | — | — | 2.9 ms | 2.6 ms |
| RT-DETR-L | Ultralytics | 0.932 | 0.968 | 0.966 | 0.660 | — | — | 17.0 ms | 2.5 ms |
| **DACP-Net** | **Ultralytics** | **0.958** | 0.923 | **0.969** | **0.663** | — | — | 5.2 ms | 9.5 ms |
| DETR-R50 | Native COCO | — | — | 0.964 | 0.536 | — | — | — | — |
| Deformable DETR-R50 | Native COCO | — | — | 0.968 | 0.643 | — | — | — | — |
| Faster R-CNN-R50 | Native COCO | — | — | 0.969 | 0.662 | — | — | — | — |
| RT-DETRv2-R18 | Native COCO | — | — | 0.958 | 0.686 | — | — | — | — |

`mAP@0.50:0.95` is COCO-style AP averaged over IoU thresholds. Native COCO
rows preserve the evaluator-reported values; their AP@0.50 values use
`maxDets=1000` for MMDetection DETR/Faster R-CNN and `maxDets=100` for
RT-DETRv2-R18, whereas their mAP@0.50:0.95 values use the values reported in
the corresponding test logs.

## Multi-Seed Test Summary (Unified Evaluation)

This is the table to use for the paper's main YOLO comparison. All listed
checkpoints were evaluated on the held-out test split (81 images, 91 objects),
not on validation data, with `imgsz=640`, `batch=16`, `conf=0.001`, NMS
`iou=0.6`, and `max_det=16`. YOLOv5n was evaluated with its official
repository; the other models were evaluated with Ultralytics. Raw output is
saved as `test_iou06_metrics.log` in every corresponding training directory.
Mean and sample standard deviation are calculated across seeds.

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | mAP@0.50:0.95 (mean ± std) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv5n | 0.611 | 0.624 | 0.631 | 0.633 | 0.643 | 0.614 | 0.6260 ± 0.0121 |
| YOLOv8n | 0.600 | 0.615 | 0.602 | 0.606 | 0.627 | 0.543 | 0.5988 ± 0.0291 |
| YOLO11n | 0.604 | 0.520 | 0.595 | 0.640 | 0.629 | 0.597 | 0.5975 ± 0.0421 |
| YOLO12n | 0.609 | 0.616 | 0.586 | 0.606 | 0.566 | 0.574 | 0.5928 ± 0.0205 |
| YOLO26n | 0.632 | 0.630 | 0.582 | 0.505 | 0.507 | 0.634 | 0.5817 ± 0.0617 |
| **DACP-Net** | 0.660 | **0.667** | 0.631 | 0.609 | 0.600 | **0.665** | **0.6387 ± 0.0296** |

| Model | Seeds | Precision (mean ± std) | Recall (mean ± std) | mAP@0.50 (mean ± std) | mAP@0.50:0.95 (mean ± std) |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv5n | 6 | **0.9393 ± 0.0309** | 0.9167 ± 0.0347 | 0.9452 ± 0.0159 | 0.6260 ± 0.0121 |
| YOLOv8n | 6 | **0.9370 ± 0.0496** | 0.9242 ± 0.0322 | 0.9447 ± 0.0272 | 0.5988 ± 0.0291 |
| YOLO11n | 6 | 0.9333 ± 0.0295 | **0.9352 ± 0.0154** | 0.9467 ± 0.0139 | 0.5975 ± 0.0421 |
| YOLO12n | 6 | 0.9088 ± 0.0347 | 0.9153 ± 0.0234 | 0.9298 ± 0.0289 | 0.5928 ± 0.0205 |
| YOLO26n | 6 | 0.8613 ± 0.1039 | 0.8495 ± 0.0712 | 0.8940 ± 0.0629 | 0.5817 ± 0.0617 |
| **DACP-Net** | **6** | 0.9393 ± 0.0195 | 0.9228 ± 0.0166 | **0.9558 ± 0.0059** | **0.6387 ± 0.0296** |

DACP-Net has the highest mean test mAP@0.50:0.95 (0.6387), improving on its
matched YOLOv8n baseline by 0.0398. It also exceeds the YOLOv5n mean by
0.0127. YOLOv5n remains an external classical baseline rather than an
architecture-only controlled comparison because its official training and
evaluation implementation differs from the Ultralytics pipeline.

## Multi-Seed Validation Summary

The following results summarize six training seeds (`0`--`5`) per model from
`/home/tkz/code/github/ultralytics/runs/detect`. Each value is the best
validation metric recorded in that seed's `results.csv`; mean and standard
deviation are calculated across the six seeds. The `*_test` directories contain
visualizations but do not persist machine-readable metrics, so they are not
used in this table.

### Per-Seed mAP@0.50:0.95

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.5883 | 0.6331 | 0.5800 | 0.6312 | 0.5945 | 0.5684 | 0.5993 |
| YOLO11n | 0.6136 | 0.5698 | 0.5677 | 0.5838 | 0.5981 | 0.6072 | 0.5900 |
| YOLO12n | 0.6210 | 0.6037 | 0.5909 | 0.5759 | 0.5770 | 0.5549 | 0.5872 |
| YOLO26n | 0.6363 | 0.5792 | 0.5987 | 0.5652 | 0.5949 | 0.5765 | 0.5918 |
| **DACP-Net** | 0.6263 | **0.6495** | **0.6404** | 0.6048 | **0.6205** | **0.6438** | **0.6309** |

| Model | Seeds | Precision (mean) | Recall (mean) | mAP@0.50 (mean) | mAP@0.50:0.95 (mean ± std) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 6 | 0.8799 | 0.8897 | 0.9348 | 0.5993 ± 0.0269 | 0.5684–0.6331 |
| YOLO11n | 6 | 0.8579 | **0.9016** | **0.9397** | 0.5900 ± 0.0193 | 0.5677–0.6136 |
| YOLO12n | 6 | 0.8211 | 0.8444 | 0.8867 | 0.5872 ± 0.0232 | 0.5549–0.6210 |
| YOLO26n | 6 | 0.8631 | 0.8468 | 0.9200 | 0.5918 ± 0.0250 | 0.5652–0.6363 |
| **DACP-Net** | **6** | **0.8848** | 0.8758 | 0.9349 | **0.6309 ± 0.0168** | **0.6048–0.6495** |

DACP-Net has the highest mean mAP@0.50:0.95 and the lowest across-seed
standard deviation. Its mean gain over the YOLOv8n base detector is 0.0316
mAP@0.50:0.95.

## DACP-Net Ablation Summary

All four variants use the YOLOv8n-based architecture, the same test split, and
the Ultralytics evaluator. DATE denotes dual-axis texture encoding; LTFR
denotes local texture-guided feature reassembly.

| Variant | DATE | LTFR | Parameters | GFLOPs | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | Change vs. baseline |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n baseline | No | No | — | — | 0.901 | 0.939 | 0.938 | 0.597 | — |
| Baseline + DATE | Yes | No | 3.28M | 8.8 | 0.951 | 0.914 | 0.955 | 0.609 | +0.012 |
| Baseline + LTFR | No | Yes | 3.11M | 8.2 | 0.896 | 0.816 | 0.900 | 0.551 | -0.046 |
| **DACP-Net** | **Yes** | **Yes** | — | — | **0.958** | 0.923 | **0.969** | **0.663** | **+0.066** |

The full model improves by 0.054 mAP@0.50:0.95 over DATE alone and by 0.112
over LTFR alone, indicating that local feature reassembly is effective when
applied to DATE-enhanced features rather than as a standalone substitution for
the original upsampling path.

## Repository-Wise Training and Test Commands

Use the commands in this section to rerun every benchmark from pretrained
weights without overwriting the previously recorded outputs. All commands use
the current dataset locations supplied for each framework. Keep the image size,
batch size, epochs, augmentation, and seed fixed within a framework.

The rerun outputs are stored inside their respective repositories:
`rtdetrv2_pytorch/output/`, `mmdetection/work_dirs/`,
`ultralytics/runs/detect/`, and `yolov5/runs/train/`.

### RT-DETRv2 Repository

The verified dataset directory is `/home/tkz/datasets/pingwen_coco_rtdetr`.
The RT-DETRv2 experiment runner trains from the full COCO-pretrained checkpoint
and automatically evaluates `best.pth` on the test split in the same output
directory.

```bash
conda activate rtdetr
cd /home/tkz/code/github/RT-DETR/rtdetrv2_pytorch
python tools/run_experiment.py \
  configs/rtdetrv2/rtdetrv2_r18vd_pingwen.yml \
  --pretrained pretrained/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir output/rtdetrv2_r18vd_pingwen_seed5 \
  --device 1 \
  --seed 5
```

The test log is saved as `output/rtdetrv2_r18vd_pingwen_seed5/test_metrics.log`.

For an existing trained checkpoint, run only the test step:

```bash
conda activate rtdetr
cd /home/tkz/code/github/RT-DETR/rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=1 \
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_pingwen_test.yml \
  -r output/rtdetrv2_r18vd_pingwen_seed5/best.pth \
  --output-dir output/rtdetrv2_r18vd_pingwen_seed5 \
  --test-only
```

### MMDetection Repository

Each verified custom config sets `load_from` to its exact COCO-pretrained
OpenMMLab model-zoo checkpoint. The runner selects the best checkpoint and runs
the test split automatically in the same `work_dir`.

```bash
conda activate mmdet
cd /home/tkz/code/github/mmdetection

python tools/run_experiment.py \
  configs/pingwen/detr_r50_300e.py \
  --work-dir work_dirs/retrain_detr_r50_300e \
  --device 1 \
  --seed 5

python tools/run_experiment.py \
  configs/pingwen/deformable_detr_r50_300e.py \
  --work-dir work_dirs/retrain_deformable_detr_r50_300e \
  --device 1 \
  --seed 5

python tools/run_experiment.py \
  configs/pingwen/faster_rcnn_r50_300e.py \
  --work-dir work_dirs/retrain_faster_rcnn_r50_300e \
  --device 1 \
  --seed 5
```

The custom MMDetection configs must use
`/home/tkz/datasets/pingwen_coco` for their dataset and annotation paths.

For an existing trained checkpoint, run only the test step. Substitute the
matching config, checkpoint, and work directory for each model:

```bash
conda activate mmdet
cd /home/tkz/code/github/mmdetection
python tools/test.py \
  configs/pingwen/detr_r50_300e.py \
  work_dirs/retrain_detr_r50_300e/best_coco_bbox_mAP_epoch_280.pth \
  --work-dir work_dirs/retrain_detr_r50_300e
```

### Ultralytics Repository

Run all commands from the Ultralytics repository in the `tkz-yolo`
environment. The official `.pt` model argument initializes each standard model
from its corresponding pretrained checkpoint. The custom YAML models initialize
from `yolov8n.pt`, keeping the backbone initialization consistent with their
YOLOv8n base architecture.

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics
```

Use [`scripts/run_experiment.sh`](../scripts/run_experiment.sh) for every
Ultralytics benchmark. It trains each model and then automatically tests its
`best.pt`; it stores test
artifacts in `<training-run>/test/` and saves the complete metric log as
`<training-run>/test_metrics.log`, so each seed remains self-contained. After
the test succeeds, it deletes every weight in `<training-run>/weights/` except
`best.pt` and records removed filenames in `deleted_checkpoints.log`.

```bash
chmod +x scripts/run_experiment.sh

scripts/run_experiment.sh --device 1 yolov8n.pt retrain_yolov8n seed=5
scripts/run_experiment.sh --device 1 yolo11n.pt retrain_yolo11n seed=5
scripts/run_experiment.sh --device 1 yolo12n.pt retrain_yolo12n seed=5
scripts/run_experiment.sh --device 1 yolo26n.pt retrain_yolo26n seed=5
scripts/run_experiment.sh --device 1 rtdetr-l.pt retrain_rtdetr_l seed=5

scripts/run_experiment.sh --device 1 \
  ultralytics/cfg/models/v8/yolov8n-fabric-baseline.yaml \
  retrain_ablation_baseline pretrained=yolov8n.pt seed=5
scripts/run_experiment.sh --device 1 \
  ultralytics/cfg/models/v8/yolov8n-fabric-directional.yaml \
  retrain_ablation_directional pretrained=yolov8n.pt seed=5
scripts/run_experiment.sh --device 1 \
  ultralytics/cfg/models/v8/yolov8n-fabric-content-aware.yaml \
  retrain_ablation_content_aware pretrained=yolov8n.pt seed=5
scripts/run_experiment.sh --device 1 \
  ultralytics/cfg/models/v8/yolov8n-fabric-directional-carafe.yaml \
  retrain_dacp_net pretrained=yolov8n.pt seed=5
```

For multiple seeds, use the same runner in a loop. For
example:

```bash
for seed in 0 1 2 3 4 5
do
  scripts/run_experiment.sh --device 1 \
    ultralytics/cfg/models/v8/yolov8n-fabric-directional-carafe.yaml \
    "dacp_net-${seed}" \
    pretrained=yolov8n.pt \
    "seed=${seed}"
done
```

For an existing trained checkpoint, run only the test step. The test artifacts
and `test_metrics.log` remain inside that training run directory:

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics
yolo detect val \
  model=/home/tkz/code/github/ultralytics/runs/detect/dacp_net-0/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=1 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/code/github/ultralytics/runs/detect/dacp_net-0 \
  name=test \
  exist_ok=True \
  2>&1 | tee /home/tkz/code/github/ultralytics/runs/detect/dacp_net-0/test_metrics.log
```

### YOLOv5 Repository

Run the original anchor-based YOLOv5n from its official repository. The
`yolov5n.pt` argument initializes from the COCO-pretrained model.

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/yolov5
python train.py \
  --weights yolov5n.pt \
  --data /home/tkz/datasets/pingwen_yolo/data.yaml \
  --img 640 \
  --epochs 300 \
  --batch-size 16 \
  --device 0 \
  --workers 8 \
  --patience 80 \
  --project /home/tkz/code/github/yolov5/runs/train \
  --name retrain_yolov5n

python val.py \
  --weights /home/tkz/code/github/yolov5/runs/train/retrain_yolov5n/weights/best.pt \
  --data /home/tkz/datasets/pingwen_yolo/data.yaml \
  --task test \
  --img 640 \
  --batch-size 16 \
  --device 0 \
  --workers 8 \
  --max-det 16 \
  --project /home/tkz/code/github/yolov5/runs/train/retrain_yolov5n \
  --name test
```

For an existing YOLOv5n checkpoint, use the `python val.py ...` command above
and replace only the `--weights` path and its containing `--project` directory.

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

## Content-Aware Pyramid Ablation

This ablation retains the clean YOLOv8n backbone and replaces only the two
top-down nearest-neighbor upsampling operations with content-aware feature
reassembly. Train it with the same settings as the other ablations:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yolov8n-fabric-content-aware.yaml \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  pretrained=yolov8n.pt \
  imgsz=640 \
  epochs=300 \
  patience=80 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=ablation_content_aware
```

Evaluate the best checkpoint on the held-out test split:

```bash
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/ablation_content_aware/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=ablation_content_aware_test
```

### Ablation Test Results

The directional-only and content-aware-only variants were evaluated on the
same held-out test split (81 images and 91 instances). The full DACP-Net result
is included to show the interaction between the two modules.

The directional-only checkpoint was evaluated with:

```bash
yolo detect val \
  model=/home/tkz/datasets/pingwen_yolo/runs/detect/ablation_directional/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  max_det=16 \
  project=/home/tkz/datasets/pingwen_yolo/runs/detect \
  name=ablation_directional_test
```

| Variant | Dual-axis texture encoding | Local feature reassembly | Parameters | GFLOPs | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n baseline | No | No | — | — | 0.901 | 0.939 | 0.938 | 0.597 |
| Directional-only | Yes | No | 3,280,313 | 8.8 | 0.951 | 0.914 | 0.955 | 0.609 |
| Content-aware-only | No | Yes | 3,113,409 | 8.2 | 0.896 | 0.816 | 0.900 | 0.551 |
| DACP-Net | Yes | Yes | — | — | 0.958 | 0.923 | 0.969 | 0.663 |

The directional-only variant has 1.7 ms preprocessing, 1.9 ms inference, and
0.4 ms postprocessing latency per image. The content-aware-only variant has
1.2 ms preprocessing, 2.2 ms inference, and 0.3 ms postprocessing latency per
image.

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
  work_dirs/detr_r50_pingwen_pretrained_300e/best_coco_bbox_mAP_epoch_280.pth \
  --cfg-options randomness.seed=5
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
  work_dirs/deformable_detr_r50_pingwen_pretrained_300e/best_coco_bbox_mAP_epoch_96.pth \
  --cfg-options randomness.seed=5
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
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
  configs/pingwen/faster_rcnn_r50_300e.py \
  work_dirs/faster_rcnn_r50_pingwen_pretrained_300e/best_coco_bbox_mAP_epoch_162.pth \
  --cfg-options randomness.seed=5
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

## YOLOv5n Baseline Test Results

Evaluate the original anchor-based YOLOv5n checkpoint using the official
YOLOv5 repository:

```bash
conda activate tkz-yolo
cd ~/code/github/yolov5
python val.py \
  --weights /home/tkz/datasets/pingwen_yolo/runs/detect/yolov5n/weights/best.pt \
  --data /home/tkz/datasets/pingwen_yolo/data.yaml \
  --task test \
  --img 640 \
  --batch-size 16 \
  --device 0 \
  --workers 8 \
  --max-det 16 \
  --project /home/tkz/datasets/pingwen_yolo/runs/detect \
  --name yolov5n_test
```

The test split contains 81 images and 91 instances. The official YOLOv5
evaluator reports each class over all 81 images.

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 81 | 91 | 0.891 | 0.921 | 0.940 | 0.611 |
| row | 81 | 60 | 0.954 | 1.000 | 0.995 | 0.673 |
| col | 81 | 12 | 0.915 | 0.894 | 0.911 | 0.584 |
| hole | 81 | 19 | 0.805 | 0.867 | 0.914 | 0.576 |

Model complexity: 1,763,224 parameters and 4.1 GFLOPs. Per-image latency: 0.2
ms preprocessing, 1.7 ms inference, and 1.4 ms NMS postprocessing.
