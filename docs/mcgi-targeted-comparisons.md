# MCGI-YOLO targeted comparisons

## Fixed design

- Seeds are fixed in advance to 0, 1, 2. No hyperparameter sweep is performed.
- CA versus MCE: GIU is fixed at both top-down upsampling sites. Replace MCE by Coordinate Attention at the final hidden C2f branch in backbone P3/P4, before concatenation. All original C2f bottlenecks and parameter paths are retained. CA uses its original full-hidden-channel operation, reduction 32; MCE retains its published partial=0.25 design.
- Upsampling comparison: the MCE backbone is fixed. Replace both top-down upsampling sites (layers 10 and 13) by Nearest, Bilinear (align_corners=False), CARAFE, or GIU.
- Reuse MCE+Nearest checkpoints in `runs/detect/ablation-axis-gate/axis-only/<seed>/weights/best.pt` and MCE+GIU in `runs/detect/axis-interp/<seed>/weights/best.pt`.
- Train only CA+GIU, MCE+Bilinear and MCE+CARAFE (three configurations, nine runs). The baseline seeds 0/2 were not found, so this design avoids retraining baseline configurations.

CA follows the [author implementation](https://github.com/houqb/CoordAttention/blob/main/coordatt.py). CARAFE matches [MMCV CARAFEPack](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/carafe.html): compression 64, encoder 3x3, reassembly 5x5, group 1, scale 2; Xavier compressor initialization and encoder std=0.001. A pure-PyTorch equivalent avoids an additional compiled MMCV dependency. Historical CAFR/ContentAwareUpsample weights are not relabeled as CARAFE.

## Protocol and reuse

The runner reads the effective training protocol from `axis-interp/1/weights/best.pt` and verifies all reused checkpoints against their saved `args.yaml`. `optimizer=auto` sets warmup_bias_lr=0 internally; this documented framework normalization is allowed. The original input file may retain 0.1. New runs use the effective checkpoint value, the same explicit `yolov8n.pt`, and the same data split. Pretraining is never inferred from a boolean `True` with a YAML entry point.

Evaluation always uses the current test split, imgsz=640, batch=16, conf=0.001, IoU=0.7, max_det=16, FP32, rect=True. Reused checkpoints are re-evaluated rather than copying rounded log lines. Results include overall mAP@50/mAP@50:95 and Row/Col/Hole AP@50/AP@50:95. Dataset files, initial checkpoint, and evaluated checkpoints are hashed. Previously printed paper results remain unchanged.

Params count the unfused model, consistent with the original architecture tables; fused counts are retained separately. GFLOPs use actual batch=1, 640x640 forward shapes: 2 x convolution MACs plus the comparison modules' arithmetic (directional means/gating, bilinear interpolation with 7 operations per output, GIU blend, and CARAFE weighted reassembly with 2*K*K-1 operations per output). Generic nonlinear activations, common backbone pooling and memory movement are omitted consistently. Both convolution-only and added arithmetic totals are saved, so these results are not silently mixed with rounded THOP-only FLOPs from old logs.

## Run on the training server

```bash
cd /home/tkz/code/github/ultralytics
/home/tkz/.conda/envs/tkz-yolo/bin/python tools/run_mcgi_targeted_queue.py
```

The queue first evaluates the six reused checkpoints, then uses GPU 0 and GPU 1 for the nine new runs. It does not delete existing weights or stop on-machine services. Existing completed runs are reused only after protocol checks; incomplete training directories require inspection rather than silent overwrite.

Individual commands, if the queue is not running:

```bash
python tools/run_mcgi_targeted.py --variant references --device 0 --seeds 0 1 2
python tools/run_mcgi_targeted.py --variant ca_giu --device 0 --seeds 0 1 2
python tools/run_mcgi_targeted.py --variant mce_bilinear --device 1 --seeds 0 1 2
python tools/run_mcgi_targeted.py --variant mce_carafe --device 1 --seeds 0 1 2
python tools/run_mcgi_targeted.py --variant summarize
```

Outputs are under `runs/detect/targeted-mcgi/`: per-run `metrics.json`, `per_seed.csv`, `summary.json`, protocol snapshots, logs and `queue.json`. Means and sample standard deviations use the same three seeds. Checkpoint selection uses validation performance, not test AP. Do not write comparative accuracy claims into the manuscript before the corresponding measured results exist.
