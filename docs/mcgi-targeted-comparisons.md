# MCGI-YOLO reference module experiments

Use `scripts/run_experiment.sh` for the new experiments. It trains each seed sequentially, validates `best.pt` on the test split, prints P/R/mAP, and writes `test_metrics.log`. The commands below do not use the historical automatic queue.

## Configurations and insertion sites

| Reference | YAML suffix | Modification | New training |
| --- | --- | --- | --- |
| ECA | `eca` | ECA after the last bottleneck of backbone P3/P4 C2f, kernel 3 | Yes |
| CBAM | `cbam` | Author CBAM at the same P3/P4 sites, reduction 16, spatial kernel 7 | Yes |
| DySample | `dysample` | Both top-down upsampling sites, LP, scale 2, groups 4, no scope branch | Yes |
| CA | `coordinate-attention` | Existing P3/P4 reference | Reuse |
| Bilinear | `bilinear` | Existing upsampling reference | Reuse |
| CARAFE | `carafe-reference` | Existing upsampling reference | Reuse |

All YAML files are under `ultralytics/cfg/models/v8/yolov8n-fabric-<suffix>.yaml` and use three classes. ECA/CBAM retain the original C2f bottlenecks, branch concatenation, pretrained parameter paths, and nearest-neighbor neck. DySample retains the original C2f backbone. These are isolated module references, not combinations with MCE or GIU. The original YOLOv8n already provides the nearest-neighbor baseline.

The implementations follow the authors' [ECA](https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py), [CBAM](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py), and [DySample](https://github.com/tiny-smart/dysample/blob/main/dysample.py) code. CBAM includes both average and maximum channel descriptors, a shared reduction MLP, and the spatial Conv-BN; it is distinct from the simplified `ultralytics.nn.modules.conv.CBAM`. ECA uses the author's default kernel 3. Original MIT notices are in `ultralytics/nn/modules/comparisons.LICENSE`.

## Train the three new configurations

After synchronizing the code to the Linux repository:

```bash
cd /home/tkz/code/github/ultralytics
conda activate tkz-yolo
export DATA_YAML=/home/tkz/datasets/pingwen_yolo/data.yaml

COMMON=(
  pretrained=yolov8n.pt
  imgsz=640 epochs=300 patience=80 batch=16 workers=8
  optimizer=auto deterministic=True
  mosaic=1.0 scale=0.5 close_mosaic=10 max_det=16
)

bash scripts/run_experiment.sh --device 0 --seeds 0,1,2 \
  ultralytics/cfg/models/v8/yolov8n-fabric-eca.yaml \
  mcgi-reference/eca "${COMMON[@]}"

bash scripts/run_experiment.sh --device 0 --seeds 0,1,2 \
  ultralytics/cfg/models/v8/yolov8n-fabric-cbam.yaml \
  mcgi-reference/cbam "${COMMON[@]}"

bash scripts/run_experiment.sh --device 0 --seeds 0,1,2 \
  ultralytics/cfg/models/v8/yolov8n-fabric-dysample.yaml \
  mcgi-reference/dysample "${COMMON[@]}"
```

Change `--device` if another GPU is available. The script refuses to overwrite an existing run directory and stops on the first failed run. It retains `best.pt` and removes other `.pt` checkpoints only after successful test evaluation, following its existing behavior.

Test evaluation is explicit: `split=test`, `conf=0.001`, `iou=0.7`, `max_det=16`, `half=False`; input size, batch, workers, and dataset follow the supplied common settings. `best.pt` is selected by validation performance.

Outputs for each variant and seed:

```text
runs/detect/mcgi-reference/eca/0/weights/best.pt
runs/detect/mcgi-reference/eca/0/test_metrics.log
runs/detect/mcgi-reference/eca/0/test/
```

Replace `eca` with `cbam` or `dysample` and `0` with the requested seed. Existing CA/Bilinear/CARAFE/MCE/GIU results do not need to be retrained for these additions.

## Test an existing checkpoint separately

```bash
yolo detect val \
  model=runs/detect/mcgi-reference/eca/1/weights/best.pt \
  data=/home/tkz/datasets/pingwen_yolo/data.yaml split=test \
  imgsz=640 batch=16 device=0 workers=8 \
  conf=0.001 iou=0.7 max_det=16 half=False \
  project=runs/detect/mcgi-reference/eca/1 name=test-recheck
```

## Parameters and computation at 640 x 640

This command uses randomly initialized architectures only; it does not train, evaluate the dataset, or download pretrained weights.

```bash
python - <<'PY'
from pathlib import Path
import torch
from ultralytics import YOLO
from tools.run_mcgi_targeted import complexity

torch.set_num_threads(2)
root = Path('ultralytics/cfg/models/v8')
variants = ['baseline', 'coordinate-attention', 'eca', 'cbam', 'axis-residual',
            'bilinear', 'carafe-reference', 'dysample', 'interp']
print('Variant                Params(M,fused)  Conv/Linear(G)  With operator arithmetic(G)')
for name in variants:
    stats = complexity(YOLO(str(root / f'yolov8n-fabric-{name}.yaml')).model)
    print(f'{name:22} {stats["params_fused"]/1e6:15.6f} '
          f'{stats["conv_gflops"]:15.6f} {stats["gflops"]:28.6f}')
PY
```

The counter reports two computation totals consistently for all variants: Conv1d/Conv2d/Linear multiply-adds (two operations per MAC), and those operations plus the explicitly modeled reference-operator arithmetic. The second includes ECA means/scaling, CBAM means/scaling/descriptor addition, and DySample offset normalization and bilinear sampling. As in the existing counter, nonlinear activations, comparisons, indexing, and memory movement are excluded. The two totals are separately labeled; use one definition consistently across a comparison. Standard THOP summaries omit DySample's `grid_sample` sampling arithmetic.

## Checks without training

```bash
python -m unittest discover -s tools/tests -p test_mcgi_targeted.py -v
python -m unittest discover -s tests -p test_run_experiment.py -v
```

Checks cover isolated YAML changes, preservation of original C2f parameter paths, split-forward equivalence, full detector forward/backward and fusion, zero-offset DySample equivalence to bilinear interpolation, group/batch isolation, and computation counting. The experiment-script tests use a fake YOLO command and temporary files, not actual training.
