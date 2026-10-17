# 原始轴向模块 + 新门控上采样的组件消融

本轮固定 `C2fAxisResidual`（P3/P4 两处、partial=0.25）和 `GatedInterpolationUpsample`（两处、initial_mix=0.05）。不混入 `C2fAxisBounded`、CAFR/CAFR-Lite 或 P2 细节融合，以形成清楚的 2×2 组件对照。

## 2026-10-17：已发现实际运行的初始化条件不一致

只读核对训练服务器的 `args.yaml`、测试日志和 `BaseTrainer.setup_model()` 后发现以下差异。三份单轴向、单门控与组合模型 YAML 的 SHA-256 均与本地已验证版本一致，已定位的差异在训练初始化参数，而非这三份模型配置文件被意外改动。

| 已核查的运行 | 实际保存的 pretrained | 测试 AP50–95 |
| --- | --- | --- |
| `ablation-axis-gate/axis-only/0` | `true` | 49.4% |
| `ablation-axis-gate/axis-only/1` | `true` | 45.2% |
| `ablation-axis-gate/axis-only/2` | `true` | 核查时未读取到完成结果 |
| `ablation-axis-gate/gate-only/0` | `true` | 核查时未读取到完成结果 |
| `axis-interp/0` | `yolov8n.pt` | 63.5% |
| `axis-interp/1` | `yolov8n.pt` | 65.5% |

这些单模块运行的模型入口为 YAML，`resume=false`。当前代码只有在 `pretrained` 是权重路径字符串时才显式加载该 checkpoint；布尔 `true` 不会为 YAML 自动选择 `yolov8n.pt`，上述单模块训练因此没有以指定预训练 checkpoint 初始化。这与组合组的初始化条件不同，不能将当前差值直接解释为模块协同，也不能据此认定单模块性能差的全部原因已被确定。

下方命令已显式设置 `pretrained=yolov8n.pt`；实际核查到的运行并没有保存这个值。漏传参数的具体启动过程未核查，不推断是某一个 shell 操作导致。脚本现增加检查：YAML 入口省略 `pretrained`、设置为空或仅设为 `True` 时，在训练前报错；有意从头训练须显式传 `pretrained=False`。

重做时保留旧结果，使用新实验目录，并在开始阶段确认有 `Transferred ... items from pretrained weights` 日志、保存的 `args.yaml` 为同一个明确的权重路径。仅编辑旧 `args.yaml` 或测试旧 `best.pt` 不能改变训练时的初始化条件。本次仅修改本地脚本与说明，没有停止或重启服务器上的任务。

## 四组对照与运行命令

| 组别 | 模型参数 | 轴向模块 | 新门控 |
| --- | --- | --- | --- |
| baseline | `yolov8n.yaml` | 无 | 无 |
| axis-only | `ultralytics/cfg/models/v8/yolov8n-fabric-axis-residual.yaml` | 有 | 无 |
| gate-only | `ultralytics/cfg/models/v8/yolov8n-fabric-interp.yaml` | 无 | 有 |
| axis-gate | `ultralytics/cfg/models/v8/yolov8n-fabric-axis-interp.yaml` | 有 | 有 |

以下 Linux/Bash 命令在仓库根目录执行。先将上述配置和现有模块实现同步到训练端，并在训练环境中使用本仓库安装的 Ultralytics。命令沿用前一轮训练参数，不使用额外的 stability 训练配方。四组使用相同数据、预训练来源和 seeds；这是训练协议一致的组件对照，不宣称不同结构的所有新层或类别头随机初值逐元素一致。

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics

export DATA_YAML=/home/tkz/datasets/pingwen_yolo/data.yaml
export RUNS_DIR="$PWD/runs/detect"

DEVICE=1
SEEDS=0,1,2,3,4
COMMON=(
  pretrained=yolov8n.pt
  imgsz=640 epochs=300 patience=80 batch=16 workers=8
  optimizer=auto deterministic=True mosaic=1.0 scale=0.5 close_mosaic=10
  max_det=16
)

bash scripts/run_experiment.sh --device "$DEVICE" --seeds "$SEEDS" \
  yolov8n.yaml ablation-axis-gate/baseline "${COMMON[@]}" &&
bash scripts/run_experiment.sh --device "$DEVICE" --seeds "$SEEDS" \
  ultralytics/cfg/models/v8/yolov8n-fabric-axis-residual.yaml \
  ablation-axis-gate/axis-only "${COMMON[@]}" &&
bash scripts/run_experiment.sh --device "$DEVICE" --seeds "$SEEDS" \
  ultralytics/cfg/models/v8/yolov8n-fabric-interp.yaml \
  ablation-axis-gate/gate-only "${COMMON[@]}" &&
bash scripts/run_experiment.sh --device "$DEVICE" --seeds "$SEEDS" \
  ultralytics/cfg/models/v8/yolov8n-fabric-axis-interp.yaml \
  ablation-axis-gate/axis-gate "${COMMON[@]}"
```

该命令总计 20 次训练，在选定 GPU 上顺序执行。可将 `SEEDS=0,1,2` 作为首轮 12 次训练；若需增加 seeds，保留现有结果，使用 `SEEDS=3,4` 补跑。输出目录已存在时脚本会拒绝覆盖，因此重跑相同 seeds 时需另设实验名称。

`run_experiment.sh` 在每次训练后，对按验证集选择的 `weights/best.pt` 自动执行 `split=test`，打印结果并保存 `test_metrics.log`。成功测试后，脚本删除该 run 中其他 `.pt`，仅保留 `best.pt`。此处复用现有脚本行为，没有修改脚本。

结果目录：

```text
runs/detect/ablation-axis-gate/<baseline|axis-only|gate-only|axis-gate>/<seed>/
  args.yaml
  results.csv
  weights/best.pt
  test/
  test_metrics.log
```

汇总打印全部测试日志：

```bash
for group in baseline axis-only gate-only axis-gate; do
  for seed in 0 1 2 3 4; do
    log="runs/detect/ablation-axis-gate/$group/$seed/test_metrics.log"
    if [[ -f "$log" ]]; then
      printf '\n===== %s / seed %s =====\n' "$group" "$seed"
      cat "$log"
    fi
  done
done
```

最终按预先指定的同一组 seeds 报告均值、标准差及配对差值，同时保留 row/col/hole 分类别结果。checkpoint 选择和后续设计判断以验证集为依据，不根据最高测试分数挑 seed。模型参数与计算量使用同一输入尺寸、类别数和融合状态统计；设备速度在训练完成后对对应权重另行测量。
