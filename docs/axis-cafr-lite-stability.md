# Axis + CAFR-Lite：多随机种子诊断与下一轮优化

分析日期：2026-10-17。来源为用户服务器 `runs/detect` 中的 `args.yaml`、`results.csv` 和 `test_metrics.log`。仅只读采集了日志、模型实现与数据统计，没有启动训练或修改服务器任务。原始分析副本保存在本地忽略目录 `paper/work/axis-cafr-stability`。

## 1. 目前的收益不稳定

取得 41 个已有完整测试日志的 axis-cafr-lite 运行，seed 为 0–40；读取时 seed 41 仅有部分训练曲线，没有纳入。YOLOv8n 有 8 个不同 seed 的完整结果，重复的 seed 11 只计一次。

当前 axis-cafr-lite 的 **测试 AP 均值 60.27%，样本标准差 3.26 个百分点，范围 53.2%–65.6%**。验证集 best AP 均值为 60.19%，标准差 2.12 个百分点。不能用最高分代表典型表现。

与 YOLOv8n 配对的共同 seed 为 **1、11、12、13、14、15、20、30**：

| 模型 | 测试 AP 均值 | 样本标准差 |
| --- | ---: | ---: |
| YOLOv8n，8 个共同 seed | 60.36% | 2.55 个百分点 |
| Axis + CAFR-Lite，同样 8 个 seed | 58.44% | 4.59 个百分点 |

共同 seed 下，改进模型赢 3 次、输 5 次，平均差为 **−1.92 个百分点**。这是已有日志的描述性配对统计，不代表因果机制或显著性检验。8 个 seed 的结果也不能与全部 41 个 seed 混用。

实际训练预算还受早停影响：例如 seed 15 的基线完成 300 轮，改进模型在 160 轮结束。因此这些分数同时反映结构和训练轨迹差异，不能直接认定新增模块必然有害；下一轮统一训练日程正是为了分清这一点。

详细的 49 条去重记录在 `docs/axis-cafr-lite-seed-results.csv`。验证分数取 CSV 中的最高 AP；测试分数取日志打印值，保留了其有限打印精度。服务器指标实现以 AP50–95 作为 fitness，因此该验证行对应日志中用于选 best 的指标。

| seed | YOLOv8n 测试 AP | Axis + CAFR-Lite 测试 AP | 差值，百分点 |
| --- | ---: | ---: | ---: |
| 1 | 61.8 | 64.8 | +3.0 |
| 11 | 59.3 | 60.5 | +1.2 |
| 12 | 58.9 | 55.5 | −3.4 |
| 13 | 62.7 | 60.0 | −2.7 |
| 14 | 58.4 | 55.1 | −3.3 |
| 15 | 63.3 | 53.3 | −10.0 |
| 20 | 62.4 | 64.3 | +1.9 |
| 30 | 56.1 | 54.0 | −2.1 |

## 2. 有证据支持优先检查的两个因素

### 训练常常在 Mosaic 关闭之前结束

这些运行使用 `epochs=300, patience=80, mosaic=1.0, scale=0.5, close_mosaic=10`。默认旋转为 0，学习率调度为线性，`optimizer=auto`。CSV 中各组初始阶段学习率一致，不能直接把 args 中的 `lr0=0.01` 当作 auto 实际采用的学习率。

41 次中，**19 次在第 290 轮前结束**，没有进入最后十轮的非 Mosaic 训练；另外 3 次结束于第 291 轮；19 次完成 300 轮。举例：seed 15 在第 80 轮达到最高验证 AP，最终第 160 轮早停；seed 30 的最高验证 AP 在第 96 轮，最终第 176 轮早停。

完成 300 轮的组，测试 AP 均值约 62.19%；第 290 轮前结束的组约 57.92%。**这是相关性，不证明提前关闭 Mosaic 一定能改善结果**：两组本来就由训练轨迹和验证分数筛选，存在选择效应。但它明确指出了一个可以在不改变模型、也不增加推理开销的情况下先测试的流程因素。

### 竖向瑕疵的定位波动最大

全部 41 次运行的分类别测试 AP：

| 类别 | 平均 AP | 样本标准差 | 范围 |
| --- | ---: | ---: | ---: |
| row | 62.60% | 2.06 | 58.7%–67.4% |
| col | 59.24% | 6.77 | 40.8%–70.4% |
| hole | 58.97% | 3.74 | 45.9%–65.0% |

测试集 col 只有 12 个实例。框高度中位值（相对原图高度）在 train、val、test 中分别为 **0.5002、0.3993、0.7617**，说明目前测试中的竖向框更长；该观察的样本量有限，不能据此断言总体分布。Mosaic 裁切和较大的缩放幅度是否破坏长缺陷连续性，是下一轮的可检验假设。

字节级哈希未发现数据集合间完全相同的图像文件，但没有验证相邻帧或近似图像是否独立。服务器 `block.py` 与本地实现一致，模型配置也核对为 P3/P4 轴向残差、两处 CAFR-Lite，没有先假设部署的是另一种结构。

## 3. 先控制训练，再做放置位置对照

新增两个训练配置，均适用于基线与候选，不改模型定义：

| 配置 | 与旧配置相比的改动 | 要验证的问题 |
| --- | --- | --- |
| `ultralytics/cfg/experiments/fabric-stability-schedule.yaml` | `patience=0`，`close_mosaic=100`；其余保留旧 Mosaic 和 scale 强度 | 完整走过收尾阶段是否改善典型表现与波动 |
| `ultralytics/cfg/experiments/fabric-stability-geometry.yaml` | 在上项基础上，`mosaic=0.25`、`scale=0.2` | 较少裁拼和较温和缩放是否更适合连续细长缺陷 |

两个配置均沿用 300 轮、batch 16、640 输入、workers 8、auto 优化器及 deterministic。第一项主要改变训练日程；第二项同时改变两项增强强度，若有收益，仍不能单独归因于 Mosaic 或 scale 中的某一项。

`patience=0` 在当前实现中关闭早停，不是训练零轮。best.pt 仍按验证 AP 选择，**不会强制选最后一轮或收尾阶段的 checkpoint**。保持验证集选 checkpoint，测试集用于最终报告；不要继续根据某个测试 seed 的最高分筛选结构。

没有把旋转、类别权重、损失函数、学习率、骨干压缩同时加入这一轮，避免无法解释改动来源。后续若加小角度增强，另作单因素实验。

### 新结构候选：只保留 P5→P4 的 CAFR-Lite

新增 `ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite-p4.yaml`：

- 保留 P3/P4 的 C2fAxisResidual，保留所有基线 bottleneck。
- 保留第 10 层 P5→P4 的低通道动态重组。
- 第 13 层 P4→P3 使用普通最近邻插值。
- 层编号、后续 Concat、C2f 和三尺度 Detect 保持一致。

它用于测试“两个尺度都引入独立动态残差是否必要”。P5→P4 选择优先作用于更深层特征，同时少一处较高分辨率的重组和 Softmax；**目前没有新精度、稳定性或 RK3566 速度结果，不将减少一处模块等同于已验证加速或提升。** 旧结构保留，不会改变已有 checkpoint 的类定义。

C2fAxisResidual 与 CAFR-Lite 当前的零初始化 gate 会使附加分支权重第一步梯度为零，随后 gate 打开才更新；这是现有测试已覆盖的设计行为，尚无日志证据证明它就是跨 seed 波动原因。本轮不据此直接更改初始化或 gate 公式。

本地 CPU 统计（输入 640×640、未融合、当前 THOP 版本）：

| 模型 | 参数量 | THOP GFLOPs |
| --- | ---: | ---: |
| axis-only | 3,011,649 | 8.19415 |
| 两处 CAFR-Lite | 3,030,555 | 8.24934 |
| 单处 CAFR-Lite | 3,025,606 | 8.21801 |

THOP 不完整覆盖函数式重组、Softmax 和访存，这些数值不代替 RK3566 实测。原有 axis-only 与双处模型的约 0.0003 G 差异来自当前计数环境；不能当作结构变化。

本轮本地 6 项测试通过：低通道重组与 unfold 参考的输出/梯度、零门控初始行为、检测损失可训练性、旧/新模型的真实 ONNX Runtime 数值一致性。新候选额外检查保留全部 axis-only 权重、非零门控实际进入导出图、仅有一处重组 Softmax。两个训练 YAML 均通过框架配置解析。测试环境为 CPU PyTorch 2.9.1、ONNX 1.18.0、ONNX Runtime 1.23.2，使用工作区隔离依赖；没有真实数据训练或 NPU 测速。

## 4. 运行方式

以下命令在包含新文件的仓库中执行；本轮没有上传配置到服务器或启动训练。使用新的实验目录，保留已有结果。先预先固定 seed 0、1、2 做流程筛查，完整报告三次结果，不按最高分决定结论；它们只是筛查，最终仍需更完整的配对重复实验。

先保持旧模型，只检查日程：

```bash
for seed in 0 1 2; do
  yolo detect train \
    cfg=ultralytics/cfg/experiments/fabric-stability-schedule.yaml \
    model=ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite.yaml \
    pretrained=yolov8n.pt data=/home/tkz/datasets/pingwen_yolo/data.yaml \
    seed=$seed device=0 project="$PWD/runs/fabric-stability" name=lite-schedule-s$seed
done
```

再用同样模型比较温和增强，将上面 `cfg` 改为 `fabric-stability-geometry.yaml`，实验名称改为 `lite-geometry-s$seed`。两组的 mean/std 都看，不能只比较各自最高分。

确认训练配方后，再保持同配方比较以下四个模型：

| 对照 | model | pretrained |
| --- | --- | --- |
| 基线 | `yolov8n.pt` | `true` |
| axis-only | `ultralytics/cfg/models/v8/yolov8n-fabric-axis-residual.yaml` | `yolov8n.pt` |
| 两处 CAFR-Lite | `ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite.yaml` | `yolov8n.pt` |
| 单处 CAFR-Lite | `ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite-p4.yaml` | `yolov8n.pt` |

例如单处候选：

```bash
yolo detect train \
  cfg=ultralytics/cfg/experiments/fabric-stability-geometry.yaml \
  model=ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite-p4.yaml \
  pretrained=yolov8n.pt data=/home/tkz/datasets/pingwen_yolo/data.yaml \
  seed=0 device=0 project="$PWD/runs/fabric-stability" name=lite-p4-geometry-s0
```

部署仍通过 `tools/export_rkopt_onnx.py` 使用九输出导出；单处候选应报告 `Native CAFR-Lite modules: ['model.10']`。使用同一校准与板端配置重新评估，不能沿用旧双处模型约 20 FPS 的口头结果。

在上述对照前，不建议继续增加新模块或替换骨干。若日程和增强调整后 axis-only 与两处/单处 CAFR 的配对均值仍无优势，再考虑改写方向信息与定位重建的机制，避免在尚未分清的训练波动上叠加新的自由度。
