# Sobel 与 Phase 的位置分工实验

本次只修改连接位置，不改已有两个类的内部计算，不覆盖 v1 配置、权重、训练结果或论文。

## v1 的观测

用户提供的三个配置均完成 seed 0–4。测试集均值和样本标准差如下，单位为百分比：

| 配置 | 测试 mAP50 | 测试 mAP50–95 | 最佳验证 mAP50–95 均值 |
|---|---:|---:|---:|
| Sobel 单独 | 94.02 ± 4.17 | 60.76 ± 5.51 | 59.88 |
| Phase 单独 | 95.32 ± 1.44 | 63.44 ± 3.27 | 61.60 |
| Sobel + Phase | 93.54 ± 3.21 | 60.64 ± 3.43 | 62.15 |

已只读检查十五次运行的 args.yaml 与 results.csv。预训练入口均为 yolov8n.pt，输入 640、batch 16、最多 300 epochs、patience 80、optimizer=auto；所核对的其余训练超参数一致，seed 确实为 0–4。组合版测试均值比 Phase 单独低 2.80 个百分点，验证均值却略高，因而尚不能把下降归结为必然的优化失败。

在八张固定验证图像上检查已训练分支的平均绝对修正量与输入量之比，组合版的骨干 P3 Sobel 修正约为 1.7%–2.3%，P4 约为 8.0%–10.4%。Phase 的两个修正分支没有出现相对单模块明显失控的全局幅值。这些观测用于提出位置调整假设，不构成“边缘被重复放大”这一具体机制的证明。

## 修改：先重构上下文，再处理融合后的边界

v1 的骨干 P3/P4 Sobel 修正会沿后续骨干传入 P5，再同时影响 Phase 的局部修正滤波与四相位权重预测。两个模块都从同一条已被改写的特征流学习，增加了联合适配的负担。

推荐候选 `yolov8n-fabric-phase-sobel-neck.yaml`：

- 骨干四个 C2f 全部保持原始结构，向 Phase 和侧向分支提供特征。
- 两处 Phase 保持在原位置：P5→P4、P4→P3。
- 只在最后一个自顶向下 P3 C2f（层 15）内部加入 Sobel。它处理最终隐藏分支的 1/4 通道，其他路径不变。
- Sobel 此时接收上采样上下文与 P3 侧向特征融合后的表征，学习边界修正。其输出继续用于 P3 检测及后续自底向上聚合。

```text
原始骨干 ─→ Phase(P5→P4) ─→ P4融合 ─→ Phase(P4→P3)
                                                   ↓
原始 P3 侧向特征 ───────────────────────────────→ Concat
                                                   ↓
                                    C2f 内的部分通道 Sobel
                                                   ↓
                                      P3检测 / 自底向上聚合
```

在一次前向计算中，修改该 Sobel 分支的参数不会改变两个 Phase 的输入或输出。这个性质可以直接通过 hook 验证。所有模块仍由同一个检测损失联合训练，反向梯度仍然相互关联；这里分离的是直接的前向输入路径。

该调整同时利用融合后的上下文选择边界，而不是仅在骨干中从局部梯度选择边界。它是一项待训练验证的假设，不预先保证改善泛化。

## 等参数量的位置对照

提供 `yolov8n-fabric-sobel-p3-phase.yaml` 作为对照：仅保留骨干 P3（层 4）中的 Sobel，P4 恢复原 C2f，两处 Phase 保持不变。

两个新候选都只使用一个 Sobel C2f、8 个活动通道，均比 Phase 单独增加 144 个可训练参数。新候选的全部参数形状与 Phase 单独的基础路径兼容，新增边缘投影仍零初始化。这样可分别比较：

1. v1 组合 vs. 骨干 P3 对照：减少 Sobel 嵌入位置是否有效。
2. 骨干 P3 对照 vs. 融合后 P3 候选：在相同新增参数量下，连接位置是否有效。
3. 融合后 P3 候选 vs. Phase 单独：边界修正是否带来稳定增益。

## 训练命令

先跑推荐候选，仍采用同样的五个种子和训练参数，避免同时改变结构与训练策略：

```bash
conda activate tkz-yolo
cd /home/tkz/code/github/ultralytics
git switch codex/jiot-sobel-phase
git pull --ff-only

bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-phase-sobel-neck.yaml \
  phase-sobel-neck-v2 pretrained=yolov8n.pt workers=4
```

结果位于 `runs/detect/phase-sobel-neck-v2/<seed>/`，每个种子包含 `weights/best.pt` 与 `test_metrics.log`。

需要区分位置作用时，运行等参数量的骨干 P3 对照：

```bash
bash scripts/run_experiment.sh --device 0 --seeds 0,1,2,3,4 \
  ultralytics/cfg/models/v8/yolov8n-fabric-sobel-p3-phase.yaml \
  sobel-p3-phase-control-v2 pretrained=yolov8n.pt workers=4
```

以验证集的均值与波动进行结构选择，最终再报告测试集表现，不用某个测试集高分种子代替多种子结果。

## 验证范围

`tools/tests/test_sobel_phase_placement.py` 验证权重迁移、等新增参数量、非零 Sobel 修正不改变上游 Phase 输出、对照配置仍存在直接路径、真实检测损失反传，以及 split、卷积融合、checkpoint 和 RKOPT ONNX 一致性。

已在服务器独立临时目录用 CPU 完成 5 项新测试和 6 项 v1 回归测试，全部通过。另只读载入 Phase 单独 seed 0 的真实 best.pt，两个新候选零初始化时均与其预测完全一致（最大差值为 0）。

两个候选均为 3,016,577 个未融合参数、3,011,377 个融合后参数，融合后的 THOP 估算为 8.0965 GFLOPs。旧配置和现有权重保持原样。上述检查验证实现及连接关系，新配置的训练精度和 RK3566 实测速度需要分别验证。
