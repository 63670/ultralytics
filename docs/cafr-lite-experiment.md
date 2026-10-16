# CAFR 四维结果与低成本替代实验

## 四维改写的用户实测反馈

来源为用户提供的两份 RKNN Layer Profiling。新报告解析出 ID 0–301 共 302 个算子，
逐行 Time(us) 之和为 206,635 us，与报告总计一致。

| 项目 | 原 CAFR | 四维 CAFR |
| --- | ---: | ---: |
| 逐层总耗时 | 590.470 ms | 206.635 ms |
| CPU | 442.856 ms | 11.170 ms |
| NPU | 147.614 ms | 195.465 ms |
| 第一处 CAFR，P5→P4 | 184.952 ms | 52.133 ms |
| 第二处 CAFR，P4→P3 | 351.610 ms | 97.374 ms |
| 其余网络及输入输出 | 53.908 ms | 57.128 ms |
| 算子总数 | 160 | 302 |
| 报告的每帧总读写量 | 319,058 KB | 386,757 KB |

两份逐层数据中的总耗时相差约 2.86 倍，但该比例不是关闭 profiling 后的正式测速结论，
附件也没有频率、SDK/runtime 等完整环境信息。

新报告不再有 CPU Transpose/Reshape，仅剩两个 Softmax 和输入输出在 CPU。
两处 CAFR 仍合计 149.507 ms，约占全图 72.35%。
其 50 次 Mul 合计 59.215 ms、48 次重组 Add 合计 44.538 ms、60 次 Slice 合计 27.760 ms。
全图 Add 为 58 次、49.309 ms，不能把其他残差加法也都算成 CAFR。
每帧读写量增加约 21.2%，因此主要问题已从 CPU 布局转换转为多次全通道乘加、切片和访存。

## DME 也有实际开销

新报告中 `/direction/` 分支均在 NPU，其逐层耗时为：

| 原 DME 所在阶段 | 耗时 |
| --- | ---: |
| P2，160×160 | 7.970 ms |
| P3，80×80 | 4.546 ms |
| P4，40×40 | 2.773 ms |
| P5，20×20 | 1.928 ms |
| 合计 | 17.217 ms |

约占非 CAFR 部分 57.128 ms 的 30.1%，P2 一处占 DME 约 46.3%。
原 DME 三个逐通道分支后还有拼接、1×1 融合和残差相加，不能因都在 NPU 上就忽略开销。
因此本次候选选用此前用户报告 AP=62.7% 的轴向残差结构，避免继续默认采用四阶段三分支 DME。
轴向残差的真实部署开销尚未由本报告验证，ReduceMean 等操作也必须检查 Target。

## 替代候选：低通道 3×3 残差重组

模型配置：`ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite.yaml`。

- C2f 使用现有 axis-residual 对照中的 P3/P4 方向上下文，保留原始 bottleneck 和可迁移权重。
- 保留全通道最近邻上采样作为主路径。
- 将 P5 的 256 通道压缩到 16 通道，P4 的 128 通道压缩到 8 通道。
- 只对这些低通道特征做 3×3、四维逐项内容重组，再用 1×1 卷积投影回原通道。
- 用从零初始化的可学习 gain 添加重组残差；初始行为为原最近邻上采样。

与全通道 5×5 重组相比，参与邻域乘加的元素数量为 `9/(16×25)`，约 2.25%。
这只描述该部分运算规模，不包括压缩、核预测、投影、主路径和调度，不能解释为全模型加速 44 倍。
本版本改变计算机制，**不与原 CAFR 数学等价，需要重新训练**，不能继承原 CAFR 的 AP。

| 模型 | 参数量 | THOP GFLOPs（640×640） |
| --- | ---: | ---: |
| axis-residual | 3,011,649 | 8.193859 |
| axis + CAFR-Lite | 3,030,555 | 8.249052 |

THOP 不完整覆盖函数式乘加、Softmax 和访存；这张表不代替板端耗时。

## 建议先测部署预算

新模块 gain 初始为零，直接导出未训练模型可能被编译器裁掉整条分支。
下面显式开启门控，生成**随机权重、仅用于部署摸底的探针**；该图不能用来报告检测精度或最终模型性能。

```bash
python - <<'PY'
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2fAxisResidual, ContentAwareUpsampleLite
from ultralytics.utils.export.rkopt import export_rkopt_onnx

model = YOLO('ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite.yaml').model
with torch.no_grad():
    for module in model.modules():
        if isinstance(module, C2fAxisResidual):
            module.gamma.fill_(0.1)
        elif isinstance(module, ContentAwareUpsampleLite):
            module.gain.fill_(1.0)
info = export_rkopt_onnx(model, 'exports/axis-cafr-lite-probe.onnx')
print('LATENCY PROBE ONLY:', info['native_cafr_lite_modules'])
PY
```

用相同 INT8 转换配置和训练集校准列表转成 RKNN，先用非调试 eval_perf 检查是否接近 100 ms 以内，
再看逐层报告中 C2f 方向上下文、CAFR-Lite 的各自成本。随机权重的量化与编译结果不代替训练后模型的最终测速。

## 预算可行后再训练

```bash
yolo detect train model=ultralytics/cfg/models/v8/yolov8n-fabric-axis-cafr-lite.yaml data=/home/tkz/datasets/pingwen_yolo/data.yaml pretrained=yolov8n.pt imgsz=640 epochs=300 patience=80 batch=16 device=0 workers=4 seed=0 project="$PWD/runs/fabric" name=axis-cafr-lite
```

训练及数据设置与 axis-residual 对照保持一致，先比较验证集。
若从已训练的 62.7% checkpoint 继续训练，可作快速探索，但对照也需匹配微调预算，不能与原单次训练直接视作受控消融。

训练后的 best.pt 通过现有九输出工具导出即可，CAFR-Lite 已原生使用四维表示：

```bash
python tools/export_rkopt_onnx.py --weights runs/fabric/axis-cafr-lite/weights/best.pt --output exports/axis-cafr-lite.onnx --imgsz 640
```

日志应出现 `Native CAFR-Lite modules: ['model.10', 'model.13']`。
仍需重新校准、核验 INT8 精度，并在同硬件条件下复测；本次没有真实训练或板端速度结果。

## 已完成检查

低通道重组与 unfold 参考的输出/输入梯度一致；载入同一个完整 axis-residual checkpoint 后，
零 gain 的新模型初始输出与对照一致；真实检测损失能先更新 gain、随后更新核预测器；
非零分支实际 ONNX 导出和 ONNX Runtime 数值比对通过。
共享四维重组函数的原有 7 项测试也继续通过，总计 11 项。
