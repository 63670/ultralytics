# RK3566 测速导出差异核查

## 已确认的导出区别

用户反馈基线由瑞芯微 `ultralytics_yolov8` 导出，自定义 DMFR 模型由当前 Ultralytics 先导出 ONNX，再转换 RKNN。
先前约 31、16、2 FPS 的比较混入了导出图差异，不能直接作为各模块部署开销的受控消融。
这不改变此前 PyTorch 精度结果，但需要重做一致口径的部署比较。

本地检查的参考实现：

- `../ultralytics_yolov8/RKOPT_README.zh-CN.md`：移出 DFL 和后处理结构，增加分类分数总和分支；无需重训。
- `../ultralytics_yolov8/ultralytics/nn/modules/head.py`：`format == 'rknn'` 时按尺度输出 box、sigmoid(class)、clipped score_sum。
- `../ultralytics_yolov8/ultralytics/engine/exporter.py`：其 RKNN 导出实际生成 opset 12 ONNX，并取输入 batch 的第一张。
- `../rknn_model_zoo/examples/yolov8/python/yolov8.py`：在 CPU 后处理执行 DFL、框解码、筛选和 NMS。

瑞芯微文档的上游链接：[RKOPT 中文说明](https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.zh-CN.md)。
本次结论来自本地源码快照，不代表检查了上游所有版本。

当前普通 ONNX 导出虽然默认不带 NMS，仍在 Detect 内执行 DFL、框解码、stride 变换并拼成单个输出。
即使直接使用当前 `format=rknn`，现有代码也没有瑞芯微九输出分支；INT8 坐标归一化不能替代这种检测头拆分。
将普通 ONNX 直接交给 Toolkit 转换，不会自动等价于瑞芯微检测头导出。

| 项目 | 普通三类 YOLOv8 ONNX | 瑞芯微兼容布局 |
| --- | --- | --- |
| 640×640、batch=1 输出 | 通常 `[1,7,8400]` | 每尺度 box/class/score_sum，共 9 个 |
| 检测头 DFL、框解码 | 图内 | 图外 CPU 后处理 |
| 三尺度 box 输出 | 已解码后合并 | `[1,64,80,80]`、`[1,64,40,40]`、`[1,64,20,20]`（reg_max=16） |
| 分类输出 | 合并到最终张量 | 对应三个尺度，各为 3 通道 sigmoid 分数 |
| score_sum | 无单独分支 | 每尺度 1 通道；分数求和后截断至 `[0,1]` |

此外必须检查旧 ONNX 的输入 batch。瑞芯微参考导出强制 batch=1，而普通导出可能继承配置或调用中的其他 batch。
本次工具固定 batch=1，不能直接把不同 batch 的每次调用耗时当作每张图片速度比较。

## 当前自定义 C2f 的导出保护

检查发现原导出器会给所有 C2f 子类强行绑定 `forward_split`。
本仓库 `C2fDirectional`、`C2fDirectionalPre`、`C2fWWTEGated` 有自定义 forward，但继承的 base forward_split 会跳过额外运算。
现已修改为：标准 C2f 或自带 forward_split 的子类才切换；其他子类保留自己的 forward。
这确认了当前代码中的风险，不能在没有旧图和具体模块类型的情况下断言用户之前的所有 DMFR 导出都触发了该问题。

## 统一导出入口

在本仓库中加载自定义训练权重，直接使用新工具，无需将模型搬进较旧的瑞芯微仓库，也无需重训。
以下文件名为示例，替换为对应实际 checkpoint：

```bash
python tools/export_rkopt_onnx.py --weights /path/to/yolov8n-best.pt --output exports/baseline-rkopt.onnx --imgsz 640
python tools/export_rkopt_onnx.py --weights /path/to/dme-best.pt --output exports/dme-rkopt.onnx --imgsz 640
python tools/export_rkopt_onnx.py --weights /path/to/dmfr-best.pt --output exports/dmfr-rkopt.onnx --imgsz 640
```

工具只处理三尺度、非 end-to-end 的 Detect 检测头，不处理分割、姿态和旋转框模型。
它对模型做私有副本和 Conv-BN 融合，保留自定义主干/颈部 forward，将检测头替换成瑞芯微兼容输出。
默认静态 batch=1、FP32 ONNX、opset=12；输出旁保存 `.rkopt.json`，记录输出形状、类别和 checkpoint SHA256。
浮点 ONNX 输入为 RGB、NCHW、`[0,1]`；RKNN 使用相同的 mean=0、std=255 配置时，运行时应按其接口要求提供原始像素输入，避免重复归一化。

接着沿用同一个 Toolkit 版本、target=rk3566、INT8 配置和**同一份布匹训练集校准图片列表**转换三个 ONNX。
可使用已有转换脚本，或调用当前仓库的转换函数：

```python
from ultralytics.utils.export.rknn import onnx2rknn

onnx2rknn(
    onnx_file="exports/dmfr-rkopt.onnx",
    output_dir="exports/dmfr-rkopt_rknn_model",
    name="rk3566",
    quantize=8,
    dataset="/path/to/fabric-calibration.txt",  # 图片路径列表，不是数据集 YAML
    batch=1,
)
```

转换函数会产生 `dmfr-rkopt-rk3566.rknn`。基线、DME 和 DMFR 也用同样设置转换。
不要因参考脚本默认使用 COCO 校准数据，就对某一个模型单独改变校准域。

## 重测和后处理边界

```bash
python tools/benchmark_rknn.py --model exports/dmfr-rkopt_rknn_model/dmfr-rkopt-rk3566.rknn --output results/dmfr-rkopt-perf.txt
python tools/benchmark_rknn.py --model exports/dmfr-rkopt_rknn_model/dmfr-rkopt-rk3566.rknn --profile --repeat 1 --output results/dmfr-rkopt-profile.txt
```

所有模型统一用无 profile 的板端 eval_perf 报告比较模型延迟。raw 输出数据量与普通已解码输出不同，PC 外部掐表会混入输出传输等开销。
九输出模型需要 RKNN Model Zoo YOLOv8 风格后处理，不能直接交给当前默认 Ultralytics RKNN 后端的单输出框解释逻辑。
C/C++ demo 的类别数量和标签也需要对应为三类，不能沿用 COCO 80 类配置。

只测模型时无需执行 CPU DFL、解码或 NMS；报告完整系统 FPS 时必须把这些移出的操作计回。
原 CAFR 的邻域展开和动态加权仍保留，不能在未重测前保证 2 FPS 会恢复到 10 FPS，也不能把导出问题当作排除 CAFR 开销的证据。

## 已验证与待验证

已使用随机初始化的基线、DME+CAFR 和 axis-residual 模型验证九输出 ONNX 可导出，并通过 ONNX Runtime 逐输出比对。
另验证了外部 DFL/框解码与原检测头 FP32 输出一致、修正后普通 ONNX 导出保留 DME forward，
以及完整 CLI 加载带 DME+CAFR 的合成 checkpoint 并产生静态 640×640 九输出图与形状报告。
这些检查验证实现，不构成真实训练权重的量化精度、RKNN 编译或 RK3566 速度结论。

```bash
python -m unittest discover -s tools/tests -p test_rkopt_export.py -v
```
