# RK3566 连板推理性能测试

在 PC 上用 RKNN-Toolkit2 加载**实际部署的 `.rknn` 文件**，通过 USB/ADB 连接 RK3566，调用官方 `eval_perf()` 获取板端模型运行耗时。脚本不执行读图、缩放、检测框解码、NMS 或绘图，也不重新转换模型。

## 1. 准备连接

- 在支持的 Linux PC 环境中安装 RKNN-Toolkit2，优先沿用模型转换时的 SDK 版本。Windows 可使用按官方教程配置的 WSL 或 Linux 虚拟机；USB 设备必须实际接入该环境。
- RK3566 使用支持 ADB 的 USB 接口连接 PC，确认板端已有可用的 NPU 驱动、`librknnrt.so` 和 `rknn_server`。沿用板卡厂商或 SDK 的安装、启动方式，保持 Toolkit、runtime/server 和驱动兼容。仅充电或仅供电接口不能用于连板测试。
- **在运行 Python 的同一个环境里**执行：

```bash
adb devices
adb shell "ps -A | grep rknn_server"
python -c "from rknn.api import RKNN; r = RKNN(); print(r.list_devices()); r.release()"
```

`adb devices` 应显示设备为 `device`；`rknn_server` 应已启动。若没有 server，先按该板卡 SDK 的说明启动，不能只靠 PC 上安装 Python 包。多块板时，使用 `list_devices()` 列出的设备 ID 传给脚本的 `--device-id`。

这里使用 PC 端的 `rknn.api.RKNN`，板端的 `RKNNLite` 是另一种部署方式。

## 2. 只测模型推理耗时

以下命令在仓库根目录执行。模型文件名是示例，请替换为三个实际部署的文件；若导出结果是目录，传入其中的 `.rknn` 文件。

```bash
python tools/benchmark_rknn.py --model models/yolov8n.rknn --output results/yolov8n-perf.txt
python tools/benchmark_rknn.py --model models/yolov8n-dme.rknn --output results/dme-perf.txt
python tools/benchmark_rknn.py --model models/yolov8n-dme-cafr.rknn --output results/full-perf.txt
```

默认 `target=rk3566`、`perf_debug=False`，先丢弃 1 次 `eval_perf()` 报告，再保留 5 次报告。`--warmup` 和 `--repeat` 控制的是该 API 的调用次数，SDK 内部的运行次数由 SDK 自身控制。全程只有一次模型加载和运行环境初始化，二者不计入报告的模型运行时间。

最小实现就是：

```python
from rknn.api import RKNN

rknn = RKNN()
try:
    assert rknn.load_rknn("your-model.rknn") == 0
    assert rknn.init_runtime(target="rk3566", perf_debug=False) == 0
    rknn.eval_perf()
finally:
    rknn.release()
```

原始示例的 ImageNet 均值、方差、224×224 输入和分类后处理不适用于我们的 YOLO 模型，全部移除。测试已有 `.rknn` 时，其输入尺寸、量化方式、预处理配置已经在转换时确定，脚本不修改这些设置。

## 3. 如何读结果

优先看 SDK 输出的 `Total Time(us)` 和 `FPS`，注意时间单位：

```text
推理时间（ms） = Total Time(us) / 1000
batch=1 时推理 FPS = 1,000,000 / Total Time(us)
```

例如 `Total Time(us): 62500` 对应 62.5 ms，即 16 FPS；500000 us 对应 500 ms，即 2 FPS。**这些只是单位换算示例，不是新的实测结果。** 10 FPS 的模型时间预算是每张 100 ms。

`eval_perf()` 提供的是 SDK 定义的板端模型性能口径，不能将其直接称为整个相机系统的实时帧率，也不能未经核实就说是“所有操作都在 NPU 上执行”的耗时。模型图内部包含的解码或其他操作仍属于模型的一部分；本脚本只去掉应用端处理。

不要在 PC 上用 `time.time()` 包围连板 `inference()`，再把这个时间写成板端推理时间：它还包含 USB/ADB 通信、数据传输和主机调用等开销。报告文本保存 SDK 返回的原始字符串；若 SDK 版本的格式不同，按其显式标注的单位读取，不猜测字段。

三个模型比较时，统一实际输入尺寸、batch=1、INT8/浮点配置、SDK/runtime 版本以及板卡频率、散热和后台负载。脚本保留模型 SHA256，便于确认测的是哪一个文件。频率信息等 SDK 控制台日志可额外用 `2>&1 | tee results/full-console.txt` 保存；`--output` 保存的是 SDK 返回的性能报告，不保证包含所有原生库日志。

## 4. 单独定位完整模型的瓶颈

```bash
python tools/benchmark_rknn.py --model models/yolov8n-dme-cafr.rknn --profile --warmup 1 --repeat 1 --output results/full-profile.txt
```

`--profile` 设置 `perf_debug=True`，让 SDK 提供逐层耗时和可用的执行信息。逐层采样会引入额外开销，应使用不带 `--profile` 的结果比较三个模型的整体速度。

优先检查 CAFR 对应的特征展开、重排、Softmax、乘法和归约附近的耗时；有 CPU/NPU 执行归属字段时一并检查。仅凭 2 FPS 不能认定算子回退到了 CPU。若纯模型测试仍约 500 ms，说明瓶颈确实在模型执行路径；若显著小于 500 ms，再回查应用的数据搬运、预处理和后处理。

## 官方资料

- [RKNN-Toolkit2：工具、平台和 SDK 下载](https://github.com/airockchip/rknn-toolkit2)
- [Toolkit2 2.3.2 API 参考：init_runtime、eval_perf](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/doc/03_Rockchip_RKNPU_API_Reference_RKNN_Toolkit2_V2.3.2_EN.pdf)
- [Toolkit2 API 差异：eval_perf 无 inputs 和 loop_cnt 参数](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/doc/RKNNToolKit2_API_Difference_With_Toolkit1-2.3.2.md)
- [官方 WSL 配置说明](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/doc/WSL%E4%B8%AD%E4%BD%BF%E7%94%A8RKNN_ToolKit2.md)
