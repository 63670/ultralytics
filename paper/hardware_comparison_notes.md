# 硬件对比表的数据口径与待补材料

日期：2026-10-14。此文件是写作核查记录，不是投稿正文。

## 本次确认

- 用户确认实际部署设备已由 RK3566 更换为 RK3588 自研边缘板，不是 Radxa ROCK 5B。
- 正文、图注、训练/部署协议中的平台名称统一为 RK3588；不编辑用户自行绘制的图片。
- 保留明确型号对应的瑕疵检测研究，不引入机场、鱼类或辣椒目标检测工作。
- 实际自研板的内存配置和成本未提供。正文平台表标 NR，不能把 ROCK 5B 8 GB 的 149 美元价格套用到自研板。
- Ascend 310 的原论文没有明确报告 Atlas 200，因此不将“Ascend 310 + Atlas 200，550 USD”作为原论文实际部署配置写入对比表。

## 用户提供的参考价格

| 产品 | USD | 目前证据状态 |
| --- | ---: | --- |
| Jetson TX2 | 599 | NVIDIA 2017 发布资料中的开发套件价格；不是单独模块价格，也不是原论文整机成本。 |
| Jetson TX2 NX | 249 | 用户提供，报价链接、批量条件和日期待补。只能作为参考价，不称官方首发价。 |
| Jetson Xavier NX | 399 | NVIDIA 发布资料中的模块批量价格，非完整检测设备价格。 |
| Jetson Orin Nano | 499 | 早期开发套件参考价；不是当前统一报价。不能与后续 Super 版本配置混为一谈。 |
| Xilinx ZCU102 | 3234 | AMD 官方美国商店当前评价套件标价可核查。 |
| Basler raL2048-48gm | 1647 | 用户提供；尚未确认历史官方首发价格来源。 |
| Basler acA2500-14gm | 629 | 用户提供；尚未确认历史官方首发价格来源。 |
| Intel RealSense D435i | 349 | 用户提供；价格时间和产品包范围待补。 |
| VEYE MV-MIPI-SC130M | 51 | 用户提供；报价来源、镜头/配件是否包含待补。 |

用户要求先保留价格、后补来源。稿件使用“indicative component reference prices”，不标成官方首发价、实际采购价或完整系统成本。仍建议在投稿前补齐来源、日期及价格范围，不能仅靠表中价格推导我们整机更便宜或给出成本下降百分比。

## 技术规格纠正

- TX2 / TX2 NX 的约 1.33 是 FP16 TFLOPS，不是 INT8 TOPS。官方产品页面：https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-tx2/
- Orin Nano 原始 8 GB 的 40 TOPS 是 sparse INT8 标称口径；与 dense TOPS 不直接等价，且与后续软件/功耗配置的数值不同。
- ZCU102 的 3.45 INT8 TOPS 对应 AMD AI SDK 中三核 B4096 DPU 配置；KAM-Net 论文实际采用自定义 16-bit 定点 FPGA 加速器，因此稿件写 implementation-dependent，不套用 3.45 TOPS。
- ZCU102 的 PS 4 GB 与 PL 0.5 GB 属于分离的内存，不能当作普通统一 4.5 GB 内存。
- Basler acA2500-14gm 官方分辨率是 2592 × 1944，不是 2592 × 1942。
- MV-MIPI-SC130M 官方整机页面使用 SC130GS-M 传感器，最大输出参数与 SC130GS 传感器本身的 240 fps 不同。按此前全文不写帧率的要求，正文新相机表不列帧率。
- 厂商规格仅作为产品额定能力，不是原论文/我们系统的实测性能；实际采集尺寸可不同。

## 后续需补/确认

1. 自研 RK3588 板内存、实际 BOM 成本及成本包含范围。
2. 若沿用 Android 13 和 INT8 RKNN 部署描述，确认换板后的实际运行环境仍一致。
3. 确认 hardware.png、system.png、site.png 等图片是否已同步 RK3588 型号；本轮只改文字，不改图。
4. 相机/平台报价的来源、日期、批量条件、税费和配件范围。
5. 原论文只报告某个 SoC/模块时，不擅自补具体载板或整机型号。

## 官方核查入口

- TX2 599 USD 开发套件发布资料：https://nvidianews.nvidia.com/_gallery/download_pdf/58bf66eced6ae50e6f680e10/
- Xavier NX 399 USD 模块发布说明：https://developer.nvidia.com/blog/introducing-jetson-xavier-nx-the-worlds-smallest-ai-supercomputer/
- 原始 Orin Nano 40 TOPS：https://nvidianews.nvidia.com/news/nvidia-jetson-orin-nano-sets-new-standard-for-entry-level-edge-ai-and-robotics-with-80x-performance-leap
- ZCU102 3234 USD：https://shop-us-en.amd.com/adaptive-embedded-computing/soc-fpga-evaluation-kits/
- ZCU102 3.45 TOPS 的特定 DPU 条件：https://docs.amd.com/r/3.0-English/ug1354-xilinx-ai-sdk/ZCU102-Evaluation-Kit
- acA2500-14gm：https://www.baslerweb.com/en-us/shop/aca2500-14gm/
- raL2048-48gm：https://www.baslerweb.com/en-us/shop/ral2048-48gm/
- D435i：https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html
- MV-MIPI-SC130M：https://www.veye.cc/en/product/mv-mipi-sc130m/
- RK3588：https://www.rock-chips.com/uploads/pdf/2022.8.26/191/RK3588%20Brief%20Datasheet.pdf
