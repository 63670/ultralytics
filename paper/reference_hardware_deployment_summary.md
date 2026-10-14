# 本地参考论文：边缘部署与成像硬件汇总

核查日期：2026-10-14。范围：`paper/reference` 当前的 37 篇 PDF。仅整理本地论文报告的信息，不补猜型号，不修改论文正文，不据器件型号推断整机成本。

页码均为 PDF 文件页码，从第一页开始计数。NR = 未报告；NR 不表示没有该设备或功能。“板端测试”不自动等同于真实产线连续运行验证。训练 GPU、控制 MCU 和实际推理设备分别记录。

## 1. 明确报告嵌入式/NPU/FPGA 部署或测试：10 篇

其中 4 篇直接面向布匹检测，3 篇面向其他瑕疵检测，3 篇属于其他目标检测场景。

| 编号 | 工作 / 来源 | 实际推理平台 | 相机或采集设备 | 证据位置与部署范围 |
| --- | --- | --- | --- | --- |
| E1 | Modified DenseNet for Automatic Fabric Defect Detection With Edge Computing for Minimizing Latency；IEEE Internet of Things Journal，2020 | 自制智能边缘相机；Cambricon 1H8 AI 处理器 IP + ARM Cortex-A7 + 视频编解码处理器 | 自制智能边缘相机；图像传感器/相机商品型号 NR | pp. 9–10，IV-B，Figs. 13–14；真实硬件及边缘/云端响应周期比较。1H8 是处理器 IP，不是可直接填入的完整开发板型号。 |
| E2 | EfficientDet for fabric defect detection based on edge computing；Journal of Engineered Fibers and Fabrics，2021 | NVIDIA Jetson TX2，8 GB；TensorRT 优化 | Basler acA2500-14gm；GigE；镜头 Basler C125-0618-5M-P；光源 OPT OPI-RI5000-W | p. 6，Experiments and discussion / Datasets and augmentation，Fig. 4；pp. 9–11 为板端实验。相机明确用于采集产线数据，不能额外推断为圆机边织边检系统。 |
| E3 | Textile Defect Detection Using Artificial Intelligence and Computer Vision—A Preliminary Deep Learning Approach；Electronics，2025 | NVIDIA Jetson Orin Nano，8 GB；TensorRT FP16 | 两台 Intel RealSense D435i RGB-D 相机 | pp. 12–16，系统架构、相机与现场安装；p. 18，3.4。布匹卷绕/检测线部署，具有本地展示、MQTT 上报及云端记录。D435i 支持深度成像不等于本文模型实际使用 RGB-D 融合。 |
| E4 | Development and Application of an AI Visual Defect Detection System for Warp-Knitted Lace Based on 5G+ Technology；Information，2026 | Huawei Ascend 310 + Atlas AI 平台/服务器；具体 Atlas 整机型号 NR | 正文写 Basler acA2440-35gm；Table 1 另列 IDS UI-3280CP | pp. 5–6，3.2 / Table 1；p. 13，4.1.2。经编蕾丝检测，含现场控制、PLC、边云协同。触摸工控机另为 Advantech Y8161A5 主板、8 GB RAM、256 GB SSD、Windows 10；不能把训练 RTX A4000 写成部署设备，也不能猜为 Atlas 200 DK 等型号。 |
| E5 | Research on Real-Time Detection System of Rail Surface Defects Based on Deep Learning；IEEE Sensors Journal，2024 | NVIDIA Jetson Xavier NX | 两台 Teledyne DALSA CMOS 线扫相机；2048 像素；具体产品型号 NR；12 mm 镜头及激光光源 | pp. 3–4，II-A，Figs. 2–3；系统有图像采集、工业电脑、边缘计算及编码器同步触发。品牌和规格已给出，但不能反推相机系列/型号。 |
| E6 | KAM-Net: Kilobyte-Scale Ultralightweight Attention-Based Network for Glass Defect Detection With Algorithm/Hardware Co-Design；IEEE Transactions on Instrumentation and Measurement，2025 | Xilinx ZCU102；Zynq UltraScale+ ZU9EG；ARM PS 管存储、FPGA PL 执行推理；16-bit 定点 | 三台线扫相机，8192 × 1、12-bit、像元 7.04 μm；品牌和型号 NR | p. 6，IV-A.4，三种光场及共享编码器采集；pp. 8–9，IV-C，Fig. 11，为 FPGA 硬件实验。论文展示数据从 ARM DDR 载入并进行推理，不应改写为已验证相机直连 FPGA 的完整在线产线链路。任务为像素级分割，不直接对比我们的检测 AP。 |
| E7 | LiteYOLO-ID: A Lightweight Object Detection Network for Insulator Defect Detection；IEEE Transactions on Instrumentation and Measurement，2024 | NVIDIA Jetson TX2 NX；TensorRT FP16 | NR；使用 IDID-Plus 等数据集，不是报告完整相机采集设备的论文 | p. 5，IV-A.3；pp. 9–10，板端推理和 Fig. 10。属于实际板端检测测试，不据此宣称已经安装到巡检机器人或无人机。 |
| E8 | AD-YOLO: A Real-Time YOLO Network With Swin Transformer and Attention Mechanism for Airport Scene Detection；IEEE Transactions on Instrumentation and Measurement，2024 | NVIDIA Jetson Orin NX，8 GB | NR；涉及机场监控相机及机场数据集，未给具体相机型号 | pp. 7–8，IV-B.1，Table II，Fig. 8；明确板端测试。机场人员/车辆目标检测，不是工业瑕疵检测，适合参考部署写法，不优先作为瑕疵检测设备对比项。 |
| E9 | MOA-YOLO: An Accurate, Real-Time, and Lightweight YOLOv10-Based Algorithm for Deep-Sea Fish Detection；IEEE Sensors Journal，2025 | 正文为 NVIDIA Jetson Xavier NX；Table X 标题另写“Jetson Xavier TX2”，名称不一致 | NR；AUV 相机及已有水下鱼类数据集，未提供相机型号 | p. 7，IV-B；p. 9，Table III；p. 10，板端比较；p. 14，Table X 和结论。可确认模型在嵌入式平台测试；不能仅凭提出的 AUV 流程声称已完成真实 AUV 海试。 |
| E10 | CMD-YOLO: An IoT-Oriented Lightweight Vision Framework for Real-Time Chili Maturity Detection and Robotic Localization on Edge Devices；IEEE Internet of Things Journal，2026 | NVIDIA Jetson AGX Orin；内存版本 NR | Canon M100：田间数据采集；Intel RealSense D435：机器人定位实验 | p. 4，II-A；p. 12，IV-E，Fig. 13；p. 13，IV-F.2，Fig. 15。不能把数据采集相机与定位实验相机混为同一设备。机器人定位实验的图注和正文为室内模拟实验，不据概述性田间表述推断全部部署验证已在真实采摘现场完成。 |

## 2. 有实际设备或系统，但模型不在嵌入式采集板上推理：10 篇

这些工作仍可比较硬件/控制架构。“现场 PC/服务器”也可能属于广义边缘计算，因此不能一律称为“没有边缘计算”；这里只区分是否在嵌入式/NPU/FPGA 端执行模型。

| 编号 | 工作 / 来源 | 推理或计算平台 | 相机 / 采集硬件 | 核查结论和位置 |
| --- | --- | --- | --- | --- |
| S1 | A Novel Industrial Application of CNN Approach: Real Time Fabric Inspection and Defect Classification on Circular Knitting Machine；Tekstil ve Konfeksiyon，2022 | 质量控制室桌面电脑；CPU/GPU 型号 NR | Basler raL2048-48gm（文中另有商品编号 105993）；线扫；编码器触发；12 个 LED | p. 3，2.2。相机安装到圆机，Ethernet 连接控制室电脑；属于直接相关的圆机检测设备论文，但不是嵌入式端模型推理。 |
| S2 | Fabric4show: real-time vision system for fabric defect detection and post-processing；Visual Intelligence，2024 | Lenovo Legion R9000P 笔记本；AMD R7-5800H + NVIDIA RTX 3060 Laptop，6 GB | HW500 工业相机；直接输出 640 × 480；自制三维滑台 | p. 8，Table 1 / 6.1。Arduino UNO R3（ATmega328P）、ESP8266 等是控制和通信单元，不是 Faster R-CNN 的推理设备。 |
| S3 | An Efficient Measurement Method for Fabric Defect Size Based on Multilight-Source Computer Vision；IEEE Transactions on Instrumentation and Measurement，2025 | NVIDIA RTX 3090 用于模型训练和测试；TensorRT/C++ 部署；其他光源算法在 Intel i9-9900K 上执行 | 八台面阵 CCD；采集尺寸 2432 × 896；具体品牌和型号 NR | pp. 3–4，II-A / Fig. 2；p. 9，IV-B。适合比较多光源成像和布匹检测设备，但没有报告独立嵌入式部署平台。 |
| S4 | A Lightweight Perception Enhancement Network for Real-Time and Accurate Internal Surface Defect Detection of Cold-Drawn Steel Pipes（LPENet）；IEEE Sensors Journal，2026 | 工业 PC + GPU；实际部署 GPU 型号 NR；RTX 4060 Ti 明确是训练设备 | 管内机器人五套视觉单元；CMOS、嵌入式板、环形光源、螺旋镜头；型号 NR | p. 2，Fig. 2 和系统描述；p. 8，图像发送至工业 PC；p. 10，limitations 明确称在机器人 NPU 上部署为未来工作。采集板不等于模型在板端推理。 |
| S5 | SRPCNet: Self-Reinforcing Perception Coordination Network for Seamless Steel Pipes Internal Surface Defect Detection；IEEE Transactions on Industrial Informatics，2025 | 图像采集完成后发送至服务器检测；服务器产品型号 NR；RTX 3090 为实验/训练配置 | 五套视觉单元，M12 工业镜头、环形光源及嵌入式采集传输板；相机/传感器型号 NR | p. 3，III-A；p. 7，实验环境；p. 9，future work。M12 是镜头规格，不是相机型号。 |
| S6 | Deep-Learning-Enabled Automatic Optical Inspection for Module-Level Defects in LCD；IEEE Internet of Things Journal，2022 | NVIDIA Tesla M60 部署；Tesla K80 训练；Ubuntu 16.04；整机型号 NR | 五相机：一台 7100 万像素灰度相机（10000 × 7096），四台 500 万像素彩色相机（2560 × 1920）；具体型号 NR | p. 6，II-B；p. 8，IV；pp. 11–12，IoT/现场系统。有真实 LCD 产线部署，不是 Jetson 类嵌入式模块。 |
| S7 | An Effective Method of Weld Defect Detection and Classification Based on Machine Vision；IEEE Transactions on Industrial Informatics，2019 | 工业 PC；Pentium E5300，2.60 GHz，2 GB RAM，Windows XP | AVT STINGRAY F-125B/C；Sony ICX445 CCD；Computar TEC-M55 镜头；LFV-34 红色同轴光源 | pp. 7–8，V-A，Table I，Figs. 13–14。工业产线检测后自动剔除不良金属罐；属于检测与执行机构联动的相关系统。Table I 为图像，需要视觉读取型号。 |
| S8 | Automated Visual Inspection of Glass Bottle Bottom With Saliency Detection and Template Matching；IEEE Transactions on Instrumentation and Measurement，2019 | 电脑；Intel Core i5-4210U，6 GB RAM；PLC 控制；具体整机型号 NR | 灰度相机 648 × 483；品牌/型号 NR | pp. 2–4，系统、成像与处理；p. 9，实验环境。有相机触发、照明和控制/分选架构，但不是报告嵌入式神经网络部署的工作。 |
| S9 | Smooth Surface Defect Detection by Deep Learning Based on Wrapped Phase Map；IEEE Sensors Journal，2021 | RTX 2080 Ti 实验平台；无明确嵌入式部署型号 | CMOS 工业相机 + LCD 条纹投射；相机型号 NR | p. 3，系统；p. 5，实验环境。适合参考特殊成像，不优先放入边缘设备对比表。 |
| S10 | Tactile-Based Fabric Defect Detection Using Convolutional Neural Network With Attention Mechanism；IEEE Transactions on Instrumentation and Measurement，2022 | 工作站；Intel i7-10700 + NVIDIA GTX 1080 Ti | 视觉触觉传感器内部相机；商品型号 NR；滚筒传感装置为后续系统设计 | pp. 2–3，传感器和采集；p. 5，IV-A；pp. 7–8，后续装置设计。不能把提出的连续滚筒装置视为已完成的嵌入式工业部署。 |

## 3. 声称嵌入式部署，但缺少可核查型号：1 篇

| 编号 | 工作 | 设备信息 | 成像信息 | 判断 |
| --- | --- | --- | --- | --- |
| U1 | A Novel Hybrid YOLO Approach for Precise Paper Defect Detection With a Dual-Layer Template and an Attention Mechanism；IEEE Sensors Journal，2024 | p. 10 写模型转换后部署至 embedded terminal devices，但未识别到明确板卡/处理器型号或独立板端实验配置 | p. 9，Table III 将 Gpixel GMAX0505 列为 Sensor；5120 × 5120。正文称其为 camera，但完整相机品牌/型号 NR | 可纳入硬件系统相关工作；边缘设备型号必须写 NR。不能把 GMAX0505 传感器型号直接填为完整工业相机型号。其机械输送、正反面检测和气流剔除结构另可比较。 |

## 4. 未识别到独立嵌入式端部署验证：其余 16 篇

此分类只表示本地文件未提供可用于当前设备对比的独立嵌入式端推理证据，不表示论文没有实际应用价值，也不表示作者没有其他未报告的设备。

| 文件 / 工作 | 当前对比用途 |
| --- | --- |
| A Feature Prefusion and Mask-Guided Network for Camera Decoration Defect Detection | 有工业相机采集图像，具体相机型号 NR；主要用于算法对比，不据此认定嵌入式端部署。 |
| Attention-Based Multiscale Feature Fusion for Efficient Surface Defect Detection | 算法相关工作。 |
| Automatic Fabric Defect Detection Using Learning-Based Local Textural Distributions in the Contourlet Domain | 算法相关工作。 |
| Computer-Vision-Based Fabric Defect Detection: A Survey | 综述；可追溯原始设备论文，不当成独立部署实验。 |
| Enhanced Fabric Defect Detection With Feature Contrast Interference Suppression | RTX 3090 算法实验；未识别到独立嵌入式部署验证。 |
| Fabric Defect Detection Systems and Methods in India: A Comprehensive Review | 综述；不是独立设备部署研究。 |
| Few-Shot Fabric Defect Detection via Vision-Text Paired In-Context Learning and Two-Stage Reflection | 算法/多模态模型评估；未识别到明确边缘设备型号和部署验证。 |
| HookNet: Efficient Multiscale Context Aggregation for High-Accuracy Detection of Fabric Defects | Xeon Silver 4214 + RTX 2080 Ti 实验；轻量化适用性不等于完成嵌入式部署。 |
| Improving Fabric Defect Classification via Automatic Mining and Leveraging of Category Hierarchy | 算法实验；RTX 4090 不当作已部署的边缘设备。 |
| Fabric Defect Detection in Real World Manufacturing Using Deep Learning（information-15-00476-v2.pdf） | 正文讨论低资源部署适用性，但未识别到实际边缘硬件型号/部署试验。 |
| JDCBL: A Joint Dynamic Contexts and Balanced Learning Framework for Real-World Fabric Defect Detection | 算法实验；未识别到独立嵌入式部署。 |
| Open-Set Fabric Defect Detection With Defect Generation and Transfer | A100 算法实验；未识别到独立嵌入式部署。 |
| Pixel-Wise Semisupervised Fabric Defect Detection Method Combined With Multitask Mean Teacher | 算法相关工作。 |
| Stage-Wise CBAM and Global–Local Fusion for Fabric Defect Detection and Assessment | 算法相关工作。 |
| Weighted Double-Low-Rank Decomposition With Application to Fabric Defect Detection | p. 14 报告工业相机 BB-500GE 采集的数据；这是采集相机信息，不是边缘推理设备。 |
| ZJU-Leaper: A Benchmark Dataset for Fabric Defect Detection and a Comparative Study | 数据集/算法评测；不将数据集采集设备自动视为算法边缘部署平台。 |

## 5. 推荐对比组合

1. **直接相关的布匹边缘系统**：E1、E2、E3、E4。E1 为 JIOT，系统架构与我们最接近；E2 的设备、相机、镜头和光源型号最完整；E3 展示实际本地检测和云端记录；E4 展示经编现场控制及边云协同。
2. **补充工业场景**：E5（铁路，完整采集和边缘推理系统）、E6（玻璃，FPGA 实现）、E7（绝缘子，板端部署测试）。E6 的任务是分割，应比较硬件实现而不是跨任务 AP。
3. **圆机与控制系统参照**：S1（圆机、编码器同步、桌面计算）、S2（完整布匹检测和后处理设备）、S7（检测后剔除）。S2 不能写成 Arduino 执行深度学习推理。
4. **部署写作参照，不优先放硬件对比表**：E8、E9、E10。它们不是瑕疵检测任务。

建议正式论文对比表采用：应用对象、成像配置、实际推理平台、推理实现/精度、本地执行动作、云端角色、验证场景。缺项用 NR；不从外观、品牌或芯片 TOPS 推断成本，不将芯片功耗当整机功耗，不横向比较不同数据集上的检测 AP。

不能声称布匹检测缺少边缘部署研究：E1–E4 已是直接反例。应在可核查架构和验证范围内讨论我们在圆机安装、视频链路、嵌入式 NPU 推理和本地停机集成方面的具体设计。

## 6. 源文件索引

以下文件均在 `C:/Users/tankz/code/github/ultralytics/paper/reference/`。

- E1: `Modified_DenseNet_for_Automatic_Fabric_Defect_Detection_With_Edge_Computing_for_Minimizing_Latency.pdf`
- E2: `song-et-al-2021-efficientdet-for-fabric-defect-detection-based-on-edge-computing.pdf`
- E3: `electronics-14-03692.pdf`
- E4: `information-17-00623.pdf`
- E5: `Research_on_Real-Time_Detection_System_of_Rail_Surface_Defects_Based_on_Deep_Learning.pdf`
- E6: `KAM-Net_Kilobyte-Scale_Ultralightweight_Attention-Based_Network_for_Glass_Defect_Detection_With_Algorithm_Hardware_Co-Design.pdf`
- E7: `LiteYOLO-ID_A_Lightweight_Object_Detection_Network_for_Insulator_Defect_Detection.pdf`
- E8: `AD-YOLO_A_Real-Time_YOLO_Network_With_Swin_Transformer_and_Attention_Mechanism_for_Airport_Scene_Detection.pdf`
- E9: `MOA-YOLO_An_Accurate_Real-Time_and_Lightweight_YOLOv10-Based_Algorithm_for_Deep-Sea_Fish_Detection.pdf`
- E10: `CMD-YOLO_An_IoT-Oriented_Lightweight_Vision_Framework_for_Real-Time_Chili_Maturity_Detection_and_Robotic_Localization_on_Edge_Devices.pdf`
- S1: `ANovelIndustrialApplicationofCNNApproachRealTimeFabricInspectionandDefectClassificationonCircularKnittingMachine.pdf`
- S2: `s44267-024-00047-w.pdf`
- S3: `An_Efficient_Measurement_Method_for_Fabric_Defect_Size_Based_on_Multilight-Source_Computer_Vision.pdf`
- S4: `A_Lightweight_Perception_Enhancement_Network_for_Real-Time_and_Accurate_Internal_Surface_Defect_Detection_of_Cold-Drawn_Steel_Pipes.pdf`
- S5: `SRPCNet_Self-Reinforcing_Perception_Coordination_Network_for_Seamless_Steel_Pipes_Internal_Surface_Defect_Detection.pdf`
- S6: `Deep-Learning-Enabled_Automatic_Optical_Inspection_for_Module-Level_Defects_in_LCD.pdf`
- S7: `An_Effective_Method_of_Weld_Defect_Detection_and_Classification_Based_on_Machine_Vision.pdf`
- S8: `Automated_Visual_Inspection_of_Glass_Bottle_Bottom_With_Saliency_Detection_and_Template_Matching.pdf`
- S9: `Smooth_Surface_Defect_Detection_by_Deep_Learning_Based_on_Wrapped_Phase_Map.pdf`
- S10: `Tactile-Based_Fabric_Defect_Detection_Using_Convolutional_Neural_Network_With_Attention_Mechanism.pdf`
- U1: `A_Novel_Hybrid_YOLO_Approach_for_Precise_Paper_Defect_Detection_With_a_Dual-Layer_Template_and_an_Attention_Mechanism.pdf`

## 7. 对比前需保留的核查备注

- E1 原文交替使用 Cambrian/Cambricon 1H8；汇总保留同一 1H8 平台，不替换为未报告的完整 SoC 型号。
- E4 的相机正文和表格信息不完全相同；写成正文 Basler、表格另列 IDS，不断言所有工位都使用同一型号。其 Atlas 具体产品型号 NR。
- E9 的正文与 Table X 平台名称不一致；采用反复出现的 Xavier NX，同时保留异常备注，不把“Xavier TX2”当作明确型号。
- E10 的 Canon M100 用于数据采集，D435 用于室内定位实验；二者用途分开。摘要/总览的田间应用表述不替代具体实验场景证据。
- U1 的 Table III 明确把 GMAX0505 列在 Sensor 项；不能将它写成完整工业相机商品型号。
- S4 的 RTX 4060 Ti 是训练硬件，文中未明确工业 PC 的实际部署 GPU 型号；S5 同样不将 RTX 3090 的训练/实验环境直接替代服务器商品型号。
- 本汇总不新增正式参考文献、不判断期刊分区，也不将参考论文的硬件选型视为我们的原创性证明。
