# 系统与算法结合论文的写作结构对照

日期：2026-10-15。依据：`paper/reference` 中的原始 PDF，重点阅读系统设计、算法、实验及部署部分，并核对相关图表。下述页码均为 PDF 文件页码，从第一页起计数。

本次以 IEEE Internet of Things Journal、IEEE Transactions on Industrial Informatics 的系统论文为主要参照，补充 IEEE Transactions on Instrumentation and Measurement 和 IEEE Sensors Journal 中贴近本项目的工作。未将所有参考文献一概认定为“一区”；正式分区需按投稿时采用的分区体系、年度和学科确认。

## 1. 对我们最有价值的结论

系统与算法没有固定的唯一出场顺序。关键在于：先交代应用问题和设计依据，再说明方案，最后用实验验证相应贡献。

对于本项目，最适合借鉴的是：**用 LCD 系统论文组织“现场问题—成像与系统方案—算法”的主线，用铁路检测论文解释硬件选型依据，再参考 Modified DenseNet 把板端运行和本地响应纳入系统验证。**

我们的系统价值应落在大圆机生产中的具体流程：相机与照明安装、旋转信号触发、视频传输、边缘检测、停机信号、操作员查看与复机、云端记录。算法价值则回应横向长条、纵向长条和孔洞的形态特征，以及多尺度融合中的弱响应问题。

## 2. 逐篇结构与可借鉴内容

### 2.1 LCD 模组缺陷检测：先系统，再算法，再讨论工厂级集成

来源：[Deep-Learning-Enabled Automatic Optical Inspection for Module-Level Defects in LCD](C:/Users/tankz/code/github/ultralytics/paper/reference/Deep-Learning-Enabled_Automatic_Optical_Inspection_for_Module-Level_Defects_in_LCD.pdf)，IEEE Internet of Things Journal，2022。

章节顺序：I Introduction → II System Overview → III Proposed Inspection Method → IV Experiments → V Discussion → VI Conclusion。

- **pp. 1–4，引言**：先分析缺陷种类、尺寸跨度、可视角度、背景、干扰等现场问题。p. 4 的 Fig. 3 将挑战与解决方法对应起来。
- **pp. 4–6，II System Overview**：先说明不同缺陷为什么需要不同显示背景，再介绍机械上下料、屏幕触发、成像与视觉分析子系统。p. 5 Fig. 4 是硬件系统，p. 6 Fig. 5 是多相机成像方案。
- **pp. 6–8，III Proposed Inspection Method**：依据前述背景与视角，将图像分配给不同 YOLOv3 检测单元，再做预处理和干扰消除。硬件生成的图像条件直接决定算法流程。
- **pp. 8–10，实验**：同时考察检测指标、运行时间和人工方案对比；消融还涉及成像与检测框架的选择，并不限于神经网络模块。
- **pp. 11–12，讨论**：再扩展到生产线节点、任务调度、图像收集、模型更新和工厂级信息管理。

可借鉴：系统章介绍“为什么这样采集、每个部件负责什么”；方法章介绍“如何处理这些图像”。云端管理可以在整体检测流程清楚之后展开。即使基础检测单元采用已有网络，成像与检测流程的针对性设计仍然可以构成论文主体。

### 2.2 Modified DenseNet：算法在前，系统实现放在边缘实验中

来源：[Modified DenseNet for Automatic Fabric Defect Detection With Edge Computing for Minimizing Latency](C:/Users/tankz/code/github/ultralytics/paper/reference/Modified_DenseNet_for_Automatic_Fabric_Defect_Detection_With_Edge_Computing_for_Minimizing_Latency.pdf)，IEEE Internet of Things Journal，2020。

章节顺序：I Introduction → II Architecture Design → III Data Enhancements → IV Experimental Results → V Related Work → VI Conclusion。

- **p. 3，II 开头**：先列出工业噪声、小样本和检测速度三类要求，再引出网络和数据处理方案。
- **pp. 3–8，方法**：分析数据特点、网络修改、损失函数与数据增强。
- **pp. 9–11，IV-B Performance Comparison Experiment on Edge Device**：在模型说明之后介绍自研边缘相机的核心板、底板、处理器及接口，接着介绍管理软件。
- **p. 10 Table VII、p. 11 Fig. 16**：将采集、传输、计算、控制响应分开，比较边缘与云端的响应周期。原文报告云端 552 ms、边缘 374 ms；这些是该文定义和环境下的结果，不能直接套用到本系统。

可借鉴：自研硬件照片与管理软件界面需要接上具体的运行流程和验证结果。管理功能服务于设备、批次及缺陷记录，不必在算法之前讲模型转换和推理引擎。

### 2.3 SRPCNet：系统和自采数据引出算法难点

来源：[SRPCNet: Self-Reinforcing Perception Coordination Network for Seamless Steel Pipes Internal Surface Defect Detection](C:/Users/tankz/code/github/ultralytics/paper/reference/SRPCNet_Self-Reinforcing_Perception_Coordination_Network_for_Seamless_Steel_Pipes_Internal_Surface_Defect_Detection.pdf)，IEEE Transactions on Industrial Informatics，2025。

章节顺序：I Introduction → II Related Works → III Proposed System and Dataset → IV Proposed Method → V Experiments → VI Conclusion。

- **p. 3，III-A**：交代管内采集装置和可视化软件的分工、采集过程以及图像发送到服务器检测的流程。
- **pp. 3–4，III-B–D**：说明数据采集和标注，再分析真实数据中的目标、伪装、位置、反光、背景等挑战。
- **pp. 4–7，IV**：算法接着回应这些挑战，介绍整体网络和各模块。
- **p. 9，V-G**：明确当前图像由服务器处理，板端预筛选属于未来工作；系统自动化程度的不足也单独说明。

可借鉴：我们可以用实际布面图像展示长条缺陷、孔洞与重复纹理，再自然引出 DME 和 CAFR。设备中存在嵌入式板，不应自动写成算法已经在该板执行；应明确实际推理位置。

### 2.4 铁路表面检测：先给成像计算，再选设备，再讲轻量算法

来源：[Research on Real-Time Detection System of Rail Surface Defects Based on Deep Learning](C:/Users/tankz/code/github/ultralytics/paper/reference/Research_on_Real-Time_Detection_System_of_Rail_Surface_Defects_Based_on_Deep_Learning.pdf)，IEEE Sensors Journal，2024。

章节顺序：I Introduction → II Hardware Systems and Datasets → III Improved YOLOv5 → IV Experiment Result → V Conclusion。

- **p. 3，II-A**：按采集宽度 500 mm、精度 0.25 mm/pixel 算出所需像素数 2000，再结合像元尺寸、镜头焦距计算物距，选用 2048 像素线扫相机。
- **pp. 3–4**：说明采集模块、编码器同步触发、工业计算机与边缘设备，再由边缘算力限制引出模型轻量化。
- **pp. 4–6，III**：先解释基线和目标尺度问题，再介绍尺度调整、锚框与轻量结构。
- **pp. 9–10，Table VI**：报告设备端检测耗时；结论将适用性限定在低速检测场景。

可借鉴：我们的 40 英寸、2 s/圈、16 cm 横向视场推导 10 FPS，正适合放在系统需求或硬件选型部分。它给出选型的工作量依据；后面的板端实验再报告实际是否达到这一要求。按当前写作约定，保留简洁的 10 FPS 推导即可，不扩展帧间重叠分析。

### 2.5 多光源布匹尺寸测量：按系统处理流程把硬件与算法写在同一章

来源：[An Efficient Measurement Method for Fabric Defect Size Based on Multilight-Source Computer Vision](C:/Users/tankz/code/github/ultralytics/paper/reference/An_Efficient_Measurement_Method_for_Fabric_Defect_Size_Based_on_Multilight-Source_Computer_Vision.pdf)，IEEE Transactions on Instrumentation and Measurement，2025。

章节顺序：I Introduction → II Related Works → III System → IV Experiments and Discussion → V Conclusion。

- **pp. 3–5，III-A Fabric Image Acquisition System**：先写多光源采集设备，解释不同光源使哪些缺陷更容易观察；p. 4 Fig. 2 同时展示原理、结构与实物。
- **pp. 5–8，III-B Fabric Defect Location Algorithms**：根据不同光源下的图像特点组织检测线程与算法。
- **p. 8，III-C**：再写多光源结果融合和尺寸测量，使“设备输出—算法输入—最终测量结果”顺序明确。
- **pp. 10–12，IV-E**：分别评估处理线程及系统耗时，再将图像处理速度与实际验布速度联系起来。

可借鉴：设备内容应解释它对后续图像和算法的作用，硬件参数不必孤立罗列。整体流程图可以采用“采集—边缘检测—停机及显示—记录上传”，让信号和数据的去向一目了然。

### 2.6 KAM-Net：先算法，再定制加速器，实验分别验证两者

来源：[KAM-Net: Kilobyte-Scale Ultralightweight Attention-Based Network for Glass Defect Detection With Algorithm/Hardware Co-Design](C:/Users/tankz/code/github/ultralytics/paper/reference/KAM-Net_Kilobyte-Scale_Ultralightweight_Attention-Based_Network_for_Glass_Defect_Detection_With_Algorithm_Hardware_Co-Design.pdf)，IEEE Transactions on Instrumentation and Measurement，2025。

章节顺序：I Introduction → II Algorithm Design → III Hardware Architecture Design → IV Experiments and Results → V Conclusion。

- **pp. 2–4，II**：定义网络和算子，说明轻量化设计。
- **pp. 4–6，III**：硬件章针对已经定义的卷积、注意力算子说明处理单元、流水线、缓存和定点实现。
- **pp. 6–8，IV-A/B**：先验证算法效果和数据集泛化。
- **pp. 8–9，IV-C**：再验证 FPGA 资源、速度与功耗，明确 ARM 与 FPGA 的职责。

可借鉴：如果要解释算子如何映射到特定硬件，应先讲清算法。它的“Hardware Architecture”主要是针对网络定制的计算架构，与我们相机、传输和机器控制接口的工程集成有明显区别，不宜直接套用同样的贡献名称。

### 2.7 LiteYOLO-ID：网络方法后接部署转换，实验单列板端测试

来源：[LiteYOLO-ID: A Lightweight Object Detection Network for Insulator Defect Detection](C:/Users/tankz/code/github/ultralytics/paper/reference/LiteYOLO-ID_A_Lightweight_Object_Detection_Network_for_Insulator_Defect_Detection.pdf)，IEEE Transactions on Instrumentation and Measurement，2024。

章节顺序：I Introduction → II Related Work → III Methods → IV Experiments → V Conclusion。

- **pp. 3–4，III-A–C**：先介绍骨干、颈部、检测头。
- **pp. 4–5，III-D TensorRT Acceleration**：在网络定义之后讲训练权重、ONNX、TensorRT 和 FP16。
- **pp. 9–10，IV-C**：单列 Jetson TX2 NX 板端测试、同环境模型速度比较及实际输出图，报告 20.2 FPS。
- **p. 10，结论**：将图像采集、检测、传输与无人机平台的完整集成列为后续工作。

可借鉴：模型转换、推理框架和精度配置放在算法之后很自然。板端推理实验本身可以独立写清楚，不需要扩展成已经完成的机器人或产线系统。

## 3. 推荐用于本稿的组织方式

以下是后续重排建议，本轮未据此改写正文：

1. **Introduction**：大圆机持续性故障与废布问题 → 相关系统与算法 → 本文系统和算法分别解决什么问题。
2. **System Overview and Hardware Design**：先给一个完整流程概览；说明安装与成像、Hall 信号、同轴链路、本地停机接口；以 10 FPS 推导交代设备选型要求。此处只需说明检测模块的输入输出。
3. **Proposed Detection Method**：整体架构 → DME → CAFR → 模型转换与边缘运行实现。模型导出、量化和 RKNN 说明均在算法定义之后。
4. **Experiments and System Validation**：数据与训练协议 → 模型对比 → 消融 → 实际板端性能 → 现场运行展示。篇幅需要时可继续拆为实验设置、结果讨论两章。
5. **Conclusion**：分别总结算法效果和已经完成的系统实现，避免把设计目标写成实测结果。

系统概览可以提前交代“采集后由检测模块给出结果，并触发本地控制”；这种功能说明不需要先讲神经网络细节。具体模型格式、量化、运行库与部署性能应在方法定义之后出现。

## 4. 最值得补强的三处内容

| 位置 | 建议写法 | 参考依据 |
| --- | --- | --- |
| 引言贡献与系统章开头 | 从大圆机安装空间、旋转运行、10 FPS 工作量和缺陷后本地干预出发，解释系统设计；让每个硬件部件对应实际功能。 | LCD、铁路检测 |
| 系统到算法的过渡 | 用真实图像的长条缺陷、孔洞、重复背景和弱响应引出 DME、CAFR，使算法动机与本系统采集对象一致。 | LCD、SRPCNet、多光源测量 |
| 实验结果末尾 | 在实际设备上给出基线与本文模型的处理速度，核对 10 FPS 要求；配合现场检测、停机信号和操作界面展示已完成的工作。无需把论文扩成复杂的控制性能研究。 | Modified DenseNet、铁路检测、LiteYOLO-ID |

Table I 按最新约定保留平台、内存、算力和价格；它适合作为平台背景资料。更直接支撑本文系统可用性的证据，是本机上模型的实际运行表现及现场流程，而不是仅凭参考平台价格推导优势。

## 5. 本轮顺带发现的型号一致性问题

审阅时，工作区 Table I 已新增 `Self-developed RK3566 edge device(Ours)`，而正文硬件、部署和实验设置仍写 RK3588。该差异来自当前未提交的正文修改，本轮保留原样，后续需按实际设备统一型号，再讨论其算力、成本和运行速度。
