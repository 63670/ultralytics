# TIE 瑕疵检测文献调研与引言引用说明

核查日期：2026-10-16。目标期刊为 **IEEE Transactions on Industrial Electronics（TIE，ISSN 0278-0046）**。检索覆盖缺陷检测、表面检测、视觉检测、布匹等关键词，以下是与本稿系统设计、上下文建模和特征融合较相关的筛选结果，并非完整综述。

## 1. 正式出版的 TIE 论文

下表年份采用最终卷期年份；DOI 中的年份可能对应提前在线出版年份。前四篇已经加入引言和 `references.bib`，第五篇保留作系统写作参考。

| 论文及正式出版信息 | 检测对象与主要内容 | 与本稿的联系及引用边界 |
| --- | --- | --- |
| **Mostafa et al., An Innovative Pipe Inspection Tool Utilizing Electromagnetic Resonance Coupling and Machine Learning.** TIE, **71(5): 5338–5348, 2024**. [正式 DOI](https://doi.org/10.1109/TIE.2023.3285970) | 面向金属管内壁金属损失，结合发射/接收线圈、电磁谐振耦合和神经网络重建，并用实验原型验证。 | **硬件与算法集成参考**。可支持“根据检测任务设计专用感知硬件并结合学习模型”的叙述；不是相机布匹检测，也不是 RK3566 部署案例。引言只概述原型、传感与学习的结合。 |
| **Su, Chen, and Zhou, BAF-Detector: An Efficient CNN-Based Detector for Photovoltaic Cell Defect Detection.** TIE, **69(3): 3161–3171, 2022**. [正式 DOI](https://doi.org/10.1109/TIE.2021.3070507) | 光伏电池电致发光（EL）图像；双向注意力特征金字塔 BAFPN，用于多尺度缺陷表征。 | **特征金字塔参考**。支持多尺度融合的研究背景；它属于 Faster R-CNN/FPN 路线，不能写成 YOLO 或嵌入式 NPU 案例，也不能据此宣称我们的模块优于它。 |
| **Shen et al., A Novel Weld Defect Detection Method for Intelligent Magnetic Flux Leakage Detection System via Contextual Relation Network.** TIE, **71(6): 6304–6314, 2024**. [正式 DOI](https://doi.org/10.1109/TIE.2023.3294578) | 管道焊缝漏磁（MFL）检测；CRNet 通过稀疏注意力、多尺度关系探索及多视角协同，区分相互耦合的焊缝和缺陷信号。 | **上下文建模参考**。可说明工业检测中的任务相关上下文设计；漏磁信号与布匹光学图像的成像机制不同，不直接比较其准确率。 |
| **Liu et al., Online Pipeline Weld Defect Detection for Magnetic Flux Leakage Inspection System via Lightweight Rotated Network.** TIE, **72(7): 7573–7584, 2025**. [正式 DOI](https://doi.org/10.1109/TIE.2024.3503635) | 管道焊缝漏磁检测；轻量旋转网络与在线检测。 | **近期轻量检测参考**。引言采用正式题名可确认的任务和网络定位。本轮未取得可核查的出版社全文，不把二手摘要中的工控机、模块细节或速度数值写入正文，更不推断其 NPU 性能。 |
| **Wertjanz et al., Bringing the Lab to the Fab: Robot-Based Inline Measurement System for Precise 3-D Surface Inspection in Vibrational Environments.** TIE, **69(10): 10666–10673, 2022**. [正式 DOI](https://doi.org/10.1109/TIE.2022.3151959) | 工业机器人上的磁悬浮测量平台，结合跟踪控制与光学三维测量，应对工厂振动。 | **系统设计写法参考，暂不加入正文**。可参考“环境约束→硬件与控制设计→原型验证”的结构；其任务是精密三维表面测量，不应写成布匹瑕疵目标检测或我们已有的抗振能力。 |

### 核查来源与证据范围

- 五篇的题名、作者顺序、最终年份、卷期、页码与 DOI，均以 IEEE 向 [Crossref 注册的 TIE 记录](https://api.crossref.org/journals/0278-0046/works)核对；具体论文可通过上表 DOI 定位。书目使用正式期刊条目，未新增 arXiv/preprint 引用。
- Mostafa：阅读了 [KAUST 作者机构库的已接收稿摘要](https://repository.kaust.edu.sa/server/api/core/bitstreams/d7812266-28da-490a-854e-814063002185/content)。机构库稿用于核查内容，参考文献指向已经分配卷期的正式论文。
- BAF-Detector：[作者主页](https://binyisu.github.io/)确认正式 TIE 卷期和作者顺序；作者公开版本的摘要用于交叉核查多尺度融合与 BAFPN。正式期刊全文未在本轮取得，不据摘要推断部署平台、实测延迟或全部模块实现。
- CRNet：[IEEE Xplore 正式页面及摘要](https://ieeexplore.ieee.org/document/10195860/)可核查任务和方法概要。
- LRNet：[作者东北大学主页](https://faculty.neu.edu.cn/liujinhai/en/lwcg/46003/content/31807.htm)确认论文及作者；最终卷期由 Crossref 核对。正文只使用可以确认的信息。
- 三维系统：[维也纳工业大学机构库](https://repositum.tuwien.at/handle/20.500.12708/136738)提供正式引文和方法摘要。

## 2. 引言中布匹 YOLO 工作的补充

为保持论文对象明确，算法相关工作以布匹文献为主；TIE 论文补充工业系统和算法背景。

| 布匹论文 | 已核查的改进方向 | 本稿必须承认的已有工作 |
| --- | --- | --- |
| [AYOLOv7-tiny，TRJ 2024](https://journals.sagepub.com/doi/10.1177/00405175231205898) | 空间到深度下采样与混合注意力，面向纯色圆纬织物的细微瑕疵。 | 已有直接涉及圆纬织物的轻量 YOLO 工作，不能声称此前没有大圆机相关布匹研究。 |
| [YOLOV4-TinyS，TRJ 2024](https://journals.sagepub.com/doi/10.1177/00405175231202035) | 轻量卷积、浅层融合与注意力，并评估输送带布匹检测系统。 | 不能概括为全部只在公开数据集训练，或都没有实际检测系统。 |
| [FA-YOLO，IEICE 2024](https://www.jstage.jst.go.jp/article/transfun/E107.A/6/E107.A_2023EAP1030/_article) | 感受野增强、注意力与自适应空间特征融合。 | 现有布匹 YOLO 已研究跨尺度自适应融合，不能笼统称其均使用固定融合。 |
| [Chen et al.，改进 YOLOv8，TRJ 2025](https://journals.sagepub.com/doi/10.1177/00405175241261092) | Swin Transformer、双向特征金字塔、注意力与定位损失改进。 | 已有上下文和定位增强研究；未取得同平台延迟证据时，不直接认定其速度不达标。 |
| [DGHR-YOLO，Nondestructive Testing and Evaluation 2025](https://www.tandfonline.com/doi/full/10.1080/10589759.2025.2451768) | 可变形卷积适应瑕疵形态，高层筛选特征金字塔与轻量特征处理。 | 不能声称前人没有考虑形态或高层语义，也不能仅凭算子类型认定一定回退 CPU。 |

本地另一篇《Real-Time Oriented Fabric Defect Detection via Position-Guided Fusion and Multiscale Rotation-Guided Strip Coupling》（DOI **10.1109/TII.2026.3717200**）属于 **TII**，不能计入 TIE。它研究旋转框布匹检测、位置引导融合和条带特征，提醒我们不能把方向/条带建模本身概括为无人研究的空白；本稿的水平框任务及方向性/局部形态组合应具体说明。

## 3. 本轮引言的组织方式

1. 区分已有纺织部署场景与本稿的圆纬织物在线检测。
2. 用专用传感原型和视觉边缘平台说明系统集成背景，再引出摄像头、感知与停机接口的任务适配需求。
3. 用 TIE 研究概述多尺度、上下文和轻量检测，再集中介绍五篇布匹 YOLO。
4. 研究动机收窄为：在本稿设备预算下，联合编码横向、纵向连续结构和局部孔洞，并改善上采样邻域重建。将“通用注意力/跨尺度选择”和“局部重建权重”作功能区分，而不宣称前者普遍无效。

以上是基于已核查内容的研究定位，不是不同数据集之间的性能结论。实际优势仍需同一数据集的消融和 RK3566 实测支持；期刊相关引用服务于研究定位，不能替代创新与实验论证。
