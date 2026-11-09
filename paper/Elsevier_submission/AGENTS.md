# Elsevier 稿件维护

- 默认修改本目录以论文标题命名的唯一 `.tex` 文件，不再将新改动写回 `TIE_submission/`。
- 全文主线是将检测从成布后的独立验布环节前移到织造中的检测与停机干预；章节为 Introduction、Related Work、System Design、Detection Method、Experiments and Results、System Deployment and Industrial Application、Conclusion。相关工作区分成布后验布和已有机上检测，不声称首次提出边织边检；不纳入经编机相关工作。验布机文献区分实际系统运行与采集数据后的模型评估；先行工作包括 Saeidi (2005)、Çelik (2022)、Ni (2025)。明海原验布机数量未知；不再写“已取消独立验布环节”，验布设备采购成本按公开项目配置作情景估算，单独于月度布损减少额，依据保存在 hardware-price-references.json。
- Related Work 按用户给定内容分为 Fabric Inspection Systems 和 Fabric Defect Detection Algorithms 两节；系统部分依次讨论独立验布机与圆机机上检测，算法部分讨论纹理表示、YOLO及形态与尺度适配。引用通过文献键自动编号，不沿用用户草稿中的手写序号。
- 使用现有 `cas-dc` 双栏模板、`natbib` 数字引文和 CAS 的 `pos=` 浮动体选项；保持图片文件名和标签稳定。
- 作者、单位、通讯作者、简介、实验数据和图注措辞以当前稿件为准，不补写未知的贡献声明、利益冲突声明或实验结果。
- 全文金额统一使用 USD，不在正文写换算日期。按当前核实汇率1 USD=6.6765 CNY换算：本系统摄像头约52美元（原350元，含串行器、不含解串器），RK3566主板约75美元（原500元，含解串器）；工厂损失与节省金额同步换算，原始CNY及计算依据保存在 hardware-price-references.json。
- 原始数据为工厂提供的月均估计；净布损为正常布25元/kg减废布回收5元/kg，即20元/kg。
- 默认由用户手动编译；未明确要求时不主动运行编译。编译脚本为 `build.ps1`，辅助文件位于 `build/`。
- 每次改动后创建对应的本地 Git commit，不自动推送。
- 硬件配置与价格比较分别放在 tab:camera-costs 和 tab:processor-costs，正文保留选型依据；工厂前后对比见 tab:factory-benefits，正常布/废布回收单价放表下注，废布率、月度和年化损失减少额放正文，其余金额保留表中。估价及产品系列参考价格在表注中标明。
- Detection Method 先写 Defect Morphology and Design Motivation：典型样本图 fig:dataset-examples 与标注框形态图 fig:annotation-shapes 均在此双栏展示，本小节仅描述横向/竖向细长连续性、形态与尺度差异及检测需求，不提前提MCE或GIU；到MCGI-YOLO Architecture再简述两个模块各自解决的问题和插入位置；Dataset仅保留来源、数量、划分及回指，不重复设计动机。
- 价格与汇率不在正文、表格或参考文献中引用商业网页；保留研究论文引用及必要的估价标记，价格来源仅保存在本地 hardware-price-references.json 供追溯。
- 用户可配置停机决策（sec:stop-rules）置于 System Deployment and Industrial Application，在部署性能之后、客户端界面之前；保留规则公式与预设表，不再放入 Detection Method。
- 排版时将对比表、FPS散点图和检测对比图放在 Comparison With Other Detectors，模块图放在首次介绍MCE的段落后，停机预设表靠近首次引用；用 FloatBarrier 限制硬件、方法、实验及客户端图表漂移到后续章节，单栏图表优先使用 pos=!htbp。
- 图像采集小节先介绍高速成像需求、相机型号与视场，接相机成本对比表，再介绍SerDes链路与带宽；圆机直径和旋转周期参数表放在Processor Subsystem的帧率计算之前，不在相机小节提前引入。图表引用融入选型论述，避免独立的“Table summarizes/compares”堆叠。
