# 全文一致性检查（2026-11-13）

对象：Elsevier_submission 当前工作稿。保留用户已有修改，只修正本次核实的问题。

## 已核实并修正

- RK3566速度按用户最新确认统一为YOLOv8n 33 FPS、MCGI-YOLO 23 FPS；纯模型FPS、处理流水线和停机响应分开表述。
- 闭环计时从预处理入口开始，25+43.5+7+36=111.5 ms；典型机械制动1.5–1.8 s，完整响应估计1.61–1.91 s。原稿将4.1 ms理论传输时间混列为实测，且起点写成图像采集，现已纠正。
- 4.1 ms移至硬件链路理论分析：1280×1024×10 bit除以额定3.2 Gb/s；不宣称实际相机传输耗时。
- 三个图像处理阶段合计75.5 ms，对应名义串行处理率约13.2 frames/s；明确为阶段耗时推算，未虚构完整流水线实测FPS。
- 区分Android 13客户端应用与Linux RKNN评估；明确两模型采用相同conf=0.001、NMS IoU=0.7和max_det=16。
- 用户更新相机成本为360 CNY，按既有统一口径约54 USD；更新来源记录。布损计算明确使用未舍入单价，避免显示的3.00 USD/kg与月度金额表观不符。
- 相机/处理器选型不再将Çelik的机上检测系统笼统归为offline系统。
- 移除没有测量依据的“impractical computational overhead”和“minimal performance loss”；保留MCE小通道子集和局部路径的真实设计。
- 保留全部消融数据，以组合效果和互补关系叙述，不再重复强调单模块Hole AP下降。
- 将“confirming ... outperforms generic ...”收敛为当前数据集上的实验结论。
- 改用已有矢量版fps_accuracy.pdf，点值与表格一致，统一mAP@50、mAP@50:95，修复旧PNG中mAP50@50的重复写法。
- 采购设备预算保留为独立情景，不说成已实现收益；删除“未核实取消采购”等冗长防御句。
- 摘要、引言、结论的工厂结果统一为工厂提供的月均估计。

## 已完成的数值和实现核对

- 数据集403图、468实例；282/40/81图、329/48/91实例相符；Row 258、Col 100、Hole 110与形态图一致。
- 五张实验/数据/停机规则表与检查前逐字一致，保留全部真实结果和用户更新的33 FPS。
- 分类别AP与总体mAP在日志独立打印的舍入精度范围内一致；未用已舍入的类别值重算并替换原始总体结果。
- 主对比表最优/次优标记、mAP/P/R差值、量化前后差值和FPS散点值已核对。
- MCE/GIU公式及模块图与C2fAxisResidual、GatedInterpolationUpsample和axis-interp配置一致：P3/P4骨干最终隐分支、1/4通道、广播方向、残差与GIU混合权重正确。
- 10 FPS设计推导、393.216 Mb/s有效载荷、12.3%容量占比、90圈/287.3 m累计横向瑕疵/2.7 m生产方向受影响布长、92.7%废布下降和所有经济金额均已复算。
- 霍尔传感器采用rotation-activated detection，未写成逐帧曝光触发。
- 41个引用键均存在；已引用条目未出现arXiv/preprint；未发现待补实验结果占位。公式标签、交叉引用、图片路径均通过检查。

## 原图中仍需修改的两处

1. 系统总览图（fig1.png，正文Fig. 2）：橙色DC支路直接指向相机，与正文“适配器先供主板、主板输出并控制LED亮度”缺少清晰对应。应在原图中明确主板供电/调光路径，并与PoC画法统一。未凭推断修改实际硬件连线。
2. SerDes图（serdes.png，正文Fig. 3）：传感器/串行器与解串器/SoC之间同时写了Parallel Data和MIPI CSI-2。THCV241A输入及THCV242A输出均为MIPI CSI-2串行接口，应删除两处Parallel Data，保留MIPI CSI-2；同轴链路的Serial Data和PoC方向正确。

## 尚需原始记录才能进一步核对

RT-DETR-R18与DETR-R50的P/R均为97.7%/94.5%。两者相同不代表必然错误，且与用户提供的汇总相符，故未擅改；投稿前建议核对两组原始评估输出是否发生复制。

工厂效益按用户确认的月均估计呈现，没有将其改成指定月份的受控实测；没有新增误停率、长期稳定性或泛化实验，也未据此宣称已有这类验证。

## 官方核验来源

- https://www.thine.co.jp/en/products/detail/THCV241A.html （MIPI CSI-2输入；4.0 Gb/s串行速率、3.2 Gb/s有效速率）
- https://www.thine.co.jp/en/products/detail/THCV242A.html （MIPI CSI-2输出）
- https://github.com/airockchip/rknn_model_zoo （35.1 FPS官方纯推理参考，与本文33 FPS实测分开）
- https://www.itmf.org/images/dl/press-releases/2026/2026-07-24-ITMSS-48-2025.pdf （2025年24.5K台大圆机出货）
- https://wedocs.unep.org/handle/20.500.11822/48707 （每年9200万吨纺织废弃物）

## 最终编译检查

最终PDF为14页，已逐页查看渲染图；未发现文字裁切、图表重叠或失效引用。文内图号正常顺延。上述两处原图标记问题仍保留在此记录中，不能将本轮检查理解为这两处已修图。
