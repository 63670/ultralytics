# Elsevier 稿件维护

- 默认修改本目录以论文标题命名的唯一 `.tex` 文件，不再将新改动写回 `TIE_submission/`。
- 全文主线是将检测从成布后的独立验布环节前移到织造中的检测与停机干预；章节为 Introduction、Related Work、System Design、Detection Method、Experiments and Results、System Deployment and Industrial Application、Conclusion。相关工作区分成布后验布和已有机上检测，不声称首次提出边织边检；先行工作包括 Saeidi (2005)、Li (2013)、Çelik (2022)、Ni (2025)。用户已确认明海工厂取消独立验布机环节，无需另购或使用验布机；仅对该工厂的实际工作流程作此表述，不虚构验布机购置价或将其计入月布损减少额。
- Related Work 采用连续研究脉络：纹理分析、学习型检测、产线部署、机上检测、当前任务需求；不再按硬件/算法拆节，不混用生产阶段和计算平台作为分类标准，平台信息随具体研究案例介绍。
- 使用现有 `cas-dc` 双栏模板、`natbib` 数字引文和 CAS 的 `pos=` 浮动体选项；保持图片文件名和标签稳定。
- 作者、单位、通讯作者、简介、实验数据和图注措辞以当前稿件为准，不补写未知的贡献声明、利益冲突声明或实验结果。
- 原始数据为工厂提供的月均估计；净布损为正常布25元/kg减废布回收5元/kg，即20元/kg。
- 编译使用 `build.ps1`，辅助文件位于 `build/`；检查最终 PDF 后再交付。
- 每次改动后创建对应的本地 Git commit，不自动推送。
