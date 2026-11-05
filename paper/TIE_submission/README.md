# MCGI-YOLO 论文维护目录

本目录是 **A Defect Detection and Closed-Loop Control System with MCGI-YOLO for Circular Knitting Machines** 的唯一维护入口，后续转投修订也在此进行。

- 同名 `.tex` 与 `.pdf` 为正文和编译结果，保留以论文标题命名的文件名。
- `figures/fig1.png` 至 `fig7.png` 为原有正文插图，`a1.jpg` 至 `a7.jpg` 为作者照片；文件名保持稳定，正文图号由 LaTeX 自动生成。
- `figures/serdes.png` 为硬件部分的 SerDes 与 PoC 链路示意图（单栏）。实验结果之后单设 `System Deployment and Operation`，集中介绍部署流程与23 FPS、客户端、云平台、规则配置与停机执行及延迟干预分析；其中 `detect.png` 为客户端页面跳转图（双栏），`iot_platform.png` 为 IoT 平台图（双栏），C)、D) 展示客户端状态与远程配置的数字孪生界面。
- 数据集样本图使用 `figures/defect.png`，在 Dataset 小节双栏展示，标签为 `fig:dataset-examples`；四行依次为 row、col、hole 及多类瑕疵共存样本。
- 真实废布照片使用 `figures/fabric_with_defect.jpg`，在延迟干预损失估算小节单栏展示，标签为 `fig:waste-fabric`，说明持续纱线异常造成重复横向瑕疵和布面报废。
- 标注框形态与尺度图为 `figures/annotation_shapes.png`（双栏，`fig:annotation-shapes`）；由 `paper/plot_annotation_shapes.py --labels <数据集labels目录> --output paper/TIE_submission/figures/annotation_shapes.png` 生成，全部框中心对齐并保留归一化宽高，附 `.stats.json` 记录统计和标注哈希。生产工况及延迟估算输入见 `tab:production-inputs`。
- FPS–mAP 散点图为 `figures/fps_accuracy.png` 与矢量版 `.pdf`，运行 `python paper/plot_fps_accuracy.py` 从当前正文 `tab:comparison` 自动读取数据；使用 RTX 3090、FP32、batch=1、640×640 的纯推理 FPS，数据快照保存在同名 `.json`。正文使用矢量 PDF 在检测器对比部分双栏展示，标签为 `fig:fps-accuracy`。
- 手动标注用纯散点底图运行 `python paper/plot_fps_accuracy.py --unlabeled`，另存 `figures/fps_accuracy_unlabeled.{png,pdf,svg,json}`，左图 mAP@50、右图 mAP@50:95；无文字、坐标轴、网格及引线，保留颜色、点形和数据位置。此版本不自动替换正文已标注插图。
- 用户可配置停机规则作为 Detection Algorithm 的末尾子节 `User-Configurable Post-NMS Stopping Decision`（`sec:stop-rules`），包含 NMS 后置信度筛选、原图尺度判断、数量条件和预设表；系统运行部分仅描述参数配置和停机执行，不将停机规则称为改进 NMS。
- `fig6.layout.json`、`fig6.sources.json` 和 `fig7.layout.json` 保留对比图与热力图的排版、来源记录。
- `references.bib`、`IEEEtranTIE.cls` 和 `Bibliography/IEEEtranTIE.bst` 为独立编译所需文件。
- `.vscode/settings.json` 提供 LaTeX Workshop 编译配置；辅助文件和编辑器预览位于 `build/`。
- `build.ps1` 自动识别本目录唯一的 `.tex` 文件，完成四步编译，并将同名 PDF 发布到正文所在目录。
- Cover letter 及其原始模板保存在上一级目录的 `Cover letter.docx` 与 `Cover letter template.doc`。

在本目录执行：

```powershell
./build.ps1 -TeXBin "C:/Users/tankz/application/MiKTex/miktex/bin/x64"
```

旧草稿目录已合并并删除。2026-11-01 的原始投稿版本可通过 Git 标签 `tie-submission-snapshot-2026-11-01` 查看或恢复。JIOT 新模型稿件继续在 `../JIOT_draft/` 单独维护。
