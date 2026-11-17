# 论文维护入口

- 后续 MCGI-YOLO 论文修改默认基于 `Elsevier_submission/A Defect Detection and Closed-Loop Control System with MCGI-YOLO for Circular Knitting Machines.tex`；转投修订也在此维护。
- 全文主线是将检测从成布后的独立验布环节前移到织造中的检测与停机干预；章节为 Introduction、Related Work、Defect Detection and Closed-Loop Control System、Detection Algorithm、Experiments and Results、Conclusion；模型部署为5.4 Model Deployment and On-Device Performance。相关工作区分成布后验布和已有机上检测，不声称首次提出边织边检；不纳入经编机相关工作。验布机文献区分实际系统运行与采集数据后的模型评估；先行工作包括 Saeidi (2005)、Çelik (2022)、Ni (2025)。工厂名称仅在Fig. 5安装照片图注中保留；摘要、引言、贡献和结论用检测精度、模型复杂度及端侧评估结果支持纺织行业大圆机在线瑕疵检测的规模化应用，不声称已验证大规模部署，不在摘要、引言、贡献、正文或结论中报告废布前后变化、减废率、经济收益或验布机采购节省；不自动恢复原经济分析表。历史数值仅保留于核验记录。
- `TIE_submission/` 保留为迁移前归档；后续正文、图表、实验数据和编译默认只更新 `Elsevier_submission/`。
- 图注和表题采用 JMS 常见的简短主题式写法，原则上一个短句；多子图仅保留必要的(a)–(d)对应。参数、测试口径、百分数和最优/次优标记规则放在正文，避免在图表标题重复说明。
- 对应图片、参考文献和编译脚本位于 `Elsevier_submission/figures/`、`Elsevier_submission/references.bib` 和 `Elsevier_submission/build.ps1`。模板采用 `els-cas-templates/` 提供的 `cas-dc` 双栏类；原有正文图片文件名 `fig1.png` 至 `fig7.png` 保持稳定；正文图号由 LaTeX 自动生成，不再等同于文件名。旧草稿目录已删除，不重新创建。
- 数据集样本图使用 `figures/defect.png`，在 Detection Method 的 Defect Morphology and Design Motivation 小节双栏展示，标签为 `fig:dataset-examples`；四行依次为 row、col、hole 及多类瑕疵共存样本。
- 原延迟干预分析及废布照片fig:waste-fabric已随原第6章从正文删除；figures/fabric_with_defect.jpg作为原始素材保留，不自动恢复插图或估算。
- 工厂名称仅保留于Fig. 5安装照片图注，不在摘要、引言、贡献或结论重复；不另设工厂应用章节；原第6章已删除，原6.1完整移至实验结果5.4。
- 标注框形态与尺度图为 `figures/annotation_shapes.png`（双栏，`fig:annotation-shapes`，与典型样本图一起置于算法设计动机小节）；由 `paper/plot_annotation_shapes.py --labels <数据集labels目录> --output paper/Elsevier_submission/figures/annotation_shapes.png` 生成，全部框中心对齐并保留归一化宽高，附 `.stats.json` 记录统计和标注哈希。机器直径和旋转周期见 `tab:production-inputs`，该两行参数表置于处理器选型小节、帧率计算之前；横向视场宽度w=16 cm在相机介绍正文说明，不在表中重复；原延迟干预估算已随原第6章删除，不恢复相应图表和计算。
- FPS–mAP散点图（正文Fig. 11）必须使用用户手动标注的 `figures/fps_accuracy.png`，双栏展示，标签 `fig:fps-accuracy`。保留其原有布局、字号、颜色和标注，不自动切换到PDF或重新生成覆盖。已有 `.pdf` 只是备份版本。需要另出底图时使用 `paper/plot_fps_accuracy.py --output <新路径>` 或 `--unlabeled`，不得覆盖正文手动标注图；数据口径仍为RTX 3090、FP32、batch=1、640×640纯推理。
- 手动标注用散点底图运行 `python paper/plot_fps_accuracy.py --unlabeled`，另存 `figures/fps_accuracy_unlabeled.{png,pdf,svg,json}`，左图 mAP@50、右图 mAP@50:95；保留坐标轴、刻度数字、网格、颜色、点形和数据位置；仅去掉模型名称、引线、图标题及轴标题。此版本不自动替换正文已标注插图。
- 硬件系统部分保留serdes.png（单栏）。模型部署与端侧性能位于实验结果5.4；原第6章及停机规则、延迟干预分析已删除。客户端和云端小节及detect.png、iot_platform.png不恢复，原始素材保留。
- RK3566 部署评估在 Linux 下进行：YOLOv8n 为33 FPS、MCGI-YOLO 为23 FPS（2026-11-13用户确认的最新实测值）；两者输入640×640、conf=0.001、NMS IoU=0.7、max_det=16。P/R 按 IoU=0.5 的最佳类别平均F1位置报告，并按类别宏平均。部署结果见正文 `tab:rknn-deployment`，不得混入 RTX 3090 的主对比表。 上述33/23 FPS均为纯模型推理帧率，不包含前处理和后处理。用户已要求删除Closed-Loop Response Time小节；正文暂不报告实际检测处理帧率或闭环时延，不将纯推理FPS作为整段检测处理帧率进行达标论证。
- 原用户可配置停机规则小节、预设表及对应公式已随原第6章删除，不移回检测算法；系统总览保留真实的检测与停机控制流程。
- 历史停机规则（已不在JMS正文展示）按原始 `1280×1024` 图像计算；预设元组依次为（置信度、破洞检测框面积、横向检测框宽度、竖向检测框高度、每帧数量）：历史记录预设为低 `(0.50, 6400, 600, 200, 3)`，中 `(0.40, 3600, 400, 100, 2)`，高 `(0.30, 1600, 200, 50, 1)`。先按置信度筛选，再判断“任意一个对应类别尺寸超过阈值，或者该帧通过置信度筛选的瑕疵总数达到数量阈值”，二者为或关系，不跨帧累计。三档均可编辑。
- `manuscript.tex` 保留为迁移前的历史稿，`TIE_LaTeX_template/` 保留为原始模板；不要将后续论文修改写回这两处。
- 每次改动后创建对应的 Git commit，遵循仓库根目录的要求。

## JIOT 草稿

- 后续 JIOT 论文修改基于 `JIOT_draft/manuscript.tex`，使用 `templates/` 提供的标准 IEEEtran 模板；具体入口见 `JIOT_draft/AGENTS.md`。
- `JIOT_draft/` 从 TIE 投稿版迁移，暂时保留 MCGI-YOLO 的方法、图片和实测结果。新模型及实验结果后续在该目录更新。
- `Elsevier_submission/` 和历史 `JIOT_DMFR_draft/` 不随 JIOT 新稿修改。

## VS Code 编译入口

- 用户在 VS Code 中以 `Elsevier_submission/` 为工作区；其 `.vscode/settings.json` 覆盖 LaTeX Workshop 的编译配置。
- 编辑器通过 `pdflatex -> bibtex -> pdflatex -> pdflatex` 编译，PDF、SyncTeX 和辅助文件统一输出到 `Elsevier_submission/build/`，预览 `build/<论文标题>.pdf`；`build.ps1` 另发布一份同名 PDF 到正文所在目录。
- `pdflatex` 必须显式传入 `-output-directory=%OUTDIR%`，BibTeX 必须读取 `%OUTDIR%/%DOCFILE%`；仅修改 `latex.outDir` 不会改变自定义工具的实际输出路径。
- 日志若显示 `I can't write on file manuscript.pdf`，先核对实际输出路径和 PDF 占用，不要据此改动公式或删除参考文献。编辑器使用 `build/<论文标题>.log`；根目录旧日志可能仍保留此前错误。

## 检测对比图生成（fig6.png）

- 直接使用现有脚本，不重新寻找旧缓存或重新运行模型推理。
- 服务器：`ssh -p 13022 tkz@8.138.122.96`。
- 已保存的检测结果：`/home/tkz/code/github/ultralytics/runs/detection_comparison`。
- 服务器出图目录：上述目录下的 `paper_figures`；脚本在其 `scripts` 子目录中。服务器仓库根目录的 `paper/` 不一定存在，不能假定脚本位于该处。
- 本地脚本：`paper/redraw_detection_labels.py` 和 `paper/compose_detection_figure.py`。先同步这两个脚本，确保显示名是 MCGI-YOLO，再在服务器执行。
- 结果目录及脚本 `MODELS` 中的 `PAGI-YOLO` 是历史数据目录名，保留用于读取结果；论文显示名称由 `compose_detection_figure.py` 的 `TITLES` 控制，使用 `MCGI-YOLO (Ours)`。
- 保留五列：Label、YOLOv8n、RT-DETR-R18、Faster R-CNN-R50、MCGI-YOLO (Ours)。显示类别和置信度，不显示字母下标。检测标签字号 44，模型名字号 48，间隔 12。
- 默认五行顺序固定为：
  1. `VFDS66RK26040002_20260801_195949765.jpg.png`
  2. `VFDS66RK26040002_20260809_022619641.jpg.png`
  3. `VFDS66RK26040002_20260727_193808156.jpg.png`
  4. `VFDS66RK26040002_20260805_002859637.jpg.png`
  5. `VFDS66RK26040002_20260727_200936881.jpg.png`
- 第五行破洞样例的导出记录属于训练集，其余四行属于测试集。图注使用 selected fabric images，不得统称测试集图片。
- 在仓库根目录执行以下 PowerShell 命令：

```powershell
scp -P 13022 paper/compose_detection_figure.py paper/redraw_detection_labels.py tkz@8.138.122.96:/home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/scripts/
ssh -p 13022 -o BatchMode=yes -o ConnectTimeout=10 tkz@8.138.122.96 'python3 /home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/scripts/redraw_detection_labels.py --source /home/tkz/code/github/ultralytics/runs/detection_comparison --output-dir /home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures --font-size 44 --model-font-size 48 --gap 12'
scp -P 13022 tkz@8.138.122.96:/home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/qualitative_large_labels.png paper/Elsevier_submission/figures/fig6.png
scp -P 13022 tkz@8.138.122.96:/home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/qualitative_large_labels.layout.json paper/Elsevier_submission/figures/fig6.layout.json
scp -P 13022 tkz@8.138.122.96:/home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/qualitative_large_labels.sources.json paper/Elsevier_submission/figures/fig6.sources.json
```

## 热力图生成（fig7.png）

- 原始素材在本地 `C:/Users/tankz/Desktop/heat`，包含 `src/`、`yolov8n/`、`ours/`。不要转到服务器寻找热力图。
- 使用现有 `paper/compose_heatmap_figure.py`。三列固定为 Label、YOLOv8n、MCGI-YOLO，不显示字母下标，模型名字号 64。
- 三行顺序固定为下面命令中的顺序。直接拼接已保存的热力图，不用生成式图像工具改写检测或激活内容。

```powershell
$paperPython = 'C:/Users/tankz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $paperPython paper/compose_heatmap_figure.py --input C:/Users/tankz/Desktop/heat --output paper/Elsevier_submission/figures/fig7.png --label-font-size 64 --images VFDS66RK26040002_20260801_201333180.png VFDS66RK26040002_20260805_003052459.png VFDS66RK26040002_20260729_040549805.png
```

## 出图后检查

- 检测对比图（`fig6.png`）保留双栏，热力图（`fig7.png`）保留单栏；不要修改其他图片或重新生成旧稿的 figures。
- 保留输出旁的 `.layout.json` 和检测对比图的 `.sources.json`，用于核对样例顺序、预测值和来源。
- 查看两张图的标签、模型名及正文中的对应排版，再编译当前稿件并核对总页数：

```powershell
& ./paper/Elsevier_submission/build.ps1 -TeXBin 'C:/Users/tankz/application/MiKTex/miktex/bin/x64'
```
