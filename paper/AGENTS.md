# 论文维护入口

- 后续 MCGI-YOLO 论文修改默认基于 `Elsevier_submission/A Defect Detection and Closed-Loop Control System with MCGI-YOLO for Circular Knitting Machines.tex`；转投修订也在此维护。
- 全文主线是将检测从成布后的独立验布环节前移到织造中的检测与停机干预；章节为 Introduction、Related Work、System Design、Detection Method、Experiments and Results、System Deployment and Industrial Application、Conclusion。相关工作区分成布后验布和已有机上检测，不声称首次提出边织边检；不纳入经编机相关工作。验布机文献区分实际系统运行与采集数据后的模型评估；先行工作包括 Saeidi (2005)、Çelik (2022)、Ni (2025)。明海原验布机数量未知；不再写“已取消独立验布环节”，验布设备采购成本按公开项目配置作情景估算，单独于月度布损减少额，依据保存在 hardware-price-references.json。
- `TIE_submission/` 保留为迁移前归档；后续正文、图表、实验数据和编译默认只更新 `Elsevier_submission/`。
- 图注和表题采用 JMS 常见的简短主题式写法，原则上一个短句；多子图仅保留必要的(a)–(d)对应。参数、测试口径、百分数和最优/次优标记规则放在正文，避免在图表标题重复说明。
- 对应图片、参考文献和编译脚本位于 `Elsevier_submission/figures/`、`Elsevier_submission/references.bib` 和 `Elsevier_submission/build.ps1`。模板采用 `els-cas-templates/` 提供的 `cas-dc` 双栏类；原有正文图片文件名 `fig1.png` 至 `fig7.png` 保持稳定；正文图号由 LaTeX 自动生成，不再等同于文件名。旧草稿目录已删除，不重新创建。
- 数据集样本图使用 `figures/defect.png`，在 Detection Method 的 Defect Morphology and Design Motivation 小节双栏展示，标签为 `fig:dataset-examples`；四行依次为 row、col、hole 及多类瑕疵共存样本。
- 真实废布照片使用 `figures/fabric_with_defect.jpg`，在延迟干预损失估算小节单栏展示，标签为 `fig:waste-fabric`，说明持续纱线异常造成重复横向瑕疵和布面报废。
- 产线应用效果见 `sec:factory-benefits` 和 `tab:factory-benefits`：用户确认的明海纺织厂月均估计，48台圆机、约350 t/月，废布约2030→148 kg/月；正常布25 CNY/kg、废布回收5 CNY/kg，按差额20 CNY/kg净损失估算月损失40,600→2,960 CNY，减少92.7%，月差额37,640 CNY，年化推算451,680 CNY。用户已确认按全厂2030 kg/月的口径，不使用每台30–40 kg/月的概略范围；未提供具体统计月份，不写成特定月份的实测结果。
- 标注框形态与尺度图为 `figures/annotation_shapes.png`（双栏，`fig:annotation-shapes`，与典型样本图一起置于算法设计动机小节）；由 `paper/plot_annotation_shapes.py --labels <数据集labels目录> --output paper/Elsevier_submission/figures/annotation_shapes.png` 生成，全部框中心对齐并保留归一化宽高，附 `.stats.json` 记录统计和标注哈希。机器直径和旋转周期见 `tab:production-inputs`，该两行参数表置于处理器选型小节、帧率计算之前；横向视场宽度w=16 cm在相机介绍正文说明，不在表中重复；纵向每圈下降约0.03 m及3分钟（180 s）未发现瑕疵的延迟在后文正文说明，对应90圈、累计横向瑕疵长度约287.3 m及纵向受影响布长约2.7 m。
- FPS–mAP 散点图为 `figures/fps_accuracy.png` 与矢量版 `.pdf`，运行 `python paper/plot_fps_accuracy.py` 从当前正文 `tab:comparison` 自动读取数据；使用 RTX 3090、FP32、batch=1、640×640 的纯推理 FPS，数据快照保存在同名 `.json`。正文使用矢量 PDF 在检测器对比部分双栏展示，标签为 `fig:fps-accuracy`。
- 手动标注用散点底图运行 `python paper/plot_fps_accuracy.py --unlabeled`，另存 `figures/fps_accuracy_unlabeled.{png,pdf,svg,json}`，左图 mAP@50、右图 mAP@50:95；保留坐标轴、刻度数字、网格、颜色、点形和数据位置；仅去掉模型名称、引线、图标题及轴标题。此版本不自动替换正文已标注插图。
- 硬件系统部分保留 `serdes.png`（单栏）。实验结果后单设 `System Deployment and Industrial Application`，依次介绍部署与23 FPS、用户可配置停机决策、客户端、云平台与数字孪生、延迟干预分析；`detect.png`、`iot_platform.png`（均双栏）放在该节。平台图子图对应为 (a)系统落地页、(b)瑕疵趋势图、(c)瑕疵记录、(d)远程灵敏度与阈值配置、(e)RKNN模型及标签文件管理与OTA、(f)设备实时状态；(d)与(f)对应数字孪生的配置与状态同步功能。客户端界面以Kotlin开发并运行于RK3566的Android系统。
- RK3566 部署评估在 Linux 下进行：YOLOv8n 为30 FPS、MCGI-YOLO 为23 FPS；两者输入640×640、conf=0.001、NMS IoU=0.7、max_det=16。P/R 按 IoU=0.5 的最佳类别平均F1位置报告，并按类别宏平均。部署结果见正文 `tab:rknn-deployment`，不得混入 RTX 3090 的主对比表。
- 用户可配置停机规则置于 `System Deployment and Industrial Application`，在部署性能之后、客户端界面之前，以 `User-Configurable Post-NMS Stopping Decision`（`sec:stop-rules`）单独成节，包含 NMS 后置信度筛选、原图尺度判断、数量条件和预设表；Detection Method 不再包含该小节，不将停机规则称为改进 NMS。
- 客户端停机规则按原始 `1280×1024` 图像计算；预设元组依次为（置信度、破洞检测框面积、横向检测框宽度、竖向检测框高度、每帧数量）：低 `(0.50, 6400, 800, 400, 3)`，中 `(0.40, 2500, 400, 200, 2)`，高 `(0.30, 1000, 200, 100, 1)`。先按置信度筛选，再判断“任意一个对应类别尺寸超过阈值，或者该帧通过置信度筛选的瑕疵总数达到数量阈值”，二者为或关系，不跨帧累计。三档均可编辑。
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
