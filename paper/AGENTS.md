# 论文维护入口

- 后续论文修改默认基于 `TIE_draft/manuscript.tex`，这是当前 TIE 投稿草稿。
- 对应图片、参考文献和编译脚本位于 `TIE_draft/figures/`、`TIE_draft/references.bib` 和 `TIE_draft/build.ps1`。
- `manuscript.tex` 保留为迁移前的历史稿，`TIE_LaTeX_template/` 保留为原始模板；不要将后续论文修改写回这两处。
- 每次改动后创建对应的 Git commit，遵循仓库根目录的要求。

## VS Code 编译入口

- 用户在 VS Code 中以 `TIE_draft/` 为工作区；其 `.vscode/settings.json` 覆盖 LaTeX Workshop 的编译配置。
- 编辑器通过 `pdflatex -> bibtex -> pdflatex -> pdflatex` 编译，PDF、SyncTeX 和辅助文件统一输出到 `TIE_draft/build/`，预览 `build/manuscript.pdf`。
- `pdflatex` 必须显式传入 `-output-directory=%OUTDIR%`，BibTeX 必须读取 `%OUTDIR%/%DOCFILE%`；仅修改 `latex.outDir` 不会改变自定义工具的实际输出路径。
- 日志若显示 `I can't write on file manuscript.pdf`，先核对实际输出路径和 PDF 占用，不要据此改动公式或删除参考文献。编辑器使用 `build/manuscript.log`；根目录旧日志可能仍保留此前错误。

## Fig. 6 检测对比图生成

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
scp -P 13022 tkz@8.138.122.96:/home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/qualitative_large_labels.png paper/TIE_draft/figures/compare.png
scp -P 13022 tkz@8.138.122.96:/home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/qualitative_large_labels.layout.json paper/TIE_draft/figures/compare.layout.json
scp -P 13022 tkz@8.138.122.96:/home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/qualitative_large_labels.sources.json paper/TIE_draft/figures/compare.sources.json
```

## Fig. 7 热力图生成

- 原始素材在本地 `C:/Users/tankz/Desktop/heat`，包含 `src/`、`yolov8n/`、`ours/`。不要转到服务器寻找热力图。
- 使用现有 `paper/compose_heatmap_figure.py`。三列固定为 Label、YOLOv8n、MCGI-YOLO，不显示字母下标，模型名字号 64。
- 三行顺序固定为下面命令中的顺序。直接拼接已保存的热力图，不用生成式图像工具改写检测或激活内容。

```powershell
$paperPython = 'C:/Users/tankz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $paperPython paper/compose_heatmap_figure.py --input C:/Users/tankz/Desktop/heat --output paper/TIE_draft/figures/heatmap_comparison.png --label-font-size 64 --images VFDS66RK26040002_20260801_201333180.png VFDS66RK26040002_20260805_003052459.png VFDS66RK26040002_20260729_040549805.png
```

## 出图后检查

- Fig. 6 保留双栏，Fig. 7 保留单栏；不要修改其他图片或重新生成旧稿的 figures。
- 保留输出旁的 `.layout.json` 和 Fig. 6 的 `.sources.json`，用于核对样例顺序、预测值和来源。
- 查看两张图的标签、模型名及正文中的对应排版，再编译 TIE 草稿并核对总页数：

```powershell
& ./paper/TIE_draft/build.ps1 -TeXBin 'C:/Users/tankz/application/MiKTex/miktex/bin/x64'
```
