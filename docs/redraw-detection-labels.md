# 已选论文图片的大字号标签重绘

直接在服务器运行，不需要 GPU、SSH 回传或重新加载模型，只依赖 Pillow：

```bash
cd /home/tkz/code/github/ultralytics
python paper/redraw_detection_labels.py --font-size 44 --model-font-size 48
```

已在服务器的独立输出目录部署并运行过一份相同脚本，当前可以直接用：

```bash
python3 /home/tkz/code/github/ultralytics/runs/detection_comparison/paper_figures/scripts/redraw_detection_labels.py \
  --font-size 44 --model-font-size 48
```

脚本默认读取 `/home/tkz/code/github/ultralytics/runs/detection_comparison` 下 GT、YOLOv8n、RT-DETR-R18、Faster-R-CNN-R50、PAGI-YOLO 的坐标 JSON 和其引用的干净原图。`PAGI-YOLO` 是已有导出目录名，当前论文及出图显示名为 `MCGI-YOLO`；保留原目录即可复用已有结果。它不在已经画过小字的图片上叠加大字。预测框标注类别及两位小数置信度，GT只显示类别。文字位置尽量避开检测框和相邻标签，框的坐标、数量、类别和预测分数保持原值，不进行新推理或结果筛选。

默认行顺序已按作者要求内置，无需桌面文件或额外选图清单：

1. `VFDS66RK26040002_20260801_195949765.jpg.png`
2. `VFDS66RK26040002_20260809_022619641.jpg.png`
3. `VFDS66RK26040002_20260727_193808156.jpg.png`
4. `VFDS66RK26040002_20260805_002859637.jpg.png`
5. `VFDS66RK26040002_20260727_200936881.jpg.png`

底部只显示模型名，不显示 `(a)`、`(b)` 等字母标识。

- `--font-size 44`：检测框类别/置信度字号。
- `--model-font-size 48`：底部模型名字号，独立调整。
- `--font`、`--model-font`：分别指定两种文字的 TrueType 字体。
- `--gap 12`：子图白色间距。
- `--no-conf`：隐藏分数，保留类别名。
- `--images 文件1 文件2 ...`：另行指定样例及顺序；或用 `--selection 路径.json` 读取旧格式选图清单，两者不能同时使用。

原图先按保存结果的 SHA-256 校验，所有模型使用同一原图；还检查原图尺寸与坐标记录一致。所有输出默认写入 `runs/detection_comparison/paper_figures`。原始数据与坐标 JSON 仅被读取，源数据缓存和单张重绘图保存在输出目录的 `work/`。

生成的文件：

- `qualitative_large_labels.png`：五列完整图。
- `qualitative_large_labels_baseline.png`：Label、YOLOv8n、MCGI-YOLO三列图。
- 对应 `.layout.json`：样例排列和面板哈希。
- `qualitative_large_labels.sources.json`：实际数据划分、原图哈希、原始检测结果和重绘文字记录。

`--source` 可改为另一份检测导出目录，`--output-dir` 指定最终图输出位置。仅需同步 `redraw_detection_labels.py` 和同目录的 `compose_detection_figure.py` 两个文件即可在服务器运行。也支持从本机读取服务器：增加 `--ssh tkz@8.138.122.96 --port 13022`，该模式输出默认位于本机 `paper/figures`。

2026-10-18实际核对：重排后的第5行 `VFDS66RK26040002_20260727_200936881.jpg.png` 孔洞样例来自 **train**，其余4张来自 **test**。默认 `--split auto` 在三个划分中逐一精确匹配并记录来源，重名匹配或哈希不符会报错。此五行图不能在论文图注中统一称作测试集样例；`--split test` 会严格拒绝当前训练集样例，而不会静默跳过。
