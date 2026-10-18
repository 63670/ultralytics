# 已选论文图片的大字号标签重绘

在本地仓库运行，不需要 GPU 或重新加载模型，只需 Pillow 和可用的 SSH 登录：

```powershell
python paper/redraw_detection_labels.py --ssh tkz@8.138.122.96 --port 13022 --font-size 44
```

脚本按 `paper/figures/qualitative_comparison.layout.json` 中的5个样例文件名，只读获取服务器 `/home/tkz/code/github/ultralytics/runs/detection_comparison` 下 GT、YOLOv8n、RT-DETR-R18、Faster-R-CNN-R50、PAGI-YOLO 的坐标 JSON 和其引用的干净原图。它不在已经画过小字的图片上叠加大字。默认字号从原来的22增加到44，预测框标注类别及两位小数置信度，GT只显示类别。文字位置会尽量避开检测框和相邻标签，框的坐标、数量、类别和预测分数保持原值，不进行新推理或结果筛选。

原图先按保存结果的 SHA-256 校验，所有模型使用同一原图；还检查原图尺寸与坐标记录一致。源数据缓存在忽略的 `paper/work/detection-label-redraw/source_bundle.json`，单张重绘图保存在该工作目录的 `panels/`，不覆盖桌面图片及服务器文件。

生成的文件：

- `paper/figures/qualitative_large_labels.png`：五列完整图。
- `paper/figures/qualitative_large_labels_baseline.png`：GT、YOLOv8n、PAGI-YOLO三列图。
- 对应 `.layout.json`：样例排列和面板哈希。
- `paper/figures/qualitative_large_labels.sources.json`：实际数据划分、原图哈希、原始检测结果和重绘文字记录。

需要继续放大可改 `--font-size 52`；`--no-conf` 隐藏分数；`--font` 指定本地 TrueType 字体。若在服务器运行，省略 `--ssh`，`--source` 使用服务器目录，同时给出本地存在的 `--selection` 清单。脚本只依赖标准库、Pillow 和同目录的 `compose_detection_figure.py`。

2026-10-18实际核对：第2行 `VFDS66RK26040002_20260727_200936881.jpg.png` 孔洞样例来自 **train**，其余4张来自 **test**。因此默认 `--split auto` 会在三个划分中逐一精确匹配，并将来源写入记录；重名匹配或哈希不符会报错。此五行图不能在论文图注中统一称作测试集样例。如需纯测试集对比，请先更换第2行样例并更新选图清单；指定 `--split test` 会严格拒绝当前训练集样例，而不会静默跳过。
