# JIOT 草稿

主文件为 `manuscript.tex`，编译结果为同目录的 `manuscript.pdf`。

本稿由 `../TIE_submission/` 的已投稿版本完整复制，采用 `../templates/bare_jrnl_new_sample4.tex` 的 `lettersize,journal` 文档选项及同目录提供的 `IEEEtran.cls`，参考文献使用标准 `IEEEtran.bst`。页眉改为 IEEE Internet of Things Journal，收稿日期恢复为模板占位符。

当前检测器采用 **P3/P4 交错条带注意力与均值保持的细节上采样**，配置为 `ultralytics/cfg/models/v8/yolov8n-fabric-ribbon-cell-detail.yaml`。骨干模块为 `C2fRibbonAttention`，两处颈部模块为 `CellDetailUpsample`。

用户提供的结果对应 `runs/detect/jiot-ribbon-attention/cell-detail-upsample/6/weights/best.pt`。测试集共 81 张图、91 个实例，Precision 为 91.0%，Recall 为 97.7%，mAP50 为 96.8%，mAP50–95 为 66.8%。融合后参数量为 3,023,329，计算量为 8.2 GFLOPs。RTX 3090 的独立测速结果为 454 FPS，论文使用 FP32、batch=1、640×640 的模型前向口径。验证日志的 1.7 ms 属于验证阶段计时。

`figures/ribbon-cell-detail.*` 包含检测器结构和两个模块的示意图。PDF 用于正文，SVG 可编辑，PNG 用于预览。在仓库根目录运行 `python paper/JIOT_draft/scripts/draw_ribbon_cell_detail.py` 可生成三种格式，需要 Matplotlib。

硬件组件图 `figures/hardware-components.png` 来自 TIE 投稿归档中的 `fig2.png`，现场部署图 `figures/deployment-site.png` 来自 `../figures/site.png`。

当前草稿按完整模型的单种子结果报告检测器对比和逐类别结果。配对消融、当前模型的检测对照图和热力图，以及板端评估待补。

### 历史结果

此前的 GESR-YOLO 对应 `sobel-bottleneck-phase-v3`，实测为 97.4% mAP50、65.5% mAP50–95 和 RTX 3090 491 FPS。Sobel-Guide 单独版为 94.3% / 61.5%，SARU 单独版为 93.4% / 61.6%。旧结构图由 `scripts/draw_v3_modules.py` 生成。旧 TIE 检测对比图与热力图属于 MCGI-YOLO；旧 22 FPS 属于此前的 RK3566 部署记录。

## 编译

在仓库根目录执行：

```powershell
& ./paper/JIOT_draft/build.ps1 -TeXBin 'C:/Users/tankz/application/MiKTex/miktex/bin/x64'
```

脚本在 `build/` 中完成多轮编译，并将最新 PDF 复制到草稿目录。以本目录打开 VS Code 时，随附设置让 LaTeX Workshop 将 PDF 输出到 `.tex` 同目录。

TIE 投稿原件和旧 DMFR 草稿保持独立；后续 JIOT 修改以本目录为准。
