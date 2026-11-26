# JIOT 草稿维护入口

- 后续 JIOT 论文修改使用本目录的 `manuscript.tex`，图片和参考文献均从本目录读取。
- 模板来源为 `../templates/`，使用标准 `IEEEtran` 类和参考文献样式；不要修改模板原件、TIE 投稿归档或旧 DMFR 草稿。
- 当前检测器使用 `yolov8n-fabric-ribbon-cell-detail.yaml`。`C2fRibbonAttention` 位于骨干 P3/P4，`CellDetailUpsample` 位于自顶向下 P5→P4 和 P4→P3 两处。
- 当前结果对应 seed 6。测试集 81 图、91 个实例，mAP50 为 96.8%，mAP50–95 为 66.8%，RTX 3090 前向吞吐量为 454 FPS。模型的板端性能和配对消融待验证。历史结果见 `README.md`。
- 正文的架构与模块图使用 `figures/ribbon-cell-detail.pdf`，由 `scripts/draw_ribbon_cell_detail.py` 生成。SVG 和 PNG 同步保留。检测对照图和热力图使用当前模型生成后再入稿。
- `build.ps1` 编译后更新同目录的 `manuscript.pdf`；VS Code 本目录设置也将 PDF 输出到主文件同目录。
- 每次修改后创建对应 Git commit，遵循仓库根目录要求。
