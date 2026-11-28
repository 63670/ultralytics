# JIOT 草稿维护入口

- 后续 JIOT 论文修改使用本目录的 `manuscript.tex`，图片和参考文献均从本目录读取。
- 模板来源为 `../templates/`，使用标准 `IEEEtran` 类和参考文献样式；不要修改模板原件、TIE 投稿归档或旧 DMFR 草稿。
- 当前检测器 ISU-YOLO 使用 `yolov8n-fabric-ribbon-cell-detail.yaml`。ISA（`C2fRibbonAttention`）位于骨干 P3/P4，CDU（`CellDetailUpsample`）位于自顶向下 P5→P4 和 P4→P3 两处。
- 当前结果对应 seed 6。测试集 81 图、91 个实例，mAP50 为 96.8%，mAP50–95 为 66.8%，RTX 3090 前向吞吐量为 454 FPS。模型的板端性能和配对消融待验证。历史结果见 `README.md`。
- 正文的 Fig. 1–5 使用用户提供的 `figures/fig1.png`、`figures/fig2.png`、`figures/typical.png`、`figures/fig3.png` 和 `figures/compare.png`，分别为系统架构、产线硬件、典型瑕疵、ISU-YOLO 网络结构和定性检测对比。Fig. 3 位于算法设计动机小节。Fig. 5 包含 5 张训练图和 4 张测试图。硬件正文采用 RK3588 和 STM32 配置，相机通过 RJ45 以太网连接。热力图使用当前模型生成后再入稿。
- `build.ps1` 编译后更新同目录的 `manuscript.pdf`；VS Code 本目录设置也将 PDF 输出到主文件同目录。
- 每次修改后创建对应 Git commit，遵循仓库根目录要求。
