# JIOT 草稿维护入口

- 后续 JIOT 论文修改使用本目录的 `manuscript.tex`，图片和参考文献均从本目录读取。
- 模板来源为 `../templates/`，使用标准 `IEEEtran` 类和参考文献样式；不要修改模板原件、TIE 投稿归档或旧 DMFR 草稿。
- 检测算法已更新为 `sobel-bottleneck-phase-v3`：SGRE 对应 `SobelResidual`（由 `C2fSobelBottleneck` 插入骨干 P3/P4 每个 bottleneck 后）；SARU 对应 `PhaseAwareUpsample`。
- 用户确认完整 v3 的 mAP50、mAP50–95 和 RTX 3090 FPS 分别为 97.4%、65.5% 和 491。RK3588 板端速度待实测，不能沿用 TIE 的 RK3566 22 FPS。
- 模块图及架构图由 `scripts/draw_v3_modules.py` 生成，保留 PDF、SVG、PNG；旧 `fig4.png`、`fig5.png` 留作历史素材，正文使用新版矢量 PDF。
- 检测对比图 `fig6.png` 和热力图 `fig7.png` 仍为 TIE 迁移素材，并非新 v3 的出图结果，正式定稿前需以新模型结果替换。
- `build.ps1` 编译后更新同目录的 `manuscript.pdf`；VS Code 本目录设置也将 PDF 输出到主文件同目录。
- 每次修改后创建对应 Git commit，遵循仓库根目录要求。
