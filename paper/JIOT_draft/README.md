# JIOT 草稿

主文件为 `manuscript.tex`，编译结果为同目录的 `manuscript.pdf`。

本稿由 `../TIE_submission/` 的已投稿版本完整复制，采用 `../templates/bare_jrnl_new_sample4.tex` 的 `lettersize,journal` 文档选项及同目录提供的 `IEEEtran.cls`，参考文献使用标准 `IEEEtran.bst`。页眉改为 IEEE Internet of Things Journal，收稿日期恢复为模板占位符。

算法部分现已更新为 `sobel-bottleneck-phase-v3`。Sobel 模块的论文名称为 **Selective Gradient Residual Enhancement (SGRE)**；Phase 模块为 **Subpixel-Adaptive Residual Upsampling (SARU)**。代码类名和配置名保持不变，兼容已训练权重。

正文、摘要、引言及消融分析使用新机制。用户确认完整模型的结果为 97.4% mAP50、65.5% mAP50–95 和 RTX 3090 491 FPS；SGRE-only 为 94.3% / 61.5%，SARU-only 为 93.4% / 61.6%，对应用户填写的消融表。原 22 FPS 不作为新版结果，RK3588 部署测试后再更新。

`figures/sgre.*` 和 `saru.*` 为独立模块图，`modules-v3.*` 为正文组合图，`network-v3.*` 为更新后的架构示意图。PDF 为矢量出图，SVG 可编辑，PNG 用于预览。在仓库根目录运行 `python paper/JIOT_draft/scripts/draw_v3_modules.py` 重新生成，需要 Matplotlib。

`fig6.png` 和 `fig7.png` 仍为旧 TIE 对比图与热力图，尚未用新版模型重新生成；定稿时需替换。原题目和模型总名称暂时保留，其余系统和作者资料沿用迁移稿。

## 编译

在仓库根目录执行：

```powershell
& ./paper/JIOT_draft/build.ps1 -TeXBin 'C:/Users/tankz/application/MiKTex/miktex/bin/x64'
```

脚本在 `build/` 中完成多轮编译，并将最新 PDF 复制到草稿目录。以本目录打开 VS Code 时，随附设置让 LaTeX Workshop 将 PDF 输出到 `.tex` 同目录。

TIE 投稿原件和旧 DMFR 草稿保持独立；后续 JIOT 修改以本目录为准。
