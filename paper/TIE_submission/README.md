# MCGI-YOLO 论文维护目录

本目录是 **A Defect Detection and Closed-Loop Control System with MCGI-YOLO for Circular Knitting Machines** 的唯一维护入口，后续转投修订也在此进行。

- 同名 `.tex` 与 `.pdf` 为正文和编译结果，保留以论文标题命名的文件名。
- `figures/fig1.png` 至 `fig7.png` 为正文插图，`a1.jpg` 至 `a7.jpg` 为作者照片。
- `fig6.layout.json`、`fig6.sources.json` 和 `fig7.layout.json` 保留对比图与热力图的排版、来源记录。
- `references.bib`、`IEEEtranTIE.cls` 和 `Bibliography/IEEEtranTIE.bst` 为独立编译所需文件。
- `.vscode/settings.json` 提供 LaTeX Workshop 编译配置；辅助文件和编辑器预览位于 `build/`。
- `build.ps1` 自动识别本目录唯一的 `.tex` 文件，完成四步编译，并将同名 PDF 发布到正文所在目录。
- Cover letter 及其原始模板保存在上一级目录的 `Cover letter.docx` 与 `Cover letter template.doc`。

在本目录执行：

```powershell
./build.ps1 -TeXBin "C:/Users/tankz/application/MiKTex/miktex/bin/x64"
```

旧草稿目录已合并并删除。2026-11-01 的原始投稿版本可通过 Git 标签 `tie-submission-snapshot-2026-11-01` 查看或恢复。JIOT 新模型稿件继续在 `../JIOT_draft/` 单独维护。
