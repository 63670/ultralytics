# JIOT 草稿

主文件为 `manuscript.tex`，编译结果为同目录的 `manuscript.pdf`。

本稿由 `../TIE_submission/` 的已投稿版本完整复制，采用 `../templates/bare_jrnl_new_sample4.tex` 的 `lettersize,journal` 文档选项及同目录提供的 `IEEEtran.cls`，参考文献使用标准 `IEEEtran.bst`。页眉改为 IEEE Internet of Things Journal，收稿日期恢复为模板占位符。

本次仅迁移模板：保留原题目、正文、七位作者、图 1–7、作者照片、参考文献，以及 MCGI-YOLO 的方法和实验结果。新的 Sobel Bottleneck + Phase 算法及其结果尚未写入本稿。

## 编译

在仓库根目录执行：

```powershell
& ./paper/JIOT_draft/build.ps1 -TeXBin 'C:/Users/tankz/application/MiKTex/miktex/bin/x64'
```

脚本在 `build/` 中完成多轮编译，并将最新 PDF 复制到草稿目录。以本目录打开 VS Code 时，随附设置让 LaTeX Workshop 将 PDF 输出到 `.tex` 同目录。

TIE 投稿原件和旧 DMFR 草稿保持独立；后续 JIOT 修改以本目录为准。
