# TIE 投稿草稿

后续论文的正文、图片和参考文献修改均以本目录为准。

- `manuscript.tex`：当前论文入口，基于 `IEEEtranTIE.cls`。
- `manuscript.pdf`：编译后的预览。
- `figures/`：本稿实际引用的图片，迁移时从当前论文逐个原样复制。
- `references.bib`：当前参考文献数据库。
- `Bibliography/IEEEtranTIE.bst`：用户提供的 TIE 文献样式。
- `build.ps1`：编译正文和参考文献，不重新生成图片。

原始模板保留在 `../TIE_LaTeX_template/`，迁移前的论文保留在 `../manuscript.tex`。本稿独立维护，不需要与旧稿双向同步。

## 编译

已将 TeX 工具加入 PATH 时，在本目录执行：

```powershell
./build.ps1
```

当前 Windows 环境也可以指定 MiKTeX 路径：

```powershell
./build.ps1 -TeXBin "C:/Users/tankz/application/MiKTex/miktex/bin/x64"
```

Linux 或手动编译时，在本目录执行：

```bash
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape -output-directory=build manuscript.tex
bibtex build/manuscript
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape -output-directory=build manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape -output-directory=build manuscript.tex
```

手动编译的 PDF 位于 `build/manuscript.pdf`；PowerShell 脚本还会将其复制为本目录的 `manuscript.pdf`。

## 迁移范围

采用原始 TIE 模板的类文件、页眉和参考文献样式，移除旧稿针对另一模板设置的标题字号覆盖及负间距，按模板说明对关键词排序。保留现有研究内容、公式、实验数值、作者信息和作者简介；仅将引言末尾的章节导航更新为当前六个章节。未添加模板中的示例内容、示例照片或虚构的作者资料。

图表允许使用页顶、页底或图页浮动，避免强制清页产生独立空白图页。模型概览图跨双栏展示；模块图和热力图均使用单栏浮动体，按单栏宽度展示。保留模板字号和页边距，适当收紧图文间距，并将作者简介之间的间距设为两个基线间距。末页按参考文献、作者简介的顺序自然排列，不强制列平衡。

编译脚本优先读取 `build/` 中的辅助文件，避免编辑器在工程根目录生成的同名 `.aux`、`.bbl` 和 `.out` 文件干扰脚本编译。

2026-10-20 迁移完成时，保留全部现有内容的 PDF 为 13 页。这里只完成模板迁移与必要的浮动排版调整，未删减正文或缩小模板规定的正文字号；投稿篇幅需在后续修订中处理。
