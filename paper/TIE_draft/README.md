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

图表允许使用页顶、页底或图页浮动，并在参考文献前清页结束。热力图以接近原单栏的宽度置于双栏浮动体中，以便与现场部署图共享图页。按模板使用末页双栏平衡，但禁用纵向拉伸，避免作者简介之间产生过大的间距。

2026-10-20 迁移完成时，保留全部现有内容的 PDF 为 13 页。这里只完成模板迁移与必要的浮动排版调整，未删减正文或缩小模板规定的正文字号；投稿篇幅需在后续修订中处理。
