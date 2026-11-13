# Elsevier / JMS 论文维护目录

本目录是 MCGI-YOLO 论文的当前维护入口，后续正文、图表和实验结果均基于此版本修改。

- 手稿及 PDF 以论文标题命名；使用 `../els-cas-templates/` 提供的 `cas-dc.cls`、`cas-common.sty` 和 `cas-model2-names.bst`。
- 全文主线是将检测从成布后的独立验布环节前移到织造中的检测与停机干预；章节为 Introduction、Related Work、System Design、Detection Method、Experiments and Results、System Deployment and Industrial Application、Conclusion。相关工作区分成布后验布和已有机上检测，不声称首次提出边织边检；先行工作包括 Saeidi (2005)、Li (2013)、Çelik (2022)、Ni (2025)。工厂应用仅说明已部署于东莞明海纺织有限公司及其用途；不报告废布变化、减废率、经济收益或验布机采购节省。
- 数字引文由 `natbib` 管理；参考文献在 `references.bib`，网络资源采用原生 `url` 字段。
- `figures/` 包含14幅正文插图、7张作者照片及出图记录，迁移时保留最新文件内容和单/双栏设置。
- 作者顺序、邮箱、单位和3位通讯作者均保留；作者简介使用 CAS 的 `bio` 环境，基金信息位于 Acknowledgements。
- IEEE 专用的页眉、收稿/修回/录用日期及简介环境不再用于本版本。
- 原稿未提供 ORCID，暂不显示空的 ORCID 行；补充 ORCID 时应移除正文中的 `\let\printorcid\relax`。
- 图注和表题维持简短主题式写法；参数和计算依据在正文说明。
- `migration.json` 记录迁移来源及图片哈希；原 `../TIE_submission/` 保留归档。
- `.vscode/settings.json` 将编辑器预览和辅助文件写入 `build/`；`build.ps1` 同时发布同名 PDF 到本目录。

在仓库根目录运行：

```powershell
& ./paper/Elsevier_submission/build.ps1 -TeXBin 'C:/Users/tankz/application/MiKTex/miktex/bin/x64'
```

正文Fig. 11使用用户手动标注的 `figures/fps_accuracy.png`，保留原图效果。脚本从对比表读取数据另存预览或底图，不能覆盖正文图：

```powershell
python paper/plot_fps_accuracy.py --output paper/Elsevier_submission/figures/fps_accuracy_preview
python paper/plot_fps_accuracy.py --unlabeled
```
