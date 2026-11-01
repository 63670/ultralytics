# JIOT 草稿维护入口

- 后续 JIOT 论文修改使用本目录的 `manuscript.tex`，图片和参考文献均从本目录读取。
- 模板来源为 `../templates/`，使用标准 `IEEEtran` 类和参考文献样式；不要修改模板原件、TIE 投稿归档或旧 DMFR 草稿。
- 本次从 TIE 投稿版迁移，仍保留 MCGI-YOLO 及其原有实验结果；新实验结果只在获得实测数据后替换。
- `build.ps1` 编译后更新同目录的 `manuscript.pdf`；VS Code 本目录设置也将 PDF 输出到主文件同目录。
- 每次修改后创建对应 Git commit，遵循仓库根目录要求。
