# AI Infra 技术手册（Material for MkDocs）

采用与 [XiangShan 官方文档](https://docs.xiangshan.cc/zh-cn/latest/contribution/documentation/formatting/) 同款的 **Material for MkDocs** 三栏工程文档架构，去除了默认蓝色主题，改为极简黑白灰配色，原生支持 MathJax 数学公式、Mermaid 矢量图、SVG 架构图与本地全文搜索。

## 本地实时预览

```bash
pip install -r requirements.txt
mkdocs serve
```

## 在线托管（Read the Docs）

登录 Read the Docs 导入 `yu830/ai-infra-handbook`，内置的 `.readthedocs.yaml` 会自动完成构建与托管；后续每次推送新章节均会自动更新。
