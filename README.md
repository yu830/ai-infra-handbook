# AI Infra 技术手册（Material for MkDocs）

采用与 [XiangShan 官方文档](https://docs.xiangshan.cc/zh-cn/latest/contribution/documentation/formatting/) 同款的 **Material for MkDocs** 三栏工程文档架构，去除了默认蓝色主题，改为极简黑白灰配色，原生支持 MathJax 数学公式、Mermaid 矢量图、SVG 架构图与本地全文搜索。

## 本地实时预览

```bash
pip install -r requirements.txt
mkdocs serve
```

## 在线托管（任选其一）

- **Vercel（已内置 `vercel.json` + `.python-version`）**：在 Vercel 网页端导入 `yu830/ai-infra-handbook` 直接点击 Deploy，无需手动填写构建命令或输出目录。
- **Read the Docs（已内置 `.readthedocs.yaml`）**：导入 `yu830/ai-infra-handbook` 即可自动构建托管。
