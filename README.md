# Arxiv LLM Daily

一个自动获取和总结 arXiv 上 **大语言模型 (LLM)** 相关论文的工具。每天自动检索最新的论文，并使用 AI 生成中文摘要。本项目旨在帮助研究人员快速了解 LLM 领域的最新动态。

## 介绍
本项目是一个自动化的 arXiv 论文摘要生成器，使用 Python 和 GitHub Actions 实现。它可以定期从 arXiv 上获取最新的论文，并使用 DeepSeek/DashScope 等大模型生成摘要。生成的摘要将被自动部署到 GitHub Pages 上，方便用户查看。

**本项目每天自动更新以下领域的论文摘要：**
- **CS.CL** (Computation and Language)
- **CS.AI** (Artificial Intelligence)
- **CS.LG** (Machine Learning)
- **核心关键词**：Large Language Model, LLM, Transformer, RAG, Agent 等。

## 网页预览
[点击这里查看每日论文日报](https://matouxiao.github.io/Arxiv_LLM_Daily/)

## 功能特点

- **全自动运行**：利用 Github Actions 定时运行 (每天北京时间 09:00)。
- **AI 智能总结**：调用大模型 API (DeepSeek/Qwen) 生成高质量中文摘要。
- **去重机制**：自动过滤重复论文，避免重复阅读。
- **网页归档**：自动生成静态网站，支持历史日报归档查看。

## 配置

在 `config/settings.py` 中可以调整：
- 搜索分类 (CATEGORIES)
- 搜索关键词 (QUERY)
- 筛选规则 (如 max_results)

## 如何使用

### 1. GitHub Actions 自动部署 (推荐)
本项目已配置自动化工作流。
1. Fork 本仓库。
2. 在 `Settings` -> `Secrets` 中配置 `LLM_API_KEY`。
3. 在 `Settings` -> `Pages` 中选择 `gh-pages` 分支进行部署。

### 2. 本地运行
```bash
# 安装依赖
pip install -r requirements.txt

# 运行脚本
python main.py