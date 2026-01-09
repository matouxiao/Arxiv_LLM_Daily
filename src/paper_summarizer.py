"""
论文总结模块 - 使用大语言模型API生成论文摘要
"""
import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
import time
from datetime import datetime
import pytz
from config.settings import LLM_CONFIG
from src.clustering import get_embeddings, cluster_papers, select_representative_papers

class ModelClient:
    """兼容 OpenAI 接口格式的 API 客户端 (适配 DashScope/DeepSeek)"""
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or LLM_CONFIG['model']
        # 自动拼接 chat/completions 端点
        base_url = LLM_CONFIG['api_url'].rstrip('/')
        self.api_url = f"{base_url}/chat/completions"
        self.timeout = LLM_CONFIG.get('timeout', 60)
        
    def _create_headers(self) -> Dict[str, str]:
        """创建请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _create_request_body(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """创建符合 OpenAI 格式的请求体"""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or LLM_CONFIG['temperature'],
            "max_tokens": max_tokens or LLM_CONFIG['max_output_tokens'],
            # DashScope/OpenAI 通常使用 top_p，不一定需要 top_k，视模型而定
            "top_p": LLM_CONFIG['top_p']
        }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """发送请求并获取回复"""
        headers = self._create_headers()
        data = self._create_request_body(messages, temperature, max_tokens)
        
        for attempt in range(LLM_CONFIG['retry_count']):
            try:
                # 打印调试信息 (注意不要打印完整的 API Key)
                print(f"正在调用模型: {self.model}...")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"API 调用失败 [{response.status_code}]: {response.text}")
                    
                result = response.json()
                
                # 解析 OpenAI 格式的响应
                content = result["choices"][0]["message"]["content"]
                
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": result.get("usage", {})
                }
                
            except requests.Timeout:
                print(f"请求超时（{self.timeout}秒），正在重试 ({attempt + 1}/{LLM_CONFIG['retry_count']})...")
                time.sleep(LLM_CONFIG['retry_delay'])
            except Exception as e:
                print(f"发生错误: {str(e)}")
                if attempt == LLM_CONFIG['retry_count'] - 1:
                    raise
                time.sleep(LLM_CONFIG['retry_delay'])

class PaperSummarizer:
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.client = ModelClient(api_key, model)
        # 根据 Token 限制调整批处理大小，DeepSeek/Qwen 上下文较长，可以设大一点
        self.max_papers_per_batch = 5 

    def _generate_batch_summaries(self, papers: List[Dict[str, Any]], start_index: int) -> str:
        """为一批论文生成总结"""
        batch_prompt = ""
        has_full_text = False
        
        for i, paper in enumerate(papers, start=start_index):
            # 优先使用关键章节（摘要、Introduction、Related Work、Conclusion），如果没有则使用摘要
            paper_content = paper.get('full_text')
            if paper_content:
                has_full_text = True
                content_type = "摘要、介绍、相关工作和总结"
                # 限制长度避免 token 超限（保留前 20000 字符，通常足够覆盖关键章节）
                if len(paper_content) > 20000:
                    paper_content = paper_content[:20000] + "\n\n[注：内容已截断，仅显示前20000字符]"
            else:
                content_type = "摘要"
                paper_content = paper.get('summary', '无摘要')
            
            batch_prompt += f"""
论文 {i}：
标题：{paper['title']}
作者：{', '.join(paper['authors'])}
发布日期：{paper['published'][:10]}
arXiv链接：{paper['entry_id']}
{content_type}：
{paper_content}

"""
        
        content_desc = "摘要、介绍、相关工作和总结" if has_full_text else "摘要"
        # 让模型只生成结构化内容（JSON格式），格式由代码控制
        final_prompt = f"""你是一名负责"金融领域 AI 应用研究"的高级技术评审专家。

【任务】
对给定的论文进行筛选判断，评估其是否值得进入"重点精读 / 复现 / 应用评估"池。

你的目标不是判断论文是否学术上严谨，而是判断该论文是否：
- 属于我关注的技术方向
- 具有明确的工程实现或应用价值
- 具备迁移或改造到金融场景的可行性

【重点关注的研究方向（至少命中其一）】

1. 音频相关（Speech / Audio）
   - 语音识别、语音理解、音频生成
   - 音频与文本/视觉/结构化数据的多模态融合
   - 音频在业务场景中的应用（如通话分析、风控、客服、合规）

2. 多模态（Multimodal）
   - 文本-图像 / 文本-音频 / 文本-结构化数据
   - 多模态推理、多模态对齐、多模态数据构造
   - 多模态在真实业务任务中的落地方法

3. 大语言模型（LLM）
   - LLM 的工程化、推理能力、对齐、评测、训练策略
   - 面向具体任务的 LLM 应用，而非泛泛理论分析
   - LLM + 业务系统 / 数据 / 工具 的组合方式

4. 数据合成（Data Synthesis / Synthetic Data）
   - 合成数据生成方法
   - 合成数据在训练、评测、对齐中的实际作用
   - 有可复现或可迁移的数据生成流程

5. 智能体（Agent）
   - LLM Agent、Tool-using Agent、多步决策 Agent
   - 面向任务执行、规划、搜索、审查等应用
   - 明确的系统架构，而非抽象定义

6. MoE（Mixture of Experts）
   - MoE 在训练/推理/系统层面的设计
   - 专家路由、负载均衡、推理加速等工程问题
   - 与实际业务负载相关的 MoE 应用

【需要明确过滤掉的论文类型】

请直接判定为"不推荐"，若论文主要属于以下情况之一：
- 纯理论分析（如仅有数学推导、复杂定理证明）
- 方法高度抽象，缺乏明确任务或系统落点
- 无实验，或实验仅为 toy task / 人工合成小任务
- 应用场景与金融、企业级系统几乎无关联，且迁移成本极高
- 完全聚焦于证明最优性、收敛性、复杂度界限，而非可用系统

请阅读以下{len(papers)}篇论文的{content_desc}，为每篇论文提取关键信息。

请以 JSON 数组格式输出，每篇论文一个对象，包含以下字段：
- chinese_title: 中文翻译的标题
- keywords: 2-5个关键词标签，用中文逗号分隔（例如：RAG优化、多模态、Agent架构）
- core_pain_point: 核心痛点，一句话概括现有技术有什么缺陷
- technical_innovation: 技术创新，详细描述论文Method部分的方法、技术、算法和流程。要求：1) 使用编号列表格式（1) 2) 3) ...），每个编号描述一个具体的技术点；2) 详细说明技术方法、算法名称、架构设计、数据处理流程等；3) 如果涉及数据集，说明数据规模和类型（如：2739个环境、11,270条多轮对话数据）；4) 如果涉及实验，说明关键实验结果（如：在5个推理基准上超越静态数据集训练）；5) 用简洁的技术语言描述，避免学术套话；6) 控制在200字以内，确保信息完整
- application_value: 应用价值，说明这项技术能带来什么（限制在88字以内）
- summary: 详细总结，控制在200字以内
- decision: 推荐决策，只能是以下三个选项之一："推荐" 或 "边缘可看" 或 "不推荐"
- decision_reason: 决策理由，用不超过150字总结为什么做出该判断（需明确说明是否命中关注方向、工程价值、金融场景可行性）

要求：
1. **拒绝废话**：所有描述必须直击要害，不要使用"本文提出了..."这种学术套话。
2. **通俗易懂**：用业界通用的技术语言，而不是晦涩的数学描述。
3. **技术创新详细化**：technical_innovation 字段必须详细描述论文Method部分的内容，使用编号列表格式，每个技术点都要说明具体的方法、算法、流程、数据规模或实验结果。格式示例："1) 编程问题自动转化为可验证推理环境（2739个环境）；2) 动态难度控制器保持目标准确率；3) 环境淘汰机制维持多样性；4) 在5个推理基准上超越静态数据集训练。"
4. **保持简短**：总结篇幅严格控制在 200 字以内，应用价值限制在88字以内，技术创新限制在200字以内，决策理由限制在150字以内。
5. **只输出 JSON**：不要添加任何解释、注释或额外文字，确保所有输出符合严格的 JSON 格式。
6. **严格转义**：确保输出的 JSON 字符串中，所有字段值内的双引号必须使用反斜杠转义（如 \"内容\"），禁止出现原始的换行符。
7. **结构完整**：输出必须是一个完整的 JSON 数组，确保每篇论文对象之间有逗号分隔。
8. **决策准确**：decision 字段只能是"推荐"、"边缘可看"、"不推荐"三者之一，不要有其他文字。

输出格式示例：
[
  {{
    "chinese_title": "示例标题",
    "keywords": "关键词1、关键词2、关键词3",
    "core_pain_point": "一句话痛点描述",
    "application_value": "应用价值描述",
    "summary": "通俗简单地总结该论文的内容",
    "decision": "推荐",
    "decision_reason": "命中LLM工程化方向，提出的参数高效方法具有明确实现路径，可迁移到金融模型训练场景降低成本",
    "technical_innovation": "1) 编程问题自动转化为可验证推理环境（2739个环境）；2) 动态难度控制器保持目标准确率；3) 环境淘汰机制维持多样性；4) 在5个推理基准上超越静态数据集训练"
  }},
  ...
]

待处理论文列表：
{batch_prompt}"""

        try:
            response = self.client.chat_completion([{
                "role": "user",
                "content": final_prompt
            }])
            content = response["choices"][0]["message"]["content"].strip()
            
            # 解析 JSON 并组装成固定格式
            return self._format_papers_from_json(content, papers, start_index)
        except Exception as e:
            print(f"批处理生成失败: {e}")
            import traceback
            traceback.print_exc()
            return f"**[本批次生成失败]** 错误信息: {str(e)}"

    def _format_papers_from_json(self, json_content: str, papers: List[Dict[str, Any]], start_index: int):
        """从 JSON 内容组装成固定格式的 Markdown，同时返回结构化数据
        
        Returns:
            tuple: (formatted_text, paper_data_list) - 格式化文本和结构化数据
        """
        try:
            # 1. 基础清理
            json_content = json_content.strip()
            if json_content.startswith('```'):
                json_content = re.sub(r'^```(?:json)?\s*\n', '', json_content)
                json_content = re.sub(r'\n```\s*$', '', json_content)
        
        # === 新增：强制清洗非法字符 ===
        # 移除 JSON 字符串值中可能存在的未转义换行符（将其替换为空格）
        # 注意：这里只处理简单的格式问题
            json_content = json_content.replace('\n', ' ').replace('\r', '') 
        # 恢复被误删的结构化换行（可选，为了解析更稳健）
            json_content = re.sub(r'}\s*{', '},{', json_content) 
        
        # 2. 尝试解析
            try:
                paper_data_list = json.loads(json_content)
            except json.JSONDecodeError:
            # 容错：如果还是不行，尝试提取最外层的 [ ]
                match = re.search(r'\[.*\]', json_content, re.DOTALL)
                if match:
                    paper_data_list = json.loads(match.group(0))
                else:
                    raise
            
            # 组装成固定格式
            formatted_papers = []
            enriched_papers = []  # 保存结构化数据
            
            for i, (paper, paper_data) in enumerate(zip(papers, paper_data_list)):
                paper_num = start_index + i
                
                # 获取决策信息（不添加图标）
                decision = paper_data.get('decision', '未评估')
                
                formatted_paper = f"""## {paper_num}. {paper['title']}
- **中文标题**: {paper_data.get('chinese_title', '')}
- **Link**: {paper['entry_id']}
- **推荐决策:** {decision}
- **决策理由:** {paper_data.get('decision_reason', '')}
- **关键词:** {paper_data.get('keywords', '')}
- **核心痛点:** {paper_data.get('core_pain_point', '')}
- **应用价值:** {paper_data.get('application_value', '')}
- **总结:** {paper_data.get('summary', '')}
- **技术创新:** {paper_data.get('technical_innovation', '')}

---"""
                formatted_papers.append(formatted_paper)
                
                # 保存结构化数据（合并原始论文信息和LLM生成的信息）
                enriched_paper = {**paper, **paper_data}
                enriched_papers.append(enriched_paper)
            
            return "\n\n".join(formatted_papers), enriched_papers
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始内容: {json_content[:500]}...")
            # 如果 JSON 解析失败，尝试提取并重试
            # 查找 JSON 数组部分
            json_match = re.search(r'\[.*\]', json_content, re.DOTALL)
            if json_match:
                try:
                    paper_data_list = json.loads(json_match.group(0))
                    return self._format_papers_from_json(json_match.group(0), papers, start_index)
                except:
                    pass
            raise
        except Exception as e:
            print(f"格式化失败: {e}")
            raise

    def _process_batch(self, papers: List[Dict[str, Any]], start_index: int):
        """处理一批论文，返回格式化文本和结构化数据
        
        Returns:
            tuple: (summaries_text, paper_data_list)
        """
        print(f"正在批量处理 {len(papers)} 篇论文...")
        summaries_text, paper_data_list = self._generate_batch_summaries(papers, start_index)
        time.sleep(1) 
        return summaries_text, paper_data_list

    def _fix_batch_format(self, text: str, start_index: int, batch_size: int) -> str:
        """修正批次格式（保留作为兼容性方法，现在格式已由代码控制）"""
        # 由于现在格式由代码控制，这个方法主要用于清理可能的问题
        # 清理多余的空行（超过2个连续空行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _generate_batch_summary(self, papers: List[Dict[str, Any]]):
        """生成所有论文的摘要，返回格式化文本和结构化数据
        
        Returns:
            tuple: (summaries_text, all_paper_data) - 格式化文本和所有论文的结构化数据
        """
        all_summaries = []
        all_paper_data = []
        total_papers = len(papers)
        
        for i in range(0, total_papers, self.max_papers_per_batch):
            batch = papers[i:i + self.max_papers_per_batch]
            print(f"\n正在处理第 {i + 1} 到 {min(i + self.max_papers_per_batch, total_papers)} 篇论文...")
            batch_summary, batch_paper_data = self._process_batch(batch, i + 1)
            
            # 后处理：修正序号和格式
            batch_summary = self._fix_batch_format(batch_summary, i + 1, len(batch))
            
            all_summaries.append(batch_summary)
            all_paper_data.extend(batch_paper_data)
        
        # 合并所有批次，确保批次之间有分隔符
        result = "\n\n".join(all_summaries)
        
        return result, all_paper_data

    def _generate_trend_analysis(self, papers: List[Dict[str, Any]], paper_data_list: List[Dict[str, Any]]) -> str:
        """
        使用 embedding 聚类筛选代表性论文，然后生成趋势报告
        
        Args:
            papers: 原始论文列表
            paper_data_list: 包含 LLM 生成的 summary 等字段的结构化数据
        """
        print("\n" + "="*60)
        print("开始基于 Embedding 聚类的趋势分析")
        print("="*60)
        
        try:
            # 1. 提取所有论文的 summary 字段用于 embedding
            summaries = [paper_data.get('summary', '') for paper_data in paper_data_list]
            
            if not summaries or len(summaries) == 0:
                print("警告：没有找到论文摘要，使用降级策略")
                return self._generate_trend_analysis_fallback(papers, paper_data_list)
            
            print(f"提取了 {len(summaries)} 篇论文的摘要")
            
            # 2. 获取 embeddings
            embeddings = get_embeddings(summaries)
            
            if not embeddings or len(embeddings) != len(summaries):
                print("警告：Embedding 获取失败，使用降级策略")
                return self._generate_trend_analysis_fallback(papers, paper_data_list)
            
            # 3. 进行聚类
            # 根据配置选择聚类方法
            from config.settings import CLUSTERING_CONFIG
            clustering_method = CLUSTERING_CONFIG.get('method', 'dbscan')
            
            if clustering_method == 'kmeans':
                from src.clustering import cluster_papers_kmeans
                n_clusters = CLUSTERING_CONFIG.get('n_clusters', 4)
                labels = cluster_papers_kmeans(embeddings, n_clusters)
            else:
                labels = cluster_papers(embeddings)
            
            # 4. 选择代表性论文
            representative_papers = select_representative_papers(paper_data_list, embeddings, labels)
            
            if not representative_papers:
                print("警告：未能选择代表性论文，使用降级策略")
                return self._generate_trend_analysis_fallback(papers, paper_data_list)
            
            print(f"\n从 {len(paper_data_list)} 篇论文中筛选出 {len(representative_papers)} 篇代表性论文")
            
            # 5. 构建代表性论文的摘要文本（用于 LLM 分析）
            representative_summaries = []
            for paper in representative_papers:
                summary_text = f"""
标题：{paper.get('title', 'Unknown')}
关键词：{paper.get('keywords', '')}
核心痛点：{paper.get('core_pain_point', '')}
技术创新：{paper.get('technical_innovation', '')}
总结：{paper.get('summary', '')}
"""
                representative_summaries.append(summary_text.strip())
            
            summaries_for_analysis = "\n\n---\n\n".join(representative_summaries)
            
            # 6. 调用 LLM 生成趋势报告
            analysis_prompt = f"""
你是一名科技情报分析师。以下是今日 Arxiv 更新的大模型(LLM)领域论文中，通过聚类算法筛选出的 {len(representative_papers)} 篇代表性论文的详细摘要。

这些论文已经过智能聚类，代表了今日论文的主要研究方向。请基于这些摘要内容，生成一份趋势简报。

要求：
1. 根据摘要中的"关键词"、"核心痛点"、"技术创新"等信息，将论文归纳为 2-4 个核心研究热点（如：RAG优化、多模态、推理加速、安全对齐等）。
2. 每个热点下，写一句简短的"赛道观察"（说明该方向今天的技术突破点或关注点）。
3. 列出属于该热点的最具代表性的论文标题（只列标题）。

请严格遵循以下 Markdown 格式输出：

## 📊 今日趋势速览 (Trend Analysis)

### 🔥 [热点方向名称，例如：RAG 检索增强]
> **赛道观察：** (一句话概括该方向今天的技术突破点或关注点)
- (论文标题1)
- (论文标题2)

### 🤖 [热点方向名称2]
> **赛道观察：** ...
- ...

---

待分析的代表性论文摘要：
{summaries_for_analysis}
"""
            
            print("\n正在调用 LLM 生成趋势报告...")
            response = self.client.chat_completion([{
                "role": "user",
                "content": analysis_prompt
            }])
            
            result = response["choices"][0]["message"]["content"].strip()
            print("趋势报告生成成功！")
            return result
            
        except Exception as e:
            print(f"聚类趋势分析失败: {e}")
            import traceback
            traceback.print_exc()
            print("使用降级策略...")
            return self._generate_trend_analysis_fallback(papers, paper_data_list)
    
    def _generate_trend_analysis_fallback(self, papers: List[Dict[str, Any]], paper_data_list: List[Dict[str, Any]]) -> str:
        """
        降级策略：当聚类失败时，使用传统方法生成趋势报告
        """
        print("使用降级策略生成趋势报告（不使用聚类）")
        
        # 构建所有论文的摘要文本
        all_summaries = []
        for paper in paper_data_list[:15]:  # 最多取前15篇避免 token 超限
            summary_text = f"""
标题：{paper.get('title', 'Unknown')}
关键词：{paper.get('keywords', '')}
核心痛点：{paper.get('core_pain_point', '')}
技术创新：{paper.get('technical_innovation', '')}
"""
            all_summaries.append(summary_text.strip())
        
        summaries_for_analysis = "\n\n---\n\n".join(all_summaries)
        
        analysis_prompt = f"""
你是一名科技情报分析师。以下是今日 Arxiv 更新的 {len(paper_data_list)} 篇大模型(LLM)领域论文的摘要信息。

请基于这些摘要内容，生成一份趋势简报。

要求：
1. 根据摘要中的"关键词"、"核心痛点"、"技术创新"等信息，将论文归纳为 2-4 个核心研究热点。
2. 每个热点下，写一句简短的"赛道观察"。
3. 列出属于该热点的最具代表性的论文标题。

请严格遵循以下 Markdown 格式输出：

## 📊 今日趋势速览 (Trend Analysis)

### 🔥 [热点方向名称]
> **赛道观察：** (一句话概括)
- (论文标题1)
- (论文标题2)

---

待分析的论文摘要：
{summaries_for_analysis}
"""
        
        try:
            response = self.client.chat_completion([{
                "role": "user",
                "content": analysis_prompt
            }])
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"降级策略也失败: {e}")
            return "*(趋势分析生成失败，请查看下方详细列表)*"

    def summarize_papers(self, papers: List[Dict[str, Any]], output_file: str) -> bool:
        try:
            print(f"开始生成论文总结，共 {len(papers)} 篇...")
            
            # 1. 生成所有单篇论文的摘要（返回文本和结构化数据）
            summaries, paper_data_list = self._generate_batch_summary(papers)
            
            # 2. 基于聚类筛选代表性论文，生成趋势报告
            trend_analysis = self._generate_trend_analysis(papers, paper_data_list)
            
            # 3. 对论文按推荐度和聚类信息排序
            sorted_paper_data = self._sort_papers_by_priority(paper_data_list)
            
            # 4. 重新生成排序后的摘要文本
            sorted_summaries = self._regenerate_summaries_text(sorted_paper_data)
            
            # 5. 组合最终报告
            markdown_content = self._generate_markdown(papers, sorted_summaries, trend_analysis)
            
            # 保存文件
            output_md = str(Path(output_file).with_suffix('.md'))
            with open(output_md, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Markdown文件已保存：{output_md}")
            
            return True
            
        except Exception as e:
            print(f"严重错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _sort_papers_by_priority(self, paper_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按推荐度和聚类信息对论文排序
        
        排序优先级：
        1. 推荐决策（推荐 > 边缘可看 > 不推荐）
        2. 聚类排名（大聚类优先）
        3. 到聚类中心的距离（越近越靠前）
        """
        print("\n正在按推荐度和聚类信息对论文排序...")
        
        # 定义推荐决策的优先级
        decision_priority = {
            '推荐': 0,
            '边缘可看': 1,
            '不推荐': 2,
            '未评估': 3
        }
        
        def sort_key(paper):
            decision = paper.get('decision', '未评估')
            cluster_rank = paper.get('_cluster_rank', 999)
            distance = paper.get('_distance_to_center', 999.0)
            
            return (
                decision_priority.get(decision, 999),  # 推荐决策优先级
                cluster_rank,                          # 聚类排名（0=最大聚类）
                distance                               # 到聚类中心的距离
            )
        
        sorted_papers = sorted(paper_data_list, key=sort_key)
        
        # 打印排序结果统计
        print(f"排序完成：")
        for i, paper in enumerate(sorted_papers[:5], 1):
            decision = paper.get('decision', '未评估')
            cluster_id = paper.get('_cluster_id', 'N/A')
            cluster_size = paper.get('_cluster_size', 'N/A')
            title = paper.get('title', 'Unknown')[:50]
            print(f"  {i}. [{decision}] 聚类{cluster_id}({cluster_size}篇) - {title}...")
        
        if len(sorted_papers) > 5:
            print(f"  ... 还有 {len(sorted_papers) - 5} 篇论文")
        
        return sorted_papers

    def _regenerate_summaries_text(self, sorted_paper_data: List[Dict[str, Any]]) -> str:
        """
        根据排序后的论文数据重新生成摘要文本
        """
        print("正在重新生成排序后的摘要文本...")
        
        formatted_papers = []
        
        for i, paper_data in enumerate(sorted_paper_data, 1):
            decision = paper_data.get('decision', '未评估')
            
            formatted_paper = f"""## {i}. {paper_data.get('title', 'Unknown')}
- **中文标题**: {paper_data.get('chinese_title', '')}
- **Link**: {paper_data.get('entry_id', '')}
- **推荐决策:** {decision}
- **决策理由:** {paper_data.get('decision_reason', '')}
- **关键词:** {paper_data.get('keywords', '')}
- **核心痛点:** {paper_data.get('core_pain_point', '')}
- **应用价值:** {paper_data.get('application_value', '')}
- **总结:** {paper_data.get('summary', '')}
- **技术创新:** {paper_data.get('technical_innovation', '')}


---"""
            formatted_papers.append(formatted_paper)
        
        return "\n\n".join(formatted_papers)

    def _generate_markdown(self, papers: List[Dict[str, Any]], summaries: str, trend_analysis: str = "") -> str:
        beijing_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        
        # 统计推荐决策分布
        recommend_count = summaries.count('**推荐决策:** 推荐')
        maybe_count = summaries.count('**推荐决策:** 边缘可看')
        not_recommend_count = summaries.count('**推荐决策:** 不推荐')
        
        return f"""# Arxiv LLM 每日研报

> **更新时间**：{beijing_time}
> **论文数量**：{len(papers)} 篇
> **推荐分布**：⭐推荐 {recommend_count} 篇 | 📌边缘可看 {maybe_count} 篇 | ❌不推荐 {not_recommend_count} 篇
> **自动生成**：By Arxiv_LLM_Daily Agent

---

{trend_analysis}

---

## 📝 论文详细列表

{summaries}

---
*Generated by AI Agent based on arXiv (cs.CL)*
"""
def create_summarizer(api_key: str, model: Optional[str] = None):
    return PaperSummarizer(api_key, model)