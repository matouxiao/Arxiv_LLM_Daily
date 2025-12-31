"""
论文总结模块 - 使用大语言模型API生成论文摘要
"""
import os
import json
from typing import List, Dict, Any, Optional
import requests
import time
from datetime import datetime
import pytz
from config.settings import LLM_CONFIG

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
        self.max_papers_per_batch = 10 

    def _generate_batch_summaries(self, papers: List[Dict[str, Any]], start_index: int) -> str:
        """为一批论文生成总结"""
        batch_prompt = ""
        for i, paper in enumerate(papers, start=start_index):
            batch_prompt += f"""
论文 {i}：
标题：{paper['title']}
作者：{', '.join(paper['authors'])}
发布日期：{paper['published'][:10]}
arXiv链接：{paper['entry_id']}
摘要：{paper['summary']}

"""
        # 修改为更适合 LLM 领域研究者的 Prompt
        final_prompt = f"""你是一位专注于大语言模型（LLM）的研究员。请阅读以下{len(papers)}篇论文的摘要，并生成中文速读周报。

请严格遵循以下 Markdown 格式输出（不要输出任何开场白或结束语）：

### [中文标题](文章链接)
- **关键词:** (提取3-5个核心技术标签，如: RAG, LoRA, Agent)
- **解决痛点:** (一句话概括这篇论文试图解决什么具体问题)
- **核心创新:** (简要列出技术创新点)
- **一句话总结:** (通俗易懂的结论)

---

请注意：
1. 直接输出 Markdown 内容，不需要包裹在 ```markdown 代码块中。
2. 即使原文是英文，也请用**中文**进行总结。
3. 确保包含原文链接。

待处理论文列表：
{batch_prompt}"""

        try:
            response = self.client.chat_completion([{
                "role": "user",
                "content": final_prompt
            }])
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"批处理生成失败: {e}")
            return f"**[本批次生成失败]** 错误信息: {str(e)}"

    def _process_batch(self, papers: List[Dict[str, Any]], start_index: int) -> str:
        print(f"正在批量处理 {len(papers)} 篇论文...")
        summaries = self._generate_batch_summaries(papers, start_index)
        time.sleep(1) 
        return summaries

    def _generate_batch_summary(self, papers: List[Dict[str, Any]]) -> str:
        all_summaries = []
        total_papers = len(papers)
        
        for i in range(0, total_papers, self.max_papers_per_batch):
            batch = papers[i:i + self.max_papers_per_batch]
            print(f"\n正在处理第 {i + 1} 到 {min(i + self.max_papers_per_batch, total_papers)} 篇论文...")
            batch_summary = self._process_batch(batch, i + 1)
            all_summaries.append(batch_summary)
            
        return "\n".join(all_summaries)

    def summarize_papers(self, papers: List[Dict[str, Any]], output_file: str) -> bool:
        try:
            print(f"开始生成论文总结，共 {len(papers)} 篇...")
            summaries = self._generate_batch_summary(papers)
            
            markdown_content = self._generate_markdown(papers, summaries)
            
            # 更改扩展名处理逻辑
            output_md = str(Path(output_file).with_suffix('.md'))
            
            with open(output_md, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Markdown文件已保存：{output_md}")
            
            return True
            
        except Exception as e:
            print(f"严重错误: {e}")
            return False

    def _generate_markdown(self, papers: List[Dict[str, Any]], summaries: str) -> str:
        beijing_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""# LLM 前沿论文周报

> 更新时间：{beijing_time}
> 
> 本周精选了 {len(papers)} 篇关于大模型 (LLM) 的最新论文。

---

{summaries}

---
*Generated by AI Agent based on arXiv (cs.CL)*
"""
def create_summarizer(api_key: str, model: Optional[str] = None):
    return PaperSummarizer(api_key, model)