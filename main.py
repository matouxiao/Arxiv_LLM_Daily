from pickle import NONE
import sys
import os
import datetime
from pathlib import Path
from openai import OpenAI  # 必须安装: pip install openai

# ---------------- 配置区域 ----------------
# 如果你的 config/settings.py 里有 API_KEY，可以保留下面这行导入
# 如果报错，请直接将 API_KEY 填在下面引号里
try:
    from config.settings import SEARCH_CONFIG, QUERY, API_KEY, BASE_URL
except ImportError:
    # 如果导入失败，请在这里手动填入（这是备用方案）
    API_KEY = None  # 你的 DeepSeek/DashScope API Key
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 你的 Base URL
    SEARCH_CONFIG = {'categories': ['cs.AI'], 'max_total_results': 5}
    QUERY = "LLM"

# ----------------------------------------

# 修复路径问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.arxiv_client import ArxivClient
except ImportError:
    print("错误：找不到 src.arxiv_client，请检查文件结构。")
    sys.exit(1)

def get_ai_summary(client_ai, paper):
    """
    调用大模型生成中文总结
    """
    print(f" > 正在请求 AI 翻译: {paper['title'][:30]}...")
    
    prompt = f"""
    请你作为一名专业的学术研究助手。
    请阅读以下这篇计算机论文的元数据，用中文生成一个简洁的总结。
    
    论文标题: {paper['title']}
    论文摘要: {paper['summary']}
    
    要求：
    1. 标题翻译为中文。
    2. 核心内容总结（3-4句话）。
    3. 输出格式必须包含两部分：【中文标题】和【中文总结】。
    """

    try:
        response = client_ai.chat.completions.create(
            model="deepseek-v3",  # 或者 "qwen-plus"，根据你的模型配置
            messages=[
                {"role": "system", "content": "你是一个专业的学术翻译助手。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" ! AI 调用失败: {e}")
        return f"AI 翻译失败，保留原文。\nTitle: {paper['title']}\nSummary: {paper['summary']}"

def main():
    print(f"=== Arxiv Daily (中文版) 开始运行 {datetime.datetime.now().strftime('%H:%M:%S')} ===")
    
    # 1. 初始化 Arxiv 客户端
    try:
        arxiv_client = ArxivClient()
    except Exception as e:
        print(f"初始化 ArxivClient 失败: {e}")
        return

    # 2. 初始化 AI 客户端 (用于翻译)
    # 确保 URL 没有中括号
    clean_base_url = BASE_URL.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    try:
        ai_client = OpenAI(api_key=API_KEY, base_url=clean_base_url)
    except Exception as e:
        print(f"初始化 AI 客户端失败: {e}")
        return

    # 3. 搜索论文
    print("正在搜索最新论文...")
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data"
    last_run_file = base_dir / "data" / "last_run.json"
    
    results = arxiv_client.search_papers(
        categories=SEARCH_CONFIG.get('categories', ['cs.AI']),
        query=QUERY,
        last_run_file=str(last_run_file)
    )

    if not results:
        print("没有发现新论文。")
        return

    # 4. 逐个翻译论文 (核心步骤)
    print(f"\n找到 {len(results)} 篇论文，开始 AI 翻译/总结 (速度取决于网络)...")
    
    processed_results = []
    for paper in results:
        # 调用 AI 获取中文内容
        ai_content = get_ai_summary(ai_client, paper)
        
        # 将翻译后的内容替换原本的英文 summary
        # 为了排版好看，我们把 AI 返回的内容直接放到 summary 字段里
        paper['summary'] = ai_content 
        
        # 将原始标题也更新（可选，如果 AI 返回了标题）
        # 这里我们简单地把 AI 的回复作为最终展示内容
        processed_results.append(paper)

    # 5. 保存结果
    print("\n正在保存中文日报...")
    try:
        arxiv_client.save_results(processed_results, str(output_dir))
        
        # 更新记录
        latest_id = results[0]['entry_id']
        arxiv_client.save_last_run_info(latest_id, str(last_run_file), len(results))
        
        print(f"✅ 完成！中文日报已保存到: {output_dir}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    main()