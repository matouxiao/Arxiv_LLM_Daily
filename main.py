import sys
import os
import datetime
from pathlib import Path

# ---------------- 配置区域 ----------------  
try:
    from config.settings import SEARCH_CONFIG, QUERY, LLM_CONFIG
    # 从 LLM_CONFIG 中获取 API_KEY
    API_KEY = LLM_CONFIG.get('api_key')
except ImportError:
    # 备用默认配置
    SEARCH_CONFIG = {'categories': ['cs.AI'], 'max_total_results': 5}
    QUERY = "LLM"
    API_KEY = None

# === 关键修正：环境变量覆盖 ===
# 这行代码的意思是：如果系统环境变量里有 LLM_API_KEY（GitHub 设置的），就用它的；
# 如果没有，就用上面从 settings 读到的或者 None。
API_KEY = os.getenv("LLM_API_KEY") or API_KEY
# ----------------------------------------

# 修复路径问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.arxiv_client import ArxivClient
    from src.paper_summarizer import PaperSummarizer
except ImportError as e:
    print(f"错误：找不到必要的模块，请检查文件结构。{e}")
    sys.exit(1)

def main():
    print(f"=== Arxiv Daily (中文版) 开始运行 {datetime.datetime.now().strftime('%H:%M:%S')} ===")
    
    # 1. 初始化 Arxiv 客户端
    try:
        arxiv_client = ArxivClient(SEARCH_CONFIG)
    except Exception as e:
        print(f"初始化 ArxivClient 失败: {e}")
        return

    # 2. 初始化 Paper Summarizer（使用 paper_summarizer.py）
    try:
        model = LLM_CONFIG.get('model', 'deepseek-v3')
        paper_summarizer = PaperSummarizer(API_KEY, model)
    except Exception as e:
        print(f"初始化 PaperSummarizer 失败: {e}")
        return

    # 3. 搜索论文
    print("正在搜索最新论文...")
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data"
    
    results = arxiv_client.search_papers(
        categories=SEARCH_CONFIG.get('categories', ['cs.AI']),
        query=QUERY
    )

    # === 新增逻辑：过滤掉下载失败或无内容的论文 ===
    valid_results = []
    for paper in results:
    # 检查是否有 full_text (PDF解析内容) 或 summary (arxiv原始摘要)
    # 并且确保内容不是错误信息
        content = paper.get('full_text') or paper.get('summary')
        if content and "下载失败" not in content and "404" not in content:
            valid_results.append(paper)
        else:
            print(f"⚠️ 跳过论文: {paper.get('title', '未知标题')} (原因: 获取内容失败)")

    results = valid_results # 用过滤后的列表替换原列表

    if not results:
        print("没有发现新论文。")
        print("提示：将使用已有的摘要文件生成网站。")
        
        # 发送无新论文通知
        try:
            from src.mailer import Mailer
            mailer = Mailer()
            mailer.send_no_papers_message()
        except Exception as e:
            print(f"⚠️ 邮件模块调用失败: {e}")
        
        # 即使没有新论文，也继续执行后续步骤（arxivsite 会处理已有文件）
        return

    # 4. 使用 PaperSummarizer 生成摘要（包含关键词、解决痛点、核心创新等详细格式）
    print(f"\n找到 {len(results)} 篇论文，开始生成摘要...")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"summary_{timestamp}.md")
    
    try:
        success = paper_summarizer.summarize_papers(results, output_file)
        if success:
            print(f"✅ 完成！中文日报已保存到: {output_file}")
            
            # === 新增：保存最后处理的论文ID，防止重复推送 ===
            try:
                # 保存最后的一篇论文ID（results是按日期正序的，最后一个是最新的）
                if results:
                    last_paper = results[-1]  # 最后一个（最新的）
                    paper_id = last_paper.get('paper_id')
                    if not paper_id:
                        # 如果没有paper_id，从entry_id提取
                        entry_id = last_paper.get('entry_id', '')
                        if entry_id:
                            paper_id = arxiv_client._extract_paper_id(entry_id)
                    
                    if paper_id:
                        arxiv_client._save_latest_paper_id(paper_id)
                        print(f"✅ 已记录最后处理的论文ID: {paper_id}，避免重复推送")
            except Exception as e:
                print(f"⚠️ 保存已处理论文记录失败: {e}")
            # ===============================================

            # === 新增：发送邮件 ===
            try:
                from src.mailer import Mailer
                mailer = Mailer()
                mailer.send_daily_summary(output_file)
            except Exception as e:
                print(f"⚠️ 邮件模块调用失败: {e}")
            # =====================
            
        else:
            print("❌ 摘要生成失败，请检查错误信息")
    except Exception as e:
        print(f"生成摘要时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()