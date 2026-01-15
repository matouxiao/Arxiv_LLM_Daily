"""
ArXiv API 客户端模块
"""
import fitz  # PyMuPDF
import arxiv
import json
import os
import requests
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from config.settings import SEARCH_CONFIG, QUERY
import time

class ArxivClient:
    def __init__(self, config=None):
        time.sleep(3)
        self.client = arxiv.Client()
        self.config = config or SEARCH_CONFIG
        # 已处理论文记录文件路径（现在只记录最新的一篇）
        from config.settings import OUTPUT_DIR, LAST_RUN_FILE
        self.processed_papers_file = Path(OUTPUT_DIR) / "processed_papers.json"

    def _load_latest_processed_paper_id(self) -> Optional[str]:
        """加载上次处理的最新论文ID（只记录一篇）"""
        if self.processed_papers_file.exists():
            try:
                with open(self.processed_papers_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 兼容旧格式
                    if isinstance(data, dict):
                        # 新格式：只记录最新的一篇论文ID
                        if 'latest_paper_id' in data:
                            return data['latest_paper_id']
                        # 兼容旧格式：如果有paper_ids列表，取第一个（最新的）
                        elif 'paper_ids' in data and isinstance(data['paper_ids'], list) and len(data['paper_ids']) > 0:
                            return data['paper_ids'][0]
                    elif isinstance(data, list) and len(data) > 0:
                        # 旧格式：列表，取第一个
                        return data[0]
                    return None
            except (json.JSONDecodeError, Exception) as e:
                print(f"加载已处理论文记录失败: {e}，将创建新记录")
                return None
        return None
    
    def _save_latest_paper_id(self, paper_id: str):
        """保存最新处理的论文ID（只保存一篇）"""
        try:
            # 确保目录存在
            self.processed_papers_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为JSON格式（只保存最新的一篇）
            data = {
                'latest_paper_id': paper_id,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.processed_papers_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存最新论文ID: {paper_id}")
        except Exception as e:
            print(f"保存已处理论文记录失败: {e}")
    
    def _extract_paper_id(self, entry_id: str) -> str:
        """从entry_id中提取论文ID（去除版本号）"""
        # entry_id格式通常是: http://arxiv.org/abs/2601.00794v1
        # 或者: arxiv:2601.00794v1
        # 提取ID部分（去除版本号v1等）
        if 'arxiv.org/abs/' in entry_id:
            paper_id = entry_id.split('arxiv.org/abs/')[-1]
        elif entry_id.startswith('arxiv:'):
            paper_id = entry_id.replace('arxiv:', '')
        else:
            paper_id = entry_id
        
        # 去除版本号（如v1, v2等）
        if 'v' in paper_id:
            paper_id = paper_id.rsplit('v', 1)[0]
        
        return paper_id
    
    def _safe_get_categories(self, paper: arxiv.Result) -> List[str]:
        """安全地获取论文分类"""
        try:
            if isinstance(paper.categories, (list, tuple, set)):
                return list(paper.categories)
            elif isinstance(paper.categories, str):
                return [paper.categories]
            else:
                return [str(paper.categories)]
        except Exception as e:
            print(f"调试 - 获取分类出错: {e}")
            return [paper.primary_category] if paper.primary_category else []

    def download_and_extract_full_pdf_text(self, pdf_url: str, max_pages: int = None) -> Optional[str]:
        """
        下载 PDF 并提取全部文本内容
        
        Args:
            pdf_url: PDF 下载链接
            max_pages: 最大页数限制（None表示提取所有页）
        
        Returns:
            完整的 PDF 文本内容，失败返回 None
        """
        try:
            print(f"正在下载 PDF: {pdf_url[:60]}...")
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                print(f"PDF 下载失败: HTTP {response.status_code}")
                return None
            
            # 读取 PDF 内容
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            
            # 提取所有页面的文本
            all_text = ""
            total_pages = len(pdf_document) if max_pages is None else min(len(pdf_document), max_pages)
            
            print(f"正在提取 PDF 文本，共 {total_pages} 页...")
            
            for page_num in range(total_pages):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    all_text += f"\n{page_text}\n"
            
            pdf_document.close()
            
            if not all_text.strip():
                print("PDF 文本提取失败：未找到文本内容")
                return None
            
            print(f"PDF 文本提取成功：共 {len(all_text)} 字符")
            return all_text
            
        except requests.Timeout:
            print(f"PDF 下载超时: {pdf_url[:60]}")
            return None
        except Exception as e:
            print(f"PDF 文本提取失败: {e}")
            return None
    
    def download_and_extract_pdf_text(self, pdf_url: str, max_pages: int = 20) -> Optional[str]:
        """
        下载 PDF 并提取特定章节（Introduction, Related Work, Method, Conclusion）
        
        Args:
            pdf_url: PDF 下载链接
            max_pages: 最大扫描页数（用于查找章节，默认20页）
        
        Returns:
            提取的章节内容（不包含摘要），失败返回 None
        """
        try:
            print(f"正在下载 PDF: {pdf_url[:60]}...")
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                print(f"PDF 下载失败: HTTP {response.status_code}")
                return None
            
            # 读取 PDF 内容（直接使用 bytes，不需要 BytesIO）
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            
            # 提取所有页面的文本（用于查找章节）
            all_text = ""
            total_pages = min(len(pdf_document), max_pages)
            
            for page_num in range(total_pages):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    all_text += f"\n{page_text}\n"
            
            pdf_document.close()
            
            if not all_text.strip():
                print("PDF 文本提取失败：未找到文本内容")
                return None
            
            # 提取特定章节
            sections = self._extract_key_sections(all_text)
            
            if sections:
                print(f"PDF 章节提取成功：找到关键章节")
                return sections
            else:
                print("未找到目标章节，返回前20000字符作为摘要")
                # 如果找不到章节，返回前几页作为摘要
                return all_text[:20000] if len(all_text) > 20000 else all_text
            
        except requests.Timeout:
            print(f"PDF 下载超时: {pdf_url[:60]}")
            return None
        except Exception as e:
            print(f"PDF 文本提取失败: {e}")
            return None

    def _extract_key_sections(self, text: str) -> Optional[str]:
        """
        从论文文本中提取 Introduction、Related Work、Method 和 Conclusion 章节
        
        Args:
            text: 完整的论文文本
        
        Returns:
            提取的章节内容
        """
        # 定义章节标题的正则表达式模式
        # 匹配各种格式：1. Introduction, Introduction, 1 Introduction, etc.
        section_patterns = {
            'introduction': [
                r'(?i)^\s*(?:1\.?\s*)?introduction\s*$',
                r'(?i)^\s*(?:1\.?\s*)?intro\s*$',
            ],
            'related_work': [
                r'(?i)^\s*(?:2\.?\s*)?related\s+work\s*$',
                r'(?i)^\s*(?:2\.?\s*)?related\s+works\s*$',
                r'(?i)^\s*(?:2\.?\s*)?related\s+literature\s*$',
                r'(?i)^\s*(?:2\.?\s*)?background\s*$',
            ],
            'method': [
                r'(?i)^\s*(?:3\.?\s*)?method\s*$',
                r'(?i)^\s*(?:3\.?\s*)?methods\s*$',
                r'(?i)^\s*(?:3\.?\s*)?methodology\s*$',
                r'(?i)^\s*(?:3\.?\s*)?approach\s*$',
                r'(?i)^\s*(?:3\.?\s*)?proposed\s+method\s*$',
                r'(?i)^\s*(?:3\.?\s*)?our\s+method\s*$',
                r'(?i)^\s*(?:3\.?\s*)?methodology\s+and\s+approach\s*$',
            ],
            'conclusion': [
                r'(?i)^\s*(?:[0-9]+\.?\s*)?conclusion\s*$',
                r'(?i)^\s*(?:[0-9]+\.?\s*)?conclusions\s*$',
                r'(?i)^\s*(?:[0-9]+\.?\s*)?summary\s*$',
                r'(?i)^\s*(?:[0-9]+\.?\s*)?discussion\s+and\s+conclusion\s*$',
            ]
        }
        
        # 按行分割文本
        lines = text.split('\n')
        sections = {}
        current_section = None
        current_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检查是否是章节标题（检查当前行和下一行，因为标题可能跨行）
            found_section = None
            for section_name, patterns in section_patterns.items():
                # 检查当前行
                for pattern in patterns:
                    if re.match(pattern, line):
                        found_section = section_name
                        break
                
                # 如果当前行没匹配，检查下一行（处理标题换行的情况）
                if not found_section and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    for pattern in patterns:
                        if re.match(pattern, next_line):
                            found_section = section_name
                            i += 1  # 跳过下一行，因为它是标题
                            line = next_line
                            break
                
                if found_section:
                    break
            
            # 如果找到新章节
            if found_section:
                # 保存之前的章节
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # 开始新章节
                current_section = found_section
                current_content = [line]  # 包含标题
            # 如果当前在某个章节中，添加内容
            elif current_section:
                # 检查是否遇到下一个主要章节（数字开头的章节，但不是我们要的章节）
                if re.match(r'^\s*[0-9]+\.?\s+[A-Z]', line):
                    # 检查是否是我们要的章节
                    is_target_section = False
                    for section_name, patterns in section_patterns.items():
                        for pattern in patterns:
                            if re.match(pattern, line):
                                is_target_section = True
                                break
                        if is_target_section:
                            break
                    
                    if not is_target_section and current_section != 'conclusion':
                        # 不是目标章节，结束当前章节
                        if current_content:
                            sections[current_section] = '\n'.join(current_content)
                        current_section = None
                        current_content = []
                    else:
                        # 是目标章节，继续添加
                        current_content.append(lines[i])
                else:
                    # 普通内容，继续添加
                    current_content.append(lines[i])
            
            i += 1
        
        # 保存最后一个章节
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # 组合提取的章节
        if sections:
            result_parts = []
            section_names = {
                'introduction': 'Introduction', 
                'related_work': 'Related Work',
                'method': 'Method',
                'conclusion': 'Conclusion'
            }
            
            # 按顺序提取章节
            for key in ['introduction', 'related_work', 'method', 'conclusion']:
                if key in sections:
                    # 限制每个章节的长度（避免过长）
                    section_text = sections[key]
                    if len(section_text) > 8000:
                        section_text = section_text[:8000] + "\n\n[注：章节内容已截断]"
                    result_parts.append(f"=== {section_names[key]} ===\n{section_text}")
            
            return '\n\n'.join(result_parts)
        
        return None

    def _create_search_query(self, query: str = "", 
                           categories: Optional[List[str]] = None,
                           keywords: Optional[Dict[str, List[str]]] = None) -> str:
        """构建高级搜索查询"""
        search_parts = []
        
        # 添加日期范围限制（只查询最近N天的论文）
        days_back = self.config.get('days_back', 1)
        if days_back > 0:
            # 计算日期范围（UTC时间，因为ArXiv使用UTC）
            today = datetime.utcnow().date()
            start_date = (today - timedelta(days=days_back)).strftime('%Y%m%d')
            end_date = today.strftime('%Y%m%d')
            # ArXiv日期查询格式：submittedDate:[YYYYMMDD TO YYYYMMDD]
            search_parts.append(f"submittedDate:[{start_date} TO {end_date}]")
            print(f"日期范围限制：{start_date} 到 {end_date}（最近{days_back}天）")
        
        # 添加基本查询
        if query:
            if self.config['title_only']:
                search_parts.append(f"ti:{query}")
            elif self.config['abstract_only']:
                search_parts.append(f"abs:{query}")
            elif self.config['author_only']:
                search_parts.append(f"au:{query}")
            else:
                search_parts.append(query)

        # 添加分类（使用 OR 连接所有分类）
        if categories:
            try:
                cat_parts = []
                for cat in categories:
                    if not cat:
                        continue
                    if self.config['include_cross_listed']:
                        cat_parts.append(f"cat:{cat}")
                    else:
                        cat_parts.append(f"primary_cat:{cat}")
                
                if cat_parts:
                    cat_query = " OR ".join(cat_parts)
                    search_parts.append(f"({cat_query})")
            except Exception as e:
                print(f"调试 - 构建分类查询出错: {e}")

        final_query = " AND ".join(search_parts) if search_parts else "*:*"
        return final_query

    def search_papers(self, 
                     categories: Optional[List[str]] = None,
                     query: str = QUERY) -> List[Dict[str, Any]]:
        """
        搜索论文并返回元数据，支持多个分类的查询
        按日期正序排列（最老在前），从上次记录的论文之后开始处理
        
        Args:
            categories: arXiv分类列表
            query: 搜索关键词
        """
        all_results = []
        
        # 构建查询
        search_query = self._create_search_query(query, categories)
        print(f"使用查询: {search_query}")
        
        # 加载上次处理的最后论文ID（最新的那篇）
        last_processed_id = self._load_latest_processed_paper_id()
        if last_processed_id:
            print(f"上次处理的最后论文ID: {last_processed_id}")
        
        # 设置排序标准（按日期正序，最老的在前）
        sort_criterion = getattr(arxiv.SortCriterion, self.config['sort_by'])
        sort_order = getattr(arxiv.SortOrder, self.config['sort_order'])
        
        try:
            # === 修改：使用较大的数字来遍历所有论文 ===
            # 这个数字只用于搜索，不限制实际处理的数量
            search_max_results = 288  # 足够大以覆盖 days_back 范围内的所有论文
            # ==========================================
            
            # 创建搜索参数字典
            search_kwargs = {
                'query': search_query,
                'max_results': search_max_results,  # 用于搜索的最大数量
                'sort_by': sort_criterion,
                'sort_order': sort_order
            }
            
            # 只在 id_list 不为 None 时添加到参数中
            if self.config['id_list'] is not None:
                search_kwargs['id_list'] = self.config['id_list']
            
            search = arxiv.Search(**search_kwargs)
            
            # 统计信息
            total_found = 0
            found_last_processed = False  # 标志位：是否已经遇到上次处理的最后论文
            processed_count = 0  # 实际处理的论文数量（用于限制）
            
            # 如果没有上次记录，说明是第一次运行
            if not last_processed_id or (last_processed_id and not found_last_processed):
                found_last_processed = True
                print("首次运行，从头开始处理论文")

            for paper in self.client.results(search):
                try:
                    total_found += 1
                    
                    # 提取论文ID
                    paper_id = self._extract_paper_id(paper.entry_id)
                    
                    # 如果还没遇到上次处理的最后论文，继续查找（但不处理）
                    if not found_last_processed:
                        if paper_id == last_processed_id:
                            print(f"找到上次处理的最后论文 (ID: {paper_id})，从下一篇开始处理")
                            found_last_processed = True
                        continue  # 跳过这篇论文，继续查找
                    
                    # === 已经找到上次记录（或首次运行），开始处理新论文 ===
                    
                    # 检查是否达到处理数量限制
                    if processed_count >= self.config['max_total_results']:
                        print(f"已处理 {processed_count} 篇新论文，达到配置的最大数量 ({self.config['max_total_results']})，停止处理")
                        break
                    
                    # 下载并提取 PDF 关键章节（摘要、Introduction、Related Work、Method、Conclusion）
                    full_text = None
                    if paper.pdf_url:
                        # 提取关键章节：Introduction、Related Work、Method、Conclusion
                        pdf_sections = self.download_and_extract_pdf_text(paper.pdf_url, max_pages=20)
                        
                        # 组合摘要和PDF章节
                        if pdf_sections:
                            # 将 arXiv 摘要放在最前面
                            abstract = paper.summary if paper.summary else ""
                            full_text = f"=== Abstract ===\n{abstract}\n\n{pdf_sections}"
                        else:
                            # 如果PDF提取失败，至少使用摘要
                            full_text = paper.summary if paper.summary else None

                    metadata = {
                        'title': paper.title,
                        'authors': [author.name for author in paper.authors],
                        'published': paper.published.isoformat(),
                        'updated': paper.updated.isoformat(),
                        'summary': paper.summary,
                        'full_text': full_text,  # 完整的 PDF 文本
                        'doi': paper.doi,
                        'primary_category': paper.primary_category,
                        'categories': self._safe_get_categories(paper),
                        'links': [link.href for link in paper.links],
                        'pdf_url': paper.pdf_url,
                        'entry_id': paper.entry_id,
                        'paper_id': paper_id,  # 添加标准化的论文ID
                        'comment': getattr(paper, 'comment', '')
                    }
                    all_results.append(metadata)
                    processed_count += 1  # 增加处理计数
                    
                except Exception as e:
                    print(f"处理单篇文章时出错: {e}")
                    continue
           
        except Exception as e:
            print(f"搜索过程出错: {e}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")

        if not all_results:
            print(f"搜索了 {total_found} 篇论文，但没有新论文需要处理")
        else:
            print(f"找到 {len(all_results)} 篇新论文（共搜索了 {total_found} 篇）")

        return all_results

    def save_results(self, results: List[Dict[str, Any]], output_dir: str, metadata_file: str = None):
        """
        保存搜索结果到 Markdown 文件
        """
        import os
        from datetime import datetime
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # === 关键修改：文件名必须包含 summary_ 前缀 ===
        # 这样 site_manager.py 才能识别到它
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f"summary_{timestamp}.md")
        # ==========================================
        
        print(f"正在保存结果到: {filename}")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # 写入标题
                display_date = datetime.now().strftime('%Y-%m-%d')
                f.write(f"# Arxiv LLM Daily - {display_date}\n\n")
                f.write(f"Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 写入每一篇论文
                for i, paper in enumerate(results, 1):
                    title = paper.get('title', 'No Title')
                    links = paper.get('links', [])
                    url = links[0] if links else paper.get('pdf_url', '#')
                    
                    authors_list = paper.get('authors', [])
                    authors = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)
                    
                    summary = paper.get('summary', 'No summary available.').replace('\n', ' ')
                    
                    f.write(f"## {i}. {title}\n")
                    f.write(f"- **Authors**: {authors}\n")
                    f.write(f"- **Link**: {url}\n")
                    f.write(f"- **Summary**: {summary}\n\n")
                    f.write("---\n\n")
                    
            print(f"成功保存 {len(results)} 篇论文到 {filename}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")