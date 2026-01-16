"""
可视化模块 - 生成饼图等图表
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def generate_decision_pie_chart(
    paper_data_list: List[Dict[str, Any]], 
    output_path: str,
    title: str = "推荐决策分布"
) -> Optional[str]:
    """
    生成推荐决策分布的饼图
    
    Args:
        paper_data_list: 论文数据列表
        output_path: 输出图片路径
        title: 图表标题
        
    Returns:
        图片文件路径
    """
    # 统计推荐决策分布
    decision_counts = {
        '推荐': 0,
        '边缘可看': 0,
        '不推荐': 0,
        '未评估': 0
    }
    
    for paper in paper_data_list:
        decision = paper.get('decision', '未评估')
        if decision in decision_counts:
            decision_counts[decision] += 1
    
    # 过滤掉数量为0的类别
    filtered_counts = {k: v for k, v in decision_counts.items() if v > 0}
    
    if not filtered_counts or sum(filtered_counts.values()) == 0:
        print("警告：没有有效的决策数据，无法生成饼图")
        return None
    
    # 准备数据
    labels = list(filtered_counts.keys())
    sizes = list(filtered_counts.values())
    colors = ['#4CAF50', '#FF9800', '#F44336', '#9E9E9E']  # 绿色(推荐)、橙色(边缘可看)、红色(不推荐)、灰色(未评估)
    color_map = {
        '推荐': '#4CAF50',
        '边缘可看': '#FF9800',
        '不推荐': '#F44336',
        '未评估': '#9E9E9E'
    }
    chart_colors = [color_map.get(label, '#9E9E9E') for label in labels]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=chart_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加总数说明
    total = sum(sizes)
    fig.text(0.5, 0.02, f'总计: {total} 篇论文', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 饼图已保存到: {output_path}")
    return output_path

def generate_trend_pie_chart(
    paper_data_list: List[Dict[str, Any]],
    labels: np.ndarray,
    output_path: str,
    title: str = "研究热点分布"
) -> tuple:
    """
    基于聚类结果生成研究热点分布的饼图
    
    Args:
        paper_data_list: 论文数据列表
        labels: 聚类标签数组
        output_path: 输出图片路径
        title: 图表标题
        
    Returns:
        tuple: (图片文件路径, 颜色列表(十六进制))，如果失败返回(None, None)
    """
    if len(paper_data_list) != len(labels):
        print(f"警告：论文数量({len(paper_data_list)})与标签数量({len(labels)})不匹配")
        return (None, None)
    
    # 统计每个聚类的论文数量（排除噪声点-1）
    from collections import defaultdict
    cluster_counts = defaultdict(int)
    
    for label in labels:
        if label != -1:  # 排除噪声点
            cluster_counts[int(label)] += 1
    
    if not cluster_counts:
        print("警告：没有有效的聚类数据，无法生成饼图")
        return (None, None)
    
    # 按聚类大小排序
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 准备数据
    cluster_labels = [f"热点 {i+1}" for i, (label, count) in enumerate(sorted_clusters)]
    sizes = [count for _, count in sorted_clusters]
    
    # 使用渐变色
    colors_rgba = plt.cm.Set3(np.linspace(0, 1, len(cluster_labels)))
    
    # 将RGBA颜色转换为十六进制
    def rgba_to_hex(rgba):
        """将matplotlib的RGBA颜色转换为十六进制"""
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    colors_hex = [rgba_to_hex(c) for c in colors_rgba]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=cluster_labels,
        colors=colors_rgba,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加总数说明
    total = sum(sizes)
    fig.text(0.5, 0.02, f'总计: {total} 篇论文，{len(cluster_labels)} 个研究热点', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 趋势饼图已保存到: {output_path}")
    return (output_path, colors_hex)

def generate_keywords_pie_chart(
    paper_data_list: List[Dict[str, Any]],
    output_path: str,
    top_n: int = 8,
    title: str = "关键词分布"
) -> Optional[str]:
    """
    生成关键词分布的饼图（显示前N个最常见的关键词）
    
    Args:
        paper_data_list: 论文数据列表
        output_path: 输出图片路径
        top_n: 显示前N个最常见的关键词
        title: 图表标题
        
    Returns:
        图片文件路径，如果失败返回None
    """
    # 统计关键词出现次数
    from collections import defaultdict
    keyword_counts = defaultdict(int)
    
    for paper in paper_data_list:
        keywords_str = paper.get('keywords', '')
        if keywords_str:
            # 分割关键词（支持中文逗号、英文逗号、顿号等）
            keywords = [k.strip() for k in keywords_str.replace('、', ',').replace('，', ',').split(',') if k.strip()]
            for keyword in keywords:
                keyword_counts[keyword] += 1
    
    if not keyword_counts:
        print("警告：没有找到关键词数据，无法生成饼图")
        return None
    
    # 选择前N个最常见的关键词
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # 准备数据
    labels = [kw for kw, _ in sorted_keywords]
    sizes = [count for _, count in sorted_keywords]
    
    # 使用渐变色
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加总数说明
    total_papers = len(paper_data_list)
    fig.text(0.5, 0.02, f'基于 {total_papers} 篇论文的关键词统计（Top {top_n}）', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 关键词饼图已保存到: {output_path}")
    return output_path
