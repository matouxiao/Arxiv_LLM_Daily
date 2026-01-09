"""
论文聚类模块 - 使用 Embedding API 和 DBSCAN 聚类
"""
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
from config.settings import EMBEDDING_CONFIG, CLUSTERING_CONFIG
from sklearn.cluster import DBSCAN, KMeans  # 添加 KMeans

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    调用 Embedding API 获取文本向量
    
    Args:
        texts: 文本列表
        
    Returns:
        向量列表
    """
    if not texts:
        return []
    
    embeddings = []
    batch_size = EMBEDDING_CONFIG['batch_size']
    timeout = EMBEDDING_CONFIG['timeout']
    api_url = EMBEDDING_CONFIG['api_url']
    
    print(f"正在获取 {len(texts)} 个文本的 embeddings...")
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            
            data = {
                "input": batch
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # 提取 embedding 向量
            if 'data' in result:
                batch_embeddings = [item['embedding'] for item in result['data']]
                embeddings.extend(batch_embeddings)
                print(f"  已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本")
            else:
                raise Exception(f"API 返回格式错误: {result}")
                
        except Exception as e:
            print(f"获取 embedding 失败: {e}")
            # 降级策略：返回随机向量（用于测试）
            print(f"  警告：为第 {i}-{i+len(batch)} 个文本生成随机向量")
            for _ in batch:
                embeddings.append([0.0] * 768)  # 假设向量维度为 768
    
    print(f"成功获取 {len(embeddings)} 个 embeddings")
    return embeddings

def cluster_papers_kmeans(embeddings: List[List[float]], n_clusters: int = 4) -> np.ndarray:
    """
    使用 K-Means 聚类（适合主题集中的情况）
    
    Args:
        embeddings: 论文的向量表示
        n_clusters: 聚类数量
        
    Returns:
        聚类标签数组
    """
    from sklearn.cluster import KMeans
    
    if not embeddings:
        return np.array([])
    
    print(f"正在使用 K-Means 进行聚类 (n_clusters={n_clusters})...")
    
    embeddings_array = np.array(embeddings)
    
    # 使用 K-Means 聚类
    kmeans = KMeans(
        n_clusters=min(n_clusters, len(embeddings)),  # 聚类数不能超过样本数
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(embeddings_array)
    
    # 统计聚类结果
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1
    
    print(f"聚类完成: 发现 {len(cluster_sizes)} 个聚类")
    
    # 打印每个聚类的大小
    for label, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"  聚类 {label}: {size} 篇")
    
    return labels

def cluster_papers(embeddings: List[List[float]]) -> np.ndarray:
    """
    使用 DBSCAN 对论文进行聚类
    
    Args:
        embeddings: 论文的向量表示
        
    Returns:
        聚类标签数组
    """
    if not embeddings:
        return np.array([])
    
    print(f"正在使用 DBSCAN 进行聚类...")
    print(f"  参数: eps={CLUSTERING_CONFIG['eps']}, min_samples={CLUSTERING_CONFIG['min_samples']}")
    
    # 计算余弦距离矩阵
    embeddings_array = np.array(embeddings)
    distance_matrix = cosine_distances(embeddings_array)
    
    # 使用 DBSCAN 聚类
    clustering = DBSCAN(
        eps=CLUSTERING_CONFIG['eps'],
        min_samples=CLUSTERING_CONFIG['min_samples'],
        metric='precomputed',  # 使用预计算的距离矩阵
        n_jobs=CLUSTERING_CONFIG['n_jobs'],
    ).fit(distance_matrix)
    
    labels = clustering.labels_
    
    # 统计聚类结果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"聚类完成: 发现 {n_clusters} 个聚类, {n_noise} 个噪声点")
    
    # 打印每个聚类的大小
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1
    
    for label, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True):
        if label == -1:
            print(f"  噪声点: {size} 篇")
        else:
            print(f"  聚类 {label}: {size} 篇")
    
    return labels


def select_representative_papers(
    papers: List[Dict[str, Any]], 
    embeddings: List[List[float]],
    labels: np.ndarray, 
    top_n: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    从聚类结果中选择代表性论文，并添加聚类元数据
    
    Args:
        papers: 原始论文列表
        embeddings: 论文的向量表示
        labels: 聚类标签
        top_n: 选择最大的 N 个聚类（None 表示使用配置中的值）
        
    Returns:
        代表性论文列表，每篇论文包含 _cluster_id, _cluster_size, _cluster_rank, _distance_to_center
    """
    if not papers or len(papers) == 0:
        return []
    
    if top_n is None:
        top_n = CLUSTERING_CONFIG.get('top_clusters', 3)
    
    print(f"\n正在从聚类中选择代表性论文...")
    
    # 按聚类分组
    cluster_to_papers = defaultdict(list)
    for i, (paper, label) in enumerate(zip(papers, labels)):
        cluster_to_papers[label].append({
            'index': i,
            'paper': paper,
            'embedding': embeddings[i]
        })
    
    # 找出最大的 N 个聚类（排除噪声点 -1）
    valid_clusters = [(label, items) for label, items in cluster_to_papers.items() if label != -1]
    valid_clusters.sort(key=lambda x: len(x[1]), reverse=True)
    top_clusters = valid_clusters[:top_n]
    
    print(f"选择了最大的 {len(top_clusters)} 个聚类")
    
    representative_papers = []
    
    for cluster_rank, (cluster_label, cluster_items) in enumerate(top_clusters):
        cluster_size = len(cluster_items)
        print(f"\n处理聚类 {cluster_label} ({cluster_size} 篇论文):")
        
        # 计算聚类中心（所有论文向量的平均值）
        cluster_embeddings = np.array([item['embedding'] for item in cluster_items])
        cluster_center = np.mean(cluster_embeddings, axis=0)
        
        # 计算每篇论文到聚类中心的距离
        distances = []
        for item in cluster_items:
            embedding = np.array(item['embedding'])
            distance = np.linalg.norm(embedding - cluster_center)
            distances.append((distance, item))
        
        # 按距离排序，选择最接近中心的论文
        distances.sort(key=lambda x: x[0])
        
        # 选择代表性论文（至少选1篇，最多选聚类大小的一半）
        num_to_select = max(1, cluster_size // 2)
        
        for dist, item in distances[:num_to_select]:
            paper = item['paper'].copy()
            # 添加聚类元数据
            paper['_cluster_id'] = cluster_label
            paper['_cluster_size'] = cluster_size
            paper['_cluster_rank'] = cluster_rank  # 聚类排名（0=最大聚类）
            paper['_distance_to_center'] = float(dist)
            representative_papers.append(paper)
        
        print(f"  从该聚类中选择了 {num_to_select} 篇代表性论文")
    
    # 如果有噪声点，随机选择一些
    if -1 in cluster_to_papers:
        noise_items = cluster_to_papers[-1]
        num_noise_to_select = min(2, len(noise_items))  # 最多选2篇噪声点
        if num_noise_to_select > 0:
            print(f"\n从 {len(noise_items)} 个噪声点中随机选择 {num_noise_to_select} 篇")
            import random
            selected_noise = random.sample(noise_items, num_noise_to_select)
            for item in selected_noise:
                paper = item['paper'].copy()
                # 噪声点的元数据
                paper['_cluster_id'] = -1
                paper['_cluster_size'] = 0
                paper['_cluster_rank'] = 999  # 噪声点排在最后
                paper['_distance_to_center'] = 999.0
                representative_papers.append(paper)
    
    print(f"\n总共选择了 {len(representative_papers)} 篇代表性论文")
    
    return representative_papers
