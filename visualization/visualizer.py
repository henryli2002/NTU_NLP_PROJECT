"""
可视化模块
支持PCA、t-SNE、UMAP降维可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
import os


class EmbeddingVisualizer:
    """Embedding可视化器"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def reduce_dimension(self, embeddings: np.ndarray, method: str = 'pca', n_components: int = 2, **kwargs):
        """
        降维
        
        Args:
            embeddings: embedding矩阵 (n_samples, n_features)
            method: 降维方法 ('pca', 'tsne', 'umap')
            n_components: 目标维度
            **kwargs: 其他参数
            
        Returns:
            降维后的坐标 (n_samples, n_components)
        """
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_.sum()
            return reduced, explained_var
        
        elif method.lower() == 'tsne':
            perplexity = kwargs.get('perplexity', 30)
            n_iter = kwargs.get('n_iter', 1000)
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(embeddings) - 1),
                random_state=42,
                n_iter=n_iter,
                verbose=0
            )
            reduced = reducer.fit_transform(embeddings)
            return reduced, None
        
        elif method.lower() == 'umap':
            if not HAS_UMAP:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")
            n_neighbors = kwargs.get('n_neighbors', 15)
            min_dist = kwargs.get('min_dist', 0.1)
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, len(embeddings) - 1),
                min_dist=min_dist,
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
            return reduced, None
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def plot_embeddings(self, embeddings: np.ndarray, labels: list = None, 
                       method: str = 'pca', title: str = '', 
                       save_path: str = None, **kwargs):
        """
        可视化embedding
        
        Args:
            embeddings: embedding矩阵
            labels: 标签列表（用于着色）
            method: 降维方法
            title: 图标题
            save_path: 保存路径
            **kwargs: 其他参数
        """
        print(f"Reducing dimensions using {method.upper()}...")
        reduced, explained_var = self.reduce_dimension(embeddings, method=method, **kwargs)
        
        plt.figure(figsize=(12, 8))
        
        if labels is not None:
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            label_to_color = dict(zip(unique_labels, colors))
            
            for label in unique_labels:
                mask = np.array(labels) == label
                plt.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    c=[label_to_color[label]],
                    label=str(label),
                    alpha=0.6,
                    s=50
                )
            plt.legend()
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)
        
        plt.title(f"{title} - {method.upper()}" + 
                 (f" (Explained Variance: {explained_var:.2%})" if explained_var else ""))
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_similarity_heatmap(self, texts: list, embeddings: np.ndarray, 
                               title: str = '', save_path: str = None):
        """
        绘制相似度热力图
        
        Args:
            texts: 文本列表
            embeddings: embedding矩阵
            title: 图标题
            save_path: 保存路径
        """
        # 计算相似度矩阵
        similarities = np.dot(embeddings, embeddings.T)
        # 归一化到0-1
        similarities = (similarities + 1) / 2
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarities,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=[t[:20] + '...' if len(t) > 20 else t for t in texts],
            yticklabels=[t[:20] + '...' if len(t) > 20 else t for t in texts],
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        
        plt.close()

