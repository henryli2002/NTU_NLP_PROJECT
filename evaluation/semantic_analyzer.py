"""
语义一致性分析模块
分析近义句和反义句的embedding聚类和相似度
"""
import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embedding_extractor import EmbeddingExtractor


class SemanticAnalyzer:
    """语义一致性分析器"""
    
    def __init__(self, model_name: str, device: str = None, batch_size: int = 8):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称
            device: 设备
            batch_size: batch大小
        """
        self.model_name = model_name
        self.extractor = EmbeddingExtractor(model_name, device)
        self.batch_size = batch_size
        
        # 定义测试用的近义句和反义句对
        self.synonym_pairs = [
            ("I love this movie", "I really enjoy this film"),
            ("The weather is nice today", "It's a beautiful day outside"),
            ("She is very happy", "She feels joyful"),
            ("This is a difficult problem", "This is a challenging issue"),
            ("The car is fast", "The vehicle moves quickly"),
            ("He is very smart", "He is highly intelligent"),
            ("This house is big", "This residence is enormous"),
            ("I'm feeling tired", "I am exhausted"),
            ("Please be quiet", "Please keep the noise down"),
            ("The book is interesting", "The novel is captivating"),
            ("She bought a new computer", "She purchased a new laptop"),
            ("Let's start the meeting", "Let's begin the discussion"),
            ("It's cold outside", "The temperature is freezing"),
            ("The food is delicious", "This meal tastes wonderful"),
            ("He is a famous writer", "He is a well-known author"),
            ("They are good friends", "They have a strong friendship"),
            ("I need assistance", "I require help"),
            ("This is a bad idea", "This is a poor suggestion"),
            ("The project is complete", "The task is finished"),
            ("He looks angry", "He appears furious"),
            ("The cat is sleeping", "The feline is resting"),
            ("The situation is complex", "The matter is complicated"),
            ("I don't understand", "I fail to grasp this"),
            ("She spoke clearly", "She articulated her points well"),
            ("The baby is cute", "The infant is adorable"),
            ("He ran quickly", "He sprinted to the finish line"),
            ("The water is deep", "The ocean is profound"),
            ("This is correct", "This is accurate"),
            ("I am certain", "I am positive about it"),
            ("They arrived late", "They got here after the scheduled time"),
            ("The movie was funny", "The film was humorous"),
            ("He is wealthy", "He is affluent"),
            ("This is very important", "This is extremely significant"),
            ("The test was easy", "The exam was simple"),
            ("I feel sick", "I am unwell"),
            ("He is a brave man", "He is a courageous person"),
            ("This is a rare opportunity", "This is an uncommon chance"),
            ("The building is tall", "The skyscraper is high"),
            ("I like your shoes", "I admire your footwear"),
            ("He is an old man", "He is an elderly gentleman"),
            ("The story is true", "The narrative is factual"),
            ("She is worried", "She feels anxious"),
            ("The noise is loud", "The sound is deafening"),
            ("He is a kind person", "He is considerate"),
            ("I will return soon", "I'll be back shortly"),
            ("They are working hard", "They are putting in a lot of effort"),
            ("The room is clean", "The space is tidy"),
            ("It's a small town", "It's a tiny village"),
            ("He answered the question", "He responded to the query"),
            ("I trust him", "I have confidence in him"),
            ("This is a new record", "This is an unbroken milestone"),
            ("The result is good", "The outcome is positive"),
            ("The company is growing", "The business is expanding"),
        ]
        
        self.antonym_pairs = [
            ("I love this movie", "I hate this film"),
            ("The weather is nice today", "The weather is terrible today"),
            ("She is very happy", "She is very sad"),
            ("This is a difficult problem", "This is an easy task"),
            ("The car is fast", "The car is slow"),
            ("He is very smart", "He is quite foolish"),
            ("This house is big", "This residence is tiny"),
            ("I'm feeling tired", "I am full of energy"),
            ("Please be quiet", "Please be loud"),
            ("The book is interesting", "The novel is boring"),
            ("She bought a new computer", "She sold her old computer"),
            ("Let's start the meeting", "Let's end the meeting"),
            ("It's cold outside", "It's hot in here"),
            ("The food is delicious", "This meal tastes awful"),
            ("He is a famous writer", "He is an unknown author"),
            ("They are good friends", "They are bitter enemies"),
            ("I need assistance", "I can manage it myself"),
            ("This is a good idea", "This is a terrible plan"),
            ("The project is complete", "The task is just beginning"),
            ("He looks calm", "He appears furious"),
            ("The cat is sleeping", "The feline is wide awake"),
            ("The situation is simple", "The matter is complicated"),
            ("I understand completely", "I am totally confused"),
            ("She spoke clearly", "She mumbled her words"),
            ("The baby is cute", "The infant is unattractive"),
            ("He ran quickly", "He walked slowly"),
            ("The water is deep", "The puddle is shallow"),
            ("This is correct", "This is incorrect"),
            ("I am certain", "I am doubtful about it"),
            ("They arrived late", "They arrived early"),
            ("The movie was funny", "The film was serious"),
            ("He is wealthy", "He is poor"),
            ("This is very important", "This is completely trivial"),
            ("The test was easy", "The exam was extremely difficult"),
            ("I feel sick", "I feel healthy"),
            ("He is a brave man", "He is a cowardly person"),
            ("This is a rare opportunity", "This is a common occurrence"),
            ("The building is tall", "The bungalow is short"),
            ("I like your shoes", "I dislike your footwear"),
            ("He is an old man", "He is a young boy"),
            ("The story is true", "The narrative is false"),
            ("She is worried", "She feels relieved"),
            ("The noise is loud", "The sound is barely audible"),
            ("He is a kind person", "He is a cruel individual"),
            ("I will return soon", "I will be gone for a long time"),
            ("They are working hard", "They are hardly working"),
            ("The room is clean", "The space is filthy"),
            ("It's a small town", "It's a massive city"),
            ("He answered the question", "He ignored the query"),
            ("I trust him", "I am suspicious of him"),
            ("This is a new record", "This matches the old record"),
            ("The result is good", "The outcome is negative"),
            ("The company is growing", "The business is shrinking"),
        ]
    
    def analyze_synonym_antonym(self) -> Dict[str, Dict]:
        """
        分析近义句和反义句的相似度
        
        Returns:
            包含相似度统计的字典
        """
        print(f"Analyzing semantic consistency for {self.model_name}...")
        
        # 提取所有文本的embedding
        all_texts = []
        for pair in self.synonym_pairs + self.antonym_pairs:
            all_texts.extend(pair)
        
        embeddings = self.extractor.encode(all_texts, batch_size=self.batch_size)
        
        # 分离近义句和反义句的embedding
        n_pairs = len(self.synonym_pairs)
        synonym_emb1 = embeddings[:n_pairs]
        synonym_emb2 = embeddings[n_pairs:2*n_pairs]
        antonym_emb1 = embeddings[2*n_pairs:3*n_pairs]
        antonym_emb2 = embeddings[3*n_pairs:]
        
        # 计算相似度
        synonym_similarities = []
        for i in range(n_pairs):
            sim = cosine_similarity([synonym_emb1[i]], [synonym_emb2[i]])[0][0]
            synonym_similarities.append(float(sim))
        
        antonym_similarities = []
        for i in range(n_pairs):
            sim = cosine_similarity([antonym_emb1[i]], [antonym_emb2[i]])[0][0]
            antonym_similarities.append(float(sim))
        
        return {
            'synonym_pairs': {
                'pairs': self.synonym_pairs,
                'similarities': synonym_similarities,
                'mean_similarity': float(np.mean(synonym_similarities)),
                'std_similarity': float(np.std(synonym_similarities))
            },
            'antonym_pairs': {
                'pairs': self.antonym_pairs,
                'similarities': antonym_similarities,
                'mean_similarity': float(np.mean(antonym_similarities)),
                'std_similarity': float(np.std(antonym_similarities))
            },
            'semantic_gap': float(np.mean(synonym_similarities) - np.mean(antonym_similarities))
        }
    
    def analyze_clustering(self, texts: List[str], n_clusters: int = 3) -> Dict:
        """
        分析文本的聚类情况
        
        Args:
            texts: 文本列表
            n_clusters: 聚类数量
            
        Returns:
            聚类结果
        """
        print(f"Performing clustering analysis for {self.model_name}...")
        
        embeddings = self.extractor.encode(texts, batch_size=self.batch_size)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 计算聚类内相似度
        intra_cluster_similarities = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_embeddings = embeddings[cluster_mask]
            if len(cluster_embeddings) > 1:
                sim_matrix = cosine_similarity(cluster_embeddings)
                # 取上三角（不包括对角线）
                triu_indices = np.triu_indices(len(cluster_embeddings), k=1)
                intra_cluster_similarities.append(float(np.mean(sim_matrix[triu_indices])))
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'intra_cluster_similarity': float(np.mean(intra_cluster_similarities)) if intra_cluster_similarities else 0.0,
            'n_clusters': n_clusters
        }
    
    def get_all_analysis(self, custom_texts: List[str] = None) -> Dict:
        """
        获取所有分析结果
        
        Args:
            custom_texts: 自定义文本列表（用于聚类分析）
            
        Returns:
            所有分析结果的字典
        """
        results = {}
        
        # 近义句和反义句分析
        results['synonym_antonym'] = self.analyze_synonym_antonym()
        
        # 聚类分析（如果提供了自定义文本）
        if custom_texts:
            results['clustering'] = self.analyze_clustering(custom_texts)
        
        return results

