"""
Embedding缓存模块
支持保存和加载embedding，避免重复计算
"""
import os
import pickle
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class EmbeddingCache:
    """Embedding缓存管理器"""
    
    def __init__(self, cache_dir: str = "results/embeddings"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _get_cache_key(self, model_name: str, texts: List[str], dataset_name: str = None) -> str:
        """
        生成缓存键
        
        Args:
            model_name: 模型名称
            texts: 文本列表
            dataset_name: 数据集名称（可选）
            
        Returns:
            缓存键（文件名）
        """
        # 创建文本内容的哈希
        text_hash = hashlib.md5('|||'.join(sorted(texts)).encode()).hexdigest()[:16]
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:16]
        
        if dataset_name:
            cache_key = f"{dataset_name}_{model_hash}_{text_hash}.pkl"
        else:
            cache_key = f"custom_{model_hash}_{text_hash}.pkl"
        
        return cache_key
    
    def save_embeddings(self, model_name: str, texts: List[str], embeddings: np.ndarray,
                       dataset_name: str = None, metadata: Dict = None):
        """
        保存embedding到缓存
        
        Args:
            model_name: 模型名称
            texts: 文本列表
            embeddings: embedding数组
            dataset_name: 数据集名称（可选）
            metadata: 额外的元数据（可选）
        """
        cache_key = self._get_cache_key(model_name, texts, dataset_name)
        cache_path = self.cache_dir / cache_key
        
        # 保存embedding和文本
        cache_data = {
            'model_name': model_name,
            'texts': texts,
            'embeddings': embeddings,
            'dataset_name': dataset_name,
            'metadata': metadata or {}
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # 更新元数据
        self.metadata[cache_key] = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'n_texts': len(texts),
            'embedding_shape': list(embeddings.shape),
            'metadata': metadata or {}
        }
        self._save_metadata()
        
        print(f"Saved embeddings to cache: {cache_key}")
        return cache_key
    
    def load_embeddings(self, model_name: str, texts: List[str], 
                       dataset_name: str = None) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        从缓存加载embedding
        
        Args:
            model_name: 模型名称
            texts: 文本列表
            dataset_name: 数据集名称（可选）
            
        Returns:
            (embeddings, metadata) 或 None
        """
        cache_key = self._get_cache_key(model_name, texts, dataset_name)
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # 验证数据匹配
                if (cache_data['model_name'] == model_name and 
                    cache_data['texts'] == texts and
                    (dataset_name is None or cache_data.get('dataset_name') == dataset_name)):
                    
                    print(f"Loaded embeddings from cache: {cache_key}")
                    return cache_data['embeddings'], cache_data.get('metadata', {})
                else:
                    print(f"Cache mismatch for {cache_key}, will recompute")
                    return None
            except Exception as e:
                print(f"Error loading cache {cache_key}: {e}")
                return None
        
        return None
    
    def list_cached_embeddings(self, model_name: str = None, dataset_name: str = None) -> List[Dict]:
        """
        列出缓存的embedding
        
        Args:
            model_name: 过滤模型名称（可选）
            dataset_name: 过滤数据集名称（可选）
            
        Returns:
            缓存项列表
        """
        results = []
        for cache_key, info in self.metadata.items():
            if model_name and info.get('model_name') != model_name:
                continue
            if dataset_name and info.get('dataset_name') != dataset_name:
                continue
            results.append({
                'cache_key': cache_key,
                **info
            })
        return results
    
    def clear_cache(self, model_name: str = None, dataset_name: str = None):
        """
        清除缓存
        
        Args:
            model_name: 只清除指定模型的缓存（可选）
            dataset_name: 只清除指定数据集的缓存（可选）
        """
        to_remove = []
        for cache_key, info in self.metadata.items():
            if model_name and info.get('model_name') != model_name:
                continue
            if dataset_name and info.get('dataset_name') != dataset_name:
                continue
            
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                cache_path.unlink()
            to_remove.append(cache_key)
        
        for key in to_remove:
            del self.metadata[key]
        
        self._save_metadata()
        print(f"Cleared {len(to_remove)} cache entries")

