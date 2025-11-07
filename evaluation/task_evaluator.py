"""
下游任务评估模块
评估模型在SST-2和STS-B和任务上的表现
"""
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr, pearsonr
from datasets import load_dataset
from typing import Dict, Tuple, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embedding_extractor import EmbeddingExtractor


class TaskEvaluator:
    """下游任务评估器"""
    
    def __init__(self, model_name: str, device: str = None, batch_size: int = 8, 
                 use_trained_model: bool = False, task: str = None):
        """
        初始化评估器
        
        Args:
            model_name: 模型名称
            device: 设备
            batch_size: batch大小
            use_trained_model: 是否使用训练后的模型
            task: 任务名称（如果使用训练后的模型，需要指定任务）
        """
        self.model_name = model_name
        self.use_trained_model = use_trained_model
        self.task = task
        
        if use_trained_model and task:
            # 使用训练后的模型进行评估
            
            model_path = f"models/trained/{model_name.replace('/', '_')}_{task}"
            if os.path.exists(model_path):
                self.trained_model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.trained_tokenizer = AutoTokenizer.from_pretrained(model_path)
                if device:
                    self.device = device
                else:
                    if torch.cuda.is_available():
                        self.device = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        self.device = 'mps'
                    else:
                        self.device = 'cpu'
                self.trained_model.to(self.device)
                self.trained_model.eval()
                print(f"Using trained model from: {model_path}")
            else:
                print(f"Trained model not found at {model_path}, using base model")
                self.use_trained_model = False
        
        if not self.use_trained_model:
            # 统一设备选择逻辑给 extractor
            if not device:
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            self.extractor = EmbeddingExtractor(model_name, device)
        
        self.batch_size = batch_size
        
    def _to_class_indices(self, arr, n_classes: int = None):
        """Robustly convert predictions/labels to 1D integer class indices."""
        import numpy as np
        a = np.asarray(arr)
        if a.ndim == 2:
            a = np.argmax(a, axis=1)
        if np.issubdtype(a.dtype, np.floating):
            a = np.rint(a).astype(int)
        if a.ndim > 1:
            a = a.reshape(-1)
        if n_classes is not None:
            a = np.clip(a, 0, n_classes - 1)
        return a
        
    def evaluate_sst2(self, max_samples: int = 1000) -> Dict[str, float]:
        """
        评估SST-2情感分类任务
        
        Args:
            max_samples: 最大评估样本数
            
        Returns:
            包含accuracy和f1的字典
        """
        print(f"Loading SST-2 dataset for {self.model_name}...")
        # 对于GLUE任务，公开测试集不含标签，因此将官方validation视为测试集
        dataset = load_dataset("glue", "sst2", split="validation")
        
        # 限制样本数
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        texts = dataset['sentence']
        labels = np.array(dataset['label'])
        
        # 如果使用训练后的模型，直接进行分类
        if self.use_trained_model and hasattr(self, 'trained_model'):
            print("Using trained model for classification...")
            import torch
            from torch.utils.data import DataLoader
            
            # 创建数据集
            from training.trainer import SST2Dataset
            eval_dataset = SST2Dataset(texts, labels, self.trained_tokenizer)
            eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
            
            predictions = []
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.trained_model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_predictions = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(batch_predictions.cpu().numpy())
            
            predictions = np.array(predictions)
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
        else:
            # 使用embedding + 最近邻分类
            print(f"Extracting embeddings for {len(texts)} samples...")
            embeddings = self.extractor.encode(texts, batch_size=self.batch_size)
            
            # 使用简单的最近邻分类：找到训练集中最近的样本
            print("Computing predictions...")
            # 为了简化，我们使用训练集的前1000个样本作为"训练集"
            train_dataset = load_dataset("glue", "sst2", split="train")
            train_texts = train_dataset['sentence']
            train_labels = np.array(train_dataset['label'])
            
            train_embeddings = self.extractor.encode(train_texts, batch_size=self.batch_size)
            
            # 计算余弦相似度并找到最近的训练样本
            predictions = []
            for i in range(0, len(embeddings), self.batch_size):
                batch_emb = embeddings[i:i + self.batch_size]
                # 计算与所有训练样本的相似度
                similarities = np.dot(batch_emb, train_embeddings.T)
                # 找到最相似的训练样本
                nearest_indices = np.argmax(similarities, axis=1)
                batch_predictions = train_labels[nearest_indices]
                predictions.extend(batch_predictions)
            
            predictions = np.array(predictions)
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        }
    
    def evaluate_stsb(self, max_samples: int = 1000) -> Dict[str, float]:
        """
        评估STS-B语义相似度任务
        
        Args:
            max_samples: 最大评估样本数
            
        Returns:
            包含spearman和pearson相关系数的字典
        """
        print(f"Loading STS-B dataset for {self.model_name}...")
        # 对于GLUE任务，公开测试集不含标签，因此将官方validation视为测试集
        dataset = load_dataset("glue", "stsb", split="validation")
        
        # 限制样本数
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        sentences1 = dataset['sentence1']
        sentences2 = dataset['sentence2']
        gold_scores = np.array(dataset['label']) / 5.0  # 归一化到0-1
        
        # 如果使用训练后的模型，直接进行回归预测
        if self.use_trained_model and hasattr(self, 'trained_model'):
            print("Using trained model for regression...")
            import torch
            from torch.utils.data import DataLoader
            
            # 创建数据集
            from training.trainer import STSBDataset
            eval_dataset = STSBDataset(sentences1, sentences2, dataset['label'], self.trained_tokenizer)
            eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
            
            similarities = []
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.trained_model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_similarities = outputs.logits.squeeze().cpu().numpy()
                    similarities.extend(batch_similarities)
            
            similarities = np.array(similarities)
        else:
            # 使用embedding + 余弦相似度
            print(f"Extracting embeddings for {len(sentences1)} pairs...")
            # 提取所有句子的embedding
            all_texts = list(sentences1) + list(sentences2)
            all_embeddings = self.extractor.encode(all_texts, batch_size=self.batch_size)
            
            # 分离两个句子的embedding
            n = len(sentences1)
            emb1 = all_embeddings[:n]
            emb2 = all_embeddings[n:]
            
            print("Computing similarities...")
            # 计算余弦相似度
            similarities = []
            for i in range(0, n, self.batch_size):
                batch_emb1 = emb1[i:i + self.batch_size]
                batch_emb2 = emb2[i:i + self.batch_size]
                # 计算余弦相似度
                dot_products = np.sum(batch_emb1 * batch_emb2, axis=1)
                norms1 = np.linalg.norm(batch_emb1, axis=1)
                norms2 = np.linalg.norm(batch_emb2, axis=1)
                batch_sim = dot_products / (norms1 * norms2 + 1e-8)  # 添加小值避免除零
                similarities.extend(batch_sim)
            
            similarities = np.array(similarities)
        
        # 计算相关系数
        spearman_corr, _ = spearmanr(gold_scores, similarities)
        pearson_corr, _ = pearsonr(gold_scores, similarities)
        
        return {
            'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
            'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0
        }
    
    def evaluate_agnews(self, max_samples: int = 1000) -> Dict[str, float]:
        """
        评估AG News分类任务
        
        Args:
            max_samples: 最大评估样本数
            
        Returns:
            包含accuracy和f1的字典
        """
        print(f"Loading AG News dataset for {self.model_name}...")
        try:
            dataset = load_dataset("ag_news", split="test")
        except:
            # 如果数据集名称不对，尝试其他名称
            dataset = load_dataset("ag_news", split="test")
        
        # 限制样本数
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        texts = dataset['text']
        labels = np.array(dataset['label'])
        
        if self.use_trained_model and hasattr(self, 'trained_model'):
            print("Using trained model for AG News classification...")
            from torch.utils.data import DataLoader
            from training.trainer import SST2Dataset
            eval_dataset = SST2Dataset(texts, labels, self.trained_tokenizer)
            eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

            predictions = []
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.trained_model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_predictions = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(batch_predictions.cpu().numpy())
            predictions = np.array(predictions)
        else:
        print(f"Extracting embeddings for {len(texts)} samples...")
        embeddings = self.extractor.encode(texts, batch_size=self.batch_size)
        
        # 使用简单的最近邻分类
        print("Computing predictions...")
        train_dataset = load_dataset("ag_news", split="train")
        train_texts = train_dataset['text']
        train_labels = np.array(train_dataset['label'])
        
        train_embeddings = self.extractor.encode(train_texts, batch_size=self.batch_size)
        
        # 计算余弦相似度并找到最近的训练样本
        predictions = []
        for i in range(0, len(embeddings), self.batch_size):
            batch_emb = embeddings[i:i + self.batch_size]
            similarities = np.dot(batch_emb, train_embeddings.T)
            nearest_indices = np.argmax(similarities, axis=1)
            batch_predictions = train_labels[nearest_indices]
            predictions.extend(batch_predictions)
        predictions = np.array(predictions)
        
        # Robust metrics computation with shape/type guards
        labels_safe = self._to_class_indices(labels, n_classes=4)
        preds_safe = self._to_class_indices(predictions, n_classes=4)
        try:
            accuracy = accuracy_score(labels_safe, preds_safe)
            f1 = f1_score(labels_safe, preds_safe, average='weighted')
        except Exception as e:
            print(f"[AGNEWS metrics error] {e}")
            print(f"  labels shape/dtype: {labels_safe.shape}, {labels_safe.dtype}")
            print(f"  preds  shape/dtype: {preds_safe.shape}, {preds_safe.dtype}")
            accuracy, f1 = 0.0, 0.0
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        }
    
    def evaluate_all(self, max_samples: int = 1000, include_agnews: bool = True) -> Dict[str, Dict[str, float]]:
        """评估所有任务"""
        results = {}
        
        print(f"\n{'='*50}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*50}")
        
        try:
            print("\nEvaluating SST-2...")
            results['sst2'] = self.evaluate_sst2(max_samples=max_samples)
            print(f"SST-2 Results: {results['sst2']}")
        except Exception as e:
            print(f"Error in SST-2 evaluation: {e}")
            results['sst2'] = {'accuracy': 0.0, 'f1_score': 0.0}
        
        try:
            print("\nEvaluating STS-B...")
            results['stsb'] = self.evaluate_stsb(max_samples=max_samples)
            print(f"STS-B Results: {results['stsb']}")
        except Exception as e:
            print(f"Error in STS-B evaluation: {e}")
            results['stsb'] = {'spearman_correlation': 0.0, 'pearson_correlation': 0.0}
        
        if include_agnews:
            try:
                print("\nEvaluating AG News...")
                results['agnews'] = self.evaluate_agnews(max_samples=max_samples)
                print(f"AG News Results: {results['agnews']}")
            except Exception as e:
                print(f"Error in AG News evaluation: {e}")
                results['agnews'] = {'accuracy': 0.0, 'f1_score': 0.0}
        
        return results

