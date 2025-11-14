"""
Embedding提取模块
支持多个预训练模型的embedding提取
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np


class EmbeddingExtractor:
    """Embedding提取器，支持多种模型"""
    
    def __init__(self, model_name: str, device: str = None):
        """
        初始化模型和tokenizer
        
        Args:
            model_name: Hugging Face模型名称
            device: 设备 ('cuda' 或 'mps' 或 'cpu')
        """
        self.model_name = model_name
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 对于E5和BGE模型，需要特殊处理
        self.is_e5 = 'e5' in model_name.lower()
        self.is_bge = 'bge' in model_name.lower()
        
    def _get_embedding(self, outputs, attention_mask):
        """从模型输出中提取embedding"""
        # 对于E5和BGE，使用last_hidden_state的mean pooling
        if self.is_e5 or self.is_bge:
            last_hidden = outputs.last_hidden_state
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        else:
            # BERT/RoBERTa使用[CLS] token
            embedding = outputs.last_hidden_state[:, 0, :]
        
        # 归一化（对于E5和BGE很重要）
        if self.is_e5 or self.is_bge:
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 8, max_length: int = 512):
        """
        提取文本的embedding
        
        Args:
            texts: 单个文本或文本列表
            batch_size: batch大小
            max_length: 最大序列长度
            
        Returns:
            numpy数组，形状为 (n_texts, hidden_size)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 对于E5模型，需要添加前缀
        if self.is_e5:
            texts = [f"passage: {text}" for text in texts]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # 获取embedding
                outputs = self.model(**encoded)
                embeddings = self._get_embedding(outputs, encoded['attention_mask'])
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_similarity(self, text1: str, text2: str):
        """计算两个文本的余弦相似度"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        # 计算余弦相似度
        dot_product = np.dot(emb1[0], emb2[0])
        norm1 = np.linalg.norm(emb1[0])
        norm2 = np.linalg.norm(emb2[0])
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

