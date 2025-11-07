"""
下游任务训练模块
支持在SST-2和STS-B任务上微调模型
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr, pearsonr
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SST2Dataset(Dataset):
    """SST-2数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class STSBDataset(Dataset):
    """STS-B数据集（回归任务）"""
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=128):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts1)
    
    def __getitem__(self, idx):
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        label = float(self.labels[idx]) / 5.0  # 归一化到0-1
        
        encoding = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


class TaskTrainer:
    """下游任务训练器"""
    
    def __init__(self, model_name: str, task: str, device: str = None, 
                 output_dir: str = "models/trained"):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型名称
            task: 任务名称 ('sst2' 或 'stsb')
            device: 设备
            output_dir: 模型保存目录
        """
        self.model_name = model_name
        self.task = task.lower()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        if self.task in ('sst2', 'agnews'):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=(4 if self.task == 'agnews' else 2)
            )
        elif self.task == 'stsb':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                problem_type="regression"
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        self.model.to(self.device)
        
        # 模型保存路径
        model_save_name = f"{model_name.replace('/', '_')}_{task}"
        self.model_save_path = self.output_dir / model_save_name
    
    def train(self, train_dataset, val_dataset, 
              num_epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 500,
              logging_steps: int = 100) -> Dict:
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            num_epochs: 训练轮数
            batch_size: batch大小
            learning_rate: 学习率
            warmup_steps: warmup步数
            save_steps: 保存步数
            eval_steps: 评估步数
            logging_steps: 日志步数
            
        Returns:
            训练历史字典
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} on {self.task.upper()}")
        print(f"{'='*60}")
        
        # 准备数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 优化器和调度器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        best_val_score = -1
        best_epoch = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                # reduce logging noise: log only per-epoch below
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_metrics.get('loss', 0))
            history['val_metrics'].append(val_metrics)
            
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_metrics.get('loss', 0):.4f}")
            if 'accuracy' in val_metrics:
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            if 'f1_score' in val_metrics:
                print(f"  Val F1: {val_metrics['f1_score']:.4f}")
            if 'spearman' in val_metrics:
                print(f"  Val Spearman: {val_metrics['spearman']:.4f}")
            
            # 保存最佳模型
            val_score = val_metrics.get('accuracy', val_metrics.get('spearman', 0))
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                self.save_model()
                print(f"  ✓ Saved best model (epoch {epoch + 1})")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best model at epoch {best_epoch + 1} with score {best_val_score:.4f}")
        print(f"Model saved to: {self.model_save_path}")
        print(f"{'='*60}\n")
        
        return history
    
    def evaluate(self, data_loader) -> Dict:
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                if self.task in ('sst2', 'agnews'):
                    # 分类任务：使用 argmax 获取类别索引
                    predictions = torch.argmax(outputs.logits, dim=-1)
                else:  # stsb
                    # 回归任务：直接使用 logits
                    predictions = outputs.logits.squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        metrics = {'loss': avg_loss}
        
        if self.task in ('sst2', 'agnews'):
            # 鲁棒地确保预测和标签都是 1D 整数类别索引
            def _to_class_indices(arr, n_classes: int = None):
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
            
            n_classes = 4 if self.task == 'agnews' else 2
            labels_safe = _to_class_indices(all_labels, n_classes=n_classes)
            preds_safe = _to_class_indices(all_predictions, n_classes=n_classes)
            
            try:
                metrics['accuracy'] = float(accuracy_score(labels_safe, preds_safe))
                metrics['f1_score'] = float(f1_score(labels_safe, preds_safe, average='weighted'))
            except Exception as e:
                print(f"[Training evaluate error for {self.task}] {e}")
                print(f"  labels shape/dtype: {labels_safe.shape}, {labels_safe.dtype}")
                print(f"  preds  shape/dtype: {preds_safe.shape}, {preds_safe.dtype}")
                metrics['accuracy'] = 0.0
                metrics['f1_score'] = 0.0
        else:  # stsb
            # 对于回归任务，计算相关系数
            spearman_corr, _ = spearmanr(all_labels, all_predictions)
            pearson_corr, _ = pearsonr(all_labels, all_predictions)
            metrics['spearman'] = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
            metrics['pearson'] = float(pearson_corr) if not np.isnan(pearson_corr) else 0.0
        
        return metrics
    
    def save_model(self):
        """保存模型"""
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
    
    def load_model(self):
        """加载已训练的模型"""
        if self.model_save_path.exists():
            if self.task == 'sst2':
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_save_path)
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_save_path),
                    num_labels=1,
                    problem_type="regression"
                )
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_save_path))
            return True
        return False


def train_model_on_task(model_name: str, task: str, 
                       max_train_samples: int = None,
                       max_val_samples: int = None,
                       num_epochs: int = 3,
                       batch_size: int = 16,
                       learning_rate: float = 2e-5,
                       device: str = None,
                       val_ratio: float = 0.1) -> Dict:
    """
    在指定任务上训练模型
    
    Args:
        model_name: 基础模型名称
        task: 任务名称 ('sst2' 或 'stsb')
        max_train_samples: 最大训练样本数
        max_val_samples: 最大验证样本数
        num_epochs: 训练轮数
        batch_size: batch大小
        learning_rate: 学习率
        device: 设备
        
    Returns:
        训练历史和评估结果
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {task.upper()}")
    print(f"{'='*60}")
    
    # 加载数据集
    if task == 'sst2':
        print("Loading SST-2 dataset (train/val split, validation as test)...")
        full_train = load_dataset("glue", "sst2", split="train")
        test_dataset_raw = load_dataset("glue", "sst2", split="validation")  # act as test
        split = full_train.train_test_split(test_size=val_ratio, seed=42)
        train_split = split['train']
        val_split = split['test']

        # apply caps if provided
        if max_train_samples not in (None, 0):
            train_split = train_split.select(range(min(max_train_samples, len(train_split))))
        if max_val_samples not in (None, 0):
            val_split = val_split.select(range(min(max_val_samples, len(val_split))))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = SST2Dataset(train_split['sentence'], train_split['label'], tokenizer)
        val_dataset = SST2Dataset(val_split['sentence'], val_split['label'], tokenizer)

        # also prepare test for later evaluation
        test_texts = test_dataset_raw['sentence']
        test_labels = test_dataset_raw['label']
        
    elif task == 'stsb':
        print("Loading STS-B dataset (train/val split, validation as test)...")
        full_train = load_dataset("glue", "stsb", split="train")
        test_dataset_raw = load_dataset("glue", "stsb", split="validation")  # act as test
        split = full_train.train_test_split(test_size=val_ratio, seed=42)
        train_split = split['train']
        val_split = split['test']

        if max_train_samples not in (None, 0):
            train_split = train_split.select(range(min(max_train_samples, len(train_split))))
        if max_val_samples not in (None, 0):
            val_split = val_split.select(range(min(max_val_samples, len(val_split))))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = STSBDataset(train_split['sentence1'], train_split['sentence2'], train_split['label'], tokenizer)
        val_dataset = STSBDataset(val_split['sentence1'], val_split['sentence2'], val_split['label'], tokenizer)

        test_texts1 = test_dataset_raw['sentence1']
        test_texts2 = test_dataset_raw['sentence2']
        test_labels = test_dataset_raw['label']

    elif task == 'agnews':
        print("Loading AG News dataset (train/val split, official test)...")
        full_train = load_dataset("ag_news", split="train")
        test_dataset_raw = load_dataset("ag_news", split="test")
        split = full_train.train_test_split(test_size=val_ratio, seed=42)
        train_split = split['train']
        val_split = split['test']

        if max_train_samples not in (None, 0):
            train_split = train_split.select(range(min(max_train_samples, len(train_split))))
        if max_val_samples not in (None, 0):
            val_split = val_split.select(range(min(max_val_samples, len(val_split))))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = SST2Dataset(train_split['text'], train_split['label'], tokenizer)
        val_dataset = SST2Dataset(val_split['text'], val_split['label'], tokenizer)
        
        test_texts = test_dataset_raw['text']
        test_labels = test_dataset_raw['label']
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # 创建训练器
    trainer = TaskTrainer(model_name, task, device=device)
    
    # 训练
    history = trainer.train(
        train_dataset,
        val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # 最终评估：在验证集和测试集上评估（测试集用于报告）
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)

    # build test loader
    if task == 'stsb':
        test_dataset = STSBDataset(test_texts1, test_texts2, test_labels, trainer.tokenizer)
    else:
        test_dataset = SST2Dataset(test_texts, test_labels, trainer.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_metrics = trainer.evaluate(test_loader)
    
    return {
        'history': history,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_path': str(trainer.model_save_path)
    }

