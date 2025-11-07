"""
阶段A：训练/评估阶段
自动下载数据集，加载模型，进行评估，保存结果和缓存
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.embedding_extractor import EmbeddingExtractor
from evaluation.task_evaluator import TaskEvaluator
from evaluation.semantic_analyzer import SemanticAnalyzer
from utils.embedding_cache import EmbeddingCache
from training.trainer import train_model_on_task

# 模型列表
MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5"
]


def download_datasets():
    """自动下载并缓存所有数据集"""
    print("\n" + "="*60)
    print("Downloading and Caching Datasets")
    print("="*60)
    
    from datasets import load_dataset
    
    datasets_to_download = [
        ("glue", "sst2", "validation"),
        ("glue", "sst2", "train"),
        ("glue", "stsb", "validation"),
        ("glue", "stsb", "train"),
        ("ag_news", None, "test"),
        ("ag_news", None, "train"),
    ]
    
    for dataset_info in datasets_to_download:
        dataset_name, subset, split = dataset_info
        try:
            print(f"\nDownloading {dataset_name}" + (f"/{subset}" if subset else "") + f" ({split})...")
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            print(f"  ✓ Downloaded {len(dataset)} samples")
        except Exception as e:
            print(f"  ✗ Error downloading {dataset_name}: {e}")
    
    print("\n" + "="*60)
    print("Dataset Download Complete")
    print("="*60 + "\n")


def run_full_evaluation(max_samples=None, batch_size=8,
                       save_embeddings=True, cache_dir="results/embeddings",
                       train_models=False, train_tasks=None, train_epochs=10,
                       train_batch_size=16, train_lr=2e-5, train_samples=None):
    """
    运行完整评估
    
    Args:
        max_samples: 最大评估样本数
        batch_size: batch大小
        include_agnews: 是否包含AG News评估
        save_embeddings: 是否保存embedding缓存
        cache_dir: 缓存目录
    """
    print("\n" + "="*60)
    print("Stage A: Training/Evaluation Phase")
    print("="*60)
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 初始化缓存
    cache = EmbeddingCache(cache_dir) if save_embeddings else None
    
    # 下载数据集
    download_datasets()
    
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'max_samples': max_samples,
            'batch_size': batch_size,
            'train_models': train_models,
            'train_tasks': train_tasks if train_models else None,
            'models': MODELS
        },
        'results': {},
        'training_history': {}
    }
    
    # 默认训练任务
    if train_tasks is None:
        train_tasks = ['sst2', 'stsb', 'agnews']
    
    # 评估每个模型
    for model_idx, model_name in enumerate(MODELS, 1):
        print(f"\n{'='*60}")
        print(f"Evaluating Model {model_idx}/{len(MODELS)}: {model_name}")
        print(f"{'='*60}")
        
        try:
            # 初始化评估器和分析器
            evaluator = TaskEvaluator(model_name, batch_size=batch_size)
            analyzer = SemanticAnalyzer(model_name, batch_size=batch_size)
            
            # 1. 训练前评估（baseline）
            print("\n[1/4] Evaluating Baseline (Before Training)...")
            baseline_task_results = evaluator.evaluate_all(
                max_samples=max_samples
            )
            
            # 2. 训练模型（如果启用）
            trained_results = {}
            training_info = {}
            if train_models:
                print("\n[2/4] Training Models on Downstream Tasks...")
                for task in train_tasks:
                    if task not in ['sst2', 'stsb', 'agnews']:
                        continue
                    try:
                        print(f"\nTraining on {task.upper()}...")
                        train_result = train_model_on_task(
                            model_name=model_name,
                            task=task,
                            max_train_samples=train_samples,
                            max_val_samples=max_samples,
                            num_epochs=train_epochs,
                            batch_size=train_batch_size,
                            learning_rate=train_lr
                        )
                        training_info[task] = {
                            'history': train_result['history'],
                            'val_metrics': train_result.get('val_metrics', {}),
                            'test_metrics': train_result.get('test_metrics', {}),
                            'model_path': train_result['model_path']
                        }
                        
                        # 使用训练后的模型进行评估
                        print(f"\nEvaluating trained model on {task.upper()}...")
                        trained_evaluator = TaskEvaluator(
                            model_name=model_name,
                            batch_size=batch_size,
                            use_trained_model=True,
                            task=task
                        )
                        
                        if task == 'sst2':
                            trained_results[task] = trained_evaluator.evaluate_sst2(max_samples=max_samples)
                        elif task == 'stsb':
                            trained_results[task] = trained_evaluator.evaluate_stsb(max_samples=max_samples)
                        elif task == 'agnews':
                            trained_results[task] = trained_evaluator.evaluate_agnews(max_samples=max_samples)
                        
                    except Exception as e:
                        print(f"  ✗ Error training on {task}: {e}")
                        import traceback
                        traceback.print_exc()
                        training_info[task] = {'error': str(e)}
            else:
                print("\n[2/4] Skipping Training (train_models=False)")
            
            # 3. 语义一致性分析
            print("\n[3/4] Analyzing Semantic Consistency...")
            semantic_results = analyzer.get_all_analysis()
            
            # 4. 保存embedding缓存（可选）
            if save_embeddings and cache:
                print("\n[4/4] Saving Embedding Cache...")
                try:
                    # 保存语义分析的embedding
                    all_semantic_texts = []
                    for pair in analyzer.synonym_pairs + analyzer.antonym_pairs:
                        all_semantic_texts.extend(pair)
                    
                    semantic_embeddings = evaluator.extractor.encode(
                        all_semantic_texts, 
                        batch_size=batch_size
                    )
                    cache.save_embeddings(
                        model_name=model_name,
                        texts=all_semantic_texts,
                        embeddings=semantic_embeddings,
                        dataset_name="semantic_analysis",
                        metadata={'n_pairs': len(analyzer.synonym_pairs)}
                    )
                except Exception as e:
                    print(f"  Warning: Could not save semantic embeddings: {e}")
            
            # 保存结果
            all_results['results'][model_name] = {
                'baseline': baseline_task_results,
                'trained': trained_results if train_models else {},
                'semantic': semantic_results,
                'status': 'success'
            }
            
            if train_models:
                all_results['training_history'][model_name] = training_info
            
            print(f"\n✓ Successfully evaluated {model_name}")
            
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results['results'][model_name] = {
                'baseline': {},
                'trained': {},
                'semantic': {},
                'status': 'error',
                'error': str(e)
            }
    
    # 保存结果到 metrics.json
    results_path = "results/metrics.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存训练历史
    if train_models and all_results.get('training_history'):
        training_history_path = "results/training_history.json"
        with open(training_history_path, 'w', encoding='utf-8') as f:
            json.dump(all_results['training_history'], f, indent=2, ensure_ascii=False)
        print(f"Training history saved to: {training_history_path}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to: {results_path}")
    if save_embeddings:
        print(f"Embeddings cached to: {cache_dir}")
    if train_models:
        print(f"Trained models saved to: models/trained/")
    print("="*60 + "\n")
    
    # 打印摘要
    print_summary(all_results)
    
    return all_results


def print_summary(results: dict):
    """打印评估摘要"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    train_models = results.get('metadata', {}).get('train_models', False)
    
    for model_name, result in results['results'].items():
        print(f"\n{model_name.split('/')[-1]}:")
        if result.get('status') == 'error':
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
            continue
        
        baseline = result.get('baseline', {})
        trained = result.get('trained', {})
        semantic = result.get('semantic', {}).get('synonym_antonym', {})
        
        # Baseline结果
        print("  [Baseline]")
        if 'sst2' in baseline:
            print(f"    SST-2 Accuracy: {baseline['sst2'].get('accuracy', 0):.4f}")
            print(f"    SST-2 F1:      {baseline['sst2'].get('f1_score', 0):.4f}")
        
        if 'stsb' in baseline:
            print(f"    STS-B Spearman: {baseline['stsb'].get('spearman_correlation', 0):.4f}")
            print(f"    STS-B Pearson:  {baseline['stsb'].get('pearson_correlation', 0):.4f}")
        
        # 训练后结果
        if train_models and trained:
            print("  [After Training]")
            if 'sst2' in trained:
                base_acc = baseline.get('sst2', {}).get('accuracy', 0)
                train_acc = trained['sst2'].get('accuracy', 0)
                improvement = train_acc - base_acc
                print(f"    SST-2 Accuracy: {train_acc:.4f} ({improvement:+.4f})")
                base_f1 = baseline.get('sst2', {}).get('f1_score', 0)
                train_f1 = trained['sst2'].get('f1_score', 0)
                improvement_f1 = train_f1 - base_f1
                print(f"    SST-2 F1:      {train_f1:.4f} ({improvement_f1:+.4f})")
            
            if 'stsb' in trained:
                base_spear = baseline.get('stsb', {}).get('spearman_correlation', 0)
                train_spear = trained['stsb'].get('spearman_correlation', 0)
                improvement = train_spear - base_spear
                print(f"    STS-B Spearman: {train_spear:.4f} ({improvement:+.4f})")
                base_pear = baseline.get('stsb', {}).get('pearson_correlation', 0)
                train_pear = trained['stsb'].get('pearson_correlation', 0)
                improvement_pear = train_pear - base_pear
                print(f"    STS-B Pearson:  {train_pear:.4f} ({improvement_pear:+.4f})")
        
        # 语义一致性
        if 'synonym_pairs' in semantic:
            syn_mean = semantic['synonym_pairs'].get('mean_similarity', 0)
            ant_mean = semantic['antonym_pairs'].get('mean_similarity', 0)
            gap = semantic.get('semantic_gap', 0)
            print(f"  Synonym Similarity: {syn_mean:.4f}")
            print(f"  Antonym Similarity: {ant_mean:.4f}")
            print(f"  Semantic Gap:       {gap:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stage A: Run full model evaluation and save results"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=0, 
        help="Maximum samples for evaluation (0 means full dataset)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--no_cache", 
        action="store_true",
        help="Do not save embedding cache"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="results/embeddings",
        help="Directory for embedding cache"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models on downstream tasks"
    )
    parser.add_argument(
        "--train_tasks",
        type=str,
        nargs='+',
        default=['sst2', 'stsb', 'agnews'],
        help="Tasks to train on (sst2, stsb, agnews)"
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--train_lr",
        type=float,
        default=2e-5,
        help="Training learning rate"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=0,
        help="Number of training samples (0 means full dataset)"
    )
    
    args = parser.parse_args()
    
    results = run_full_evaluation(
        max_samples=(None if args.max_samples == 0 else args.max_samples),
        batch_size=args.batch_size,
        save_embeddings=not args.no_cache,
        cache_dir=args.cache_dir,
        train_models=args.train,
        train_tasks=args.train_tasks,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_samples=(None if args.train_samples == 0 else args.train_samples)
    )
