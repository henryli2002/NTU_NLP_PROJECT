"""
轻量脚本：只更新并缓存指定模型的语义 embeddings（不运行完整评估）

用法示例：
  python3 update_embeddings.py --model "intfloat/e5-base-v2" --batch_size 8 --samples_per_task 100
  python3 update_embeddings.py --rebuild_metadata  # 从现有 .pkl 文件重建 metadata.json

功能：
  - 从 SemanticAnalyzer 加载预定义的近义/反义对（默认 53 条）
  - 可选地从下游任务数据集中抽样若干文本（sst2, stsb, ag_news）以扩充 embedding 数量
  - 使用 EmbeddingExtractor 生成 embedding 并通过 EmbeddingCache 保存到 results/embeddings
  - 支持从现有 .pkl 文件重建 metadata.json
"""
import argparse
import os
import pickle
import json
from pathlib import Path
from typing import List

import numpy as np

MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5"
]

from models.embedding_extractor import EmbeddingExtractor
from evaluation.semantic_analyzer import SemanticAnalyzer
from utils.embedding_cache import EmbeddingCache



def sample_task_texts(samples_per_task: int) -> List[str]:
    """从各任务数据集中抽样文本用于生成更多 embeddings。

    返回一个文本列表（可能重复已有 pairs）
    """
    texts = []
    if samples_per_task <= 0:
        return texts

    try:
        from datasets import load_dataset

        # SST-2 (validation)
        sst2 = load_dataset("glue", "sst2", split="validation")
        n = min(samples_per_task, len(sst2))
        texts.extend([t for t in sst2['sentence'][:n]])

        # STS-B (validation) - 合并两句以获得更多语句
        stsb = load_dataset("glue", "stsb", split="validation")
        n = min(samples_per_task, len(stsb))
        texts.extend([t for t in stsb['sentence1'][:n]])
        texts.extend([t for t in stsb['sentence2'][:n]])

        # AG News (test)
        ag = load_dataset("ag_news", split="test")
        n = min(samples_per_task, len(ag))
        texts.extend([t for t in ag['text'][:n]])

    except Exception as e:
        print(f"Warning: failed to sample datasets: {e}")

    return texts


def process_model(model_name: str, batch_size: int, samples_per_task: int, cache_dir: str, dataset_name: str, force: bool):
    print(f"\n=== Updating embeddings for model: {model_name} ===")

    analyzer = SemanticAnalyzer(model_name)
    all_texts = []
    for pair in analyzer.synonym_pairs + analyzer.antonym_pairs:
        all_texts.extend(pair)

    if samples_per_task and samples_per_task > 0:
        sampled = sample_task_texts(samples_per_task)
        all_texts.extend(sampled)

    # Deduplicate while preserving order
    seen = set()
    dedup_texts = []
    for t in all_texts:
        if t not in seen:
            dedup_texts.append(t)
            seen.add(t)

    print(f"Total texts to encode for {model_name}: {len(dedup_texts)}")

    extractor = EmbeddingExtractor(model_name)
    cache = EmbeddingCache(cache_dir)

    if not force:
        loaded = cache.load_embeddings(model_name, dedup_texts, dataset_name=dataset_name)
        if loaded is not None:
            embeddings, meta = loaded
            print(f"Embeddings already cached for {model_name} (n={embeddings.shape[0]}). Skipping.")
            return

    # Compute embeddings
    print("Computing embeddings...")
    embeddings = extractor.encode(dedup_texts, batch_size=batch_size)

    # Save to cache
    meta = {'n_pairs': len(analyzer.synonym_pairs), 'samples_per_task': samples_per_task}
    cache_key = cache.save_embeddings(model_name, dedup_texts, embeddings, dataset_name=dataset_name, metadata=meta)
    print(f"Saved {embeddings.shape[0]} embeddings to cache key: {cache_key}")


def main():
    parser = argparse.ArgumentParser(description="Update semantic embeddings for one or more models (fast, standalone)")
    parser.add_argument("--model", type=str, help="HuggingFace model name (e.g. intfloat/e5-base-v2). If omitted, use --models or --all")
    parser.add_argument("--models", type=str, nargs='+', help="List of models to run (space separated)")
    parser.add_argument("--all", action="store_true", help="Run for all MODELS defined in this file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding extraction")
    parser.add_argument("--samples_per_task", type=int, default=0, help="Number of samples to pull from each downstream dataset (sst2/stsb/ag_news). 0 = only pairs")
    parser.add_argument("--cache_dir", type=str, default="results/embeddings", help="Embedding cache directory")
    parser.add_argument("--dataset_name", type=str, default="semantic_analysis", help="dataset_name tag saved to cache metadata")
    parser.add_argument("--force", action="store_true", help="Force recompute and overwrite cache even if existing entry found")
    parser.add_argument("--rebuild_metadata", action="store_true", help="Rebuild metadata.json from existing .pkl files")

    args = parser.parse_args()

    # Check if rebuild_metadata is requested
    if args.rebuild_metadata:
        rebuild_metadata(args.cache_dir)
        return

    # determine models to run
    models_to_run = []
    if args.all:
        models_to_run = MODELS
    elif args.models:
        models_to_run = args.models
    elif args.model:
        models_to_run = [args.model]
    else:
        parser.error("Specify --model, --models or --all to choose models to process")

    for m in models_to_run:
        process_model(m, args.batch_size, args.samples_per_task, args.cache_dir, args.dataset_name, args.force)


def rebuild_metadata(cache_dir: str):
    """
    从现有的 .pkl 文件重建 metadata.json
    
    Args:
        cache_dir: 缓存目录路径
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    metadata = {}
    count = 0

    for p in cache_path.iterdir():
        if p.is_file() and p.suffix == '.pkl':
            try:
                with open(p, 'rb') as f:
                    data = pickle.load(f)

                # expected keys: model_name, texts, embeddings, dataset_name, metadata
                info = {
                    'model_name': data.get('model_name'),
                    'dataset_name': data.get('dataset_name'),
                    'n_texts': len(data.get('texts', [])),
                    'embedding_shape': list(data.get('embeddings').shape) if data.get('embeddings') is not None else None,
                    'metadata': data.get('metadata', {})
                }
                metadata[p.name] = info
                count += 1
            except Exception as e:
                print(f"  Warning: failed to read {p.name}: {e}")

    # write metadata.json
    meta_file = cache_path / 'metadata.json'
    try:
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Wrote metadata.json with {count} entries to: {meta_file}")
    except Exception as e:
        print(f"Failed to write metadata.json: {e}")


if __name__ == "__main__":
    main()
