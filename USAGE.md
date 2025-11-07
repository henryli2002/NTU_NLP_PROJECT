# 使用说明

## 两阶段工作流程详解

### 阶段A：训练/评估阶段

**目的**：运行完整的模型评估，生成所有结果和缓存

**命令**：
```bash
python run_evaluation.py [选项]
```

**主要功能**：
1. 自动下载数据集（SST-2, STS-B, AG News）
2. 加载4个预训练模型
3. 提取embedding（最后隐藏层 mean pooling）
4. 评估下游任务：
   - SST-2 → accuracy / F1
   - STS-B → Pearson / Spearman
   - AG News → accuracy / F1（可选）
5. 语义一致性分析（近义句、反义句余弦相似度）
6. 保存结果到 `results/metrics.json`
7. 可选保存embedding缓存到 `results/embeddings/`

**选项**：
- `--max_samples N`: 评估样本数（默认：1000）
- `--batch_size N`: Batch大小（默认：8）
- `--include_agnews`: 包含AG News评估
- `--no_cache`: 不保存embedding缓存
- `--cache_dir DIR`: 缓存目录（默认：results/embeddings）

**示例**：
```bash
# 基本评估
python run_evaluation.py --max_samples 1000 --batch_size 8

# 包含AG News，使用更大batch size
python run_evaluation.py --max_samples 2000 --batch_size 16 --include_agnews

# 快速测试（小样本，不保存缓存）
python run_evaluation.py --max_samples 100 --batch_size 4 --no_cache
```

**输出文件**：
- `results/metrics.json`: 所有评估指标（必需）
- `results/embeddings/`: Embedding缓存（可选）

### 阶段B：展示阶段

**目的**：从缓存读取数据，提供交互式可视化界面

**命令**：
```bash
streamlit run app/app.py
```

**主要功能**：
1. 从 `results/metrics.json` 读取评估结果
2. 从 `results/embeddings/` 加载embedding缓存（如果存在）
3. 提供5个功能标签页：
   - 📊 模型对比：性能对比表和图表
   - 📈 任务性能：详细任务指标
   - 🎨 空间可视化：PCA/t-SNE/UMAP降维可视化
   - 🔍 语义一致性分析：近义句/反义句分析
   - 🔎 相似度查询：实时文本相似度计算

**特点**：
- ✅ 不执行训练或数据下载
- ✅ 快速加载，无需重新计算
- ✅ 不占用GPU资源（除非实时查询）
- ✅ 纯展示，适合演示和报告

**注意事项**：
- 必须先运行阶段A生成 `results/metrics.json`
- 如果没有embedding缓存，某些可视化功能可能受限
- 实时相似度查询需要加载模型（会占用GPU/CPU资源）

## 工作流程示例

### 完整流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行阶段A：评估所有模型
python run_evaluation.py --max_samples 1000 --batch_size 8 --include_agnews

# 等待评估完成（可能需要较长时间，取决于GPU和样本数）

# 3. 运行阶段B：启动展示应用
streamlit run app/app.py

# 4. 在浏览器中查看结果（自动打开 http://localhost:8501）
```

### 快速测试流程

```bash
# 1. 快速评估（小样本，不保存缓存）
python run_evaluation.py --max_samples 100 --batch_size 4 --no_cache

# 2. 启动展示应用
streamlit run app/app.py
```

### 仅展示已有结果

如果已经有 `results/metrics.json` 文件：

```bash
# 直接运行展示应用
streamlit run app/app.py
```

## 常见问题

### Q: 阶段B提示找不到 metrics.json

**A**: 需要先运行阶段A：
```bash
python run_evaluation.py --max_samples 1000 --batch_size 8
```

### Q: 可视化功能无法使用

**A**: 可能原因：
1. 没有embedding缓存：运行阶段A时不要使用 `--no_cache`
2. 缓存不匹配：重新运行阶段A生成缓存

### Q: 评估时间太长

**A**: 可以：
1. 减少样本数：`--max_samples 500`
2. 增加batch size（如果GPU内存足够）：`--batch_size 16`
3. 不保存缓存：`--no_cache`（但会影响阶段B的可视化）

### Q: GPU内存不足

**A**: 
1. 减小batch size：`--batch_size 4` 或更小
2. 使用CPU：设置环境变量 `CUDA_VISIBLE_DEVICES=""`

### Q: 如何只评估特定模型？

**A**: 修改 `run_evaluation.py` 中的 `MODELS` 列表，只保留需要的模型。

## 文件说明

### 关键文件

- `run_evaluation.py`: 阶段A脚本
- `app/app.py`: 阶段B应用
- `results/metrics.json`: 评估结果（阶段A输出，阶段B读取）
- `results/embeddings/`: Embedding缓存（可选）

### 模块说明

- `models/embedding_extractor.py`: Embedding提取
- `evaluation/task_evaluator.py`: 下游任务评估
- `evaluation/semantic_analyzer.py`: 语义一致性分析
- `visualization/visualizer.py`: 可视化工具
- `utils/embedding_cache.py`: 缓存管理

## 性能优化建议

1. **首次运行**：使用小样本测试（`--max_samples 100`）
2. **完整评估**：使用合适的batch size（GPU: 8-16, CPU: 2-4）
3. **展示阶段**：确保有embedding缓存以获得最佳体验
4. **存储空间**：embedding缓存可能较大，如不需要可禁用

## 扩展功能

### 添加新模型

在 `run_evaluation.py` 的 `MODELS` 列表中添加模型名称。

### 添加新评估任务

1. 在 `evaluation/task_evaluator.py` 中添加评估方法
2. 在 `evaluate_all()` 中调用新方法
3. 在 `app/app.py` 中添加展示逻辑

### 自定义可视化

修改 `visualization/visualizer.py` 或直接在 `app/app.py` 中使用Plotly创建自定义图表。

