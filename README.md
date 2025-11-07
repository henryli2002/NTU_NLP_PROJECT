# Embedding模型评估与可视化项目

这是一个完整的NLP项目，用于对比和评估多个预训练语言模型在语义嵌入和下游任务上的表现。

## 📋 项目概述

本项目对比了四个预训练语言模型：
- **bert-base-uncased**: BERT基础模型
- **roberta-base**: RoBERTa基础模型
- **E5-base-v2**: 微软的E5嵌入模型
- **BGE-base-en-v1.5**: BAAI的BGE嵌入模型

## 🎯 评估维度

1. **下游任务性能**
   - SST-2（情感分类任务）
   - STS-B（语义相似度任务）
   - 使用 accuracy、F1 score、Spearman correlation、Pearson correlation 等指标

2. **空间结构对比**
   - 使用 PCA、t-SNE、UMAP 进行降维可视化
   - 展示嵌入空间的分布特征

3. **语义一致性分析**
   - 分析近义句与反义句的embedding相似度
   - 进行embedding聚类分析

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. （可选）运行基础测试

```bash
python test_basic.py
```

这将测试embedding提取功能是否正常工作。

## 📊 两阶段工作流程

项目采用**两阶段设计**，训练/评估和展示完全分离：

### 阶段A：训练/评估阶段

运行完整的模型评估，自动下载数据集，计算所有指标，保存结果和缓存：

```bash
# 基本评估（SST-2, STS-B）- 仅评估，不训练
python run_evaluation.py --max_samples 1000 --batch_size 8

# 包含AG News评估
python run_evaluation.py --max_samples 1000 --batch_size 8 --include_agnews

# 包含训练流程（训练模型并对比训练前后效果）
python run_evaluation.py --max_samples 1000 --batch_size 8 --train

# 训练指定任务（只训练SST-2）
python run_evaluation.py --max_samples 1000 --batch_size 8 --train --train_tasks sst2

# 自定义训练参数
python run_evaluation.py --max_samples 1000 --batch_size 8 --train \
    --train_epochs 5 --train_batch_size 16 --train_lr 2e-5 --train_samples 2000

# 不保存embedding缓存（节省空间）
python run_evaluation.py --max_samples 1000 --batch_size 8 --no_cache
```

**输出**：
- `results/metrics.json` - 所有评估指标（包含baseline和trained结果）
- `results/training_history.json` - 训练历史（如果启用训练）
- `results/embeddings/` - Embedding缓存（可选）
- `models/trained/` - 训练后的模型（如果启用训练）

### 阶段B：展示阶段

从缓存读取数据，提供交互式可视化界面，**不执行训练或数据下载**：

```bash
streamlit run app/app.py
```

应用将在浏览器中自动打开，默认地址为 `http://localhost:8501`

**注意**：展示阶段需要先运行阶段A生成 `results/metrics.json` 文件。

### 旧版应用（兼容模式）

如果需要同时支持训练和展示，可以使用旧版应用：

```bash
streamlit run app.py
```

## 📁 项目结构

```
Project/
├── app.py                    # Streamlit主应用（兼容模式，支持训练+展示）
├── app/                      # 展示阶段应用
│   ├── __init__.py
│   └── app.py               # 阶段B：纯展示应用
├── run_evaluation.py        # 阶段A：训练/评估脚本
├── requirements.txt          # 依赖包列表
├── README.md                # 项目说明
├── test_basic.py            # 基础功能测试脚本
├── setup.sh                 # 项目设置脚本
├── .streamlit/              # Streamlit配置
│   └── config.toml
├── models/                  # 模型相关模块
│   ├── __init__.py
│   └── embedding_extractor.py
├── evaluation/              # 评估模块
│   ├── __init__.py
│   ├── task_evaluator.py    # 下游任务评估（支持SST-2, STS-B, AG News）
│   └── semantic_analyzer.py # 语义一致性分析
├── training/                # 训练模块
│   ├── __init__.py
│   └── trainer.py           # 下游任务训练器
├── visualization/           # 可视化模块
│   ├── __init__.py
│   └── visualizer.py        # 降维和可视化
├── utils/                   # 工具模块
│   ├── __init__.py
│   └── embedding_cache.py   # Embedding缓存管理
├── data/                    # 数据目录
├── models/                  # 模型目录
│   └── trained/             # 训练后的模型（阶段A输出）
└── results/                 # 结果目录（自动生成）
    ├── metrics.json         # 评估指标（阶段A输出）
    ├── training_history.json # 训练历史（阶段A输出，如果启用训练）
    └── embeddings/          # Embedding缓存（阶段A输出）
```

## 🔧 功能说明

### 1. 模型对比 (`📊 模型对比`)
- 运行所有模型的评估
- 展示性能对比表格（Baseline结果）
- 可视化各模型的指标对比

### 2. 任务性能 (`📈 任务性能`)
- 展示详细的Baseline任务性能
- **SST-2**: 情感分类任务，输出accuracy和F1 score
- **STS-B**: 语义相似度任务，输出Spearman和Pearson相关系数
- **AG News**: 新闻分类任务（可选）

### 3. 训练前后对比 (`🎯 训练前后对比`) ⭐ 新增
- 对比训练前后的性能变化
- 显示改进幅度和百分比
- 可视化训练前后对比图表
- 展示训练损失曲线
- 支持SST-2和STS-B任务

### 4. 空间可视化 (`🎨 空间可视化`)
- 支持PCA、t-SNE、UMAP三种降维方法
- 输入自定义文本，可视化其embedding分布
- 交互式图表展示

### 5. 语义一致性分析 (`🔍 语义一致性分析`)
- 分析近义句和反义句的相似度
- 计算语义区分度
- 可视化对比结果

### 6. 相似度查询 (`🔎 相似度查询`)
- 输入两个文本，计算余弦相似度
- 使用仪表盘可视化相似度分数

## ⚙️ 配置说明

在Streamlit应用的侧边栏可以配置：
- **选择模型**: 选择要评估的模型
- **评估样本数**: 控制评估时使用的样本数量（100-2000）
- **Batch Size**: 控制推理时的batch大小（1-16），建议根据GPU内存调整

## 📊 使用示例

### 运行完整评估

1. 打开应用
2. 进入"📊 模型对比"标签页
3. 点击"运行所有模型评估"按钮
4. 等待评估完成（可能需要较长时间）
5. 查看对比结果和可视化图表

### 可视化自定义文本

1. 进入"🎨 空间可视化"标签页
2. 选择模型和降维方法
3. 输入要分析的文本（每行一个）
4. 点击"生成可视化"
5. 查看交互式图表

### 查询文本相似度

1. 进入"🔎 相似度查询"标签页
2. 选择模型
3. 输入两个文本
4. 点击"计算相似度"
5. 查看相似度分数和可视化

## 🔍 技术细节

### 两阶段设计优势
1. **训练/评估阶段（阶段A）**：
   - 自动下载所有数据集
   - 批量处理，效率高
   - 结果和缓存持久化保存
   - 可重复运行，自动使用缓存

2. **展示阶段（阶段B）**：
   - 快速加载，无需重新计算
   - 不占用GPU资源（除非实时查询）
   - 纯展示，适合演示和报告

### 模型加载
- 所有模型通过Hugging Face自动下载和缓存
- 支持CPU和GPU推理（自动检测）
- 针对E5和BGE模型进行了特殊处理（mean pooling + 归一化）

### 评估方法
- **Baseline评估**:
  - **SST-2**: 使用最近邻分类（基于训练集embedding）
  - **STS-B**: 计算余弦相似度，与人工标注分数进行相关性分析
  - **AG News**: 使用最近邻分类（可选）
- **训练后评估**:
  - 使用微调后的模型直接进行分类/回归
  - 支持对比训练前后的性能变化

### 缓存机制
- Embedding自动缓存，避免重复计算
- 基于文本内容和模型名称的哈希键
- 支持缓存查询和管理

### 可视化
- 使用Plotly进行交互式可视化
- 支持多种降维方法（PCA, t-SNE, UMAP）
- 从缓存加载数据，快速响应

## 💾 数据存储

- **评估指标**: `results/metrics.json`（阶段A输出，阶段B读取）
  - 包含 `baseline`（训练前）和 `trained`（训练后）结果
- **训练历史**: `results/training_history.json`（如果启用训练）
  - 包含训练损失曲线和最终指标
- **训练后的模型**: `models/trained/`（如果启用训练）
  - 每个模型按 `{model_name}_{task}` 格式保存
- **Embedding缓存**: `results/embeddings/`（可选，阶段A生成）
- **模型缓存**: Hugging Face默认缓存目录（自动管理）
- **数据集缓存**: Hugging Face默认缓存目录（自动管理）

## ⚠️ 注意事项

1. **GPU内存**: 如果使用GPU，建议batch size设置为8或更小
2. **首次运行**: 首次运行需要下载模型，可能需要较长时间
3. **评估时间**: 完整评估所有模型可能需要较长时间，建议先使用较小的样本数测试
4. **UMAP**: 如果未安装UMAP，可以通过 `pip install umap-learn` 安装

## 📝 依赖说明

主要依赖包：
- `torch`: PyTorch深度学习框架
- `transformers`: Hugging Face transformers库
- `datasets`: Hugging Face datasets库
- `streamlit`: Web应用框架
- `plotly`: 交互式可视化
- `scikit-learn`: 机器学习工具
- `umap-learn`: UMAP降维算法

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目仅供学习和研究使用。

