# 更新日志

## 两阶段架构重构

### 主要变更

#### 1. 阶段A：训练/评估阶段 (`run_evaluation.py`)

**新增功能**：
- ✅ 自动下载并缓存所有数据集（SST-2, STS-B, AG News）
- ✅ 支持AG News分类任务评估
- ✅ Embedding缓存系统（可选）
- ✅ 统一结果保存格式（`results/metrics.json`）
- ✅ 详细的进度输出和错误处理
- ✅ 命令行参数支持

**改进**：
- 模块化设计，易于扩展
- 自动错误恢复
- 结果包含元数据（时间戳、配置等）

#### 2. 阶段B：展示阶段 (`app/app.py`)

**新增功能**：
- ✅ 从 `results/metrics.json` 读取评估结果
- ✅ 从embedding缓存加载数据
- ✅ 5个功能标签页：
  - 📊 模型对比
  - 📈 任务性能
  - 🎨 空间可视化
  - 🔍 语义一致性分析
  - 🔎 相似度查询
- ✅ 不执行训练或数据下载
- ✅ 快速加载，适合演示

**特点**：
- 纯展示模式，不占用GPU资源（除非实时查询）
- 交互式可视化（Plotly）
- 自动检测缓存状态

#### 3. 新增模块

**`utils/embedding_cache.py`**：
- Embedding缓存管理
- 基于哈希的缓存键
- 支持保存/加载/查询/清除

**`evaluation/task_evaluator.py`**：
- 新增 `evaluate_agnews()` 方法
- 支持 `include_agnews` 参数

#### 4. 文档更新

- ✅ `README.md`: 更新两阶段工作流程说明
- ✅ `USAGE.md`: 详细使用说明
- ✅ `CHANGELOG.md`: 更新日志

#### 5. 辅助脚本

- ✅ `run_stage_a.sh`: 阶段A快速启动脚本
- ✅ `run_stage_b.sh`: 阶段B快速启动脚本

### 文件结构变更

```
新增文件：
├── app/
│   ├── __init__.py
│   └── app.py              # 阶段B应用
├── utils/
│   ├── __init__.py
│   └── embedding_cache.py  # 缓存管理
├── run_stage_a.sh          # 阶段A启动脚本
├── run_stage_b.sh          # 阶段B启动脚本
├── USAGE.md                # 使用说明
└── CHANGELOG.md            # 更新日志

修改文件：
├── run_evaluation.py       # 重构为阶段A
├── evaluation/task_evaluator.py  # 添加AG News支持
└── README.md               # 更新文档

保留文件：
├── app.py                  # 兼容模式（训练+展示）
└── 其他模块文件
```

### 使用方式变更

**之前**：
```bash
streamlit run app.py  # 训练和展示在一起
```

**现在**：
```bash
# 阶段A：训练/评估
python run_evaluation.py --max_samples 1000 --batch_size 8

# 阶段B：展示
streamlit run app/app.py
```

### 兼容性

- ✅ 保留旧版 `app.py`（兼容模式）
- ✅ 所有模块向后兼容
- ✅ 结果格式统一

### 性能优化

1. **缓存机制**：避免重复计算embedding
2. **分离阶段**：训练和展示独立，提高效率
3. **批量处理**：阶段A批量处理，阶段B快速加载

### 已知问题

- 无

### 未来计划

- [ ] 支持更多评估任务
- [ ] 支持更多可视化方法
- [ ] 添加模型对比报告生成
- [ ] 支持分布式评估

