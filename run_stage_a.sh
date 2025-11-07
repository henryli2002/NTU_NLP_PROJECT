#!/bin/bash
# 阶段A快速启动脚本

echo "=========================================="
echo "阶段A：训练/评估阶段"
echo "=========================================="
echo ""

# 默认参数
MAX_SAMPLES=${1:-1000}
BATCH_SIZE=${2:-8}
INCLUDE_AGNEWS=${3:-false}

echo "配置："
echo "  样本数: $MAX_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  包含AG News: $INCLUDE_AGNEWS"
echo ""

# 构建命令
CMD="python run_evaluation.py --max_samples $MAX_SAMPLES --batch_size $BATCH_SIZE"

if [ "$INCLUDE_AGNEWS" = "true" ]; then
    CMD="$CMD --include_agnews"
fi

echo "执行命令: $CMD"
echo ""
echo "开始评估..."
echo ""

# 执行
$CMD

echo ""
echo "=========================================="
echo "阶段A完成！"
echo "结果保存在: results/metrics.json"
echo "=========================================="
echo ""
echo "下一步：运行阶段B"
echo "  streamlit run app/app.py"
echo ""

