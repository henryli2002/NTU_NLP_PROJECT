#!/bin/bash
# 阶段B快速启动脚本

echo "=========================================="
echo "阶段B：展示阶段"
echo "=========================================="
echo ""

# 检查是否存在metrics.json
if [ ! -f "results/metrics.json" ]; then
    echo "错误: 未找到 results/metrics.json"
    echo ""
    echo "请先运行阶段A："
    echo "  python run_evaluation.py"
    echo "  或"
    echo "  ./run_stage_a.sh"
    echo ""
    exit 1
fi

echo "找到评估结果: results/metrics.json"
echo ""
echo "启动Streamlit应用..."
echo ""

# 启动Streamlit
streamlit run app/app.py

