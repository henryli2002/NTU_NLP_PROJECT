#!/bin/bash
# 项目设置脚本

echo "Setting up Embedding Model Evaluation Project..."

# 创建必要的目录
mkdir -p data models evaluation visualization results

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Not in a virtual environment. Consider creating one:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
fi

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To run the Streamlit app:"
echo "  streamlit run app.py"
echo ""
echo "To run evaluation script:"
echo "  python run_evaluation.py --max_samples 1000 --batch_size 8"

