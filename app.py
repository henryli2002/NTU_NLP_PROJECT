"""
Streamlit主应用
交互式Web应用，展示模型评估和可视化结果
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
import pickle
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from models.embedding_extractor import EmbeddingExtractor
from evaluation.task_evaluator import TaskEvaluator
from evaluation.semantic_analyzer import SemanticAnalyzer
from visualization.visualizer import EmbeddingVisualizer

# 页面配置
st.set_page_config(
    page_title="Embedding模型评估与可视化",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 模型列表
MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5"
]

# 缓存装饰器
@st.cache_resource
def load_model(model_name):
    """加载模型"""
    try:
        extractor = EmbeddingExtractor(model_name)
        return extractor
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

@st.cache_data
def load_evaluation_results():
    """加载评估结果"""
    results_path = "results/evaluation_results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}

def save_evaluation_results(results):
    """保存评估结果"""
    os.makedirs("results", exist_ok=True)
    results_path = "results/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    st.title("🤖 Embedding模型评估与可视化系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 配置")
        
        selected_model = st.selectbox(
            "选择模型",
            MODELS,
            index=0
        )
        
        max_samples = st.slider(
            "评估样本数",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=16,
            value=8,
            step=1
        )
        
        st.markdown("---")
        
        if st.button("🔄 清除缓存", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("缓存已清除！")
    
    # 主内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 模型对比", 
        "📈 下游任务评估", 
        "🎨 空间可视化", 
        "🔍 语义一致性分析", 
        "🔎 相似度查询"
    ])
    
    # Tab 1: 模型对比
    with tab1:
        st.header("模型性能对比")
        
        if st.button("运行所有模型评估", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = {}
            
            for i, model_name in enumerate(MODELS):
                status_text.text(f"正在评估: {model_name} ({i+1}/{len(MODELS)})")
                progress_bar.progress((i + 1) / len(MODELS))
                
                try:
                    evaluator = TaskEvaluator(model_name, batch_size=batch_size)
                    model_results = evaluator.evaluate_all(max_samples=max_samples)
                    results[model_name] = model_results
                except Exception as e:
                    st.error(f"评估 {model_name} 时出错: {e}")
                    results[model_name] = {
                        'sst2': {'accuracy': 0.0, 'f1_score': 0.0},
                        'stsb': {'spearman_correlation': 0.0, 'pearson_correlation': 0.0}
                    }
            
            save_evaluation_results(results)
            status_text.text("评估完成！")
            st.success("所有模型评估完成！")
        
        # 显示对比表
        eval_results = load_evaluation_results()
        if eval_results:
            # 准备数据
            comparison_data = []
            for model_name, results in eval_results.items():
                comparison_data.append({
                    '模型': model_name.split('/')[-1],
                    'SST-2 Accuracy': results.get('sst2', {}).get('accuracy', 0.0),
                    'SST-2 F1': results.get('sst2', {}).get('f1_score', 0.0),
                    'STS-B Spearman': results.get('stsb', {}).get('spearman_correlation', 0.0),
                    'STS-B Pearson': results.get('stsb', {}).get('pearson_correlation', 0.0)
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # 可视化对比
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('SST-2 Accuracy', 'SST-2 F1 Score', 
                               'STS-B Spearman Correlation', 'STS-B Pearson Correlation'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=df['模型'], y=df['SST-2 Accuracy'], name='Accuracy'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=df['模型'], y=df['SST-2 F1'], name='F1'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=df['模型'], y=df['STS-B Spearman'], name='Spearman'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=df['模型'], y=df['STS-B Pearson'], name='Pearson'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False, title_text="模型性能对比")
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: 下游任务评估
    with tab2:
        st.header("下游任务评估")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SST-2 情感分类")
            task_model = st.selectbox("选择模型", MODELS, key="task_model1")
            
            if st.button("评估 SST-2", key="eval_sst2"):
                with st.spinner("正在评估..."):
                    try:
                        evaluator = TaskEvaluator(task_model, batch_size=batch_size)
                        results = evaluator.evaluate_sst2(max_samples=max_samples)
                        
                        st.metric("Accuracy", f"{results['accuracy']:.4f}")
                        st.metric("F1 Score", f"{results['f1_score']:.4f}")
                    except Exception as e:
                        st.error(f"评估失败: {e}")
        
        with col2:
            st.subheader("STS-B 语义相似度")
            task_model2 = st.selectbox("选择模型", MODELS, key="task_model2")
            
            if st.button("评估 STS-B", key="eval_stsb"):
                with st.spinner("正在评估..."):
                    try:
                        evaluator = TaskEvaluator(task_model2, batch_size=batch_size)
                        results = evaluator.evaluate_stsb(max_samples=max_samples)
                        
                        st.metric("Spearman Correlation", f"{results['spearman_correlation']:.4f}")
                        st.metric("Pearson Correlation", f"{results['pearson_correlation']:.4f}")
                    except Exception as e:
                        st.error(f"评估失败: {e}")
    
    # Tab 3: 空间可视化
    with tab3:
        st.header("嵌入空间可视化")
        
        viz_model = st.selectbox("选择模型", MODELS, key="viz_model")
        dim_method = st.selectbox("降维方法", ["PCA", "t-SNE", "UMAP"], key="dim_method")
        
        # 示例文本
        sample_texts = st.text_area(
            "输入文本（每行一个）",
            value="I love this movie\nI hate this film\nThe weather is nice\nIt's raining today\nShe is happy\nHe is sad",
            height=150
        )
        
        if st.button("生成可视化", key="generate_viz"):
            if sample_texts:
                texts = [t.strip() for t in sample_texts.split('\n') if t.strip()]
                
                with st.spinner("正在提取embedding和生成可视化..."):
                    try:
                        extractor = load_model(viz_model)
                        if extractor:
                            embeddings = extractor.encode(texts, batch_size=batch_size)
                            
                            # 降维
                            visualizer = EmbeddingVisualizer()
                            reduced, explained_var = visualizer.reduce_dimension(
                                embeddings, 
                                method=dim_method.lower(),
                                n_components=2
                            )
                            
                            # 绘图
                            fig = px.scatter(
                                x=reduced[:, 0],
                                y=reduced[:, 1],
                                text=texts,
                                title=f"{viz_model.split('/')[-1]} - {dim_method} Visualization",
                                labels={'x': f'{dim_method} Component 1', 'y': f'{dim_method} Component 2'}
                            )
                            fig.update_traces(textposition="top center")
                            if explained_var:
                                fig.update_layout(title_text=f"{fig.layout.title.text} (Explained Variance: {explained_var:.2%})")
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"可视化失败: {e}")
    
    # Tab 4: 语义一致性分析
    with tab4:
        st.header("语义一致性分析")
        
        sem_model = st.selectbox("选择模型", MODELS, key="sem_model")
        
        if st.button("分析语义一致性", key="analyze_semantic"):
            with st.spinner("正在分析..."):
                try:
                    analyzer = SemanticAnalyzer(sem_model, batch_size=batch_size)
                    results = analyzer.get_all_analysis()
                    
                    # 显示近义句和反义句分析
                    st.subheader("近义句 vs 反义句相似度")
                    
                    syn_data = results['synonym_antonym']['synonym_pairs']
                    ant_data = results['synonym_antonym']['antonym_pairs']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("近义句平均相似度", f"{syn_data['mean_similarity']:.4f}")
                        st.write("**近义句对:**")
                        for i, (pair, sim) in enumerate(zip(syn_data['pairs'], syn_data['similarities'])):
                            st.write(f"{i+1}. {pair[0]} ↔ {pair[1]}")
                            st.write(f"   相似度: {sim:.4f}")
                    
                    with col2:
                        st.metric("反义句平均相似度", f"{ant_data['mean_similarity']:.4f}")
                        st.write("**反义句对:**")
                        for i, (pair, sim) in enumerate(zip(ant_data['pairs'], ant_data['similarities'])):
                            st.write(f"{i+1}. {pair[0]} ↔ {pair[1]}")
                            st.write(f"   相似度: {sim:.4f}")
                    
                    st.metric("语义区分度", f"{results['synonym_antonym']['semantic_gap']:.4f}")
                    
                    # 可视化
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['近义句', '反义句'],
                        y=[syn_data['mean_similarity'], ant_data['mean_similarity']],
                        marker_color=['green', 'red']
                    ))
                    fig.update_layout(
                        title="近义句 vs 反义句相似度对比",
                        yaxis_title="平均相似度",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"分析失败: {e}")
    
    # Tab 5: 相似度查询
    with tab5:
        st.header("文本相似度查询")
        
        query_model = st.selectbox("选择模型", MODELS, key="query_model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area("文本 1", height=100, key="text1")
        
        with col2:
            text2 = st.text_area("文本 2", height=100, key="text2")
        
        if st.button("计算相似度", key="calc_sim"):
            if text1 and text2:
                with st.spinner("正在计算..."):
                    try:
                        extractor = load_model(query_model)
                        if extractor:
                            similarity = extractor.get_similarity(text1, text2)
                            st.metric("余弦相似度", f"{similarity:.4f}")
                            
                            # 可视化
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = similarity,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "相似度"},
                                gauge = {
                                    'axis': {'range': [None, 1]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 0.3], 'color': "lightgray"},
                                        {'range': [0.3, 0.7], 'color': "gray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0.9
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"计算失败: {e}")
            else:
                st.warning("请输入两个文本")

if __name__ == "__main__":
    main()

