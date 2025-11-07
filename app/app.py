"""
Stage B: Visualization Phase
Read data from results/metrics.json and cache, provide interactive visualization interface
No training or data downloading is performed
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embedding_cache import EmbeddingCache
from visualization.visualizer import EmbeddingVisualizer
from models.embedding_extractor import EmbeddingExtractor
from evaluation.semantic_analyzer import SemanticAnalyzer

# Page configuration
st.set_page_config(
    page_title="Embedding Model Evaluation & Visualization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model list
MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5"
]


@st.cache_data
def load_metrics():
    """Load evaluation metrics"""
    metrics_path = "results/metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load metrics file: {e}")
            return None
    else:
        st.warning(f"Metrics file not found: {metrics_path}")
        st.info("Please run `python run_evaluation.py` to evaluate models first")
        return None


@st.cache_resource
def load_cache():
    """Load embedding cache"""
    cache_dir = "results/embeddings"
    if os.path.exists(cache_dir):
        return EmbeddingCache(cache_dir)
    return None


@st.cache_resource
def load_model_for_query(model_name):
    """Load model for real-time query (only when needed)"""
    try:
        extractor = EmbeddingExtractor(model_name)
        return extractor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def get_model_display_name(model_name):
    """Get model display name"""
    return model_name.split('/')[-1]


def main():
    st.title("Embedding Model Evaluation & Visualization System")
    st.markdown("**Stage B: Visualization Phase** - Read data from cache, no training performed")
    st.markdown("---")
    
    # Load data
    metrics_data = load_metrics()
    cache = load_cache()
    
    if metrics_data is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display metadata
        if 'metadata' in metrics_data:
            meta = metrics_data['metadata']
            st.subheader("Evaluation Info")
            st.write(f"**Timestamp**: {meta.get('timestamp', 'N/A')}")
            st.write(f"**Max Samples**: {meta.get('max_samples', 'N/A')}")
            st.write(f"**Batch Size**: {meta.get('batch_size', 'N/A')}")
            if meta.get('include_agnews'):
                st.write("**Includes**: AG News")
        
        st.markdown("---")
        
        # Model selection
        available_models = list(metrics_data.get('results', {}).keys())
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                format_func=get_model_display_name,
                index=0
            )
        else:
            selected_model = None
            st.warning("No available model results")
        
        st.markdown("---")
        
        # Cache status
        if cache:
            cached_items = cache.list_cached_embeddings()
            st.subheader("Cache Status")
            st.write(f"Cached: {len(cached_items)} items")
        else:
            st.info("No embedding cache found")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Model Comparison", 
        "📈 Task Performance", 
        "🎯 Before/After Training",
        "🎨 Embedding Space Visualization", 
        "🔍 Semantic Consistency Analysis", 
        "🔎 Similarity Query"
    ])
    
    # Tab 1: Model Comparison
    with tab1:
        st.header("Model Performance Comparison")
        
        results = metrics_data.get('results', {})
        if not results:
            st.warning("No available evaluation results")
        else:
            # Prepare comparison data
            comparison_data = []
            for model_name, result in results.items():
                if result.get('status') == 'error':
                    continue
                
                baseline = result.get('baseline', {})
                row = {
                    'Model': get_model_display_name(model_name),
                    'Model Full Name': model_name
                }
                
                # SST-2 (Baseline)
                if 'sst2' in baseline:
                    row['SST-2 Accuracy'] = baseline['sst2'].get('accuracy', 0.0)
                    row['SST-2 F1'] = baseline['sst2'].get('f1_score', 0.0)
                else:
                    row['SST-2 Accuracy'] = None
                    row['SST-2 F1'] = None
                
                # STS-B (Baseline)
                if 'stsb' in baseline:
                    row['STS-B Spearman'] = baseline['stsb'].get('spearman_correlation', 0.0)
                    row['STS-B Pearson'] = baseline['stsb'].get('pearson_correlation', 0.0)
                else:
                    row['STS-B Spearman'] = None
                    row['STS-B Pearson'] = None
                
                # AG News (Baseline)
                if 'agnews' in baseline:
                    row['AG News Accuracy'] = baseline['agnews'].get('accuracy', 0.0)
                    row['AG News F1'] = baseline['agnews'].get('f1_score', 0.0)
                else:
                    row['AG News Accuracy'] = None
                    row['AG News F1'] = None
                
                comparison_data.append(row)
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                # Reorder columns: by task order, same task metrics together
                # Define column order: Model -> SST-2 -> STS-B -> AG News
                ordered_cols = ['Model']
                
                # SST-2 metrics
                sst2_cols = ['SST-2 Accuracy', 'SST-2 F1']
                for col in sst2_cols:
                    if col in df.columns:
                        ordered_cols.append(col)
                
                # STS-B metrics
                stsb_cols = ['STS-B Spearman', 'STS-B Pearson']
                for col in stsb_cols:
                    if col in df.columns:
                        ordered_cols.append(col)
                
                # AG News metrics
                agnews_cols = ['AG News Accuracy', 'AG News F1']
                for col in agnews_cols:
                    if col in df.columns:
                        ordered_cols.append(col)
                
                # Ensure all columns are included (handle possible other columns)
                remaining_cols = [c for c in df.columns if c not in ordered_cols and c != 'Model Full Name']
                ordered_cols.extend(remaining_cols)
                
                # Only keep columns that actually exist
                display_cols = [c for c in ordered_cols if c in df.columns]
                df_display = df[display_cols]
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # Visualization comparison
                st.subheader("Performance Comparison Charts")
                
                # Select metrics to visualize
                metric_options = {
                    'SST-2 Accuracy': 'SST-2 Accuracy',
                    'SST-2 F1': 'SST-2 F1',
                    'STS-B Spearman': 'STS-B Spearman',
                    'STS-B Pearson': 'STS-B Pearson',
                }
                if 'AG News Accuracy' in df.columns:
                    metric_options['AG News Accuracy'] = 'AG News Accuracy'
                    metric_options['AG News F1'] = 'AG News F1'
                
                # Default selection: all available metrics
                default_metrics = list(metric_options.keys())
                
                selected_metrics = st.multiselect(
                    "Select Metrics to Compare",
                    list(metric_options.keys()),
                    default=default_metrics
                )
                
                if selected_metrics:
                    n_metrics = len(selected_metrics)
                    n_cols = min(2, n_metrics)
                    n_rows = (n_metrics + n_cols - 1) // n_cols
                    
                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=selected_metrics,
                        specs=[[{"type": "bar"} for _ in range(n_cols)] for _ in range(n_rows)]
                    )
                    
                    for idx, metric in enumerate(selected_metrics):
                        row = idx // n_cols + 1
                        col = idx % n_cols + 1
                        
                        if metric in df.columns:
                            fig.add_trace(
                                go.Bar(
                                    x=df['Model'],
                                    y=df[metric],
                                    name=metric,
                                    showlegend=False
                                ),
                                row=row, col=col
                            )
                    
                    fig.update_layout(height=300 * n_rows, title_text="Model Performance Comparison")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Task Performance
    with tab2:
        st.header("Detailed Task Performance")
        
        if selected_model:
            result = metrics_data.get('results', {}).get(selected_model, {})
            if result.get('status') == 'error':
                st.error(f"Model evaluation error: {result.get('error', 'Unknown error')}")
            else:
                baseline = result.get('baseline', {})
                trained = result.get('trained', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("SST-2 Sentiment Classification")
                    if 'sst2' in baseline:
                        sst2 = baseline['sst2']
                        st.metric("Accuracy", f"{sst2.get('accuracy', 0):.4f}")
                        st.metric("F1 Score", f"{sst2.get('f1_score', 0):.4f}")
                    else:
                        st.info("Not evaluated")
                
                with col2:
                    st.subheader("STS-B Semantic Similarity")
                    if 'stsb' in baseline:
                        stsb = baseline['stsb']
                        st.metric("Spearman", f"{stsb.get('spearman_correlation', 0):.4f}")
                        st.metric("Pearson", f"{stsb.get('pearson_correlation', 0):.4f}")
                    else:
                        st.info("Not evaluated")
                
                with col3:
                    st.subheader("AG News Classification")
                    if 'agnews' in baseline:
                        agnews = baseline['agnews']
                        st.metric("Accuracy", f"{agnews.get('accuracy', 0):.4f}")
                        st.metric("F1 Score", f"{agnews.get('f1_score', 0):.4f}")
                    else:
                        st.info("Not evaluated")
        else:
            st.warning("Please select a model")
    
    # Tab 3: Before/After Training
    with tab3:
        st.header("Before/After Training Comparison")
        
        train_models = metrics_data.get('metadata', {}).get('train_models', False)
        
        if not train_models:
            st.info("This evaluation did not include training. To view before/after training comparison, run evaluation with `--train` parameter.")
        elif selected_model:
            result = results.get(selected_model, {})
            baseline = result.get('baseline', {})
            trained = result.get('trained', {})
            training_history = metrics_data.get('training_history', {}).get(selected_model, {})
            
            if not trained:
                st.warning("This model has no trained results")
            else:
                # SST-2 comparison
                if 'sst2' in baseline and 'sst2' in trained:
                    st.subheader("SST-2 Sentiment Classification Task")
                    col1, col2, col3 = st.columns(3)
                    
                    base_sst2 = baseline['sst2']
                    train_sst2 = trained['sst2']
                    
                    with col1:
                        st.write("**Before Training (Baseline)**")
                        st.metric("Accuracy", f"{base_sst2.get('accuracy', 0):.4f}")
                        st.metric("F1 Score", f"{base_sst2.get('f1_score', 0):.4f}")
                    
                    with col2:
                        st.write("**After Training (Trained)**")
                        acc_improvement = train_sst2.get('accuracy', 0) - base_sst2.get('accuracy', 0)
                        f1_improvement = train_sst2.get('f1_score', 0) - base_sst2.get('f1_score', 0)
                        st.metric("Accuracy", f"{train_sst2.get('accuracy', 0):.4f}", 
                                 delta=f"{acc_improvement:+.4f}")
                        st.metric("F1 Score", f"{train_sst2.get('f1_score', 0):.4f}",
                                 delta=f"{f1_improvement:+.4f}")
                    
                    with col3:
                        st.write("**Improvement**")
                        acc_pct = (acc_improvement / base_sst2.get('accuracy', 1)) * 100 if base_sst2.get('accuracy', 0) > 0 else 0
                        f1_pct = (f1_improvement / base_sst2.get('f1_score', 1)) * 100 if base_sst2.get('f1_score', 0) > 0 else 0
                        st.metric("Accuracy", f"{acc_pct:+.2f}%")
                        st.metric("F1 Score", f"{f1_pct:+.2f}%")
                    
                    # Visualization comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Before Training',
                        x=['Accuracy', 'F1 Score'],
                        y=[base_sst2.get('accuracy', 0), base_sst2.get('f1_score', 0)],
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        name='After Training',
                        x=['Accuracy', 'F1 Score'],
                        y=[train_sst2.get('accuracy', 0), train_sst2.get('f1_score', 0)],
                        marker_color='darkblue'
                    ))
                    fig.update_layout(
                        title="SST-2 Before/After Training Comparison",
                        yaxis_title="Score",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Training history
                    if 'sst2' in training_history and 'history' in training_history['sst2']:
                        st.subheader("Training History (SST-2)")
                        history = training_history['sst2']['history']
                        if 'train_loss' in history and 'val_loss' in history:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=history['train_loss'],
                                mode='lines',
                                name='Train Loss',
                                line=dict(color='blue')
                            ))
                            fig.add_trace(go.Scatter(
                                y=history['val_loss'],
                                mode='lines',
                                name='Val Loss',
                                line=dict(color='red')
                            ))
                            fig.update_layout(
                                title="Training Loss Curve",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # STS-B comparison
                if 'stsb' in baseline and 'stsb' in trained:
                    st.subheader("STS-B Semantic Similarity Task")
                    col1, col2, col3 = st.columns(3)
                    
                    base_stsb = baseline['stsb']
                    train_stsb = trained['stsb']
                    
                    with col1:
                        st.write("**Before Training (Baseline)**")
                        st.metric("Spearman", f"{base_stsb.get('spearman_correlation', 0):.4f}")
                        st.metric("Pearson", f"{base_stsb.get('pearson_correlation', 0):.4f}")
                    
                    with col2:
                        st.write("**After Training (Trained)**")
                        spear_improvement = train_stsb.get('spearman_correlation', 0) - base_stsb.get('spearman_correlation', 0)
                        pear_improvement = train_stsb.get('pearson_correlation', 0) - base_stsb.get('pearson_correlation', 0)
                        st.metric("Spearman", f"{train_stsb.get('spearman_correlation', 0):.4f}",
                                 delta=f"{spear_improvement:+.4f}")
                        st.metric("Pearson", f"{train_stsb.get('pearson_correlation', 0):.4f}",
                                 delta=f"{pear_improvement:+.4f}")
                    
                    with col3:
                        st.write("**Improvement**")
                        spear_pct = (spear_improvement / abs(base_stsb.get('spearman_correlation', 1))) * 100 if base_stsb.get('spearman_correlation', 0) != 0 else 0
                        pear_pct = (pear_improvement / abs(base_stsb.get('pearson_correlation', 1))) * 100 if base_stsb.get('pearson_correlation', 0) != 0 else 0
                        st.metric("Spearman", f"{spear_pct:+.2f}%")
                        st.metric("Pearson", f"{pear_pct:+.2f}%")
                    
                    # Visualization comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Before Training',
                        x=['Spearman', 'Pearson'],
                        y=[base_stsb.get('spearman_correlation', 0), base_stsb.get('pearson_correlation', 0)],
                        marker_color='lightcoral'
                    ))
                    fig.add_trace(go.Bar(
                        name='After Training',
                        x=['Spearman', 'Pearson'],
                        y=[train_stsb.get('spearman_correlation', 0), train_stsb.get('pearson_correlation', 0)],
                        marker_color='darkred'
                    ))
                    fig.update_layout(
                        title="STS-B Before/After Training Comparison",
                        yaxis_title="Correlation",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Training history
                    if 'stsb' in training_history and 'history' in training_history['stsb']:
                        st.subheader("Training History (STS-B)")
                        history = training_history['stsb']['history']
                        if 'train_loss' in history and 'val_loss' in history:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=history['train_loss'],
                                mode='lines',
                                name='Train Loss',
                                line=dict(color='blue')
                            ))
                            fig.add_trace(go.Scatter(
                                y=history['val_loss'],
                                mode='lines',
                                name='Val Loss',
                                line=dict(color='red')
                            ))
                            fig.update_layout(
                                title="Training Loss Curve",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # AG News comparison
                if 'agnews' in baseline and 'agnews' in trained:
                    st.subheader("AG News Classification Task")
                    col1, col2, col3 = st.columns(3)
                    
                    base_agnews = baseline['agnews']
                    train_agnews = trained['agnews']
                    
                    with col1:
                        st.write("**Before Training (Baseline)**")
                        st.metric("Accuracy", f"{base_agnews.get('accuracy', 0):.4f}")
                        st.metric("F1 Score", f"{base_agnews.get('f1_score', 0):.4f}")
                    
                    with col2:
                        st.write("**After Training (Trained)**")
                        acc_improvement = train_agnews.get('accuracy', 0) - base_agnews.get('accuracy', 0)
                        f1_improvement = train_agnews.get('f1_score', 0) - base_agnews.get('f1_score', 0)
                        st.metric("Accuracy", f"{train_agnews.get('accuracy', 0):.4f}", 
                                 delta=f"{acc_improvement:+.4f}")
                        st.metric("F1 Score", f"{train_agnews.get('f1_score', 0):.4f}",
                                 delta=f"{f1_improvement:+.4f}")
                    
                    with col3:
                        st.write("**Improvement**")
                        acc_pct = (acc_improvement / base_agnews.get('accuracy', 1)) * 100 if base_agnews.get('accuracy', 0) > 0 else 0
                        f1_pct = (f1_improvement / base_agnews.get('f1_score', 1)) * 100 if base_agnews.get('f1_score', 0) > 0 else 0
                        st.metric("Accuracy", f"{acc_pct:+.2f}%")
                        st.metric("F1 Score", f"{f1_pct:+.2f}%")
                    
                    # Visualization comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Before Training',
                        x=['Accuracy', 'F1 Score'],
                        y=[base_agnews.get('accuracy', 0), base_agnews.get('f1_score', 0)],
                        marker_color='lightgreen'
                    ))
                    fig.add_trace(go.Bar(
                        name='After Training',
                        x=['Accuracy', 'F1 Score'],
                        y=[train_agnews.get('accuracy', 0), train_agnews.get('f1_score', 0)],
                        marker_color='darkgreen'
                    ))
                    fig.update_layout(
                        title="AG News Before/After Training Comparison",
                        yaxis_title="Score",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Training history
                    if 'agnews' in training_history and 'history' in training_history['agnews']:
                        st.subheader("Training History (AG News)")
                        history = training_history['agnews']['history']
                        if 'train_loss' in history and 'val_loss' in history:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=history['train_loss'],
                                mode='lines',
                                name='Train Loss',
                                line=dict(color='blue')
                            ))
                            fig.add_trace(go.Scatter(
                                y=history['val_loss'],
                                mode='lines',
                                name='Val Loss',
                                line=dict(color='red')
                            ))
                            fig.update_layout(
                                title="Training Loss Curve",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select a model")
    
    # Tab 4: Embedding Space Visualization
    with tab4:
        st.header("Embedding Space Visualization")
        
        if selected_model:
            viz_model = selected_model
            dim_method = st.selectbox("Dimensionality Reduction Method", ["PCA", "UMAP"], key="dim_method")
            
            # Store cache_data for label generation
            cache_data_for_labels = None
            
            # Try to load embeddings from cache for semantic analysis
            embeddings = None
            texts = None
            
            if cache:
                cached_items = cache.list_cached_embeddings(
                    model_name=viz_model,
                    dataset_name="semantic_analysis"
                )
                if cached_items:
                    # Load first matching cache
                    cache_key = cached_items[0]['cache_key']
                    cache_path = Path("results/embeddings") / cache_key
                    if cache_path.exists():
                        import pickle
                        with open(cache_path, 'rb') as f:
                            cache_data = pickle.load(f)
                            embeddings = cache_data['embeddings']
                            texts = cache_data['texts']
                            cache_data_for_labels = cache_data
                            st.success(f"Loaded embeddings for {len(texts)} texts from cache")
            
            # If no cache, allow user to input texts
            if embeddings is None:
                st.info("No cache data found, please enter texts for visualization")
                sample_texts = st.text_area(
                    "Enter texts (one per line)",
                    value="I love this movie\nI hate this film\nThe weather is nice\nIt's raining today\nShe is happy\nHe is sad",
                    height=150,
                    key="viz_texts"
                )
                
                if st.button("Generate Visualization", key="generate_viz"):
                    if sample_texts:
                        texts = [t.strip() for t in sample_texts.split('\n') if t.strip()]
                        
                        with st.spinner("Extracting embeddings..."):
                            extractor = load_model_for_query(viz_model)
                            if extractor:
                                embeddings = extractor.encode(texts, batch_size=8)
            
            # Visualization
            if embeddings is not None and texts is not None:
                # Determine if this is semantic analysis data and calculate pairs
                total_samples = len(texts)
                is_semantic_analysis = cache_data_for_labels and cache_data_for_labels.get('dataset_name') == 'semantic_analysis'
                total_pairs = None
                text_to_index = {text: idx for idx, text in enumerate(texts)}
                triplets = None  # List of (base_text, synonym_text, antonym_text, indices)
                
                if is_semantic_analysis:
                    try:
                        analyzer = SemanticAnalyzer(viz_model)
                        # Build triplets: each triplet is (base_text, synonym_text, antonym_text)
                        # Base text is the first element of both synonym and antonym pairs
                        triplets = []
                        synonym_map = {pair[0]: pair[1] for pair in analyzer.synonym_pairs}
                        antonym_map = {pair[0]: pair[1] for pair in analyzer.antonym_pairs}
                        
                        # Find common base texts (first element of pairs)
                        base_texts = set(synonym_map.keys()) & set(antonym_map.keys())
                        
                        for base_text in base_texts:
                            synonym_text = synonym_map[base_text]
                            antonym_text = antonym_map[base_text]
                            
                            # Find indices in the texts list
                            indices = []
                            if base_text in text_to_index:
                                indices.append(text_to_index[base_text])
                            if synonym_text in text_to_index:
                                indices.append(text_to_index[synonym_text])
                            if antonym_text in text_to_index:
                                indices.append(text_to_index[antonym_text])
                            
                            # Only add if we have all 3 texts
                            if len(indices) == 3:
                                triplets.append((base_text, synonym_text, antonym_text, indices))
                        
                        total_pairs = len(triplets)
                    except Exception as e:
                        st.warning(f"Could not build triplets: {e}")
                        triplets = None
                
                # Allow user to select number of pairs or samples
                if is_semantic_analysis and total_pairs is not None and triplets:
                    max_pairs = total_pairs
                    selected_pairs = st.slider(
                        "Number of pairs to visualize",
                        min_value=1,
                        max_value=max_pairs,
                        value=min(25, max_pairs),  # Default to 25 pairs or max available
                        step=1,
                        help=f"Total available: {total_pairs} pairs ({total_pairs * 3} texts: base + synonym + antonym)"
                    )
                    
                    # Select complete triplets
                    import random
                    random.seed(42)
                    selected_triplet_indices = random.sample(range(total_pairs), selected_pairs)
                    selected_triplets = [triplets[i] for i in selected_triplet_indices]
                    
                    # Collect all indices from selected triplets in order: base, synonym, antonym
                    all_indices = []
                    sampled_texts_list = []
                    labels = []
                    
                    for base, syn, ant, triplet_indices in selected_triplets:
                        # Get indices in order: base, synonym, antonym
                        base_idx = text_to_index.get(base)
                        syn_idx = text_to_index.get(syn)
                        ant_idx = text_to_index.get(ant)
                        
                        # Add in order: base, synonym, antonym
                        if base_idx is not None:
                            all_indices.append(base_idx)
                            sampled_texts_list.append(base)
                            labels.append('Base')
                        if syn_idx is not None:
                            all_indices.append(syn_idx)
                            sampled_texts_list.append(syn)
                            labels.append('Synonym')
                        if ant_idx is not None:
                            all_indices.append(ant_idx)
                            sampled_texts_list.append(ant)
                            labels.append('Antonym')
                    
                    sampled_texts = sampled_texts_list
                    sampled_embeddings = embeddings[all_indices]
                    
                else:
                    max_samples = st.slider(
                        "Number of samples to visualize",
                        min_value=1,
                        max_value=total_samples,
                        value=min(100, total_samples),  # Default to 100 or all if less
                        step=1,
                        help=f"Total available: {total_samples} samples"
                    )
                    
                    # Sample the data if needed
                    if max_samples < total_samples:
                        import random
                        random.seed(42)  # For reproducibility
                        indices = random.sample(range(total_samples), max_samples)
                        sampled_texts = [texts[i] for i in indices]
                        sampled_embeddings = embeddings[indices]
                        labels = None
                    else:
                        sampled_texts = texts
                        sampled_embeddings = embeddings
                        labels = None
                
                if is_semantic_analysis and total_pairs is not None and triplets:
                    st.info(f"Visualizing {len(sampled_texts)} texts ({selected_pairs} pairs × 3) out of {total_samples} texts ({total_pairs} pairs)")
                else:
                    st.info(f"Visualizing {len(sampled_texts)} out of {total_samples} samples")
                
                # Option to show text labels
                show_text_labels = st.checkbox("Show text labels", value=False, key="show_text_labels")
                
                with st.spinner(f"Generating visualization using {dim_method}..."):
                    try:
                        visualizer = EmbeddingVisualizer()
                        reduced, explained_var = visualizer.reduce_dimension(
                            sampled_embeddings,
                            method=dim_method.lower(),
                            n_components=2
                        )
                        
                        # Labels are already created above for semantic analysis data with triplets
                        # For non-semantic or when triplets couldn't be built, labels will be None
                        # No need to recreate labels here
                        
                        # Prepare text labels (optional)
                        text_labels = sampled_texts if show_text_labels else None
                        
                        fig = px.scatter(
                            x=reduced[:, 0],
                            y=reduced[:, 1],
                            text=text_labels,
                            color=labels,
                            title=f"{get_model_display_name(viz_model)} - {dim_method} Visualization ({len(sampled_texts)} samples)",
                            labels={'x': f'{dim_method} Component 1', 'y': f'{dim_method} Component 2'}
                        )
                        if show_text_labels:
                            fig.update_traces(textposition="top center")
                        
                        # Hide empty string labels from legend (unlabeled points)
                        if labels and any(l == "" for l in labels):
                            # Update legend to hide empty string entries
                            for trace in fig.data:
                                if trace.name == "" or trace.name is None:
                                    trace.showlegend = False
                        
                        if explained_var:
                            fig.update_layout(
                                title_text=f"{fig.layout.title.text} (Explained Variance: {explained_var:.2%})"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"Visualization failed: {error_msg}")
        else:
            st.warning("Please select a model")
    
    # Tab 5: Semantic Consistency Analysis
    with tab5:
        st.header("Semantic Consistency Analysis")
        
        if selected_model:
            result = results.get(selected_model, {})
            semantic = result.get('semantic', {}).get('synonym_antonym', {})
            
            if semantic:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Synonym Pairs Analysis")
                    syn_data = semantic.get('synonym_pairs', {})
                    st.metric("Mean Similarity", f"{syn_data.get('mean_similarity', 0):.4f}")
                    st.metric("Std Deviation", f"{syn_data.get('std_similarity', 0):.4f}")
                    
                    st.write("**Synonym Pairs:**")
                    pairs = syn_data.get('pairs', [])
                    similarities = syn_data.get('similarities', [])
                    for i, (pair, sim) in enumerate(zip(pairs, similarities), 1):
                        st.write(f"{i}. {pair[0]}")
                        st.write(f"   ↔ {pair[1]}")
                        st.write(f"   Similarity: {sim:.4f}")
                        st.write("")
                
                with col2:
                    st.subheader("Antonym Pairs Analysis")
                    ant_data = semantic.get('antonym_pairs', {})
                    st.metric("Mean Similarity", f"{ant_data.get('mean_similarity', 0):.4f}")
                    st.metric("Std Deviation", f"{ant_data.get('std_similarity', 0):.4f}")
                    
                    st.write("**Antonym Pairs:**")
                    pairs = ant_data.get('pairs', [])
                    similarities = ant_data.get('similarities', [])
                    for i, (pair, sim) in enumerate(zip(pairs, similarities), 1):
                        st.write(f"{i}. {pair[0]}")
                        st.write(f"   ↔ {pair[1]}")
                        st.write(f"   Similarity: {sim:.4f}")
                        st.write("")
                
                # Semantic gap
                gap = semantic.get('semantic_gap', 0)
                st.metric("Semantic Gap", f"{gap:.4f}", 
                         help="Mean similarity of synonym pairs - Mean similarity of antonym pairs")
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Synonym Pairs', 'Antonym Pairs'],
                    y=[syn_data.get('mean_similarity', 0), ant_data.get('mean_similarity', 0)],
                    marker_color=['green', 'red'],
                    text=[f"{syn_data.get('mean_similarity', 0):.4f}", 
                          f"{ant_data.get('mean_similarity', 0):.4f}"],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Synonym vs Antonym Pairs Similarity Comparison",
                    yaxis_title="Mean Similarity",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No semantic consistency analysis results found")
        else:
            st.warning("Please select a model")
    
    # Tab 6: Similarity Query
    with tab6:
        st.header("Text Similarity Query")
        
        if selected_model:
            query_model = selected_model
            
            col1, col2 = st.columns(2)
            
            with col1:
                text1 = st.text_area("Text 1", height=100, key="text1")
            
            with col2:
                text2 = st.text_area("Text 2", height=100, key="text2")
            
            if st.button("Calculate Similarity", key="calc_sim"):
                if text1 and text2:
                    with st.spinner("Calculating..."):
                        try:
                            extractor = load_model_for_query(query_model)
                            if extractor:
                                similarity = extractor.get_similarity(text1, text2)
                                st.metric("Cosine Similarity", f"{similarity:.4f}")
                                
                                # Visualization
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=similarity,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Similarity"},
                                    gauge={
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
                            st.error(f"Calculation failed: {e}")
                else:
                    st.warning("Please enter two texts")
        else:
            st.warning("Please select a model")


if __name__ == "__main__":
    main()

