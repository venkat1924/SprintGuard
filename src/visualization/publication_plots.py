"""
Publication-Quality Plot Generators for SprintGuard Research Paper

All plots follow IEEE conference/journal standards:
- Vector formats (PDF/SVG)
- Proper sizing for 2-column layout
- Colorblind-friendly palettes
- Publication-ready fonts and labels
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to use SciencePlots for IEEE style
try:
    plt.style.use(['science', 'ieee'])
    SCIENCE_PLOTS_AVAILABLE = True
    logging.info("[PLOTS] Using SciencePlots IEEE style")
except:
    SCIENCE_PLOTS_AVAILABLE = False
    logging.warning("[PLOTS] SciencePlots not available, using default matplotlib style")
    # Set some sensible defaults
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9


def generate_sankey_diagram(
    stage_counts: Dict[str, int],
    output_path: str = "visualizations/sankey_pipeline_flow.pdf"
):
    """
    Generate Sankey diagram showing data flow through pipeline stages.
    
    Shows: Raw Data → Snorkel Labels → Cleanlab Filtered → Final Training Set
    
    Args:
        stage_counts: Dictionary with keys 'raw', 'snorkel', 'cleanlab', 'final'
                     Example: {'raw': 12106, 'snorkel': 12106, 'cleanlab': 10500, 'final': 9000}
        output_path: Where to save the figure
    
    Returns:
        Plotly figure object
    """
    logging.info("[PLOTS] Generating Sankey diagram...")
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        logging.error("[PLOTS] Plotly not installed. Run: pip install plotly kaleido")
        return None
    
    # Prepare data
    labels = ["Raw Data", "Snorkel\nLabels", "Cleanlab\nFiltered", "Training\nSet"]
    
    # Values for links (flow between stages)
    raw_count = stage_counts.get('raw', 0)
    snorkel_count = stage_counts.get('snorkel', 0)
    cleanlab_count = stage_counts.get('cleanlab', 0)
    final_count = stage_counts.get('final', 0)
    
    # Calculate data lost at each stage
    snorkel_loss = raw_count - snorkel_count
    cleanlab_loss = snorkel_count - cleanlab_count
    final_loss = cleanlab_count - final_count
    
    logging.info(f"  Raw data: {raw_count}")
    logging.info(f"  After Snorkel: {snorkel_count} (lost {snorkel_loss})")
    logging.info(f"  After Cleanlab: {cleanlab_count} (lost {cleanlab_loss})")
    logging.info(f"  Final training: {final_count} (lost {final_loss})")
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        ),
        link=dict(
            source=[0, 1, 2],  # indices of source nodes
            target=[1, 2, 3],  # indices of target nodes
            value=[snorkel_count, cleanlab_count, final_count],
            color=["rgba(31, 119, 180, 0.4)", "rgba(255, 127, 14, 0.4)", "rgba(44, 160, 44, 0.4)"]
        )
    )])
    
    fig.update_layout(
        title_text="SprintGuard Data Pipeline Flow",
        font_size=12,
        height=400,
        width=700
    )
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        fig.write_image(str(output_path), format='pdf')
        logging.info(f"[PLOTS] Saved Sankey diagram to {output_path}")
        
        # Also save as HTML for interactive viewing
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        logging.info(f"[PLOTS] Saved interactive version to {html_path}")
    except Exception as e:
        logging.error(f"[PLOTS] Failed to save Sankey diagram: {e}")
        logging.info("[PLOTS] Note: Install kaleido for PDF export: pip install kaleido")
    
    return fig


def generate_ablation_study(
    results: Dict[str, Tuple[float, float]],
    output_path: str = "visualizations/ablation_study.pdf"
):
    """
    Generate ablation study bar chart comparing different configurations.
    
    Compares:
    - Baseline (No Weak Supervision)
    - Snorkel Only (No Cleanlab filtering)
    - SprintGuard Full (Snorkel + Cleanlab)
    
    Args:
        results: Dictionary mapping method name to (mean_f1, std_f1)
                Example: {'Baseline': (0.72, 0.03), 'Snorkel Only': (0.78, 0.02), 
                         'SprintGuard': (0.83, 0.02)}
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating ablation study bar chart...")
    
    # Reset style
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    methods = list(results.keys())
    means = [results[m][0] for m in methods]
    stds = [results[m][1] for m in methods]
    
    logging.info(f"  Methods: {methods}")
    for method, mean, std in zip(methods, means, stds):
        logging.info(f"    {method}: {mean:.3f} ± {std:.3f}")
    
    # Create bar chart with error bars
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color='#808080', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Customize plot
    ax.set_ylabel('Macro-F1 Score', fontsize=11)
    ax.set_title('Ablation Study: Impact of Weak Supervision', fontsize=12, pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim([0.6, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Also save as SVG
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved ablation study to {output_path}")
    
    return fig


def generate_lf_correlation_heatmap(
    L_matrix: np.ndarray,
    lf_names: List[str],
    output_path: str = "visualizations/lf_correlation_heatmap.pdf"
):
    """
    Generate correlation heatmap of labeling function votes.
    
    Shows which labeling functions are redundant (high correlation) or 
    complementary (low correlation).
    
    Args:
        L_matrix: Label matrix from Snorkel (n_samples × n_lfs)
        lf_names: Names of labeling functions
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating LF correlation heatmap...")
    
    # Reset style
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    # Calculate correlation matrix
    df = pd.DataFrame(L_matrix, columns=lf_names)
    corr_matrix = df.corr()
    
    logging.info(f"  Correlation matrix shape: {corr_matrix.shape}")
    logging.info(f"  Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Generate heatmap (grayscale for publication)
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap='Greys',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
        vmin=-1,
        vmax=1,
        annot=False  # Too cluttered with 18 LFs
    )
    
    # Customize
    ax.set_title('Labeling Function Correlation Matrix', fontsize=12, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Also save as SVG
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved LF correlation heatmap to {output_path}")
    
    return fig


def generate_calibration_plot(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/calibration_plot.pdf"
):
    """
    Generate calibration plot showing model confidence reliability.
    
    A well-calibrated model should have predicted probabilities that match
    the actual frequency of outcomes.
    
    Args:
        y_true: True class labels (shape: n_samples)
        y_pred_proba: Predicted probabilities (shape: n_samples × n_classes)
        class_names: Names for each class (default: ['Low', 'Medium', 'High'])
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating calibration plot...")
    
    from sklearn.calibration import calibration_curve
    
    # Reset style
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    n_classes = y_pred_proba.shape[1]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Define colorblind-friendly colors
    colors = ['#0173B2', '#DE8F05', '#CC78BC']  # Blue, Orange, Purple
    
    # Plot calibration curve for each class
    for class_idx in range(n_classes):
        # Convert to binary problem
        y_binary = (y_true == class_idx).astype(int)
        
        # Calculate calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, 
                y_pred_proba[:, class_idx],
                n_bins=10,
                strategy='uniform'
            )
            
            # Plot
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=6,
                label=f'{class_names[class_idx]} Risk',
                color=colors[class_idx % len(colors)]
            )
            
            logging.info(f"  {class_names[class_idx]}: ECE = {np.abs(fraction_of_positives - mean_predicted_value).mean():.3f}")
            
        except Exception as e:
            logging.warning(f"  Could not generate calibration curve for {class_names[class_idx]}: {e}")
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration', alpha=0.7)
    
    # Customize
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Calibration Plot: Model Confidence vs. Actual Outcomes', fontsize=12, pad=10)
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Also save as SVG
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved calibration plot to {output_path}")
    
    return fig


def generate_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/embeddings_tsne.pdf",
    sample_size: int = 2000
):
    """
    Generate t-SNE visualization of DistilBERT embeddings.
    
    Shows how well the embedding space separates different risk classes.
    
    Args:
        embeddings: Embedding vectors (shape: n_samples × embedding_dim)
        labels: Class labels (shape: n_samples)
        class_names: Names for each class (default: ['Low', 'Medium', 'High'])
        output_path: Where to save the figure
        sample_size: Number of samples to plot (for speed)
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating t-SNE embedding visualization...")
    
    from sklearn.manifold import TSNE
    
    # Reset style
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    # Sample data if too large
    if len(embeddings) > sample_size:
        logging.info(f"  Sampling {sample_size} points from {len(embeddings)} for visualization")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Compute t-SNE
    logging.info(f"  Computing t-SNE for {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    logging.info(f"  t-SNE complete")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Define colorblind-friendly colors
    colors = ['#0173B2', '#DE8F05', '#CC78BC']  # Blue, Orange, Purple
    
    # Plot each class
    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[class_idx % len(colors)],
            label=f'{class_names[class_idx]} Risk',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        logging.info(f"  {class_names[class_idx]}: {mask.sum()} points")
    
    # Customize
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('t-SNE Visualization of DistilBERT Embeddings', fontsize=12, pad=10)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Also save as SVG
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved t-SNE visualization to {output_path}")
    
    return fig


def generate_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/confusion_matrix.pdf",
    normalize: bool = True
):
    """
    Generate publication-quality confusion matrix heatmap.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: Names for each class (default: ['Low', 'Medium', 'High'])
        output_path: Where to save the figure
        normalize: Whether to show percentages (True) or counts (False)
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating confusion matrix heatmap...")
    
    from sklearn.metrics import confusion_matrix
    
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        cbar_label = 'Percentage (%)'
    else:
        cm_display = cm
        fmt = 'd'
        cbar_label = 'Count'
    
    logging.info(f"  Confusion matrix shape: {cm.shape}")
    logging.info(f"  Overall accuracy: {np.trace(cm) / cm.sum() * 100:.1f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Generate heatmap
    sns.heatmap(
        cm_display,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        linewidths=0.5,
        cbar_kws={'label': cbar_label, 'shrink': 0.8},
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Customize
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=12, pad=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved confusion matrix to {output_path}")
    
    plt.close(fig)
    return fig


def generate_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/roc_curves.pdf"
):
    """
    Generate ROC curves for multi-class classification (one-vs-rest).
    
    Args:
        y_true: True class labels
        y_pred_proba: Predicted probabilities (n_samples × n_classes)
        class_names: Names for each class
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating ROC curves...")
    
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    n_classes = len(class_names)
    
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Colorblind-friendly colors
    colors = ['#0173B2', '#DE8F05', '#CC78BC']
    
    # Plot ROC curve for each class
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_pred_proba[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr,
            color=colors[class_idx % len(colors)],
            linewidth=2,
            label=f'{class_names[class_idx]} (AUC = {roc_auc:.3f})'
        )
        
        logging.info(f"  {class_names[class_idx]}: AUC = {roc_auc:.3f}")
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random Classifier')
    
    # Customize
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=12, pad=10)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved ROC curves to {output_path}")
    
    plt.close(fig)
    return fig


def generate_precision_recall_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/precision_recall_curves.pdf"
):
    """
    Generate Precision-Recall curves for multi-class classification.
    
    Important for imbalanced datasets where ROC can be misleading.
    
    Args:
        y_true: True class labels
        y_pred_proba: Predicted probabilities (n_samples × n_classes)
        class_names: Names for each class
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating Precision-Recall curves...")
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    n_classes = len(class_names)
    
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Colorblind-friendly colors
    colors = ['#0173B2', '#DE8F05', '#CC78BC']
    
    # Plot PR curve for each class
    for class_idx in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, class_idx], 
            y_pred_proba[:, class_idx]
        )
        ap = average_precision_score(y_true_bin[:, class_idx], y_pred_proba[:, class_idx])
        
        ax.plot(
            recall, precision,
            color=colors[class_idx % len(colors)],
            linewidth=2,
            label=f'{class_names[class_idx]} (AP = {ap:.3f})'
        )
        
        logging.info(f"  {class_names[class_idx]}: AP = {ap:.3f}")
    
    # Customize
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves', fontsize=12, pad=10)
    ax.legend(loc='lower left', frameon=True, fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved Precision-Recall curves to {output_path}")
    
    plt.close(fig)
    return fig


def generate_feature_importance_plot(
    model,
    feature_names: List[str],
    output_path: str = "visualizations/feature_importance.pdf",
    top_k: int = 20,
    importance_type: str = 'gain'
):
    """
    Generate feature importance bar chart from XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        output_path: Where to save the figure
        top_k: Number of top features to show
        importance_type: Type of importance ('gain', 'weight', 'cover')
    
    Returns:
        Matplotlib figure object
    """
    logging.info(f"[PLOTS] Generating feature importance plot (top {top_k})...")
    
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    # Get feature importance scores
    importance_dict = model.get_score(importance_type=importance_type)
    
    # Map feature indices to names and sort
    importance_scores = []
    for fname in feature_names:
        score = importance_dict.get(fname, 0)
        importance_scores.append((fname, score))
    
    # Sort by importance and take top k
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    top_features = importance_scores[:top_k]
    
    names = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]
    
    logging.info(f"  Top 5 features:")
    for name, score in top_features[:5]:
        logging.info(f"    {name}: {score:.4f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Horizontal bar chart
    y_pos = np.arange(len(names))
    
    # Color symbolic features differently from embeddings
    colors = ['#D62728' if not name.startswith('embedding_') else '#1F77B4' for name in names]
    
    ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel(f'Importance ({importance_type})', fontsize=11)
    ax.set_title(f'Top {top_k} Feature Importance', fontsize=12, pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend for feature types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D62728', edgecolor='black', label='Symbolic Features'),
        Patch(facecolor='#1F77B4', edgecolor='black', label='BERT Embeddings')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved feature importance to {output_path}")
    
    plt.close(fig)
    return fig


def generate_learning_curves(
    evals_result: Dict[str, Dict[str, List[float]]],
    output_path: str = "visualizations/learning_curves.pdf"
):
    """
    Generate learning curves showing training and validation loss over boosting rounds.
    
    Args:
        evals_result: Dictionary from XGBoost training with format:
                     {'train': {'mlogloss': [...]}, 'val': {'mlogloss': [...]}}
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating learning curves...")
    
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Get metric name (usually 'mlogloss' for multi-class)
    metric_name = list(evals_result.get('train', {}).keys())[0] if evals_result.get('train') else 'mlogloss'
    
    train_scores = evals_result.get('train', {}).get(metric_name, [])
    val_scores = evals_result.get('val', {}).get(metric_name, [])
    
    rounds = list(range(1, len(train_scores) + 1))
    
    logging.info(f"  Metric: {metric_name}")
    logging.info(f"  Rounds: {len(rounds)}")
    logging.info(f"  Final train loss: {train_scores[-1]:.4f}" if train_scores else "  No train data")
    logging.info(f"  Final val loss: {val_scores[-1]:.4f}" if val_scores else "  No val data")
    
    # Plot curves
    if train_scores:
        ax.plot(rounds, train_scores, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    if val_scores:
        ax.plot(rounds, val_scores, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    # Find best validation point
    if val_scores:
        best_round = np.argmin(val_scores) + 1
        best_score = min(val_scores)
        ax.axvline(x=best_round, color='green', linestyle='--', alpha=0.7, 
                   label=f'Best Val (round {best_round})')
        ax.scatter([best_round], [best_score], color='green', s=100, zorder=5, marker='*')
    
    # Customize
    ax.set_xlabel('Boosting Round', fontsize=11)
    ax.set_ylabel(f'Loss ({metric_name})', fontsize=11)
    ax.set_title('Training and Validation Learning Curves', fontsize=12, pad=10)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved learning curves to {output_path}")
    
    plt.close(fig)
    return fig


def generate_shap_summary_plot(
    model,
    X: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/shap_summary.pdf",
    max_display: int = 20
):
    """
    Generate SHAP summary plot (beeswarm) showing global feature importance.
    
    This is the most important SHAP visualization for papers - it shows:
    - Feature importance ranking (y-axis)
    - Impact magnitude (x-axis)  
    - Feature value coloring (red=high, blue=low)
    
    Args:
        model: Trained XGBoost model (Booster or XGBClassifier)
        X: Feature matrix (n_samples × n_features)
        feature_names: List of feature names
        class_names: Names for each class
        output_path: Where to save the figure
        max_display: Maximum features to show
    
    Returns:
        SHAP values object
    """
    import shap
    
    logging.info("[PLOTS] Generating SHAP summary plot...")
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For multi-class, shap_values is a list of arrays (one per class)
    # We'll create a summary for each class
    n_classes = len(shap_values) if isinstance(shap_values, list) else 1
    
    logging.info(f"  Computing SHAP values for {X.shape[0]} samples...")
    logging.info(f"  Number of classes: {n_classes}")
    
    # Create multi-panel figure for multi-class
    if n_classes > 1:
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6))
        
        for class_idx in range(n_classes):
            plt.sca(axes[class_idx])
            shap.summary_plot(
                shap_values[class_idx],
                X,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
                plot_size=None
            )
            axes[class_idx].set_title(f'{class_names[class_idx]} Risk', fontsize=11)
        
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved SHAP summary plot to {output_path}")
    
    plt.close(fig)
    return shap_values


def generate_shap_bar_plot(
    model,
    X: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/shap_importance.pdf",
    max_display: int = 20
):
    """
    Generate SHAP bar plot showing mean absolute SHAP values.
    
    Simpler than summary plot - good for showing pure feature importance
    without the distribution details.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix (n_samples × n_features)
        feature_names: List of feature names
        class_names: Names for each class
        output_path: Where to save the figure
        max_display: Maximum features to show
    
    Returns:
        Dictionary of mean absolute SHAP values per class
    """
    import shap
    
    logging.info("[PLOTS] Generating SHAP bar plot...")
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    n_classes = len(shap_values) if isinstance(shap_values, list) else 1
    
    # Calculate mean absolute SHAP values
    mean_shap = {}
    for class_idx in range(n_classes):
        sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
        mean_shap[class_names[class_idx]] = np.abs(sv).mean(axis=0)
    
    # Create figure
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Average across all classes for overall importance
    all_class_mean = np.mean([v for v in mean_shap.values()], axis=0)
    
    # Get top features
    top_indices = np.argsort(all_class_mean)[-max_display:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_values = all_class_mean[top_indices]
    
    # Color by feature type
    colors = ['#D62728' if not name.startswith('embedding_') else '#1F77B4' 
              for name in top_names]
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title('Feature Importance (SHAP)', fontsize=12, pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D62728', edgecolor='black', label='Symbolic Features'),
        Patch(facecolor='#1F77B4', edgecolor='black', label='BERT Embeddings')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved SHAP bar plot to {output_path}")
    
    # Log top features
    logging.info("  Top 5 features by SHAP importance:")
    for name, val in zip(top_names[:5], top_values[:5]):
        logging.info(f"    {name}: {val:.4f}")
    
    plt.close(fig)
    return mean_shap


def generate_shap_waterfall_plot(
    model,
    X_single: np.ndarray,
    feature_names: List[str],
    class_idx: int = 2,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/shap_waterfall.pdf",
    max_display: int = 15
):
    """
    Generate SHAP waterfall plot for a single prediction.
    
    Shows how each feature contributes to pushing the prediction
    from the base value (average) to the final output.
    
    Args:
        model: Trained XGBoost model
        X_single: Single sample feature vector (1 × n_features)
        feature_names: List of feature names
        class_idx: Which class to explain (0=Low, 1=Medium, 2=High)
        class_names: Names for each class
        output_path: Where to save the figure
        max_display: Maximum features to show
    
    Returns:
        SHAP Explanation object
    """
    import shap
    
    logging.info(f"[PLOTS] Generating SHAP waterfall plot for class {class_idx}...")
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Ensure X_single is 2D
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    shap_values = explainer.shap_values(X_single)
    
    # Get values for specified class
    sv = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0]
    base_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=X_single[0],
        feature_names=feature_names
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    
    plt.title(f'SHAP Explanation for {class_names[class_idx]} Risk Prediction', fontsize=12, pad=20)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved SHAP waterfall plot to {output_path}")
    
    plt.close(fig)
    return explanation


def generate_shap_dependence_plot(
    model,
    X: np.ndarray,
    feature_names: List[str],
    feature_name: str,
    interaction_feature: Optional[str] = None,
    class_idx: int = 2,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/shap_dependence.pdf"
):
    """
    Generate SHAP dependence plot showing feature effect on predictions.
    
    Shows the relationship between a feature's value and its SHAP value,
    optionally colored by an interaction feature.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix (n_samples × n_features)
        feature_names: List of feature names
        feature_name: Feature to plot on x-axis
        interaction_feature: Feature to color by (auto-detected if None)
        class_idx: Which class to explain
        class_names: Names for each class
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure
    """
    import shap
    
    logging.info(f"[PLOTS] Generating SHAP dependence plot for '{feature_name}'...")
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    # Find feature index
    try:
        feature_idx = feature_names.index(feature_name)
    except ValueError:
        logging.error(f"  Feature '{feature_name}' not found in feature_names")
        return None
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get values for specified class
    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    
    # Create figure
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Determine interaction feature
    if interaction_feature is not None:
        try:
            interaction_idx = feature_names.index(interaction_feature)
        except ValueError:
            interaction_idx = "auto"
    else:
        interaction_idx = "auto"
    
    shap.dependence_plot(
        feature_idx,
        sv,
        X,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        ax=ax,
        show=False
    )
    
    ax.set_title(f'SHAP Dependence: {feature_name} ({class_names[class_idx]} Risk)', fontsize=12, pad=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved SHAP dependence plot to {output_path}")
    
    plt.close(fig)
    return fig


def generate_shap_force_plot(
    model,
    X_single: np.ndarray,
    feature_names: List[str],
    class_idx: int = 2,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/shap_force.html"
):
    """
    Generate SHAP force plot (interactive HTML).
    
    Shows how features push the prediction from base value to final output.
    Best viewed in browser - provides interactive tooltips.
    
    Args:
        model: Trained XGBoost model
        X_single: Single sample feature vector (1 × n_features)
        feature_names: List of feature names
        class_idx: Which class to explain
        class_names: Names for each class
        output_path: Where to save the HTML file
    
    Returns:
        SHAP force plot object
    """
    import shap
    
    logging.info(f"[PLOTS] Generating SHAP force plot for class {class_idx}...")
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Ensure X_single is 2D
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    shap_values = explainer.shap_values(X_single)
    
    # Get values for specified class
    sv = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0]
    base_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    # Initialize JS visualization
    shap.initjs()
    
    # Create force plot
    force_plot = shap.force_plot(
        base_value,
        sv,
        X_single[0],
        feature_names=feature_names,
        show=False
    )
    
    # Save as HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    shap.save_html(str(output_path), force_plot)
    
    logging.info(f"[PLOTS] Saved SHAP force plot to {output_path}")
    
    return force_plot


def generate_class_distribution_plot(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "visualizations/class_distribution.pdf"
):
    """
    Generate grouped bar chart showing class distribution across splits.
    
    Args:
        train_labels: Training set labels
        val_labels: Validation set labels
        test_labels: Test set labels
        class_names: Names for each class
        output_path: Where to save the figure
    
    Returns:
        Matplotlib figure object
    """
    logging.info("[PLOTS] Generating class distribution plot...")
    
    if SCIENCE_PLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])
    
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    n_classes = len(class_names)
    
    # Count samples per class in each split
    train_counts = [np.sum(train_labels == i) for i in range(n_classes)]
    val_counts = [np.sum(val_labels == i) for i in range(n_classes)]
    test_counts = [np.sum(test_labels == i) for i in range(n_classes)]
    
    logging.info(f"  Train: {dict(zip(class_names, train_counts))}")
    logging.info(f"  Val: {dict(zip(class_names, val_counts))}")
    logging.info(f"  Test: {dict(zip(class_names, test_counts))}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(n_classes)
    width = 0.25
    
    bars1 = ax.bar(x - width, train_counts, width, label='Train', color='#0173B2', edgecolor='black')
    bars2 = ax.bar(x, val_counts, width, label='Validation', color='#DE8F05', edgecolor='black')
    bars3 = ax.bar(x + width, test_counts, width, label='Test', color='#CC78BC', edgecolor='black')
    
    # Add count labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    # Customize
    ax.set_xlabel('Risk Class', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Class Distribution Across Data Splits', fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    logging.info(f"[PLOTS] Saved class distribution to {output_path}")
    
    plt.close(fig)
    return fig

