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

