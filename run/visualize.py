"""
Visualization suite for MSA Research.
Generates publication-quality plots from training logs and model outputs.

Usage:
    python run/visualize.py --log_dir logs --output_dir figures
    python run/visualize.py --log_dir logs --output_dir figures --model_path models/robust_hybrid_best.pt --data_path data/unaligned_50.pkl --tsne
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality defaults
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {'text': '#E74C3C', 'audio': '#3498DB', 'vision': '#2ECC71', 'full': '#9B59B6'}


def load_log(log_path):
    """Load a training log JSON file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_training_curves(log, output_dir):
    """Plot training loss curves (main, gate, contrastive) over epochs."""
    epochs_data = log['epochs']
    epochs = [e['epoch'] for e in epochs_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Main loss
    if 'train_loss_main' in epochs_data[0]:
        axes[0].plot(epochs, [e['train_loss_main'] for e in epochs_data], 'b-', linewidth=2)
        axes[0].set_title('Main Classification Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
    elif 'train_loss' in epochs_data[0]:
        axes[0].plot(epochs, [e['train_loss'] for e in epochs_data], 'b-', linewidth=2)
        axes[0].set_title('Training Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
    
    # Gate loss
    if 'train_loss_gate' in epochs_data[0]:
        axes[1].plot(epochs, [e['train_loss_gate'] for e in epochs_data], 'r-', linewidth=2)
        axes[1].set_title('Auxiliary Gate Loss', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
    
    # Contrastive loss
    if 'train_loss_contrastive' in epochs_data[0]:
        axes[2].plot(epochs, [e['train_loss_contrastive'] for e in epochs_data], 'g-', linewidth=2)
        axes[2].set_title('Contrastive Loss (InfoNCE)', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_losses.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_validation_metrics(log, output_dir):
    """Plot validation metrics over epochs."""
    epochs_data = log['epochs']
    epochs = [e['epoch'] for e in epochs_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    if 'val_acc2' in epochs_data[0]:
        axes[0].plot(epochs, [e['val_acc2'] for e in epochs_data], 'b-o', linewidth=2, markersize=4, label='Acc-2')
    if 'val_acc7' in epochs_data[0]:
        axes[0].plot(epochs, [e['val_acc7'] for e in epochs_data], 'r-s', linewidth=2, markersize=4, label='Acc-7')
    axes[0].set_title('Validation Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # MAE
    if 'val_mae' in epochs_data[0]:
        axes[1].plot(epochs, [e['val_mae'] for e in epochs_data], 'g-^', linewidth=2, markersize=4)
        axes[1].set_title('Validation MAE', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
    
    # F1
    if 'val_f1' in epochs_data[0]:
        axes[2].plot(epochs, [e['val_f1'] for e in epochs_data], 'm-D', linewidth=2, markersize=4)
        axes[2].set_title('Validation F1 Score', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'validation_metrics.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_gradient_norms(log, output_dir):
    """Plot per-modality gradient norms over epochs — visual proof of OGM-GE working."""
    epochs_data = log['epochs']
    if 'avg_grad_norm_text' not in epochs_data[0]:
        print("  Skipping gradient norms: not in log.")
        return
    
    epochs = [e['epoch'] for e in epochs_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [e['avg_grad_norm_text'] for e in epochs_data], 
            color=COLORS['text'], linewidth=2.5, label='Text Encoder', marker='o', markersize=4)
    ax.plot(epochs, [e['avg_grad_norm_audio'] for e in epochs_data], 
            color=COLORS['audio'], linewidth=2.5, label='Audio Encoder', marker='s', markersize=4)
    ax.plot(epochs, [e['avg_grad_norm_vision'] for e in epochs_data], 
            color=COLORS['vision'], linewidth=2.5, label='Vision Encoder', marker='^', markersize=4)
    
    ax.set_title('Per-Modality Gradient Norms During Training', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Gradient L2 Norm', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.fill_between(epochs, 0, [e['avg_grad_norm_text'] for e in epochs_data],
                     color=COLORS['text'], alpha=0.05)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'gradient_norms.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_gate_weights(log, output_dir):
    """Plot gate weight distribution over training — shows how fusion balances modalities."""
    epochs_data = log['epochs']
    if 'avg_gate_text' not in epochs_data[0]:
        print("  Skipping gate weights: not in log.")
        return
    
    epochs = [e['epoch'] for e in epochs_data]
    g_t = [e['avg_gate_text'] for e in epochs_data]
    g_a = [e['avg_gate_audio'] for e in epochs_data]
    g_v = [e['avg_gate_vision'] for e in epochs_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line plot over time
    ax1.plot(epochs, g_t, color=COLORS['text'], linewidth=2.5, label='Text', marker='o', markersize=4)
    ax1.plot(epochs, g_a, color=COLORS['audio'], linewidth=2.5, label='Audio', marker='s', markersize=4)
    ax1.plot(epochs, g_v, color=COLORS['vision'], linewidth=2.5, label='Vision', marker='^', markersize=4)
    ax1.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Uniform (1/3)')
    ax1.set_title('Gate Weights Over Training', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Gate Weight')
    ax1.legend()
    ax1.set_ylim(0, 0.8)
    
    # Final epoch bar chart
    final_gates = [g_t[-1], g_a[-1], g_v[-1]]
    bars = ax2.bar(['Text', 'Audio', 'Vision'], final_gates,
                    color=[COLORS['text'], COLORS['audio'], COLORS['vision']],
                    edgecolor='white', linewidth=2, width=0.5)
    ax2.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Uniform')
    ax2.set_title(f'Final Gate Weights (Epoch {epochs[-1]})', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Gate Weight')
    ax2.set_ylim(0, 0.7)
    ax2.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars, final_gates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'gate_weights.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_adaptive_dropout(log, output_dir):
    """Plot adaptive text dropout probability over training."""
    epochs_data = log['epochs']
    if 'text_dropout_p' not in epochs_data[0]:
        print("  Skipping adaptive dropout: not in log.")
        return
    
    epochs = [e['epoch'] for e in epochs_data]
    dropout_p = [e['text_dropout_p'] for e in epochs_data]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, dropout_p, color='#E67E22', linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(epochs, 0, dropout_p, color='#E67E22', alpha=0.15)
    ax.set_title('Adaptive Text Modality Dropout Probability', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dropout Probability')
    ax.set_ylim(0, 0.55)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'adaptive_dropout.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_modality_ablation_bars(results, output_dir):
    """Bar chart showing accuracy for each modality ablation condition."""
    conditions = list(results.keys())
    labels = {'full': 'Full (T+A+V)', 'text': 'Text Only',
              'audio': 'Audio Only', 'vision': 'Vision Only'}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = [('acc2', 'Binary Accuracy (Acc-2)'), 
                       ('f1', 'F1 Score'),
                       ('mae', 'MAE (lower is better)')]
    
    for ax, (metric, title) in zip(axes, metrics_to_plot):
        values = [results[c][metric] for c in conditions]
        colors = [COLORS.get(c, '#95A5A6') for c in conditions]
        bars = ax.bar([labels[c] for c in conditions], values,
                       color=colors, edgecolor='white', linewidth=2, width=0.5)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_ylabel(metric.upper())
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'modality_ablation.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_tsne(model_stage2, model_stage3, loader, device, output_dir):
    """t-SNE visualization of latent representations per modality (Before vs After Stage 3)."""
    from sklearn.manifold import TSNE
    import torch
    
    def extract_features(model):
        model.eval()
        all_h = {0: [], 1: [], 2: []}
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                text, audio, vision, y = batch['text'], batch['audio'], batch['vision'], batch['labels']
                text, audio, vision = text.to(device), audio.to(device), vision.to(device)
                
                outputs = model([text, audio, vision])
                hs = outputs['hs_detached']
                for i in range(3):
                    all_h[i].append(hs[i][0].cpu().numpy())
                all_labels.append(y.squeeze(-1).numpy())
        
        for i in range(3):
            all_h[i] = np.concatenate(all_h[i], axis=0)
        return all_h, np.concatenate(all_labels, axis=0)

    print("Extracting features for Stage 2 (Before Alignment)...")
    h2, labels2 = extract_features(model_stage2)
    print("Extracting features for Stage 3 (After Alignment)...")
    h3, labels3 = extract_features(model_stage3)
    
    binary_labels = (labels2 > 0).astype(int)  # 0=negative, 1=positive
    
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    modality_names = ['Text', 'Audio', 'Vision']
    mod_colors = [COLORS['text'], COLORS['audio'], COLORS['vision']]
    row_titles = ['Stage 2: Before Contrastive Alignment', 'Stage 3: After Contrastive Alignment']
    
    for row_idx, (h_dict, row_title) in enumerate([(h2, row_titles[0]), (h3, row_titles[1])]):
        for col_idx, (name, color) in enumerate(zip(modality_names, mod_colors)):
            ax = axes[row_idx, col_idx]
            
            n = min(500, len(h_dict[col_idx]))
            indices = np.random.choice(len(h_dict[col_idx]), n, replace=False)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            embedded = tsne.fit_transform(h_dict[col_idx][indices])
            
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                                c=binary_labels[indices], cmap='RdYlGn',
                                alpha=0.6, s=20, edgecolors='white', linewidths=0.3)
            ax.set_title(f'{name} ({row_title})', fontweight='bold', fontsize=13)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            if col_idx == 2:
                plt.colorbar(scatter, ax=ax, label='Sentiment (0=Neg, 1=Pos)')
    

    
    plt.tight_layout()
    path = os.path.join(output_dir, 'tsne_representations.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='MSA Visualization Suite')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing training logs')
    parser.add_argument('--output_dir', type=str, default='figures', help='Directory to save figures')
    parser.add_argument('--stage', type=int, default=3, help='Which stage log to visualize')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tsne', action='store_true', help='Generate t-SNE plots (requires model + data)')
    parser.add_argument('--model_stage2', type=str, default='models/students_distilled_aligned.pt')
    parser.add_argument('--model_stage3', type=str, default='models/robust_hybrid_best.pt')
    parser.add_argument('--data_path', type=str, default='data/unaligned_50.pkl')
    parser.add_argument('--dataset', type=str, default='mosi')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and plot from training logs
    log_file = os.path.join(args.log_dir, f'training_log_stage{args.stage}_seed{args.seed}.json')
    if os.path.exists(log_file):
        print(f"\nLoading training log: {log_file}")
        log = load_log(log_file)
        
        print("Generating plots...")
        plot_training_curves(log, args.output_dir)
        plot_validation_metrics(log, args.output_dir)
        plot_gradient_norms(log, args.output_dir)
        plot_gate_weights(log, args.output_dir)
        plot_adaptive_dropout(log, args.output_dir)
    else:
        print(f"Warning: Log file not found: {log_file}")
        print("Run training first to generate logs.")
    
    # t-SNE visualization
    if args.tsne:
        import torch
        from datasets.dataloader import getdataloader
        from models.msamodel import MSAModel
        
        use_cuda = torch.cuda.is_available() and not args.no_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        
        if os.path.exists(args.model_stage2) and os.path.exists(args.model_stage3):
            print(f"\nGenerating t-SNE comparing: {args.model_stage2} vs {args.model_stage3}")
            dataloader, orig_dim = getdataloader(args.dataset, 32, args.data_path)
            model2 = MSAModel(output_dim=7, orig_dim=orig_dim, proj_dim=40, layers=5).to(device)
            model2.load_state_dict(torch.load(args.model_stage2, map_location=device), strict=False)
            model3 = MSAModel(output_dim=7, orig_dim=orig_dim, proj_dim=40, layers=5).to(device)
            model3.load_state_dict(torch.load(args.model_stage3, map_location=device), strict=False)
            plot_tsne(model2, model3, dataloader['test'], device, args.output_dir)
        else:
            print(f"Warning: Models not found for t-SNE comparison")
    
    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
