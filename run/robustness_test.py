"""
Robustness Evaluation Suite for MSA Research.

Tests model performance under text degradation to prove that
modality-balanced models are more robust than text-dominant ones.

This produces the KEY results table for the paper:
- Accuracy vs. noise level curves
- Degradation comparison across models
- Missing modality robustness

Usage:
    python run/robustness_test.py --model_path models/robust_hybrid_best.pt --data_path data/unaligned_50.pkl
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.dataloader import getdataloader
from models.msamodel import MSAModel
from src.eval_metrics import eval_senti


# ============================================================
# TEXT CORRUPTION FUNCTIONS
# ============================================================

def add_gaussian_noise(text_tensor, noise_ratio=0.3, noise_std=1.0):
    """Add Gaussian noise to a random subset of text embedding dimensions.
    
    Simulates ASR feature-level errors where extracted features are noisy.
    
    Args:
        text_tensor: [Batch, SeqLen, Dim] text embeddings
        noise_ratio: fraction of dimensions to corrupt (0.0 to 1.0)
        noise_std: standard deviation of Gaussian noise
    
    Returns:
        corrupted text tensor
    """
    mask = (torch.rand_like(text_tensor) < noise_ratio).float()
    noise = torch.randn_like(text_tensor) * noise_std
    return text_tensor + mask * noise


def token_dropout(text_tensor, drop_ratio=0.3):
    """Zero out random time steps in the text sequence.
    
    Simulates missing or garbled words (e.g., ASR drops, incomplete transcripts).
    
    Args:
        text_tensor: [Batch, SeqLen, Dim]
        drop_ratio: fraction of timesteps to zero out
    
    Returns:
        corrupted text tensor
    """
    B, T, D = text_tensor.shape
    mask = (torch.rand(B, T, 1, device=text_tensor.device) > drop_ratio).float()
    return text_tensor * mask


def shuffle_tokens(text_tensor, shuffle_ratio=0.3):
    """Randomly shuffle a fraction of time steps in the text sequence.
    
    Simulates word order errors from ASR or misalignment.
    
    Args:
        text_tensor: [Batch, SeqLen, Dim]
        shuffle_ratio: fraction of timesteps to shuffle
    
    Returns:
        corrupted text tensor
    """
    B, T, D = text_tensor.shape
    result = text_tensor.clone()
    n_shuffle = max(1, int(T * shuffle_ratio))
    for b in range(B):
        indices = torch.randperm(T, device=text_tensor.device)[:n_shuffle]
        shuffled = indices[torch.randperm(n_shuffle, device=text_tensor.device)]
        result[b, indices] = text_tensor[b, shuffled]
    return result


# ============================================================
# EVALUATION UNDER CORRUPTION
# ============================================================

def evaluate_with_corruption(model, loader, device, corruption_fn=None, corruption_kwargs=None):
    """Evaluate model with optional text corruption applied at test time.
    
    Args:
        model: MSAModel
        loader: test DataLoader
        device: torch device
        corruption_fn: function to apply to text tensor (None = clean)
        corruption_kwargs: dict of kwargs for corruption_fn
    
    Returns:
        metrics dict
    """
    model.eval()
    all_preds, all_truths = [], []
    all_gates = []
    
    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            y = batch['labels']
            
            # Apply text corruption
            if corruption_fn is not None:
                text = corruption_fn(text, **(corruption_kwargs or {}))
            
            outputs = model([text, audio, vision])
            preds_logits = outputs['output']
            preds_class = torch.argmax(preds_logits, dim=1)
            preds_reg = preds_class.cpu().float() - 3
            
            all_preds.append(preds_reg)
            all_truths.append(y.squeeze(-1))
            
            if 'gate_weights' in outputs and outputs['gate_weights'] is not None:
                all_gates.append(outputs['gate_weights'])
    
    all_preds = torch.cat(all_preds)
    all_truths = torch.cat(all_truths)
    metrics = eval_senti(all_preds, all_truths, exclude_zero=True)
    
    if len(all_gates) > 0:
        avg_gates = torch.cat(all_gates, dim=0).mean(dim=0).cpu().numpy()
        metrics['gate_text'] = float(avg_gates[0])
        metrics['gate_audio'] = float(avg_gates[1])
        metrics['gate_vision'] = float(avg_gates[2])
    
    return metrics


def run_noise_sweep(model, loader, device, noise_type='gaussian', levels=None):
    """Run evaluation at multiple noise levels and return results.
    
    Args:
        model: MSAModel
        loader: test DataLoader
        device: torch device
        noise_type: 'gaussian', 'token_dropout', 'shuffle', or 'full_dropout'
        levels: list of noise intensities to test
    
    Returns:
        list of (level, metrics_dict) tuples
    """
    if levels is None:
        levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    corruption_fns = {
        'gaussian': (add_gaussian_noise, 'noise_ratio'),
        'token_dropout': (token_dropout, 'drop_ratio'),
        'shuffle': (shuffle_tokens, 'shuffle_ratio'),
    }
    
    results = []
    for level in levels:
        if level == 0.0:
            # Clean evaluation
            metrics = evaluate_with_corruption(model, loader, device)
        elif noise_type == 'full_dropout':
            # Complete modality removal: zero out text
            def zero_text(t, **kwargs):
                return torch.zeros_like(t)
            metrics = evaluate_with_corruption(model, loader, device, zero_text)
        else:
            fn, param_name = corruption_fns[noise_type]
            metrics = evaluate_with_corruption(
                model, loader, device, fn, {param_name: level}
            )
        
        results.append((level, metrics))
        gate_str = ""
        if 'gate_text' in metrics:
            gate_str = f"| Gates: T={metrics['gate_text']:.2f} A={metrics['gate_audio']:.2f} V={metrics['gate_vision']:.2f}"
            
        print(f"  {noise_type} @ {level:.0%}: Acc-2={metrics['acc2']:.4f} | "
              f"F1={metrics['f1']:.4f} | MAE={metrics['mae']:.4f} {gate_str}")
    
    return results


def plot_robustness_curves(all_results, output_dir):
    """Plot accuracy vs noise level curves — THE key figure for the paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'figure.dpi': 150, 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics_to_plot = ['acc2', 'f1', 'mae']
    titles = ['Binary Accuracy (Acc-2) ↑', 'F1 Score ↑', 'MAE ↓']
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        for idx, (model_name, noise_type, results) in enumerate(all_results):
            levels = [r[0] for r in results]
            values = [r[1][metric] for r in results]
            ax.plot(levels, values, '-o', color=colors[idx % len(colors)],
                    linewidth=2.5, markersize=6, label=f'{model_name}')
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Text Corruption Level', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'robustness_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


def compute_degradation_table(all_results):
    """Compute degradation ratios for each model — the proof of robustness."""
    import pandas as pd
    
    rows = []
    for model_name, noise_type, results in all_results:
        clean_acc = results[0][1]['acc2']
        levels = [r[0] for r in results]
        acc_values = [r[1]['acc2'] for r in results]
        audc = np.trapz(acc_values, levels)
        
        for level, metrics in results:
            degradation = clean_acc - metrics['acc2']
            deg_pct = (degradation / (clean_acc + 1e-8)) * 100
            rows.append({
                'Model': model_name,
                'Noise': noise_type,
                'Level': f"{level:.0%}",
                'Acc-2': f"{metrics['acc2']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'MAE': f"{metrics['mae']:.4f}",
                'Degradation': f"{degradation:+.4f}",
                'Deg%': f"{deg_pct:.1f}%",
                'AUDC': f"{audc:.4f}" if level == 0.0 else ""
            })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Robustness Evaluation Suite')
    parser.add_argument('--model_path', type=str, default='models/robust_hybrid_best.pt',
                        help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='Ours (Balanced)',
                        help='Name for the model in result tables')
    parser.add_argument('--data_path', type=str, default='data/unaligned_50.pkl')
    parser.add_argument('--dataset', type=str, default='mosi')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--output_dir', type=str, default='figures')
    parser.add_argument('--noise_types', type=str, nargs='+',
                        default=['gaussian', 'token_dropout'],
                        choices=['gaussian', 'token_dropout', 'shuffle'])
    parser.add_argument('--levels', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    # Optional: compare against a text-dominant baseline
    parser.add_argument('--baseline_path', type=str, default=None,
                        help='Path to a text-dominant baseline model for comparison')
    parser.add_argument('--baseline_name', type=str, default='Text-Dominant Baseline')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load data
    dataloader, orig_dim = getdataloader(args.dataset, args.batch_size, args.data_path)
    test_loader = dataloader['test']
    
    # Load model
    print(f"\nLoading model: {args.model_path}")
    model = MSAModel(output_dim=7, orig_dim=orig_dim, proj_dim=40, layers=5).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()
    
    # Collect all results for plotting
    all_results = []
    
    for noise_type in args.noise_types:
        print(f"\n{'='*60}")
        print(f"  NOISE SWEEP: {noise_type} — Model: {args.model_name}")
        print(f"{'='*60}")
        results = run_noise_sweep(model, test_loader, device, noise_type, args.levels)
        all_results.append((args.model_name, noise_type, results))
    
    # Test with complete text removal
    print(f"\n{'='*60}")
    print(f"  MODALITY ABLATION: Text completely removed")
    print(f"{'='*60}")
    text_off_metrics = evaluate_with_corruption(
        model, test_loader, device,
        lambda t, **kw: torch.zeros_like(t)
    )
    print(f"  Text-Off: Acc-2={text_off_metrics['acc2']:.4f} | "
          f"F1={text_off_metrics['f1']:.4f} | MAE={text_off_metrics['mae']:.4f}")
    
    # Optional: compare against baseline
    if args.baseline_path and os.path.exists(args.baseline_path):
        print(f"\nLoading baseline: {args.baseline_path}")
        baseline = MSAModel(output_dim=7, orig_dim=orig_dim, proj_dim=40, layers=5).to(device)
        baseline.load_state_dict(torch.load(args.baseline_path, map_location=device), strict=False)
        baseline.eval()
        
        for noise_type in args.noise_types:
            print(f"\n  BASELINE NOISE SWEEP: {noise_type}")
            results = run_noise_sweep(baseline, test_loader, device, noise_type, args.levels)
            all_results.append((args.baseline_name, noise_type, results))
    
    # Generate plots
    print(f"\nGenerating robustness plots...")
    # Group by noise type for plotting
    for noise_type in args.noise_types:
        type_results = [(name, nt, res) for name, nt, res in all_results if nt == noise_type]
        if type_results:
            plot_robustness_curves(type_results, args.output_dir)
    
    # Generate degradation table
    deg_table = compute_degradation_table(all_results)
    print(f"\n{'='*80}")
    print(f"  DEGRADATION ANALYSIS")
    print(f"{'='*80}")
    print(deg_table.to_string(index=False))
    
    # Save results
    save_data = {
        'model_path': args.model_path,
        'model_name': args.model_name,
        'text_off_metrics': text_off_metrics,
        'noise_sweeps': [
            {'model': name, 'noise_type': nt,
             'results': [(l, m) for l, m in res]}
            for name, nt, res in all_results
        ]
    }
    save_path = os.path.join('logs', 'robustness_results.json')
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {save_path}")
    print(f"Figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
