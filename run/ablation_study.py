import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd

# Add project root to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.dataloader import getdataloader
from models.msamodel import MSAModel
from src.eval_metrics import eval_senti

def token_dropout_ablation(text_tensor, drop_ratio=0.5):
    """Zero out random time steps in the text sequence."""
    B, T, D = text_tensor.shape
    mask = (torch.rand(B, T, 1, device=text_tensor.device) > drop_ratio).float()
    return text_tensor * mask


def run_ablation(model, loader, hyp_params, mode):
    """Run evaluation in a specific ablation mode.
    
    Args:
        model: the MSAModel
        loader: test dataloader
        hyp_params: namespace with use_cuda etc.
        mode: 'full', 'text', 'audio', or 'vision'
    
    Returns:
        metrics dict with acc2, acc7, f1, mae, corr
    """
    model.eval()
    all_preds_reg, all_truths_reg = [], []
    device = torch.device("cuda" if hyp_params.use_cuda else "cpu")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Ablation ({mode})", leave=False):
            text, audio, vision, y = batch['text'], batch['audio'], batch['vision'], batch['labels']
            if hyp_params.use_cuda:
                text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
            
            if mode == 'full':
                output_dict = model([text, audio, vision])
                preds_logits = output_dict['output']
            elif mode == 'full_robust':
                text_corrupted = token_dropout_ablation(text, drop_ratio=0.5)
                output_dict = model([text_corrupted, audio, vision])
                preds_logits = output_dict['output']
            else:
                mode_map = {'text': 0, 'audio': 1, 'vision': 2}
                output_dict = model([text, audio, vision], unimodal_mode=mode_map[mode])
                preds_logits = output_dict['unimodal_output']

            preds_class = torch.argmax(preds_logits, dim=1)
            preds_reg_style = preds_class.cpu().float() - 3
            all_preds_reg.append(preds_reg_style)
            all_truths_reg.append(y.cpu().squeeze(-1))

    all_preds_reg = torch.cat(all_preds_reg)
    all_truths_reg = torch.cat(all_truths_reg)
    metrics = eval_senti(all_preds_reg, all_truths_reg, exclude_zero=True)
    return metrics


def run_technique_ablation(hyp_params, orig_dim, dataloaders, device):
    """Run 2^3 factorial ablation over {Dropout, OGM, Contrastive}.
    
    This trains Stage 3 from scratch for each combination and evaluates.
    NOTE: This is computationally expensive. For quick evaluation, 
    use pre-trained models with run_modality_ablation instead.
    
    Returns a list of result dicts.
    """
    import json
    
    combinations = [
        {'dropout': False, 'ogm': False, 'contrastive': False, 'name': 'Base (no tricks)'},
        {'dropout': True,  'ogm': False, 'contrastive': False, 'name': '+ Dropout'},
        {'dropout': False, 'ogm': True,  'contrastive': False, 'name': '+ OGM-GE'},
        {'dropout': False, 'ogm': False, 'contrastive': True,  'name': '+ Contrastive'},
        {'dropout': True,  'ogm': True,  'contrastive': False, 'name': '+ Dropout + OGM'},
        {'dropout': True,  'ogm': False, 'contrastive': True,  'name': '+ Dropout + Contr'},
        {'dropout': False, 'ogm': True,  'contrastive': True,  'name': '+ OGM + Contr'},
        {'dropout': True,  'ogm': True,  'contrastive': True,  'name': 'Full (All Three)'},
    ]
    
    print("\n" + "=" * 70)
    print("  2^3 FACTORIAL ABLATION: {Dropout, OGM-GE, Contrastive}")
    print("  This will train 8 model variants. This takes a while.")
    print("=" * 70)
    
    results = []
    for i, combo in enumerate(combinations):
        print(f"\n--- [{i+1}/8] Training: {combo['name']} ---")
        
        # Build command args for this combination
        cmd_parts = [
            'python', 'run/train.py', '--stage', '3',
            '--data_path', hyp_params.data_path,
            '--dataset', hyp_params.dataset,
            '--final_model_path', f'models/ablation_{i}.pt',
            '--num_epochs', str(hyp_params.num_epochs),
            '--seed', str(hyp_params.seed),
            '--log_dir', 'logs',
        ]
        if combo['dropout']:
            cmd_parts += ['--modality_dropout', '0.15', '--use_adaptive_dropout']
        if combo['ogm']:
            cmd_parts.append('--use_ogm')
        if combo['contrastive']:
            cmd_parts.append('--use_contrastive')
        if not hyp_params.use_cuda:
            cmd_parts.append('--no_cuda')
        
        import subprocess
        try:
            subprocess.check_call(cmd_parts)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Training failed for {combo['name']}: {e}")
            results.append({**combo, 'acc2': 0, 'acc7': 0, 'f1': 0, 'mae': 999, 'corr': 0, 'robust_acc2': 0})
            continue
        
        # Load and evaluate
        model = MSAModel(
            output_dim=hyp_params.num_classes, orig_dim=orig_dim,
            proj_dim=hyp_params.proj_dim, layers=hyp_params.nlevels
        ).to(device)
        model.load_state_dict(torch.load(f'models/ablation_{i}.pt', map_location=device), strict=False)
        
        metrics = run_ablation(model, dataloaders['test'], hyp_params, 'full')
        robust_metrics = run_ablation(model, dataloaders['test'], hyp_params, 'full_robust')
        metrics['robust_acc2'] = robust_metrics['acc2']
        results.append({**combo, **metrics})
        print(f"  Result: Acc-2={metrics['acc2']:.4f} | Acc-7={metrics['acc7']:.4f} | "
              f"F1={metrics['f1']:.4f} | MAE={metrics['mae']:.4f} | Robust-Acc2={metrics['robust_acc2']:.4f}")
    
    return results


def main():
    print("\n" + "=" * 70)
    print("     COMPREHENSIVE ABLATION STUDY")
    print("=" * 70)

    parser = argparse.ArgumentParser(description='Ablation Study Script')
    parser.add_argument('--data_path', type=str, default='data/unaligned_50.pkl')
    parser.add_argument('--dataset', type=str, default='mosi')
    parser.add_argument('--final_model_path', type=str, default='models/robust_hybrid_best.pt')
    parser.add_argument('--student_model_path', type=str, default='models/students_distilled_aligned.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--run_technique_ablation', action='store_true',
                        help='Run full 2^3 factorial ablation (trains 8 models, very slow)')
    parser.add_argument('--num_epochs', type=int, default=40, help='Epochs for technique ablation training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--proj_dim', type=int, default=40)
    parser.add_argument('--nlevels', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=7)
    args = parser.parse_args()

    hyp_params = argparse.Namespace(
        dataset=args.dataset, num_classes=args.num_classes, nlevels=args.nlevels,
        num_heads=5, proj_dim=args.proj_dim,
        attn_dropout=0.15, relu_dropout=0.15, embed_dropout=0.2, 
        res_dropout=0.1, out_dropout=0.1,
        use_cuda=torch.cuda.is_available() and not args.no_cuda,
        data_path=args.data_path, seed=args.seed, num_epochs=args.num_epochs
    )
    hyp_params.output_dim = hyp_params.num_classes
    hyp_params.num_mod = 3
    device = torch.device("cuda" if hyp_params.use_cuda else "cpu")

    dataloader, orig_dim = getdataloader(args.dataset, args.batch_size, args.data_path)
    test_loader = dataloader['test']
    hyp_params.orig_dim = orig_dim
    hyp_params.layers = hyp_params.nlevels

    # =============================================
    # PART 1: Modality Ablation on Final Model
    # =============================================
    if os.path.exists(args.final_model_path):
        print(f"\n--- Part 1: Modality Ablation on Final Model ---")
        print(f"Loading: {args.final_model_path}")
        
        model = MSAModel(**vars(hyp_params))
        final_dict = torch.load(args.final_model_path, map_location=device)
        model.load_state_dict(final_dict, strict=False)

        # Load unimodal heads from student if available
        if os.path.exists(args.student_model_path):
            print(f"Loading unimodal heads from: {args.student_model_path}")
            student_dict = torch.load(args.student_model_path, map_location=device)
            with torch.no_grad():
                if 'proj1.weight' in student_dict and 'out_layer.weight' in student_dict:
                    for i in range(hyp_params.num_mod):
                        model.unimodal_heads[i][0].weight.copy_(student_dict['proj1.weight'])
                        model.unimodal_heads[i][0].bias.copy_(student_dict['proj1.bias'])
                        model.unimodal_heads[i][2].weight.copy_(student_dict['out_layer.weight'])
                        model.unimodal_heads[i][2].bias.copy_(student_dict['out_layer.bias'])
                    print("  Injected student classifier weights into evaluation heads.")

        model.to(device)
        
        ablation_modes = ['full', 'text', 'audio', 'vision']
        results = {}
        for mode in ablation_modes:
            metrics = run_ablation(model, test_loader, hyp_params, mode)
            results[mode] = metrics

        # Print results table
        print("\n" + "=" * 80)
        print("     MODALITY ABLATION RESULTS")
        print("=" * 80)
        
        rows = []
        for mode in ablation_modes:
            m = results[mode]
            label = {'full': 'Full Model (T+A+V)', 'text': 'Text Only',
                     'audio': 'Audio Only', 'vision': 'Vision Only'}[mode]
            rows.append([label, f"{m['acc2']:.4f}", f"{m['acc7']:.4f}",
                        f"{m['f1']:.4f}", f"{m['mae']:.4f}", f"{m['corr']:.4f}"])
        
        df = pd.DataFrame(rows, columns=['Condition', 'Acc-2', 'Acc-7', 'F1', 'MAE', 'Corr'])
        print(df.to_string(index=False))

        # Modality Dominance Analysis
        mae_t = results['text']['mae']
        mae_a = results['audio']['mae']
        mae_v = results['vision']['mae']
        mae_full = results['full']['mae']

        best_unimodal_mae = min(mae_t, mae_a, mae_v)
        synergy_gap = best_unimodal_mae - mae_full
        ratio_a_t = mae_a / (mae_t + 1e-8)
        ratio_v_t = mae_v / (mae_t + 1e-8)

        print("\n--- Modality Dominance Analysis ---")
        print(f"  Fusion Synergy Gap:         {synergy_gap:+.4f}  (Goal: > 0, means fusion helps)")
        print(f"  Audio-Text Disparity Ratio: {ratio_a_t:.4f}   (Goal: → 1.0)")
        print(f"  Vision-Text Disparity Ratio:{ratio_v_t:.4f}   (Goal: → 1.0)")
        
        # Accuracy drop analysis
        acc_full = results['full']['acc2']
        drop_text = acc_full - results['text']['acc2']
        drop_audio = acc_full - results['audio']['acc2']
        drop_vision = acc_full - results['vision']['acc2']
        print(f"\n--- Accuracy Drop (Full - Unimodal) ---")
        print(f"  Text contribution:   {drop_text:+.4f}")
        print(f"  Audio contribution:  {drop_audio:+.4f}")
        print(f"  Vision contribution: {drop_vision:+.4f}")
        print(f"  (More balanced drops = less dominance)")
    else:
        print(f"Skipping Part 1: {args.final_model_path} not found. Train all stages first.")

    # =============================================
    # PART 2: 2^3 Factorial Technique Ablation
    # =============================================
    if args.run_technique_ablation:
        print(f"\n--- Part 2: 2^3 Factorial Technique Ablation ---")
        tech_results = run_technique_ablation(hyp_params, orig_dim, dataloader, device)
        
        print("\n" + "=" * 90)
        print("     TECHNIQUE ABLATION RESULTS (2^3 Factorial)")
        print("=" * 90)
        
        rows = []
        for r in tech_results:
            rows.append([
                r['name'],
                '✓' if r.get('dropout') else '✗',
                '✓' if r.get('ogm') else '✗',
                '✓' if r.get('contrastive') else '✗',
                f"{r['acc2']:.4f}", f"{r['f1']:.4f}", f"{r['robust_acc2']:.4f}"
            ])
        
        df = pd.DataFrame(rows, columns=['Variant', 'Drop', 'OGM', 'Contr',
                                          'Acc-2', 'F1', 'Robust@50%'])
        print(df.to_string(index=False))
        
        # Save results into a markdown file for easy copy-paste
        with open('logs/technique_ablation_results.md', 'w') as f:
            f.write(df.to_markdown(index=False))
        
        # Save results
        import json
        os.makedirs('logs', exist_ok=True)
        with open('logs/technique_ablation_results.json', 'w') as f:
            json.dump(tech_results, f, indent=2, default=str)
        print("\nResults saved to logs/technique_ablation_results.json")

    print("\n" + "=" * 70)
    print("     ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()