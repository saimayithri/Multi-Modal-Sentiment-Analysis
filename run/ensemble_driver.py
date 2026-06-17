import argparse
import subprocess
import os
import torch
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.dataloader import getdataloader
from models.msamodel import MSAModel
from src.eval_metrics import eval_senti

def run_training(args, seeds):
    """
    Runs Stage 3 training for multiple seeds.
    """
    model_paths = []
    
    for seed in seeds:
        print(f"\n\n>>> MANAGED ENSEMBLE: Training Seed {seed}...\n")
        model_name = f"models/ensemble_seed_{seed}.pt"
        model_paths.append(model_name)
        
        # We reuse the same command logic as Stage 3 but change seed and save path
        cmd = [
            'python', 'run/train.py',
            '--stage', '3',
            '--data_path', args.data_path,
            '--final_model_path', model_name,
            '--seed', str(seed),
            '--modality_dropout', str(args.modality_dropout),
            '--num_epochs', str(args.num_epochs)
        ]
        
        if args.use_ogm:
            cmd.append('--use_ogm')
        if args.use_contrastive:
            cmd.append('--use_contrastive')
        if not args.use_cuda:
            cmd.append('--no_cuda')
            
        subprocess.check_call(cmd)
        
    return model_paths

def evaluate_ensemble(model_paths, args):
    """
    Loads all models, predicts, averages logits, and computes metrics.
    """
    print(f"\n\n>>> ENSEMBLE EVALUATION: Averaging {len(model_paths)} models...\n")
    
    # Load Data
    hyp_params = argparse.Namespace(
        batch_size=32, data_path=args.data_path, dataset='mosi', # assumed
        use_cuda=args.use_cuda
    )
    dataloader, orig_dim = getdataloader('mosi', hyp_params.batch_size, hyp_params.data_path)
    test_loader = dataloader['test']
    
    # Model Config (Must match train.py defaults/args)
    model_args = {
        'output_dim': 7,
        'orig_dim': orig_dim,
        'proj_dim': 40,
        'layers': 5, # Default nlevels
        'num_heads': 5,
        'attn_dropout': 0.15, 'relu_dropout': 0.15, 'embed_dropout': 0.2, 
        'res_dropout': 0.1, 'out_dropout': 0.1
        # Add others if they differ from default
    }
    
    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    # Load Models
    models = []
    for path in model_paths:
        m = MSAModel(num_mod=3, **model_args).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
        
    # Inference Loop
    all_preds_reg = []
    all_truths_reg = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Inference"):
            text, audio, vision, y = batch['text'], batch['audio'], batch['vision'], batch['labels']
            if args.use_cuda:
                text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
            
            # Get predictions from all models
            batch_logits = []
            for m in models:
                outputs = m([text, audio, vision])
                batch_logits.append(outputs['output']) # [Batch, 7]
            
            # Average the logits
            avg_logits = torch.stack(batch_logits).mean(dim=0) # [Batch, 7]
            
            preds_class = torch.argmax(avg_logits, dim=1)
            preds_reg = preds_class.cpu().float() - 3
            
            all_preds_reg.append(preds_reg)
            all_truths_reg.append(y.cpu().squeeze(-1))
            
    all_preds_reg = torch.cat(all_preds_reg)
    all_truths_reg = torch.cat(all_truths_reg)
    
    acc, mae, corr = eval_senti(all_preds_reg, all_truths_reg, exclude_zero=True)
    
    print("\n==============================================================")
    print(f"     Ensemble Result ({len(models)} models)")
    print("==============================================================")
    print(f"Accuracy (Acc-2): {acc:.4f} (Goal: >0.8000)")
    print(f"MAE             : {mae:.4f}")
    print(f"Correlation     : {corr:.4f}")
    print("==============================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/unaligned_50.pkl')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--modality_dropout', type=float, default=0.15)
    parser.add_argument('--use_ogm', action='store_true', default=True)
    parser.add_argument('--use_contrastive', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    
    print("--- Starting Ensemble Training + Evaluation ---")
    model_paths = run_training(args, args.seeds)
    evaluate_ensemble(model_paths, args)
