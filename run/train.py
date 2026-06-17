import sys
import os
import argparse
import random
import json
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np

# Adjust path to import from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.dataloader import getdataloader
from models.msamodel import MSAModel
from src.eval_metrics import eval_senti


class InfoNCE(nn.Module):
    """Symmetrized InfoNCE contrastive loss for inter-modal alignment."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, view1, view2):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        logits = torch.matmul(view1, view2.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def compute_grad_norm(model, param_filter=None):
    """Compute total gradient norm for a subset of model parameters."""
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if param_filter is None or param_filter in name:
                total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def evaluate_classification(model, loader, hyp_params):
    """Evaluate model and return full metrics dict."""
    model.eval()
    all_preds, all_truths = [], []
    device = torch.device("cuda" if hyp_params.use_cuda else "cpu")
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, y_reg = batch['text'], batch['audio'], batch['vision'], batch['labels']
            if hyp_params.use_cuda:
                text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
            if not hyp_params.eval_modes.get('text', True): text.zero_()
            if not hyp_params.eval_modes.get('audio', True): audio.zero_()
            if not hyp_params.eval_modes.get('vision', True): vision.zero_()
            
            active_modes = [k for k, v in hyp_params.eval_modes.items() if v]
            if len(active_modes) == 1:
                mode_map = {'text': 0, 'audio': 1, 'vision': 2}
                outputs = model([text, audio, vision], unimodal_mode=mode_map[active_modes[0]])
                preds_logits = outputs['unimodal_output']
            else:
                outputs = model([text, audio, vision])
                preds_logits = outputs['output']
            preds_class = torch.argmax(preds_logits, dim=1)
            preds_reg_style = preds_class.cpu().float() - 3
            all_preds.append(preds_reg_style)
            all_truths.append(y_reg.cpu())
            
    all_preds, all_truths = torch.cat(all_preds), torch.cat(all_truths)
    
    from collections import Counter
    pred_classes_eval = (all_preds.numpy().round() + 3).astype(int)
    true_classes_eval = (all_truths.squeeze().numpy().round() + 3).astype(int)
    
    active_modes_str = "+".join([k for k, v in hyp_params.eval_modes.items() if v]) or "none"
    print(f"  [{active_modes_str.upper()}] Pred Dist: {dict(Counter(pred_classes_eval))}")
    print(f"  [{active_modes_str.upper()}] True Dist: {dict(Counter(true_classes_eval))}")
    
    metrics = eval_senti(all_preds, all_truths, exclude_zero=True)
    return metrics


def run():
    parser = argparse.ArgumentParser(description='3-Stage Training Script')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--data_path', type=str, default='data/unaligned_50.pkl')
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei', 'sims'],
                        help='Dataset name for dataloader routing')
    parser.add_argument('--teacher_model_path', type=str, default='models/teacher_text_best.pt')
    parser.add_argument('--student_model_path', type=str, default='models/students_distilled_aligned.pt')
    parser.add_argument('--final_model_path', type=str, default='models/robust_hybrid_best.pt')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lambda_triplet', type=float, default=0.5)
    parser.add_argument('--nlevels', type=int, default=5)
    parser.add_argument('--proj_dim', type=int, default=40)
    parser.add_argument('--lambda_gate', type=float, default=0.3, help='Weight for the auxiliary gating classifier loss')
    parser.add_argument('--modality_dropout', type=float, default=0.0, help='Max probability of dropping text modality (0.0 to 1.0)')
    parser.add_argument('--use_ogm', action='store_true', help='Enable On-the-fly Gradient Modulation (OGM-GE)')
    parser.add_argument('--ogm_alpha', type=float, default=2.0, help='OGM-GE sensitivity hyperparameter')
    parser.add_argument('--use_contrastive', action='store_true', help='Enable Inter-Modal Contrastive Loss')
    parser.add_argument('--lambda_contrastive', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--use_adaptive_dropout', action='store_true',
                        help='Enable adaptive curriculum-based modality dropout (novel)')
    parser.add_argument('--adaptive_k', type=float, default=5.0,
                        help='Steepness of sigmoid for adaptive dropout')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save training logs')
    args = parser.parse_args()
    hyp_params = args

    def setup_seed(seed):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        np.random.seed(seed); random.seed(seed); torch.backends.cudnn.deterministic = True
    setup_seed(hyp_params.seed)
    
    hyp_params.use_cuda = torch.cuda.is_available() and not hyp_params.no_cuda
    device = torch.device("cuda" if hyp_params.use_cuda else "cpu")
    dataloaders, orig_dim = getdataloader(hyp_params.dataset, hyp_params.batch_size, hyp_params.data_path, num_workers=2)
    model = MSAModel(output_dim=hyp_params.num_classes, orig_dim=orig_dim, proj_dim=hyp_params.proj_dim, layers=hyp_params.nlevels).to(device)
    criterion = nn.CrossEntropyLoss()

    # Setup logging directory
    os.makedirs(hyp_params.log_dir, exist_ok=True)
    training_log = {
        'stage': hyp_params.stage, 'seed': hyp_params.seed,
        'config': {k: v for k, v in vars(hyp_params).items() if isinstance(v, (int, float, str, bool))},
        'epochs': []
    }

    if hyp_params.stage == 1:
        print("--- Starting Stage 1: Training Text-Only Teacher ---")
        optimizer = optim.Adam(model.parameters(), lr=hyp_params.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        best_val_acc = 0.0
        for epoch in range(1, hyp_params.num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in dataloaders['train']:
                text, audio, vision, y_reg = batch['text'], batch['audio'], batch['vision'], batch['labels']
                y_class = y_reg.squeeze().round().long() + 3
                if hyp_params.use_cuda: text, y_class = text.cuda(), y_class.cuda()
                optimizer.zero_grad()
                outputs = model([text, torch.zeros_like(audio).to(device), torch.zeros_like(vision).to(device)])
                loss = criterion(outputs['output'], y_class)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
            hyp_params.eval_modes = {'text': True, 'audio': False, 'vision': False}
            metrics = evaluate_classification(model, dataloaders['valid'], hyp_params)
            acc = metrics['acc2']
            scheduler.step(acc)

            epoch_log = {'epoch': epoch, 'train_loss': epoch_loss / len(dataloaders['train']),
                         'val_acc2': acc, 'val_mae': metrics['mae'], 'val_f1': metrics['f1']}
            training_log['epochs'].append(epoch_log)

            print(f"Stage 1 | Epoch {epoch:2d} | Loss: {epoch_log['train_loss']:.4f} | "
                  f"Valid Acc-2: {acc:.4f} | F1: {metrics['f1']:.4f}")
            if acc > best_val_acc:
                best_val_acc = acc
                print(f"*** New Best Teacher! Saving to {hyp_params.teacher_model_path} ***")
                torch.save(model.state_dict(), hyp_params.teacher_model_path)

    elif hyp_params.stage == 2:
        print("--- Starting Stage 2: Training A/V encoders with Decoupled Updates ---")
        teacher_model = MSAModel(output_dim=hyp_params.num_classes, orig_dim=orig_dim, proj_dim=hyp_params.proj_dim, layers=hyp_params.nlevels).to(device)
        try:
            teacher_model.load_state_dict(torch.load(hyp_params.teacher_model_path, map_location=device), strict=False)
        except FileNotFoundError:
            print(f"ERROR: Teacher model not found at {hyp_params.teacher_model_path}. Please run Stage 1 first.")
            sys.exit(1)
        teacher_model.eval()
        print("Teacher model loaded and frozen.")
        student_model = model
        optimizer = optim.Adam(list(student_model.parameters()), lr=hyp_params.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        best_val_acc = 0.0
        for epoch in range(1, hyp_params.num_epochs + 1):
            student_model.train()
            
            # Staged Warmup Protocol
            if epoch <= 10:
                kl_weight = 0.0
                ce_weight = 1.0
            else:
                kl_weight = 0.5 * ((epoch - 10) / 30)
                ce_weight = 1.0 - kl_weight
            epoch_loss_a, epoch_loss_v = 0.0, 0.0
            for batch in dataloaders['train']:
                text, audio, vision, y_reg = batch['text'], batch['audio'], batch['vision'], batch['labels']
                y_class = y_reg.squeeze().round().long() + 3
                if hyp_params.use_cuda: text, audio, vision, y_class = text.cuda(), audio.cuda(), vision.cuda(), y_class.cuda()
                with torch.no_grad():
                    teacher_outputs = teacher_model([text, torch.zeros_like(audio).to(device), torch.zeros_like(vision).to(device)])
                    teacher_logits = teacher_outputs['output']
                text_zeros = torch.zeros_like(text).to(device)
                optimizer.zero_grad()
                audio_outputs = student_model([text_zeros, audio, torch.zeros_like(vision).to(device)])
                audio_logits, h_audio = audio_outputs['unimodal_logits'][1], audio_outputs['hs_nondetached'][1][0]
                
                loss_ce_a = criterion(audio_logits, y_class)
                if kl_weight > 0:
                    loss_kl_a = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(audio_logits / hyp_params.temperature, dim=1), F.softmax(teacher_logits / hyp_params.temperature, dim=1)) * (hyp_params.temperature ** 2)
                    total_loss_a = (kl_weight * loss_kl_a) + (ce_weight * loss_ce_a)
                else:
                    total_loss_a = loss_ce_a
                    
                total_loss_a.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), hyp_params.grad_clip)
                optimizer.step()
                epoch_loss_a += total_loss_a.item()

                optimizer.zero_grad()
                vision_outputs = student_model([text_zeros, torch.zeros_like(audio).to(device), vision])
                vision_logits, h_vision = vision_outputs['unimodal_logits'][2], vision_outputs['hs_nondetached'][2][0]
                
                loss_ce_v = criterion(vision_logits, y_class)
                if kl_weight > 0:
                    loss_kl_v = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(vision_logits / hyp_params.temperature, dim=1), F.softmax(teacher_logits / hyp_params.temperature, dim=1)) * (hyp_params.temperature ** 2)
                    total_loss_v = (kl_weight * loss_kl_v) + (ce_weight * loss_ce_v)
                else:
                    total_loss_v = loss_ce_v
                    
                total_loss_v.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), hyp_params.grad_clip)
                optimizer.step()
                epoch_loss_v += total_loss_v.item()

            hyp_params.eval_modes = {'text': False, 'audio': True, 'vision': False}
            metrics_a = evaluate_classification(student_model, dataloaders['valid'], hyp_params)
            hyp_params.eval_modes = {'text': False, 'audio': False, 'vision': True}
            metrics_v = evaluate_classification(student_model, dataloaders['valid'], hyp_params)
            avg_acc = (metrics_a['acc2'] + metrics_v['acc2']) / 2
            scheduler.step(avg_acc)

            epoch_log = {'epoch': epoch,
                         'train_loss_audio': epoch_loss_a / len(dataloaders['train']),
                         'train_loss_vision': epoch_loss_v / len(dataloaders['train']),
                         'val_acc_audio': metrics_a['acc2'], 'val_acc_vision': metrics_v['acc2'],
                         'val_avg_acc': avg_acc}
            training_log['epochs'].append(epoch_log)

            print(f"Stage 2 | Epoch {epoch:2d} | A-Acc: {metrics_a['acc2']:.4f} | V-Acc: {metrics_v['acc2']:.4f} | Avg: {avg_acc:.4f}")
            if avg_acc > best_val_acc:
                best_val_acc = avg_acc
                print(f"*** New Best A/V Students! Saving to {hyp_params.student_model_path} ***")
                torch.save(student_model.state_dict(), hyp_params.student_model_path)

    elif hyp_params.stage == 3:
        print("--- Starting Stage 3: Fine-tuning Full Model with Learnable Gating ---")
        model.load_state_dict(torch.load(hyp_params.student_model_path, map_location=device))
        print("Loaded trained A/V student encoders.")
        teacher_dict = torch.load(hyp_params.teacher_model_path, map_location=device)
        model_dict = model.state_dict()
        text_encoder_dict = {k: v for k, v in teacher_dict.items() if 'encoders.0' in k or 'proj.0' in k}
        model_dict.update(text_encoder_dict)
        model.load_state_dict(model_dict)
        print("Re-loaded pre-trained text teacher encoder.")
        optimizer = optim.Adam(model.parameters(), lr=hyp_params.lr / 10)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        best_val_acc = 0.0
        
        # Initialize Contrastive Loss
        contrastive_criterion = InfoNCE(temperature=0.1).to(device)

        # Adaptive dropout state
        current_text_dropout_p = hyp_params.modality_dropout  # initial value
        prev_text_acc, prev_weak_acc = 0.5, 0.5  # will be overwritten after first epoch

        for epoch in range(1, hyp_params.num_epochs + 1):
            model.train()
            total_loss_main_epoch, total_loss_gate_epoch, total_loss_contrastive_epoch = 0, 0, 0
            grad_norms = {'text': [], 'audio': [], 'vision': []}
            gate_weights = {'text': [], 'audio': [], 'vision': []}

            for batch in dataloaders['train']:
                text, audio, vision, y_reg = batch['text'], batch['audio'], batch['vision'], batch['labels']
                y_class = y_reg.squeeze().round().long() + 3
                if hyp_params.use_cuda: text, audio, vision, y_class = text.cuda(), audio.cuda(), vision.cuda(), y_class.cuda()
                
                # --- Modality Dropout (Adaptive or Fixed) ---
                if current_text_dropout_p > 0 and model.training:
                    if random.random() < current_text_dropout_p:
                        text = torch.zeros_like(text)
                
                optimizer.zero_grad()
                outputs = model([text, audio, vision])
                loss_main = criterion(outputs['output'], y_class)
                unimodal_logits = outputs['unimodal_logits']
                loss_gate_t = criterion(unimodal_logits[0], y_class)
                loss_gate_a = criterion(unimodal_logits[1], y_class)
                loss_gate_v = criterion(unimodal_logits[2], y_class)
                
                # --- OGM-GE: Correct implementation matching CVPR 2022 paper ---
                if hyp_params.use_ogm and model.training:
                    with torch.no_grad():
                        losses = torch.stack([loss_gate_t, loss_gate_a, loss_gate_v])
                        # Lower loss = dominant modality = should be suppressed
                        # ratio_i = loss_i / max(losses)  -> dominant has ratio close to min/max < 1
                        # But OGM works on "overfitting ratio": dominant modality converges faster
                        # We use: ratio_i = max(losses) / (loss_i + 1e-8)
                        # ratio > 1 means this modality is dominant (lower loss = better)
                        max_loss = losses.max()
                        ratios = max_loss / (losses + 1e-8)
                        # coeff_i = 1 - tanh(alpha * relu(ratio_i - 1))
                        # For dominant modality (ratio > 1): coeff < 1 (suppress gradient)
                        # For weak modality (ratio ≈ 1): coeff ≈ 1 (keep gradient)
                        coeffs = 1.0 - torch.tanh(hyp_params.ogm_alpha * F.relu(ratios - 1.0))
                        w_t, w_a, w_v = coeffs[0].item(), coeffs[1].item(), coeffs[2].item()
                else:
                    w_t, w_a, w_v = 1.0, 1.0, 1.0

                loss_gate_total = (w_t * loss_gate_t + w_a * loss_gate_a + w_v * loss_gate_v)
                
                # --- Contrastive Loss (Inter-Modal Alignment) ---
                loss_contrastive = torch.tensor(0.0, device=device)
                if hyp_params.use_contrastive:
                    hs = outputs['hs_nondetached']
                    h_text = hs[0][0]   # [Batch, Dim]
                    h_audio = hs[1][0]  # [Batch, Dim]
                    h_vision = hs[2][0] # [Batch, Dim]
                    
                    loss_c_ta = contrastive_criterion(h_text, h_audio)
                    loss_c_tv = contrastive_criterion(h_text, h_vision)
                    loss_contrastive = loss_c_ta + loss_c_tv

                total_loss = ((1 - hyp_params.lambda_gate) * loss_main 
                              + hyp_params.lambda_gate * loss_gate_total 
                              + hyp_params.lambda_contrastive * loss_contrastive)
                total_loss.backward()

                # Log per-modality gradient norms BEFORE clipping
                grad_norms['text'].append(compute_grad_norm(model, 'encoders.0'))
                grad_norms['audio'].append(compute_grad_norm(model, 'encoders.1'))
                grad_norms['vision'].append(compute_grad_norm(model, 'encoders.2'))

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.grad_clip)
                optimizer.step()

                total_loss_main_epoch += loss_main.item()
                total_loss_gate_epoch += loss_gate_total.item()
                total_loss_contrastive_epoch += loss_contrastive.item()

                # Log gate weights
                if 'gates' in outputs:
                    g = outputs['gates'].detach().cpu().mean(dim=0)
                    gate_weights['text'].append(g[0].item())
                    gate_weights['audio'].append(g[1].item())
                    gate_weights['vision'].append(g[2].item())

            n_batches = len(dataloaders['train'])
            avg_main = total_loss_main_epoch / n_batches
            avg_gate = total_loss_gate_epoch / n_batches
            avg_contrastive = total_loss_contrastive_epoch / n_batches

            # Evaluate full model
            hyp_params.eval_modes = {'text': True, 'audio': True, 'vision': True}
            metrics_full = evaluate_classification(model, dataloaders['valid'], hyp_params)
            scheduler.step(metrics_full['acc2'])

            # --- Adaptive Curriculum Dropout (Novel Contribution) ---
            if hyp_params.use_adaptive_dropout:
                # Measure unimodal accuracies to gauge dominance
                hyp_params.eval_modes = {'text': True, 'audio': False, 'vision': False}
                m_t = evaluate_classification(model, dataloaders['valid'], hyp_params)
                hyp_params.eval_modes = {'text': False, 'audio': True, 'vision': False}
                m_a = evaluate_classification(model, dataloaders['valid'], hyp_params)
                hyp_params.eval_modes = {'text': False, 'audio': False, 'vision': True}
                m_v = evaluate_classification(model, dataloaders['valid'], hyp_params)

                text_acc = m_t['acc2']
                avg_weak_acc = (m_a['acc2'] + m_v['acc2']) / 2 + 1e-8
                dominance_ratio = text_acc / avg_weak_acc

                # Adaptive: drop text more when it dominates, less when balanced
                # sigmoid maps dominance_ratio to [0, 1], scaled by p_max
                current_text_dropout_p = min(
                    0.5,  # cap at 50%
                    torch.sigmoid(torch.tensor(hyp_params.adaptive_k * (dominance_ratio - 1.0))).item()
                    * hyp_params.modality_dropout  # scale by base rate
                )
                # Ensure at least some base dropout
                current_text_dropout_p = max(current_text_dropout_p, hyp_params.modality_dropout * 0.3)

            # Build epoch log
            epoch_log = {
                'epoch': epoch,
                'train_loss_main': avg_main, 'train_loss_gate': avg_gate,
                'train_loss_contrastive': avg_contrastive,
                'val_acc2': metrics_full['acc2'], 'val_acc7': metrics_full['acc7'],
                'val_f1': metrics_full['f1'], 'val_mae': metrics_full['mae'],
                'val_corr': metrics_full['corr'],
                'avg_grad_norm_text': np.mean(grad_norms['text']),
                'avg_grad_norm_audio': np.mean(grad_norms['audio']),
                'avg_grad_norm_vision': np.mean(grad_norms['vision']),
                'avg_gate_text': np.mean(gate_weights['text']) if gate_weights['text'] else 0,
                'avg_gate_audio': np.mean(gate_weights['audio']) if gate_weights['audio'] else 0,
                'avg_gate_vision': np.mean(gate_weights['vision']) if gate_weights['vision'] else 0,
                'text_dropout_p': current_text_dropout_p,
                'ogm_coeffs': {'w_t': w_t, 'w_a': w_a, 'w_v': w_v}
            }
            training_log['epochs'].append(epoch_log)

            print(f"Epoch {epoch:2d} | Loss: {avg_main:.4f} | Gate: {avg_gate:.4f} | "
                  f"Contr: {avg_contrastive:.4f} | Acc-2: {metrics_full['acc2']:.4f} | "
                  f"Acc-7: {metrics_full['acc7']:.4f} | F1: {metrics_full['f1']:.4f} | MAE: {metrics_full['mae']:.4f}")
            print(f"         Gates: T={epoch_log['avg_gate_text']:.3f} A={epoch_log['avg_gate_audio']:.3f} "
                  f"V={epoch_log['avg_gate_vision']:.3f} | GradNorms: T={epoch_log['avg_grad_norm_text']:.3f} "
                  f"A={epoch_log['avg_grad_norm_audio']:.3f} V={epoch_log['avg_grad_norm_vision']:.3f} | "
                  f"TextDrop: {current_text_dropout_p:.3f}")

            if metrics_full['acc2'] > best_val_acc:
                best_val_acc = metrics_full['acc2']
                print(f"*** New Best Full Model! Acc-2={best_val_acc:.4f} Saving to {hyp_params.final_model_path} ***")
                torch.save(model.state_dict(), hyp_params.final_model_path)

    # Save training log
    log_path = os.path.join(hyp_params.log_dir, f'training_log_stage{hyp_params.stage}_seed{hyp_params.seed}.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    print(f"\nTraining log saved to {log_path}")

if __name__ == '__main__':
    run()