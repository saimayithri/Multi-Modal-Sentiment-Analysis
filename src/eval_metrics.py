import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def multiclass_acc(preds, truths):
    """Compute multiclass accuracy (exact match after rounding)."""
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def eval_senti(results, truths, exclude_zero=False):
    """Compute full MSA evaluation metrics.
    
    Args:
        results: predicted sentiment scores (continuous or discrete)
        truths: ground truth sentiment scores
        exclude_zero: whether to exclude zero-sentiment samples for binary acc
        
    Returns:
        metrics: dict with keys 'acc2', 'acc7', 'f1', 'mae', 'corr'
    """
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or not exclude_zero])

    # MAE
    mae = np.mean(np.absolute(test_preds - test_truth))
    
    # Pearson Correlation
    if np.std(test_preds) == 0 or np.std(test_truth) == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(test_preds, test_truth)[0][1]

    # Binary Accuracy (Acc-2): positive vs negative sentiment (excluding zero if requested)
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc2 = accuracy_score(binary_truth, binary_preds)

    # Binary F1 (weighted)
    f1 = f1_score(binary_truth, binary_preds, average='weighted', zero_division=0)

    # 7-class Accuracy (Acc-7): round to nearest integer in [-3, 3], then accuracy
    preds_7 = np.clip(np.round(test_preds), -3, 3).astype(int)
    truth_7 = np.clip(np.round(test_truth), -3, 3).astype(int)
    acc7 = accuracy_score(truth_7, preds_7)

    # Return as dict for easy table generation
    metrics = {
        'acc2': float(acc2),
        'acc7': float(acc7),
        'f1': float(f1),
        'mae': float(mae),
        'corr': float(corr)
    }
    return metrics


def eval_senti_legacy(results, truths, exclude_zero=False):
    """Legacy interface returning (acc, mae, corr) tuple for backward compatibility."""
    m = eval_senti(results, truths, exclude_zero)
    return m['acc2'], m['mae'], m['corr']