"""Evaluation metrics for image-to-LaTeX models."""

from typing import Optional
from collections import Counter
import math


def compute_token_accuracy(
    predictions: list[list[int]],
    targets: list[list[int]],
    pad_id: int = 0,
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        pad_id: Padding token ID to ignore
        
    Returns:
        Token accuracy (0-1)
    """
    correct = 0
    total = 0
    
    for pred, tgt in zip(predictions, targets):
        for p, t in zip(pred, tgt):
            if t != pad_id:
                total += 1
                if p == t:
                    correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_sequence_accuracy(
    predictions: list[list[int]],
    targets: list[list[int]],
    pad_id: int = 0,
    eos_id: int = 2,
) -> float:
    """
    Compute sequence-level (exact match) accuracy.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        pad_id: Padding token ID
        eos_id: End-of-sequence token ID
        
    Returns:
        Sequence accuracy (0-1)
    """
    correct = 0
    
    for pred, tgt in zip(predictions, targets):
        # Remove padding and EOS from both
        pred_clean = []
        for t in pred:
            if t == eos_id:
                break
            if t != pad_id:
                pred_clean.append(t)
        
        tgt_clean = []
        for t in tgt:
            if t == eos_id:
                break
            if t != pad_id:
                tgt_clean.append(t)
        
        if pred_clean == tgt_clean:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def compute_bleu(
    predictions: list[list[int]],
    targets: list[list[int]],
    max_n: int = 4,
    pad_id: int = 0,
    eos_id: int = 2,
) -> float:
    """
    Compute BLEU score.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        max_n: Maximum n-gram order
        pad_id: Padding token ID
        eos_id: End-of-sequence token ID
        
    Returns:
        BLEU score (0-100)
    """
    def clean_sequence(seq):
        result = []
        for t in seq:
            if t == eos_id:
                break
            if t != pad_id:
                result.append(t)
        return result
    
    def get_ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]
    
    def count_ngrams(seq, n):
        return Counter(get_ngrams(seq, n))
    
    def modified_precision(pred_seqs, ref_seqs, n):
        clipped_count = 0
        total_count = 0
        
        for pred, ref in zip(pred_seqs, ref_seqs):
            pred_ngrams = count_ngrams(pred, n)
            ref_ngrams = count_ngrams(ref, n)
            
            for ngram, count in pred_ngrams.items():
                clipped_count += min(count, ref_ngrams.get(ngram, 0))
                total_count += count
        
        return clipped_count / total_count if total_count > 0 else 0.0
    
    # Clean sequences
    pred_clean = [clean_sequence(p) for p in predictions]
    ref_clean = [clean_sequence(t) for t in targets]
    
    # Compute brevity penalty
    pred_len = sum(len(p) for p in pred_clean)
    ref_len = sum(len(r) for r in ref_clean)
    
    if pred_len == 0:
        return 0.0
    
    if pred_len <= ref_len:
        bp = math.exp(1 - ref_len / pred_len)
    else:
        bp = 1.0
    
    # Compute modified precisions
    precisions = []
    for n in range(1, max_n + 1):
        p = modified_precision(pred_clean, ref_clean, n)
        precisions.append(p)
    
    # Handle zero precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    log_precisions = [math.log(p) for p in precisions]
    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    
    bleu = bp * geo_mean * 100
    return bleu


def compute_metrics(
    predictions: list[list[int]],
    targets: list[list[int]],
    pad_id: int = 0,
    eos_id: int = 2,
) -> dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        pad_id: Padding token ID
        eos_id: End-of-sequence token ID
        
    Returns:
        Dictionary of metric names to values
    """
    return {
        'token_accuracy': compute_token_accuracy(predictions, targets, pad_id),
        'sequence_accuracy': compute_sequence_accuracy(predictions, targets, pad_id, eos_id),
        'bleu': compute_bleu(predictions, targets, pad_id=pad_id, eos_id=eos_id),
    }

