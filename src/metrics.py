"""
Evaluation metrics for SWELM
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for translation tasks.
    
    Args:
        predictions: Predicted translations
        references: Reference translations
        
    Returns:
        BLEU score
    """
    # TODO: Implement BLEU computation or use sacrebleu
    raise NotImplementedError


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: Predicted answers
        references: Reference answers
        
    Returns:
        Exact match score
    """
    matches = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
    return matches / len(predictions)


def compute_f1_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute token-level F1 score.
    
    Args:
        predictions: Predicted answers
        references: Reference answers
        
    Returns:
        F1 score
    """
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
            continue
            
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
            
    return np.mean(f1_scores)


def evaluate_performance(
    predictions: List[str],
    references: List[str],
    languages: Optional[List[str]] = None,
    metrics: List[str] = ["exact_match", "f1"]
) -> Dict[str, float]:
    """
    Evaluate performance across multiple metrics.
    
    Args:
        predictions: Predicted outputs
        references: Reference outputs
        languages: Optional language codes for per-language evaluation
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    if "exact_match" in metrics:
        results["exact_match"] = compute_exact_match(predictions, references)
        
    if "f1" in metrics:
        results["f1"] = compute_f1_score(predictions, references)
        
    if "bleu" in metrics:
        results["bleu"] = compute_bleu_score(predictions, references)
        
    # Per-language metrics
    if languages is not None:
        lang_groups = defaultdict(lambda: {"predictions": [], "references": []})
        for pred, ref, lang in zip(predictions, references, languages):
            lang_groups[lang]["predictions"].append(pred)
            lang_groups[lang]["references"].append(ref)
            
        for lang, data in lang_groups.items():
            lang_results = evaluate_performance(
                data["predictions"],
                data["references"],
                metrics=metrics
            )
            for metric, score in lang_results.items():
                results[f"{lang}_{metric}"] = score
                
    return results
