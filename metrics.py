import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from typing import List, Dict, Set, Tuple, Union

EPS = 1e-12  # numerical stability

# ─────────── single metrics ───────────

def precision_at_k(expl: List[Tuple[str, float]], human: Set[str], k: int = 10) -> float:
    """
    Calculate simple Precision@k.
    
    Args:
        expl: List of (token, score) tuples, sorted by importance
        human: Set of gold standard rationale tokens
        k: Number of top tokens to consider
        
    Returns:
        Precision@k score [0,1]
    """
    if not expl or not human:
        return 0.0
        
    top_k_tokens = {token for token, _ in expl[:k]}
    return len(top_k_tokens & human) / min(k, len(expl))

def robustness(attr_a: Dict[str, float], attr_b: Dict[str, float]) -> float:
    """
    Calculate robustness between two attributions.
    
    Args:
        attr_a, attr_b: Token -> score mappings
        
    Returns:
        Normalized robustness score [0,1] (0 = perfect, 1 = worst)
    """
    common = set(attr_a) & set(attr_b)
    if not common:
        return 1.0
    
    # Calculate absolute differences
    diffs = [abs(attr_a[t] - attr_b[t]) for t in common]
    mean_diff = np.mean(diffs)
    
    # Improved normalization using max absolute values
    max_a = max(abs(attr_a[t]) for t in common)
    max_b = max(abs(attr_b[t]) for t in common)
    max_possible = max(max_a, max_b) + EPS
    
    return min(mean_diff / max_possible, 1.0)

def consistency(attr_a: Dict[str, float], attr_b: Dict[str, float], 
                min_tokens: int = 3) -> float:
    """
    Calculate ranking consistency using Spearman correlation.
    
    Args:
        attr_a, attr_b: Token -> score mappings
        min_tokens: Minimum number of common tokens required
        
    Returns:
        Spearman correlation coefficient [-1,1]
    """
    common_tokens = list(set(attr_a) & set(attr_b))
    
    if len(common_tokens) < min_tokens:
        return 0.0
    
    scores_a = [attr_a[token] for token in common_tokens]
    scores_b = [attr_b[token] for token in common_tokens]
    
    corr = spearmanr(scores_a, scores_b).correlation
    return 0.0 if np.isnan(corr) else corr

def contrastivity(attr_pos: Dict[str, float], attr_neg: Dict[str, float]) -> float:
    """
    Calculate contrastivity using Jensen-Shannon divergence.
    
    Args:
        attr_pos: Attribution for positive class
        attr_neg: Attribution for negative class
        
    Returns:
        JS divergence [0,1]
    """
    # Get all unique tokens
    all_tokens = set(attr_pos.keys()) | set(attr_neg.keys())
    
    # Create aligned probability distributions
    p = np.array([attr_pos.get(token, 0.0) for token in all_tokens]) + EPS
    q = np.array([attr_neg.get(token, 0.0) for token in all_tokens]) + EPS
    
    # Normalize to probability distributions
    p = p / p.sum()
    q = q / q.sum()
    
    return jensenshannon(p, q)

def faithfulness(attr: Dict[str, float], model_func, input_tokens: List[str], 
                 original_pred: float, mask_value: str = "[MASK]") -> float:
    """
    Calculate faithfulness by measuring prediction change when masking important tokens.
    
    Args:
        attr: Token -> importance score mapping
        model_func: Function that takes tokens and returns prediction
        input_tokens: Original input tokens
        original_pred: Original model prediction
        mask_value: Token to use for masking
        
    Returns:
        Faithfulness score [0,1]
    """
    # Sort tokens by importance (descending)
    sorted_tokens = sorted(attr.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Calculate prediction changes when masking top-k tokens
    faithfulness_scores = []
    
    for k in range(1, min(11, len(sorted_tokens) + 1)):  # Test masking 1-10 tokens
        # Create masked input
        masked_tokens = input_tokens.copy()
        for token, _ in sorted_tokens[:k]:
            if token in masked_tokens:
                # Find and mask the first occurrence
                idx = masked_tokens.index(token)
                masked_tokens[idx] = mask_value
        
        # Get prediction for masked input
        masked_pred = model_func(masked_tokens)
        
        # Calculate prediction change
        pred_change = abs(original_pred - masked_pred)
        faithfulness_scores.append(pred_change)
    
    return np.mean(faithfulness_scores)

# ─────────── aggregate scores ───────────

def cws(cn: float, ct: float, r_norm: float) -> float:
    """
    Comprehensive Weighted Score - aggregates 3 metrics (without Human Agreement).
    
    Args:
        cn: Consistency score [-1,1] (will be normalized to [0,1])
        ct: Contrastivity score [0,1]
        r_norm: Robustness score [0,1] (0 = robust)
        
    Returns:
        CWS score [0,1]
    """
    # Normalize consistency to [0,1]
    cn_norm = (cn + 1) / 2
    
    return (cn_norm + ct + (1 - r_norm)) / 3

def weighted_cws(cn: float, ct: float, r_norm: float, 
                 weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)) -> float:
    """
    Weighted CWS allowing custom importance weights (3 metrics only).
    
    Args:
        cn: Consistency score [-1,1]
        ct: Contrastivity score [0,1]
        r_norm: Robustness score [0,1] (0 = robust)
        weights: Tuple of weights (cn, ct, robustness)
        
    Returns:
        Weighted CWS score [0,1]
    """
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    # Normalize consistency to [0,1]
    cn_norm = (cn + 1) / 2
    
    return (weights[0] * cn_norm + 
            weights[1] * ct + 
            weights[2] * (1 - r_norm))

# ─────────── utility functions ───────────

def evaluate_explanation(expl: List[Tuple[str, float]], 
                        baseline_attr: Dict[str, float] = None,
                        contrasting_attr: Dict[str, float] = None,
                        k: int = 10) -> Dict[str, float]:
    """
    Comprehensive evaluation of a single explanation (without Human Agreement).
    
    Args:
        expl: List of (token, score) tuples
        baseline_attr: Baseline attribution for robustness comparison
        contrasting_attr: Contrasting attribution for contrastivity
        k: Number of top tokens to consider
        
    Returns:
        Dictionary of metric scores
    """
    attr_dict = dict(expl)
    
    results = {}
    
    if baseline_attr:
        results['robustness'] = robustness(attr_dict, baseline_attr)
        results['consistency'] = consistency(attr_dict, baseline_attr)
    
    if contrasting_attr:
        results['contrastivity'] = contrastivity(attr_dict, contrasting_attr)
    
    # Calculate CWS if we have all components
    if all(key in results for key in ['consistency', 'contrastivity', 'robustness']):
        results['cws'] = cws(
            results['consistency'],
            results['contrastivity'],
            results['robustness']
        )
    
    return results