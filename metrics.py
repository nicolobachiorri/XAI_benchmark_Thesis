# metrics.py
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

EPS = 1e-12  # stabilità numerica

# ─────────── metriche singole ───────────
def human_agreement(expl, human, k: int = 10):
    """
    expl  : List[(token, score)]
    human : Set[token]  (gold rationales)
    """
    hits = [1 if t in human else 0 for t, _ in expl[:k]]
    if not any(hits):
        return 0.0
    precisions = [sum(hits[:i + 1]) / (i + 1) for i in range(k)]
    return sum(p * h for p, h in zip(precisions, hits)) / sum(hits)

def robustness(attr_a, attr_b):
    """
    attr_* : Dict[token] = score
    Restituisce R̂ normalizzato in [0,1] (0 = perfetto, 1 = pessimo)
    """
    common = set(attr_a) & set(attr_b)
    if not common:
        return 1.0
    diff = np.mean([abs(attr_a[t] - attr_b[t]) for t in common])
    # normalizza dividendo per (|a|+|b|)/2 per tenere valori in [0,1]
    norm = (np.mean([abs(attr_a[t]) for t in common]) +
            np.mean([abs(attr_b[t]) for t in common])) / 2 + EPS
    return min(diff / norm, 1.0)

def consistency(attr_a, attr_b):
    keys = list(set(attr_a) & set(attr_b))
    if len(keys) < 2:
        return 0.0
    corr = spearmanr([attr_a[k] for k in keys],
                     [attr_b[k] for k in keys]).correlation
    return 0.0 if np.isnan(corr) else corr

def contrastivity(attr_pos, attr_neg):
    p = np.array(list(attr_pos.values())) + EPS
    q = np.array(list(attr_neg.values())) + EPS
    # allinea lunghezze
    m = min(len(p), len(q))
    p, q = p[:m], q[:m]
    p, q = p / p.sum(), q / q.sum()
    return jensenshannon(p, q)

# ─────────── score aggregato ───────────
def cws(ha, cn, ct, r_norm):
    """
    r_norm deve già essere in [0,1] (0 = robusto). CWS massimizza.
    """
    return 0.25 * ha + 0.25 * cn + 0.25 * ct + 0.25 * (1 - r_norm)
