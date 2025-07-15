"""
metrics.py – Metriche del paper XAI (senza Human‑Agreement)
===========================================================

Implementa **tre metriche automatiche** tratte da *“Evaluating the effectiveness
of XAI techniques for encoder‑based language models”*.

1. **Robustness** – stabilità delle saliency sotto perturbazione del testo
2. **Consistency** – allineamento delle saliency fra due modelli gemelli
3. **Contrastivity** – diversità delle saliency fra classi opposte

La metrica di *Human‑reasoning Agreement* non è implementata perché richiede
annotazioni manuali.

Dipendenze:
    transformers, torch, numpy, scipy, tqdm
"""

# ==== 1. Librerie ====
from __future__ import annotations
from typing import List, Callable, Sequence
from explainers import Attribution 
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.stats import spearmanr, entropy  # Spearman ρ, KL divergence
from tqdm import tqdm

# ==== 2. Helper probabilità positivo ====

def _prob_positive(model: PreTrainedModel, input_ids: torch.Tensor, attn_mask: torch.Tensor):
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        probs = F.softmax(logits, dim=-1)
        idx = 1 if probs.size(-1) > 1 else 0
        return probs[:, idx].item()

# ==== 3. Perturbazione semplice ====

def _random_mask(text: str, ratio: float = 0.15) -> str:
    """Maschera ~15% delle parole con un token speciale <MASK>."""
    tokens = text.split()
    n = max(1, int(len(tokens) * ratio))
    idx_to_mask = random.sample(range(len(tokens)), n)
    for i in idx_to_mask:
        tokens[i] = "<MASK>"
    return " ".join(tokens)

# ==== 4. Robustness ====

def compute_robustness(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], "Attribution"],
    text: str,
    perturb_fn: Callable[[str], str] = _random_mask,
) -> float:
    """Mean‑Average‑Difference (MAD) fra original e perturbed saliency."""
    orig_attr = explainer(text)
    pert_text = perturb_fn(text)
    pert_attr = explainer(pert_text)

    # Allineiamo i token che coincidono (per semplicità, stessa stringa)
    score_diffs = []
    for tok, score in zip(orig_attr.tokens, orig_attr.scores):
        if tok in pert_attr.tokens:
            j = pert_attr.tokens.index(tok)
            score_diffs.append(abs(score - pert_attr.scores[j]))
    if not score_diffs:
        return 0.0
    return float(np.mean(score_diffs))  # più basso ⇒ più robusto

# ==== 5. Consistency ====

def compute_consistency(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer_a: PreTrainedTokenizer,
    tokenizer_b: PreTrainedTokenizer,
    explainer_a: Callable[[str], "Attribution"],
    explainer_b: Callable[[str], "Attribution"],
    text: str,
) -> float:
    """Spearman ρ tra importanza di token comuni dei due modelli (−1…1).
    Più alto ⇒ maggiore consistenza."""
    attr_a = explainer_a(text)
    attr_b = explainer_b(text)

    # allinea token comuni preservando ordine in attr_a
    shared_scores_a, shared_scores_b = [], []
    for tok, score_a in zip(attr_a.tokens, attr_a.scores):
        if tok in attr_b.tokens:
            idx = attr_b.tokens.index(tok)
            shared_scores_a.append(score_a)
            shared_scores_b.append(attr_b.scores[idx])
    if len(shared_scores_a) < 2:
        return 0.0
    rho, _ = spearmanr(shared_scores_a, shared_scores_b)
    return float(rho)

# ==== 6. Contrastivity ====

def _normalize(scores: Sequence[float]) -> np.ndarray:
    arr = np.array(scores) - np.min(scores)  # shift to >=0
    if arr.sum() == 0:
        arr = np.ones_like(arr)
    return arr / arr.sum()


def compute_contrastivity(
    positive_attrs: List["Attribution"],
    negative_attrs: List["Attribution"],
) -> float:
    """KL divergence tra medie di distribuzioni di importanza delle due classi.
    • positive_attrs / negative_attrs sono liste di Attribution.
    • Ritorna KL( P_pos || P_neg ) (più alto ⇒ feature diverse fra classi)."""
    # Concateniamo tutte le importanze su vocabolario condiviso (token string)
    token_scores_pos, token_scores_neg = {}, {}

    def _accumulate(d: dict, attr: "Attribution"):
        for tok, s in zip(attr.tokens, attr.scores):
            d[tok] = d.get(tok, 0.0) + s

    for attr in positive_attrs:
        _accumulate(token_scores_pos, attr)
    for attr in negative_attrs:
        _accumulate(token_scores_neg, attr)

    vocab = list(set(token_scores_pos) | set(token_scores_neg))
    p = _normalize([token_scores_pos.get(t, 0.0) for t in vocab])
    q = _normalize([token_scores_neg.get(t, 0.0) for t in vocab])

    return float(entropy(p, q, base=2))  # bit‑KL

# ==== 7. Convenience batch wrappers ====

def evaluate_robustness_over_dataset(model, tokenizer, explainer, texts: List[str]):
    diffs = [compute_robustness(model, tokenizer, explainer, t) for t in tqdm(texts)]
    return float(np.mean(diffs))


def evaluate_consistency_over_dataset(model_a, model_b, tok_a, tok_b, expl_a, expl_b, texts: List[str]):
    rhos = [compute_consistency(model_a, model_b, tok_a, tok_b, expl_a, expl_b, t) for t in tqdm(texts)]
    return float(np.mean(rhos))


def evaluate_contrastivity_over_dataset(pos_attrs: List["Attribution"], neg_attrs: List["Attribution"]):
    return compute_contrastivity(pos_attrs, neg_attrs)
