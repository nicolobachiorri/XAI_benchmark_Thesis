# ────────────────────────────────────────────────────────────
"""
Funzioni di valutazione delle spiegazioni XAI.
Calcola le quattro metriche definite nel paper (HA, Robustness,
Consistency, Contrastivity) e il punteggio aggregato CWS.

Gli attributi devono essere dizionari: {token: attribution_score}.
Se non hai rationales o esempi contrastivi, puoi omettere i relativi
argomenti: la funzione userà fallback interni.
"""

from __future__ import annotations
from typing import Dict, List, Set
from metrics import (
    human_agreement,
    robustness,
    consistency,
    contrastivity,
    cws,
)

# ───────────────────────────────────────── 
def list_to_dict(attrs_list):
    """
    Converte List[(tok, score)] -> Dict[tok] = score.
    Se un token compare più volte, tiene il valore a modulo massimo.
    """
    out = {}
    for tok, score in attrs_list:
        out[tok] = max(out.get(tok, 0.0), float(score), key=abs)
    return out


# ───────────────────────────────────────── single example
def compute_metrics_for_example(
    expl_attrs: Dict[str, float],
    *,
    human_rationales: Set[str] | None = None,
    pert_attrs: Dict[str, float] | None = None,
    other_pos_attrs: Dict[str, float] | None = None,
    other_neg_attrs: Dict[str, float] | None = None,
):
    """
    Calcola HA, R, Cn, Ct, CWS per UN esempio.

    Parameters
    ----------
    expl_attrs        : attribution sul testo originale
    human_rationales  : set di token «oro» (può essere None -> HA=0)
    pert_attrs        : attribution su testo perturbato  (per Robustness)
    other_pos_attrs   : attribution di un altro esempio stessa label
    other_neg_attrs   : attribution di un esempio label opposta
    """
    human_rationales = human_rationales or set()
    pert_attrs = pert_attrs or expl_attrs
    other_pos_attrs = other_pos_attrs or expl_attrs
    other_neg_attrs = other_neg_attrs or expl_attrs

    ha = human_agreement(list(expl_attrs.items()), human_rationales)
    r = robustness(expl_attrs, pert_attrs)
    cn = consistency(expl_attrs, other_pos_attrs)
    ct = contrastivity(other_pos_attrs, other_neg_attrs)
    cws_score = cws(ha, cn, ct, r)

    return dict(HA=ha, R=r, Cn=cn, Ct=ct, CWS=cws_score)


# ───────────────────────────────────────── batch
def compute_metrics_batch(
    batch_attrs: List[Dict[str, float]],
    *,
    batch_human: List[Set[str]] | None = None,
    batch_pert: List[Dict[str, float]] | None = None,
    batch_other_pos: List[Dict[str, float]] | None = None,
    batch_other_neg: List[Dict[str, float]] | None = None,
):
    """
    Calcola le metriche su un batch, restituendo la media.

    Le liste passate (se fornite) devono avere la stessa lunghezza
    di batch_attrs; gli argomenti mancanti vengono rimpiazzati
    con fallback come nella versione per singolo esempio.
    """
    n = len(batch_attrs)
    totals = dict(HA=0.0, R=0.0, Cn=0.0, Ct=0.0, CWS=0.0)

    for i in range(n):
        res = compute_metrics_for_example(
            batch_attrs[i],
            human_rationales=None if batch_human is None else batch_human[i],
            pert_attrs=None if batch_pert is None else batch_pert[i],
            other_pos_attrs=None
            if batch_other_pos is None
            else batch_other_pos[i],
            other_neg_attrs=None
            if batch_other_neg is None
            else batch_other_neg[i],
        )
        for k, v in res.items():
            totals[k] += v

    return {k: v / n for k, v in totals.items()}
