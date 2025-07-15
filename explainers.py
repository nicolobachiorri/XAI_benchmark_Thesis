"""
explainers.py – Six XAI methods for Transformer models
======================================================

Explainers
----------
lime               -> LIME‑Text
shap               -> SHAP KernelExplainer
grad_input         -> Captum InputXGradient  (gradient × input)
attention_rollout  -> Mean attention roll‑out (Abnar & Zuidema, 2020)
attention_flow     -> Attention Flow (Abnar & Zuidema, 2020)
lrp                -> Layer‑wise Relevance Propagation (last encoder layer)

Returned API
------------
get_explainer(name, model, tokenizer) -> callable
    expl(text) -> Attribution(tokens, scores)
"""

from __future__ import annotations
from typing import List, Dict, Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

# -------------------------------------------------------------------------
# Captum imports (optional)
try:
    from captum.attr import (
        IntegratedGradients,
        GradientShap,
        InputXGradient,
        LayerLRP,
    )
except ImportError:
    IntegratedGradients = GradientShap = InputXGradient = LayerLRP = None  # type: ignore

# LIME & SHAP imports (optional)
try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None  # type: ignore

try:
    import shap
    import numpy as np
except ImportError:
    shap = None  # type: ignore
    np = None    # type: ignore


# -------------------------------------------------------------------------
def _forward_func(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor):
    """Return logit of positive class (idx 1)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    idx = 1 if logits.size(-1) > 1 else 0
    return logits[:, idx]


class Attribution:
    """Container for tokens and their importance scores."""
    def __init__(self, tokens: List[str], scores: List[float]):
        self.tokens = tokens
        self.scores = scores

    def __repr__(self):
        joined = ", ".join(f"{t}:{s:.3f}" for t, s in zip(self.tokens, self.scores))
        return f"Attribution([{joined}])"


# -------------------------------------------------------------------------
# 1) 
# GRADIENT × INPUT (embedding‑level, senza Captum)
def _grad_input(model, tokenizer):
    def explain(text: str) -> Attribution:
        model.eval()
        enc = tokenizer(text, return_tensors="pt")
        ids, attn = enc["input_ids"], enc["attention_mask"]

        # ottieni embeddings e abilita i gradienti
        embeds = model.get_input_embeddings()(ids).detach()
        embeds.requires_grad_(True)

        # forward usando inputs_embeds
        outputs = model(inputs_embeds=embeds, attention_mask=attn)
        logits = outputs.logits
        target = 1 if logits.size(-1) > 1 else 0    # classe positiva
        loss = logits[:, target].sum()

        # backward
        model.zero_grad()
        loss.backward()

        grads = embeds.grad        # shape (B, seq, dim)
        scores = (grads * embeds).sum(dim=-1).squeeze(0)  # grad × input
        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
        return Attribution(tokens, scores.tolist())
    return explain



# -------------------------------------------------------------------------
# ATTENTION ROLLOUT (Abnar & Zuidema, 2020)
# -------------------------------------------------------------------------
def _attention_rollout(model, tokenizer, head_avg: str = "mean"):
    """
    Calcola l’Attention Rollout:
        R_L   =  A_L
        R_l   =  (A_l   @ R_{l+1})      per l=L-1 … 0
    dove A_l è l’attenzione (heads raggruppate) già
    corretta per le residual connection:   Â = 0.5·A + 0.5·I
    Il risultato finale è la rilevanza dei token input rispetto a [CLS].
    """
    factor = 0.5   # peso per residual, come nel paper

    def explain(text: str):
        model.eval()
        enc = tokenizer(text, return_tensors="pt", output_attentions=True)
        with torch.no_grad():
            outs = model(**enc)

        # media o max sui heads
        if head_avg == "mean":
            att_stack = torch.stack([a.mean(dim=1) for a in outs.attentions])   # (L,B,seq,seq)
        elif head_avg == "max":
            att_stack = torch.stack([a.max(dim=1).values for a in outs.attentions])
        else:
            raise ValueError("head_avg deve essere 'mean' o 'max'")

        # aggiungi residual e rinormalizza riga per riga
        I = torch.eye(att_stack.size(-1)).unsqueeze(0).unsqueeze(0)             # (1,1,seq,seq)
        att_stack = factor * att_stack + factor * I
        att_stack = att_stack / att_stack.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # rollout cumulativo dal basso verso l’alto
        rollout = att_stack[-1]                  # start: ultimo layer (B,seq,seq)
        for l in range(att_stack.size(0) - 2, -1, -1):
            rollout = att_stack[l].bmm(rollout)

        scores = rollout.squeeze(0)[0]           # attenzione da [CLS] (indice 0)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        return Attribution(tokens, scores.tolist())

    return explain


# -------------------------------------------------------------------------
# ATTENTION FLOW (Abnar & Zuidema, 2020)
# -------------------------------------------------------------------------
def _attention_flow(model, tokenizer, head_avg: str = "mean"):
    """
    Implementa il Maximum‑Flow descritto da Abnar & Zuidema.
    Complessità O(L²·n⁴) ma per frasi brevi è accettabile.
    """
    import networkx as nx                      # richiede  networkx>=2

    factor = 0.5

    def explain(text: str):
        model.eval()
        enc = tokenizer(text, return_tensors="pt", output_attentions=True)
        with torch.no_grad():
            outs = model(**enc)

        # heads -> (L,B,seq,seq)
        if head_avg == "mean":
            attn = torch.stack([a.mean(dim=1) for a in outs.attentions])
        elif head_avg == "max":
            attn = torch.stack([a.max(dim=1).values for a in outs.attentions])
        else:
            raise ValueError("head_avg deve essere 'mean' o 'max'")

        seq_len = attn.size(-1)
        I = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
        attn = factor * attn + factor * I
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Costruisci grafo a più layer
        G = nx.DiGraph()
        L = attn.size(0)

        # nodi: (layer, pos)  ─ sorgente = (L,pos)  target   = (0,pos0)
        for l in range(L):
            for i in range(seq_len):
                G.add_node((l, i))

        # archi con capacità = peso attenzione
        for l in range(L):
            for i in range(seq_len):           # from (l,i)
                for j in range(seq_len):       # to   (l-1,j)
                    w = attn[l, 0, i, j].item()   # B=1
                    if w > 0:
                        G.add_edge((l, i), (l - 1, j), capacity=w)

        # flusso massimo da ogni token del top‑layer a ogni input token
        cls_scores = torch.zeros(seq_len)
        for src in range(seq_len):
            flow_val, flow_dict = nx.maximum_flow(G, (L - 1, src), (0, 0))
            cls_scores[src] = flow_val

        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        return Attribution(tokens, cls_scores.tolist())

    return explain



# -------------------------------------------------------------------------
# 4) LRP (LayerLRP on last encoder layer)
def _lrp(model, tokenizer):
    if LayerLRP is None:
        raise ImportError("Captum non installato: pip install captum")

    # heuristic: last encoder block
    try:
        target_layer = model.bert.encoder.layer[-1]
    except AttributeError:
        raise ValueError("LRP: modello non ha struttura BERT standard")

    lrp = LayerLRP(model, target_layer)

    def explain(text: str) -> Attribution:
        model.eval()
        enc = tokenizer(text, return_tensors="pt")
        ids, attn = enc["input_ids"], enc["attention_mask"]

        relev = lrp.attribute(ids, additional_forward_args=(attn,),
                              internal_batch_size=1)
        scores = relev.sum(dim=-1).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
        return Attribution(tokens, scores.tolist())
    return explain


# -------------------------------------------------------------------------
# 5) LIME‑Text
def _lime_text(model, tokenizer):
    if LimeTextExplainer is None:
        raise ImportError("LIME non installato: pip install lime")

    class_names = ["neg", "pos"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict(texts: List[str]):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def explain(text: str) -> Attribution:
        exp = explainer.explain_instance(text, predict, num_features=10)
        tokens, scores = zip(*exp.as_list())
        return Attribution(list(tokens), list(scores))
    return explain


# -------------------------------------------------------------------------
# 6) SHAP KernelExplainer
def _kernel_shap(model, tokenizer):
    if shap is None or np is None:
        raise ImportError("SHAP non installato: pip install shap")

    def predict(texts: List[str]):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()

    explainer = shap.KernelExplainer(predict, np.array(["."]))

    def explain(text: str) -> Attribution:
        shap_vals = explainer.shap_values([text], nsamples=100)
        scores = shap_vals[1][0]  # positive class
        tokens = text.split()
        return Attribution(tokens, scores.tolist())
    return explain


# -------------------------------------------------------------------------
_EXPLAINER_FACTORY = {
    "lime":               _lime_text,
    "shap":               _kernel_shap,
    "grad_input":         _grad_input,          # <-- aggiornata
    "attention_rollout":  _attention_rollout,
    "attention_flow":     _attention_flow,
    "lrp":                _lrp,
}

# -------------------------------------------------------------------------
def list_explainers():
    return list(_EXPLAINER_FACTORY.keys())


def get_explainer(name: str,
                  model: PreTrainedModel,
                  tokenizer: PreTrainedTokenizer):
    name = name.lower()
    if name not in _EXPLAINER_FACTORY:
        raise ValueError(f"Explainer '{name}' non supportato")
    if IntegratedGradients is None and name in {"grad_input"}:
        raise ImportError("Captum non installato: pip install captum")
    return _EXPLAINER_FACTORY[name](model, tokenizer)
