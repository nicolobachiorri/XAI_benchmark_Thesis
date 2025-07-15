"""
explainers.py – 6 XAI methods (lime, shap, grad_input, attention_rollout,
attention_flow, lrp) con token truncation a 512 per evitare RuntimeError.
"""

from __future__ import annotations
from typing import List, Dict, Callable
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

# -------------------------------------------------------------------------
# Opzionali
try:
    from captum.attr import InputXGradient, LayerLRP
except ImportError:
    InputXGradient = LayerLRP = None  # type: ignore

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None  # type: ignore

try:
    import shap, numpy as np
except ImportError:
    shap = np = None  # type: ignore

MAX_LEN = 512  # taglio uniforme per tutti gli explainers

# -------------------------------------------------------------------------
def _forward_pos(model, ids, mask):
    logits = model(input_ids=ids, attention_mask=mask).logits
    idx = 1 if logits.size(-1) > 1 else 0
    return logits[:, idx]


class Attribution:
    def __init__(self, tokens: List[str], scores: List[float]):
        self.tokens = tokens
        self.scores = scores
    def __repr__(self):
        return "Attribution(" + ", ".join(f"{t}:{s:.3f}" for t, s in zip(self.tokens, self.scores)) + ")"


# -------------------------------------------------------------------------
# GRADIENT × INPUT (embedding‑level)  ### NEW
def _grad_input(model, tokenizer):
    def explain(text: str) -> Attribution:
        model.eval()
        enc = tokenizer(text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_LEN)               # NEW
        ids, attn = enc["input_ids"], enc["attention_mask"]

        embeds = model.get_input_embeddings()(ids).detach()
        embeds.requires_grad_(True)

        logits = model(inputs_embeds=embeds, attention_mask=attn).logits
        target = 1 if logits.size(-1) > 1 else 0
        loss = logits[:, target].sum()

        model.zero_grad()
        loss.backward()

        scores = (embeds.grad * embeds).sum(dim=-1).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
        return Attribution(tokens, scores.tolist())
    return explain


# -------------------------------------------------------------------------
# ATTENTION ROLLOUT (Abnar & Zuidema, 2020)  ### CHANGED (truncation)
def _attention_rollout(model, tokenizer):
    def explain(text: str) -> Attribution:
        model.eval()
        enc = tokenizer(text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_LEN,
                        output_attentions=True)            # NEW
        with torch.no_grad():
            outs = model(**enc)
        att = torch.stack([a.mean(dim=1) for a in outs.attentions])  # (L,B,seq,seq)

        I = torch.eye(att.size(-1)).unsqueeze(0).unsqueeze(0)
        att = 0.5 * att + 0.5 * I
        att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        rollout = att[-1]
        for l in range(att.size(0) - 2, -1, -1):
            rollout = att[l].bmm(rollout)

        scores = rollout.squeeze(0)[0]
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        return Attribution(tokens, scores.tolist())
    return explain


# -------------------------------------------------------------------------
# ATTENTION FLOW  ### CHANGED (truncation)
def _attention_flow(model, tokenizer):
    import networkx as nx

    def explain(text: str) -> Attribution:
        model.eval()
        enc = tokenizer(text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_LEN,
                        output_attentions=True)            # NEW
        with torch.no_grad():
            outs = model(**enc)

        att = torch.stack([a.mean(dim=1) for a in outs.attentions])
        seq = att.size(-1)
        I = torch.eye(seq).unsqueeze(0).unsqueeze(0)
        att = 0.5 * att + 0.5 * I
        att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        G = nx.DiGraph()
        L = att.size(0)
        for l in range(L):
            for i in range(seq):
                G.add_node((l, i))
        for l in range(L):
            for i in range(seq):
                for j in range(seq):
                    w = att[l, 0, i, j].item()
                    if w > 0:
                        G.add_edge((l, i), (l - 1, j), capacity=w)

        flow_scores = torch.zeros(seq)
        for src in range(seq):
            flow_val, _ = nx.maximum_flow(G, (L - 1, src), (0, 0))
            flow_scores[src] = flow_val

        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        return Attribution(tokens, flow_scores.tolist())
    return explain


# -------------------------------------------------------------------------
# LIME‑Text (truncation added)  ### CHANGED
def _lime_text(model, tokenizer):
    if LimeTextExplainer is None:
        raise ImportError("LIME non installato")
    expl = LimeTextExplainer(class_names=["neg", "pos"])

    def predict(texts):
        enc = tokenizer(texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_LEN)               # NEW
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()

    def explain(text: str) -> Attribution:
        exp = expl.explain_instance(text, predict, num_features=10)
        tokens, scores = zip(*exp.as_list())
        return Attribution(list(tokens), list(scores))
    return explain


# -------------------------------------------------------------------------
# SHAP KernelExplainer (truncation added)  ### CHANGED
def _kernel_shap(model, tokenizer):
    if shap is None or np is None:
        raise ImportError("SHAP non installato")

    def predict(texts):
        enc = tokenizer(texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_LEN)               # NEW
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()

    expl = shap.KernelExplainer(predict, np.array(["."]))

    def explain(text: str) -> Attribution:
        sv = expl.shap_values([text], nsamples=100)
        scores = sv[1][0] if len(sv) > 1 else sv[0][0]
        tokens = text.split()
        return Attribution(tokens, scores.tolist())
    return explain


# -------------------------------------------------------------------------
# LRP (truncation added)  ### CHANGED
def _lrp(model, tokenizer):
    if LayerLRP is None:
        raise ImportError("Captum non installato")
    layer = model.bert.encoder.layer[-1]
    lrp = LayerLRP(model, layer)

    def explain(text: str) -> Attribution:
        enc = tokenizer(text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_LEN)               # NEW
        ids, attn = enc["input_ids"], enc["attention_mask"]
        attrs = lrp.attribute(ids, additional_forward_args=(attn,))
        scores = attrs.sum(dim=-1).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
        return Attribution(tokens, scores.tolist())
    return explain


# -------------------------------------------------------------------------
_EXPLAINER_FACTORY: Dict[str, Callable] = {
    "lime":               _lime_text,
    "shap":               _kernel_shap,
    "grad_input":         _grad_input,
    "attention_rollout":  _attention_rollout,
    "attention_flow":     _attention_flow,
    "lrp":                _lrp,
}

# public helpers unchanged
def list_explainers():
    return list(_EXPLAINER_FACTORY.keys())


def get_explainer(name: str,
                  model: PreTrainedModel,
                  tokenizer: PreTrainedTokenizer):
    name = name.lower()
    if name not in _EXPLAINER_FACTORY:
        raise ValueError(f"Explainer '{name}' non supportato")
    return _EXPLAINER_FACTORY[name](model, tokenizer)
