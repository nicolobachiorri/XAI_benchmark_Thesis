"""
explainers.py
─────────────
Metodi XAI implementati:

    lime_explain        – LIME (Ribeiro 2016)
    shap_explain        – SHAP  (Lundberg 2017)
    lrp_detach          – Layer-wise Relevance Propagation
    input_x_gradient    – Input × Gradient (embedding-level)
    attention_flow      – Attention Flow  (Abnar & Zuidema 2020)
    attention_rollout   – Attention Rollout (Abnar & Zuidema 2020)

Ogni funzione restituisce: List[(token, attribution_score)].
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------
# 1) GRADIENT-BASED METHODS
# ------------------------------------------------------------



# ---------- Input × Gradient (embedding level) ----------
def input_x_gradient(
    model,
    tokenizer,
    text,
    target: int = 1,
    device: str = "cuda",
) -> List[Tuple[str, float]]:
    """
    attr_i = embedding_i * ∂logit_target/∂embedding_i
    Funziona con qualsiasi versione di Captum / PyTorch.
    """
    model.eval()

    # 1) tokenizza
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # 2) embedding con require_grad
    emb_layer = model.get_input_embeddings()
    emb = emb_layer(input_ids)
    emb.requires_grad_(True)

    # 3) forward via inputs_embeds
    logits = model(
        inputs_embeds=emb,
        attention_mask=attention_mask,
    ).logits
    logit_target = logits[:, target].sum()

    # 4) backward → otteniamo i gradienti direttamente
    grads = torch.autograd.grad(
        outputs=logit_target,
        inputs=emb,
        retain_graph=False,
        create_graph=False,
    )[0]                              # (batch, seq_len, hidden)

    # 5) attribution = emb * grad  → somma dim embedding
    attr = (emb * grads).sum(-1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    return list(zip(tokens, attr.detach().cpu().tolist()))


# ---------- LRP (detach trick + ε) ----------
from captum.attr import LRP 

def lrp_detach(
    model, tokenizer, text,
    target: int = 1,
    device: str = "cuda",
    eps: float = 1e-6,
):
    """
    LRP con 'detach trick' e regola ε (epsilon) globale.
    Compatibile con Captum 0.6+.
    """
    # congela i gradienti
    for p in model.parameters():
        p.requires_grad_(False)

    lrp = LRP(model)

    enc = tokenizer(text, return_tensors="pt",
                    truncation=True, padding=True).to(device)

    attrs = (
        lrp.attribute(
            enc["input_ids"],
            additional_forward_args=(enc["attention_mask"],),
            target=target,
            rule="epsilon",        # ← seleziona ε-rule
            eps=eps,               # ← valore di stabilizzazione
        )
        .sum(-1)
        .squeeze()
    )

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
    return list(zip(tokens, attrs.detach().cpu().tolist())) 

# ------------------------------------------------------------
# 2) PERTURBATION-BASED
# ------------------------------------------------------------
from lime.lime_text import LimeTextExplainer


def lime_explain(
    model,
    tokenizer,
    text,
    target: int = 1,
    device: str = "cuda",
    num_samples: int = 200,
) -> List[Tuple[str, float]]:
    """Bag-of-Words LIME; num_samples ≈ 400-800."""
    model.eval()

    def predict_proba(texts: List[str]):
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()

    explainer = LimeTextExplainer(class_names=["neg", "pos"])
    exp = explainer.explain_instance(text, predict_proba, num_samples=num_samples, labels=[target])
    return exp.as_list(label=target)  # [(token, weight), ...]


# ---------- SHAP ----------
import shap


def shap_explain(
    model,
    tokenizer,
    text,
    target: int = 1,
    device: str = "cuda",
    nsamples: int | str = "auto",
) -> List[Tuple[str, float]]:
    """SHAP Text masker."""
    model.eval()

    def f_prob(texts: List[str]):
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f_prob, masker, output_names=["neg", "pos"])
    shap_values = explainer([text], max_evals=nsamples)[0]
    tokens = shap_values.data
    scores = shap_values.values[..., target]
    return list(zip(tokens, scores))


# ------------------------------------------------------------
# 3) ATTENTION METHODS – Abnar & Zuidema 2020
# ------------------------------------------------------------
def _normalize_and_residual(att: torch.Tensor) -> torch.Tensor:
    """Media su head, aggiunge residuo identità, normalizza righe."""
    att = att.mean(dim=1)
    eye = torch.eye(att.size(-1), device=att.device).unsqueeze(0)
    att = att + eye
    att = att / att.sum(dim=-1, keepdim=True)
    return att


@torch.no_grad()
def attention_rollout(
    model, tokenizer, text, target: int = 1, device: str = "cuda"
) -> List[Tuple[str, float]]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    att_layers = model(**enc, output_attentions=True).attentions

    rollout = torch.eye(att_layers[0].size(-1), device=device).unsqueeze(0)
    for att in att_layers:
        rollout = torch.bmm(_normalize_and_residual(att), rollout)

    scores = rollout[0, 0]  # CLS row
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
    return list(zip(tokens, scores.cpu().tolist()))


@torch.no_grad()
def attention_flow(
    model,
    tokenizer,
    text,
    target: int = 1,
    device: str = "cuda",
    start_layer: int = 0,
) -> List[Tuple[str, float]]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    att_layers = model(**enc, output_attentions=True).attentions

    flow = _normalize_and_residual(att_layers[-1])
    for l in range(len(att_layers) - 2, start_layer - 1, -1):
        flow = torch.bmm(_normalize_and_residual(att_layers[l]), flow)

    scores = flow[0, 0]
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
    return list(zip(tokens, scores.cpu().tolist()))


# ------------------------------------------------------------
# Registro funzioni
# ------------------------------------------------------------
EXPLAINERS = {
    "ixg": input_x_gradient,
    "lrp": lrp_detach,
    "lime": lime_explain,
    "shap": shap_explain,
    "aflow": attention_flow,
    "aroll": attention_rollout,
}
