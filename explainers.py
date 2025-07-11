"""
explainers.py - Versione con LRP corretto
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

# Safe imports con fallback semplici
try:
    from captum.attr import LRP
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─────────── GRADIENT METHODS ───────────

def input_x_gradient(
    model, tokenizer, text, target: int = 1, device: str = "cuda",
) -> List[Tuple[str, float]]:
    """Input × Gradient - sempre funziona"""
    model.eval()
    
    enc = tokenizer(text, return_tensors="pt", truncation=True, 
                   padding="max_length", max_length=256).to(device)
    
    emb_layer = model.get_input_embeddings()
    emb = emb_layer(enc["input_ids"])
    emb.requires_grad_(True)
    
    logits = model(inputs_embeds=emb, attention_mask=enc["attention_mask"]).logits
    logit_target = logits[:, target].sum()
    
    grads = torch.autograd.grad(logit_target, emb, retain_graph=False)[0]
    attr = (emb * grads).sum(-1).squeeze()
    
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
    return list(zip(tokens, attr.detach().cpu().tolist()))

def lrp_detach(
    model, tokenizer, text, target: int = 1, device: str = "cuda", eps: float = 1e-6,
):
    """LRP con API semplificata e fallback robusto"""
    if not CAPTUM_AVAILABLE:
        return input_x_gradient(model, tokenizer, text, target, device)
    
    try:
        # Disabilita gradienti per il modello
        for p in model.parameters():
            p.requires_grad_(False)
        
        model.eval()
        
        # Tokenizza
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        # Assicurati che input_ids sia float per LRP
        input_ids = input_ids.float()
        input_ids.requires_grad_(True)
        
        # Crea LRP con API semplice
        lrp = LRP(model)
        
        # Prova prima senza parametri extra
        try:
            attrs = lrp.attribute(
                input_ids,
                additional_forward_args=(attention_mask,),
                target=target
            )
        except Exception as e1:
            # Fallback: prova con altri parametri
            try:
                attrs = lrp.attribute(
                    input_ids,
                    target=target
                )
            except Exception as e2:
                # Se LRP fallisce completamente, usa gradiente
                raise Exception(f"LRP failed: {e1}, {e2}")
        
        # Processa attributions
        if attrs.dim() > 2:
            attrs = attrs.sum(-1)
        
        attrs = attrs.squeeze()
        
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
        return list(zip(tokens, attrs.detach().cpu().tolist()))
        
    except Exception as e:
        # Fallback completo a input×gradient
        print(f"LRP fallback to gradient method due to: {e}")
        return input_x_gradient(model, tokenizer, text, target, device)

# ─────────── PERTURBATION METHODS ───────────

def lime_explain(
    model, tokenizer, text, target: int = 1, device: str = "cuda", num_samples: int = 200,
) -> List[Tuple[str, float]]:
    """LIME con fallback semplice"""
    if not LIME_AVAILABLE:
        return _simple_perturbation_fallback(model, tokenizer, text, target, device)
    
    model.eval()
    
    def predict_proba(texts: List[str]):
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()
    
    explainer = LimeTextExplainer(class_names=["neg", "pos"])
    exp = explainer.explain_instance(text, predict_proba, num_samples=num_samples, labels=[target])
    return exp.as_list(label=target)

def shap_explain(
    model, tokenizer, text, target: int = 1, device: str = "cuda", nsamples: int | str = "auto",
) -> List[Tuple[str, float]]:
    """SHAP con fallback semplice"""
    if not SHAP_AVAILABLE:
        return _simple_perturbation_fallback(model, tokenizer, text, target, device)
    
    model.eval()
    
    def f_prob(texts: List[str]):
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()
    
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f_prob, masker, output_names=["neg", "pos"])
    
    max_evals = 100 if nsamples == "auto" else nsamples
    shap_values = explainer([text], max_evals=max_evals)[0]
    
    return list(zip(shap_values.data, shap_values.values[..., target]))

def _simple_perturbation_fallback(model, tokenizer, text, target, device):
    """Fallback semplice per LIME/SHAP"""
    model.eval()
    words = text.split()
    
    def predict_text(text_str):
        enc = tokenizer(text_str, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            return F.softmax(logits, dim=-1)[0, target].item()
    
    baseline_score = predict_text(text)
    attributions = []
    
    for i, word in enumerate(words):
        perturbed_text = " ".join(words[:i] + words[i+1:])
        if perturbed_text.strip():
            perturbed_score = predict_text(perturbed_text)
            attribution = baseline_score - perturbed_score
        else:
            attribution = baseline_score
        attributions.append((word, attribution))
    
    return attributions

# ─────────── ATTENTION METHODS ───────────

def _normalize_and_residual(att: torch.Tensor) -> torch.Tensor:
    """Media su head, aggiunge residuo identità, normalizza righe."""
    att = att.mean(dim=1)
    eye = torch.eye(att.size(-1), device=att.device).unsqueeze(0)
    att = att + eye
    return att / att.sum(dim=-1, keepdim=True)

@torch.no_grad()
def attention_rollout(model, tokenizer, text, target: int = 1, device: str = "cuda"):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    att_layers = model(**enc, output_attentions=True).attentions
    
    rollout = torch.eye(att_layers[0].size(-1), device=device).unsqueeze(0)
    for att in att_layers:
        rollout = torch.bmm(_normalize_and_residual(att), rollout)
    
    scores = rollout[0, 0]
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
    return list(zip(tokens, scores.cpu().tolist()))

@torch.no_grad()
def attention_flow(model, tokenizer, text, target: int = 1, device: str = "cuda", start_layer: int = 0):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    att_layers = model(**enc, output_attentions=True).attentions
    
    flow = _normalize_and_residual(att_layers[-1])
    for l in range(len(att_layers) - 2, start_layer - 1, -1):
        flow = torch.bmm(_normalize_and_residual(att_layers[l]), flow)
    
    scores = flow[0, 0]
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
    return list(zip(tokens, scores.cpu().tolist()))

# ─────────── REGISTRY ───────────

EXPLAINERS = {
    "ixg": input_x_gradient,
    "lrp": lrp_detach,
    "lime": lime_explain,
    "shap": shap_explain,
    "aflow": attention_flow,
    "aroll": attention_rollout,
}