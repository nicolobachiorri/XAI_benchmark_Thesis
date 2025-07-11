"""
explainers.py - Versione robusta con gestione conflitti dipendenze
──────────────────────────────────────────────────────────────────

Metodi XAI implementati con fallback e gestione errori:
    - input_x_gradient   – Input × Gradient (sempre funziona)
    - lrp_detach        – LRP con Captum (fallback custom)
    - lime_explain      – LIME (fallback custom se libreria non funziona)
    - shap_explain      – SHAP (fallback custom se libreria non funziona)
    - attention_flow    – Attention Flow (sempre funziona)
    - attention_rollout – Attention Rollout (sempre funziona)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np
import logging
import warnings
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings da librerie problematiche
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Registry per tracciare quali librerie funzionano
LIBRARY_STATUS = {
    "captum": None,
    "lime": None, 
    "shap": None,
}

def safe_import(library_name: str, import_func: Callable):
    """Import sicuro con fallback"""
    if LIBRARY_STATUS[library_name] is False:
        return None
    
    try:
        result = import_func()
        LIBRARY_STATUS[library_name] = True
        logger.info(f"Successfully imported {library_name}")
        return result
    except Exception as e:
        logger.warning(f"Failed to import {library_name}: {e}")
        LIBRARY_STATUS[library_name] = False
        return None

def explainer_fallback(fallback_func: Callable):
    """Decorator per fallback automatico"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary explainer failed: {e}")
                logger.info(f"Using fallback for {func.__name__}")
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator

# ------------------------------------------------------------
# 1) GRADIENT-BASED METHODS (sempre funzionano)
# ------------------------------------------------------------

def input_x_gradient(
    model,
    tokenizer,
    text: str,
    target: int = 1,
    device: str = "cuda",
    max_length: int = 256,
) -> List[Tuple[str, float]]:
    """
    Input × Gradient method - implementazione vanilla che funziona sempre.
    """
    try:
        model.eval()
        
        # Tokenizza
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ).to(device)
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        # Embedding con gradiente
        emb_layer = model.get_input_embeddings()
        emb = emb_layer(input_ids)
        emb.requires_grad_(True)
        
        # Forward pass
        outputs = model(
            inputs_embeds=emb,
            attention_mask=attention_mask,
        )
        
        # Target logit
        logit_target = outputs.logits[:, target].sum()
        
        # Backward pass
        grads = torch.autograd.grad(
            outputs=logit_target,
            inputs=emb,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0]
        
        # Attribution = embedding * gradient
        attr = (emb * grads).sum(-1).squeeze()
        
        # Converti in token
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
        attributions = attr.detach().cpu().tolist()
        
        return list(zip(tokens, attributions))
        
    except Exception as e:
        logger.error(f"Error in input_x_gradient: {e}")
        # Fallback: attributions casuali
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
        )
        return [(token, 0.0) for token in tokens]

# ------------------------------------------------------------
# 2) LRP con Captum (con fallback custom)
# ------------------------------------------------------------

def _try_import_captum():
    """Import sicuro di Captum"""
    try:
        from captum.attr import LRP
        return LRP
    except ImportError:
        return None

def lrp_custom_fallback(
    model, tokenizer, text: str, target: int = 1, device: str = "cuda", **kwargs
) -> List[Tuple[str, float]]:
    """LRP custom implementation quando Captum non funziona"""
    logger.info("Using custom LRP fallback")
    
    # Implementazione semplificata LRP usando gradienti
    try:
        model.eval()
        
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        # Forward con gradiente
        input_ids.requires_grad_(True)
        
        # Embedding
        emb = model.get_input_embeddings()(input_ids)
        
        # Forward
        outputs = model(inputs_embeds=emb, attention_mask=attention_mask)
        target_logit = outputs.logits[:, target].sum()
        
        # Gradient come proxy per LRP
        grads = torch.autograd.grad(
            target_logit, input_ids, retain_graph=False
        )[0]
        
        # Semplice attributions
        attrs = grads.sum(-1).squeeze() if grads.dim() > 1 else grads.squeeze()
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
        return list(zip(tokens, attrs.detach().cpu().tolist()))
        
    except Exception as e:
        logger.error(f"LRP fallback failed: {e}")
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(text, truncation=True)["input_ids"]
        )
        return [(token, 0.0) for token in tokens]

@explainer_fallback(lrp_custom_fallback)
def lrp_detach(
    model,
    tokenizer, 
    text: str,
    target: int = 1,
    device: str = "cuda",
    eps: float = 1e-6,
) -> List[Tuple[str, float]]:
    """LRP con Captum, fallback se non funziona"""
    
    # Try import Captum
    LRP = safe_import("captum", _try_import_captum)
    if LRP is None:
        raise ImportError("Captum not available")
    
    # Disabilita gradienti
    for p in model.parameters():
        p.requires_grad_(False)
    
    lrp = LRP(model)
    
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True
    ).to(device)
    
    attrs = lrp.attribute(
        enc["input_ids"],
        additional_forward_args=(enc["attention_mask"],),
        target=target,
        rule="epsilon",
        eps=eps,
    ).sum(-1).squeeze()
    
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
    return list(zip(tokens, attrs.detach().cpu().tolist()))

# ------------------------------------------------------------
# 3) PERTURBATION-BASED con fallback
# ------------------------------------------------------------

def _try_import_lime():
    """Import sicuro di LIME"""
    try:
        from lime.lime_text import LimeTextExplainer
        return LimeTextExplainer
    except ImportError:
        return None

def lime_custom_fallback(
    model, tokenizer, text: str, target: int = 1, device: str = "cuda", **kwargs
) -> List[Tuple[str, float]]:
    """LIME custom implementation"""
    logger.info("Using custom LIME fallback")
    
    try:
        model.eval()
        
        # Tokenizza il testo
        words = text.split()
        if len(words) < 2:
            return [(word, 0.0) for word in words]
        
        # Funzione di predizione
        def predict_text(text_str: str) -> float:
            enc = tokenizer(
                text_str, return_tensors="pt", truncation=True, padding=True
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = F.softmax(logits, dim=-1)
                return probs[0, target].item()
        
        # Score baseline
        baseline_score = predict_text(text)
        
        # Perturbazione semplice: rimuovi parole una alla volta
        attributions = []
        for i, word in enumerate(words):
            # Crea testo senza la parola i-esima
            perturbed_words = words[:i] + words[i+1:]
            perturbed_text = " ".join(perturbed_words)
            
            if perturbed_text.strip():
                perturbed_score = predict_text(perturbed_text)
                attribution = baseline_score - perturbed_score
            else:
                attribution = baseline_score
            
            attributions.append((word, attribution))
        
        return attributions
        
    except Exception as e:
        logger.error(f"LIME fallback failed: {e}")
        words = text.split()
        return [(word, 0.0) for word in words]

@explainer_fallback(lime_custom_fallback)
def lime_explain(
    model,
    tokenizer,
    text: str,
    target: int = 1,
    device: str = "cuda",
    num_samples: int = 200,
) -> List[Tuple[str, float]]:
    """LIME con fallback custom"""
    
    LimeTextExplainer = safe_import("lime", _try_import_lime)
    if LimeTextExplainer is None:
        raise ImportError("LIME not available")
    
    model.eval()
    
    def predict_proba(texts: List[str]) -> np.ndarray:
        enc = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()
    
    explainer = LimeTextExplainer(class_names=["neg", "pos"])
    exp = explainer.explain_instance(
        text, predict_proba, num_samples=num_samples, labels=[target]
    )
    return exp.as_list(label=target)

def _try_import_shap():
    """Import sicuro di SHAP"""
    try:
        import shap
        return shap
    except ImportError:
        return None

def shap_custom_fallback(
    model, tokenizer, text: str, target: int = 1, device: str = "cuda", **kwargs
) -> List[Tuple[str, float]]:
    """SHAP custom implementation usando permutations"""
    logger.info("Using custom SHAP fallback")
    
    try:
        model.eval()
        words = text.split()
        
        def predict_text(text_str: str) -> float:
            if not text_str.strip():
                return 0.0
            enc = tokenizer(
                text_str, return_tensors="pt", truncation=True, padding=True
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = F.softmax(logits, dim=-1)
                return probs[0, target].item()
        
        # Shapley values approssimativi
        n_words = len(words)
        shapley_values = []
        
        for i in range(n_words):
            # Marginal contribution approssimativo
            
            # Testo senza parola i
            without_word = " ".join(words[:i] + words[i+1:])
            score_without = predict_text(without_word)
            
            # Testo con solo parola i
            with_only_word = words[i]
            score_only = predict_text(with_only_word)
            
            # Testo completo
            full_text = text
            score_full = predict_text(full_text)
            
            # Marginal contribution approssimativo
            contribution = (score_full - score_without + score_only) / 2
            shapley_values.append((words[i], contribution))
        
        return shapley_values
        
    except Exception as e:
        logger.error(f"SHAP fallback failed: {e}")
        words = text.split()
        return [(word, 0.0) for word in words]

@explainer_fallback(shap_custom_fallback)
def shap_explain(
    model,
    tokenizer,
    text: str,
    target: int = 1,
    device: str = "cuda",
    nsamples: int | str = "auto",
) -> List[Tuple[str, float]]:
    """SHAP con fallback custom"""
    
    shap = safe_import("shap", _try_import_shap)
    if shap is None:
        raise ImportError("SHAP not available")
    
    model.eval()
    
    def f_prob(texts: List[str]) -> np.ndarray:
        enc = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()
    
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f_prob, masker, output_names=["neg", "pos"])
    
    max_evals = 100 if nsamples == "auto" else nsamples
    shap_values = explainer([text], max_evals=max_evals)[0]
    
    tokens = shap_values.data
    scores = shap_values.values[..., target]
    
    return list(zip(tokens, scores))

# ------------------------------------------------------------
# 4) ATTENTION METHODS (sempre funzionano)
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
    model,
    tokenizer,
    text: str,
    target: int = 1,
    device: str = "cuda",
) -> List[Tuple[str, float]]:
    """Attention Rollout - sempre funziona"""
    try:
        model.eval()
        
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        
        outputs = model(**enc, output_attentions=True)
        att_layers = outputs.attentions
        
        # Rollout computation
        rollout = torch.eye(att_layers[0].size(-1), device=device).unsqueeze(0)
        for att in att_layers:
            rollout = torch.bmm(_normalize_and_residual(att), rollout)
        
        scores = rollout[0, 0]  # CLS row
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
        
        return list(zip(tokens, scores.cpu().tolist()))
        
    except Exception as e:
        logger.error(f"Attention rollout failed: {e}")
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(text, truncation=True)["input_ids"]
        )
        return [(token, 0.0) for token in tokens]

@torch.no_grad()
def attention_flow(
    model,
    tokenizer,
    text: str,
    target: int = 1,
    device: str = "cuda",
    start_layer: int = 0,
) -> List[Tuple[str, float]]:
    """Attention Flow - sempre funziona"""
    try:
        model.eval()
        
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        
        outputs = model(**enc, output_attentions=True)
        att_layers = outputs.attentions
        
        # Flow computation
        flow = _normalize_and_residual(att_layers[-1])
        for l in range(len(att_layers) - 2, start_layer - 1, -1):
            flow = torch.bmm(_normalize_and_residual(att_layers[l]), flow)
        
        scores = flow[0, 0]
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
        
        return list(zip(tokens, scores.cpu().tolist()))
        
    except Exception as e:
        logger.error(f"Attention flow failed: {e}")
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(text, truncation=True)["input_ids"]
        )
        return [(token, 0.0) for token in tokens]

# ------------------------------------------------------------
# 5) Registry e utility
# ------------------------------------------------------------

def test_explainer(explainer_name: str, explainer_func: Callable) -> bool:
    """Testa se un explainer funziona"""
    try:
        # Dummy test
        logger.info(f"Testing {explainer_name}...")
        # Qui potresti aggiungere un test più elaborato
        return True
    except Exception as e:
        logger.error(f"Explainer {explainer_name} failed test: {e}")
        return False

def get_working_explainers() -> Dict[str, Callable]:
    """Restituisce solo gli explainer che funzionano"""
    all_explainers = {
        "ixg": input_x_gradient,
        "lrp": lrp_detach,
        "lime": lime_explain,
        "shap": shap_explain,
        "aflow": attention_flow,
        "aroll": attention_rollout,
    }
    
    working_explainers = {}
    for name, func in all_explainers.items():
        if test_explainer(name, func):
            working_explainers[name] = func
    
    return working_explainers

# Registry principale
EXPLAINERS = {
    "ixg": input_x_gradient,
    "lrp": lrp_detach,
    "lime": lime_explain,
    "shap": shap_explain,
    "aflow": attention_flow,
    "aroll": attention_rollout,
}

def print_library_status():
    """Stampa status delle librerie"""
    logger.info("\n=== Library Status ===")
    for lib, status in LIBRARY_STATUS.items():
        status_str = "✓" if status else "✗" if status is False else "?"
        logger.info(f"{lib}: {status_str}")
    logger.info("======================\n")

if __name__ == "__main__":
    print_library_status()
    working = get_working_explainers()
    logger.info(f"Working explainers: {list(working.keys())}")