"""
explainers.py – 
======================================================
metodi XAI implementati per modelli di sentiment analysis: 
- Grad × Input
- Attention Rollout
- Attention Flow
- LRP Conservativa
- LIME
- SHAP
======================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import models

# Dipendenze opzionali
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

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Config
MAX_LEN = 512
DEBUG_TIMING = False

def log_timing(name: str, duration: float):
    if DEBUG_TIMING:
        print(f"[TIMING] {name}: {duration:.3f}s")

class Attribution:
    def __init__(self, tokens: List[str], scores: List[float]):
        self.tokens = tokens
        self.scores = scores
    
    def __repr__(self):
        items = [f"{t}:{s:.3f}" for t, s in zip(self.tokens[:3], self.scores[:3])]
        return "Attribution(" + ", ".join(items) + "...)"

def _safe_tokenize(text: str, tokenizer, max_length=MAX_LEN):
    """Tokenizzazione sicura con GPU support."""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding='max_length')
    return models.move_batch_to_device(encoded)

def _get_embedding_layer(model):
    """Trova layer di embedding."""
    if hasattr(model, 'bert'):
        return model.bert.embeddings.word_embeddings
    elif hasattr(model, 'distilbert'):
        return model.distilbert.embeddings.word_embeddings
    elif hasattr(model, 'roberta'):
        return model.roberta.embeddings.word_embeddings
    else:
        return model.get_input_embeddings()

def _normalize_scores(scores):
    """Normalizza scores in [-0.5, +0.5]."""
    scores = np.array(scores)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = scores - 0.5
    else:
        scores = np.zeros_like(scores)
    return scores.tolist()

# ============================================================================
# GRADIENT × INPUT
# ============================================================================

def _grad_input(model, tokenizer):
    def explain(text: str) -> Attribution:
        start_time = time.time()
        model.eval()
        
        enc = _safe_tokenize(text, tokenizer)
        embed_layer = _get_embedding_layer(model)
        embeds = embed_layer(enc["input_ids"]).detach()
        embeds.requires_grad_(True)
        
        outputs = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"])
        target = outputs.logits[:, 1].sum() if outputs.logits.size(-1) > 1 else outputs.logits.sum()
        
        model.zero_grad()
        target.backward()
        
        if embeds.grad is not None:
            scores = (embeds.grad * embeds).sum(dim=-1).squeeze(0)
        else:
            scores = torch.zeros(enc["input_ids"].size(-1))
        
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        log_timing("grad_input", time.time() - start_time)
        return Attribution(tokens, _normalize_scores(scores.tolist()))
    
    return explain

# ============================================================================
# ATTENTION ROLLOUT (Ali et al. 2022)
# ============================================================================

def _attention_rollout(model, tokenizer):
    def explain(text: str) -> Attribution:
        start_time = time.time()
        model.eval()
        
        enc = _safe_tokenize(text, tokenizer)
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)
        
        if not outputs.attentions:
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            return Attribution(tokens, [0.1] * len(tokens))
        
        # Media su heads: [layers, seq_len, seq_len]
        att = torch.stack([a.mean(dim=1) for a in outputs.attentions]).squeeze(1)
        
        # Joint attention con residual
        I = torch.eye(att.size(-1), device=att.device)
        att = 0.5 * att + 0.5 * I.unsqueeze(0)
        att = att / att.sum(dim=-1, keepdim=True)
        
        # Rollout
        joint = att[0]
        for layer in att[1:]:
            joint = layer @ joint
        
        scores = joint.sum(dim=0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        
        log_timing("attention_rollout", time.time() - start_time)
        return Attribution(tokens, _normalize_scores(scores))
    
    return explain

# ============================================================================
# ATTENTION FLOW (Ali et al. 2022)
# ============================================================================

def _attention_flow(model, tokenizer):
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX richiesto per attention_flow")
    
    def explain(text: str) -> Attribution:
        start_time = time.time()
        model.eval()
        
        enc = _safe_tokenize(text, tokenizer, min(128, MAX_LEN))
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)
        
        if not outputs.attentions:
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            return Attribution(tokens, [0.0] * len(tokens))
        
        # Media su heads e batch
        att = torch.stack([a.mean(dim=1) for a in outputs.attentions]).squeeze(1).cpu().numpy()
        n_layers, seq_len, _ = att.shape
        
        # Residual + normalizzazione
        att = att + np.eye(seq_len)[None, ...]
        att = att / att.sum(axis=-1, keepdims=True)
        
        # Crea grafo semplificato
        G = nx.DiGraph()
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        
        # Solo connessioni significative (>0.1)
        for l in range(n_layers):
            for i in range(seq_len):
                for j in range(seq_len):
                    if att[l, i, j] > 0.1:
                        u, v = l * seq_len + i, (l-1) * seq_len + j if l > 0 else j
                        G.add_edge(u, v, capacity=att[l, i, j])
        
        # Flow dal ultimo layer ai token iniziali
        scores = np.zeros(seq_len)
        try:
            for i in range(min(5, seq_len)):  # Limita per performance
                if G.has_node((n_layers-1) * seq_len + i) and G.has_node(i):
                    flow = nx.maximum_flow_value(G, (n_layers-1) * seq_len + i, i)
                    scores[i] = flow
        except:
            pass
        
        log_timing("attention_flow", time.time() - start_time)
        return Attribution(tokens, _normalize_scores(scores))
    
    return explain

# ============================================================================
# LRP CONSERVATIVA (Ali et al. 2022)
# ============================================================================

def _lrp(model, tokenizer):
    def explain(text: str) -> Attribution:
        start_time = time.time()
        
        # Patch LayerNorm per conservazione
        original_forwards = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                original_forwards[name] = module.forward
                
                def make_conservative_forward(orig_module):
                    def conservative_forward(x):
                        if not x.requires_grad:
                            return orig_module._orig_forward(x)
                        
                        mean = x.mean(dim=-1, keepdim=True)
                        var = x.var(dim=-1, keepdim=True, unbiased=False)
                        centered = x - mean
                        std = torch.sqrt(var + orig_module.eps).detach()  # Detach per conservazione
                        normalized = centered / std
                        
                        if hasattr(orig_module, 'weight') and orig_module.weight is not None:
                            normalized = normalized * orig_module.weight
                        if hasattr(orig_module, 'bias') and orig_module.bias is not None:
                            normalized = normalized + orig_module.bias
                        
                        return normalized
                    return conservative_forward
                
                module._orig_forward = module.forward
                module.forward = make_conservative_forward(module)
        
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer)
            
            # Embedding con gradienti
            embed_layer = _get_embedding_layer(model)
            embeds = embed_layer(enc["input_ids"]).detach()
            embeds.requires_grad_(True)
            
            outputs = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"])
            target = outputs.logits[:, 1].sum() if outputs.logits.size(-1) > 1 else outputs.logits.sum()
            
            model.zero_grad()
            target.backward()
            
            if embeds.grad is not None:
                scores = (embeds.grad * embeds).sum(dim=-1).squeeze(0)
            else:
                scores = torch.zeros(enc["input_ids"].size(-1))
            
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            log_timing("lrp", time.time() - start_time)
            return Attribution(tokens, _normalize_scores(scores.tolist()))
            
        finally:
            # Ripristina forward originali
            for name, module in model.named_modules():
                if hasattr(module, '_orig_forward'):
                    module.forward = module._orig_forward
                    delattr(module, '_orig_forward')
    
    return explain

# ============================================================================
# LIME & SHAP (compatti)
# ============================================================================

def _lime_text(model, tokenizer):
    if not LIME_AVAILABLE:
        raise ImportError("LIME non installato")
    
    explainer_obj = LimeTextExplainer(class_names=["negative", "positive"])
    
    def predict_proba(texts):
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        encoded = models.move_batch_to_device(encoded)
        with torch.no_grad():
            logits = model(**encoded).logits
            return F.softmax(logits, dim=-1).cpu().numpy()
    
    def explain(text: str) -> Attribution:
        start_time = time.time()
        exp = explainer_obj.explain_instance(text, predict_proba, num_features=15, num_samples=100)
        features = exp.as_list()
        tokens, scores = zip(*features) if features else ([], [])
        log_timing("lime", time.time() - start_time)
        return Attribution(list(tokens), list(scores))
    
    return explain

def _kernel_shap(model, tokenizer):
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP non installato")
    
    def predict_proba(texts):
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        encoded = models.move_batch_to_device(encoded)
        with torch.no_grad():
            logits = model(**encoded).logits
            return F.softmax(logits, dim=-1).cpu().numpy()
    
    def explain(text: str) -> Attribution:
        start_time = time.time()
        words = text.split()[:15]  # Limita per velocità
        
        def predict_for_shap(word_presence):
            texts = []
            for presence in word_presence:
                text_words = [words[i] if presence[i] > 0.5 else "[MASK]" for i in range(len(words))]
                texts.append(" ".join(text_words))
            return predict_proba(texts)
        
        background = np.zeros((1, len(words)))
        explainer_obj = shap.KernelExplainer(predict_for_shap, background)
        shap_values = explainer_obj.shap_values(np.ones((1, len(words))), nsamples=30, silent=True)
        
        if isinstance(shap_values, list):
            scores = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            scores = shap_values[0]
        
        log_timing("shap", time.time() - start_time)
        return Attribution(words, scores.tolist())
    
    return explain

# ============================================================================
# FACTORY & API
# ============================================================================

_EXPLAINERS = {
    "grad_input": _grad_input,
    "attention_rollout": _attention_rollout,
    "attention_flow": _attention_flow,
    "lrp": _lrp,
    "lime": _lime_text,
    "shap": _kernel_shap,
}

def list_explainers():
    """Lista explainer disponibili."""
    available = []
    for name in _EXPLAINERS:
        if name == "lime" and not LIME_AVAILABLE:
            continue
        elif name == "shap" and not SHAP_AVAILABLE:
            continue
        elif name == "attention_flow" and not NETWORKX_AVAILABLE:
            continue
        available.append(name)
    return available

def get_explainer(name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Crea explainer."""
    if name not in list_explainers():
        raise ValueError(f"Explainer '{name}' non disponibile. Disponibili: {list_explainers()}")
    return _EXPLAINERS[name](model, tokenizer)

def check_dependencies():
    """Controlla dipendenze."""
    return {
        "LIME": LIME_AVAILABLE,
        "SHAP": SHAP_AVAILABLE,
        "NetworkX": NETWORKX_AVAILABLE,
    }

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing explainers...")
    deps = check_dependencies()
    for lib, status in deps.items():
        print(f"  {lib}: {'OK' if status else 'MISSING'}")
    
    print(f"Available explainers: {list_explainers()}")
    
    try:
        import models
        model = models.load_model("distilbert")
        tokenizer = models.load_tokenizer("distilbert")
        
        for explainer_name in list_explainers()[:3]:  # Test primi 3
            explainer = get_explainer(explainer_name, model, tokenizer)
            result = explainer("Great movie!")
            print(f"{explainer_name}: {len(result.tokens)} tokens")
    except Exception as e:
        print(f"Test failed: {e}")

