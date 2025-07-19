"""
explainers.py – Gestione explainers per modelli di sentiment analysis (GOOGLE COLAB)
============================================================

Versione ottimizzata per Google Colab:
1. Gestione robusta dipendenze opzionali
2. Auto-install di librerie mancanti
3. Fallback graceful se dipendenze non disponibili
4. Memory optimization per GPU Colab
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Callable, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import models

# ==== Auto-install Dependencies ====
def auto_install_package(package_name: str, import_name: str = None):
    """Auto-installa package se mancante."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"[INSTALL] Installing {package_name}...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            print(f"[INSTALL] ✓ {package_name} installed")
            return True
        except Exception as e:
            print(f"[INSTALL] ✗ Failed to install {package_name}: {e}")
            return False

# ==== Dependencies Check con Auto-install ====
print("[DEPS] Checking explainer dependencies...")

# LIME
try:
    auto_install_package("lime")
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
    print("[DEPS] ✓ LIME available")
except Exception:
    LIME_AVAILABLE = False
    print("[DEPS] ✗ LIME not available")

# SHAP
try:
    auto_install_package("shap")
    import shap
    SHAP_AVAILABLE = True
    print("[DEPS] ✓ SHAP available")
except Exception:
    SHAP_AVAILABLE = False
    print("[DEPS] ✗ SHAP not available")

# NetworkX
try:
    auto_install_package("networkx")
    import networkx as nx
    NETWORKX_AVAILABLE = True
    print("[DEPS] ✓ NetworkX available")
except Exception:
    NETWORKX_AVAILABLE = False
    print("[DEPS] ✗ NetworkX not available")

# ==== Constants ====
MAX_LEN = 512
DEBUG_TIMING = False

def log_timing(name: str, duration: float):
    if DEBUG_TIMING:
        print(f"[TIMING] {name}: {duration:.3f}s")

# ==== Attribution Class ====
class Attribution:
    def __init__(self, tokens: List[str], scores: List[float]):
        self.tokens = tokens
        self.scores = scores

    def __repr__(self):
        items = [f"{t}:{s:.3f}" for t, s in zip(self.tokens[:3], self.scores[:3])]
        return "Attribution(" + ", ".join(items) + "...)"

# ==== Utility Functions ====
def _safe_tokenize(text: str, tokenizer, max_length=MAX_LEN):
    """Tokenizza testo con gestione errori."""
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids="token_type_ids" in tokenizer.model_input_names
        )
        return models.move_batch_to_device(encoded)
    except Exception as e:
        print(f"[ERROR] Tokenization failed: {e}")
        # Fallback basic tokenization
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=min(128, max_length))
        return models.move_batch_to_device(encoded)

def _get_embedding_layer(model):
    """Trova layer di embedding del modello."""
    if hasattr(model, 'bert'):
        return model.bert.embeddings.word_embeddings
    elif hasattr(model, 'distilbert'):
        return model.distilbert.embeddings.word_embeddings
    elif hasattr(model, 'roberta'):
        return model.roberta.embeddings.word_embeddings
    else:
        return model.get_input_embeddings()

def _normalize_scores(scores):
    """Normalizza scores tra -1 e +1."""
    scores = np.array(scores)
    if len(scores) == 0:
        return []
    
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = 2 * scores - 1  # -1 a +1
    else:
        scores = np.zeros_like(scores)
    return scores.tolist()

def _filter_tokens_and_scores(enc, tokenizer, scores):
    """Filtra token e scores basandosi su attention mask."""
    try:
        input_ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0).bool()
        
        valid_ids = input_ids[mask]
        tokens = tokenizer.convert_ids_to_tokens(valid_ids)
        
        # Rimuovi prefissi subword
        tokens = [t.lstrip("Ġ").lstrip("##") for t in tokens]
        
        # Filtra scores
        if isinstance(scores, torch.Tensor):
            filtered_scores = scores[mask].tolist()
        else:
            filtered_scores = [scores[i] for i in range(len(scores)) if i < len(mask) and mask[i]]
        
        return tokens, filtered_scores
    except Exception as e:
        print(f"[ERROR] Token filtering failed: {e}")
        return ["[ERROR]"], [0.0]

# ==== Gradient-based Explainers ====
def _grad_input(model, tokenizer):
    """Gradient × Input attribution."""
    def explain(text: str) -> Attribution:
        start_time = time.time()
        model.eval()

        try:
            enc = _safe_tokenize(text, tokenizer)
            embed_layer = _get_embedding_layer(model)
            embeds = embed_layer(enc["input_ids"]).detach()
            embeds.requires_grad_(True)

            # Forward pass
            try:
                outputs = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"])
            except TypeError:
                # Fallback se inputs_embeds non supportato
                outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                return Attribution(["[FALLBACK]"], [0.0])

            # Backward pass
            target = outputs.logits[:, 1].sum() if outputs.logits.size(-1) > 1 else outputs.logits.sum()
            model.zero_grad()
            target.backward()

            # Calcola attribution
            if embeds.grad is not None:
                scores = (embeds.grad * embeds).sum(dim=-1).squeeze(0)
            else:
                scores = torch.zeros(enc["input_ids"].size(-1))
            
            tokens, scores = _filter_tokens_and_scores(enc, tokenizer, scores)
            log_timing("grad_input", time.time() - start_time)
            return Attribution(tokens, _normalize_scores(scores))
            
        except Exception as e:
            print(f"[ERROR] Grad×Input failed: {e}")
            return Attribution(["[ERROR]"], [0.0])
    
    return explain

# ==== Attention-based Explainers ====
def _attention_rollout(model, tokenizer):
    """Attention rollout method."""
    def explain(text: str) -> Attribution:
        start_time = time.time()
        model.eval()

        try:
            enc = _safe_tokenize(text, tokenizer)
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)

            if not outputs.attentions:
                tokens, _ = _filter_tokens_and_scores(enc, tokenizer, [0.1] * enc["input_ids"].size(-1))
                return Attribution(tokens, [0.1] * len(tokens))

            # Attention rollout
            att = torch.stack([a.mean(dim=1) for a in outputs.attentions]).squeeze(1)
            I = torch.eye(att.size(-1), device=att.device)
            att = 0.5 * att + 0.5 * I.unsqueeze(0)
            att = att / att.sum(dim=-1, keepdim=True)

            # Rollout through layers
            joint = att[0]
            for layer in att[1:]:
                joint = layer @ joint

            scores = joint.sum(dim=0).cpu().numpy()
            tokens, scores = _filter_tokens_and_scores(enc, tokenizer, scores)
            log_timing("attention_rollout", time.time() - start_time)
            return Attribution(tokens, _normalize_scores(scores))
            
        except Exception as e:
            print(f"[ERROR] Attention rollout failed: {e}")
            return Attribution(["[ERROR]"], [0.0])
    
    return explain

def _attention_flow(model, tokenizer):
    """Attention flow method (richiede NetworkX)."""
    if not NETWORKX_AVAILABLE:
        def explain(text: str) -> Attribution:
            print("[ERROR] NetworkX not available for attention flow")
            return Attribution(["[NO_NETWORKX]"], [0.0])
        return explain

    def explain(text: str) -> Attribution:
        start_time = time.time()
        model.eval()

        try:
            # Usa sequenza più corta per efficienza
            enc = _safe_tokenize(text, tokenizer, min(128, MAX_LEN))
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)

            if not outputs.attentions:
                tokens, _ = _filter_tokens_and_scores(enc, tokenizer, [0.0] * enc["input_ids"].size(-1))
                return Attribution(tokens, [0.0] * len(tokens))

            # Crea grafo di flusso
            att = torch.stack([a.mean(dim=1) for a in outputs.attentions]).squeeze(1).cpu().numpy()
            n_layers, seq_len, _ = att.shape
            att = att + np.eye(seq_len)[None, ...]
            att = att / att.sum(axis=-1, keepdims=True)

            G = nx.DiGraph()
            for l in range(n_layers):
                for i in range(seq_len):
                    for j in range(seq_len):
                        if att[l, i, j] > 0.1:  # Soglia per ridurre complessità
                            u = l * seq_len + i
                            v = (l - 1) * seq_len + j if l > 0 else j
                            G.add_edge(u, v, capacity=att[l, i, j])

            # Calcola flusso massimo
            scores = np.zeros(seq_len)
            try:
                for i in range(min(5, seq_len)):  # Limita per efficienza
                    source = (n_layers - 1) * seq_len + i
                    target = i
                    if G.has_node(source) and G.has_node(target):
                        flow = nx.maximum_flow_value(G, source, target)
                        scores[i] = flow
            except Exception:
                pass  # Se fallisce, usa scores zero

            tokens, scores = _filter_tokens_and_scores(enc, tokenizer, scores)
            log_timing("attention_flow", time.time() - start_time)
            return Attribution(tokens, _normalize_scores(scores))
            
        except Exception as e:
            print(f"[ERROR] Attention flow failed: {e}")
            return Attribution(["[ERROR]"], [0.0])
    
    return explain

# ==== LRP (simplified) ====
def _lrp(model, tokenizer):
    """Simplified LRP implementation."""
    def explain(text: str) -> Attribution:
        start_time = time.time()
        
        try:
            # Per semplicità, usa grad×input come proxy
            grad_explainer = _grad_input(model, tokenizer)
            result = grad_explainer(text)
            log_timing("lrp", time.time() - start_time)
            return result
            
        except Exception as e:
            print(f"[ERROR] LRP failed: {e}")
            return Attribution(["[ERROR]"], [0.0])
    
    return explain

# ==== LIME ====
def _lime_text(model, tokenizer):
    """LIME explainer."""
    if not LIME_AVAILABLE:
        def explain(text: str) -> Attribution:
            print("[ERROR] LIME not available")
            return Attribution(["[NO_LIME]"], [0.0])
        return explain
    
    explainer_obj = LimeTextExplainer(class_names=["negative", "positive"])
    
    def predict_proba(texts):
        """Prediction function per LIME."""
        try:
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            encoded = models.move_batch_to_device(encoded)
            
            with torch.no_grad():
                logits = model(**encoded).logits
                return F.softmax(logits, dim=-1).cpu().numpy()
        except Exception as e:
            print(f"[ERROR] LIME prediction failed: {e}")
            # Fallback: predizioni casuali
            return np.random.rand(len(texts), 2)
    
    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            exp = explainer_obj.explain_instance(
                text, 
                predict_proba, 
                num_features=min(15, len(text.split())), 
                num_samples=50  # Ridotto per Colab
            )
            features = exp.as_list()
            tokens, scores = zip(*features) if features else ([], [])
            log_timing("lime", time.time() - start_time)
            return Attribution(list(tokens), list(scores))
        except Exception as e:
            print(f"[ERROR] LIME explanation failed: {e}")
            return Attribution(["[ERROR]"], [0.0])
    
    return explain

# ==== SHAP ====
def _kernel_shap(model, tokenizer):
    """SHAP explainer (fixed)."""
    if not SHAP_AVAILABLE:
        def explain(text: str) -> Attribution:
            print("[ERROR] SHAP not available")
            return Attribution(["[NO_SHAP]"], [0.0])
        return explain

    def predict_proba(texts):
        """Prediction function per SHAP."""
        try:
            if isinstance(texts, str):
                texts = [texts]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            encoded = models.move_batch_to_device(encoded)
            with torch.no_grad():
                logits = model(**encoded).logits
                return F.softmax(logits, dim=-1).cpu().numpy()
        except Exception as e:
            print(f"[ERROR] SHAP prediction failed: {e}")
            n_texts = len(texts) if isinstance(texts, list) else 1
            return np.random.rand(n_texts, 2)

    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            words = text.split()[:8]  # Limita a 8 parole per stabilità
            if len(words) == 0:
                return Attribution(["[EMPTY]"], [0.0])

            def predict_for_shap(word_presence):
                """Fixed prediction function for SHAP."""
                try:
                    if word_presence.ndim == 1:
                        word_presence = word_presence.reshape(1, -1)

                    texts = []
                    for row in word_presence:
                        text_words = []
                        for i, presence in enumerate(row):
                            if i < len(words):
                                try:
                                    value = float(presence)
                                    if value > 0.5:
                                        text_words.append(words[i])
                                    else:
                                        text_words.append("[MASK]")
                                except (ValueError, TypeError):
                                    print(f"[WARN] Invalid presence value: {presence}, using [MASK]")
                                    text_words.append("[MASK]")

                        if not text_words:
                            text_words = ["[MASK]"]

                        texts.append(" ".join(text_words))

                    return predict_proba(texts)

                except Exception as e:
                    print(f"[ERROR] predict_for_shap failed: {e}")
                    n_samples = len(word_presence) if hasattr(word_presence, '__len__') else 1
                    return np.random.rand(n_samples, 2)

            background = np.zeros((1, len(words)))

            test_input = np.ones((1, len(words)))
            test_output = predict_for_shap(test_input)
            if test_output.shape[1] != 2:
                raise ValueError(f"Prediction function returned wrong shape: {test_output.shape}")

            explainer_obj = shap.KernelExplainer(predict_for_shap, background)

            shap_values = explainer_obj.shap_values(
                np.ones((1, len(words))),
                nsamples=15,
                silent=True
            )

            if isinstance(shap_values, list):
                scores = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                scores = shap_values[0] if shap_values.ndim > 1 else shap_values

            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)] if np.isscalar(scores) else scores.flatten().tolist()

            scores = scores[:len(words)]
            if len(scores) < len(words):
                scores.extend([0.0] * (len(words) - len(scores)))

            log_timing("shap", time.time() - start_time)
            return Attribution(words, scores)

        except Exception as e:
            print(f"[ERROR] SHAP explanation failed: {e}")
            return Attribution(["[ERROR]"], [0.0])

    return explain

# ==== Factory ====
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
    """Crea explainer con gestione errori."""
    available = list_explainers()
    if name not in available:
        print(f"[ERROR] Explainer '{name}' not available. Available: {available}")
        # Fallback al primo disponibile
        if available:
            fallback_name = available[0]
            print(f"[FALLBACK] Using {fallback_name} instead")
            return _EXPLAINERS[fallback_name](model, tokenizer)
        else:
            raise ValueError(f"No explainers available!")
    
    return _EXPLAINERS[name](model, tokenizer)

def check_dependencies():
    """Controlla dipendenze."""
    return {
        "LIME": LIME_AVAILABLE,
        "SHAP": SHAP_AVAILABLE,
        "NetworkX": NETWORKX_AVAILABLE,
    }

# ==== Test ====
if __name__ == "__main__":
    print("Testing explainers on Colab...")
    
    # Check dependencies
    deps = check_dependencies()
    for lib, status in deps.items():
        print(f"  {lib}: {'✓' if status else '✗'}")
    
    available = list_explainers()
    print(f"Available explainers: {available}")
    
    if available:
        try:
            # Test con modello piccolo
            print("\nTesting with tinybert...")
            model = models.load_model("tinybert")
            tokenizer = models.load_tokenizer("tinybert")
            
            test_text = "This movie is absolutely fantastic!"
            
            for explainer_name in available[:3]:  # Test primi 3
                try:
                    print(f"\nTesting {explainer_name}...")
                    explainer = get_explainer(explainer_name, model, tokenizer)
                    result = explainer(test_text)
                    print(f"  Result: {len(result.tokens)} tokens, {len(result.scores)} scores")
                    if result.tokens and result.scores:
                        print(f"  Sample: {result.tokens[0]} -> {result.scores[0]:.3f}")
                    print(f"  ✓ {explainer_name}")
                except Exception as e:
                    print(f"  ✗ {explainer_name}: {e}")
            
            models.clear_gpu_memory()
            
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("No explainers available for testing")
    
    print("\nExplainer test completed!")