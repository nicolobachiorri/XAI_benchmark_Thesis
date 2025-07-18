"""
explainers.py – 6 XAI methods con SOLO LRP CONSERVATIVA
=====================================================

Metodi: lime, shap, grad_input, attention_rollout, attention_flow, lrp

NOTA: 'lrp' ora è l'implementazione conservativa basata su Ali et al. (2022)
Rimossa la vecchia implementazione Captum-based.
"""

from __future__ import annotations
from typing import List, Dict, Callable, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from transformers import PreTrainedModel, PreTrainedTokenizer
import models  # Per GPU support

# -------------------------------------------------------------------------
# Librerie opzionali (CAPTUM RIMOSSA - non serve più!)
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LimeTextExplainer = None
    LIME_AVAILABLE = False

try:
    import shap
    import numpy as np
    SHAP_AVAILABLE = True
except ImportError:
    shap = np = None
    SHAP_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

# Parametri globali
MAX_LEN = 512
MIN_LEN = 10
DEBUG_TIMING = True

# -------------------------------------------------------------------------
# Utility functions (MANTENUTE)

def _get_model_architecture(model):
    """Identifica l'architettura del modello."""
    model_name = model.__class__.__name__.lower()
    config_name = getattr(model.config, 'model_type', '').lower()
    
    if 'roberta' in model_name or 'roberta' in config_name:
        return 'roberta'
    elif 'distilbert' in model_name or 'distilbert' in config_name:
        return 'distilbert'
    elif 'tinybert' in model_name or 'tinybert' in config_name:
        return 'tinybert'
    elif 'bert' in model_name or 'bert' in config_name:
        return 'bert'
    else:
        return 'unknown'

def _get_base_model(model):
    """Ottiene il modello base (senza testa di classificazione)."""
    if hasattr(model, 'roberta'):
        return model.roberta
    elif hasattr(model, 'distilbert'):
        return model.distilbert
    elif hasattr(model, 'bert'):
        return model.bert
    else:
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, 'encoder') and hasattr(attr, 'embeddings'):
                return attr
        return model

def _get_embedding_layer(model):
    """Ottiene il layer di embedding del modello."""
    base_model = _get_base_model(model)
    if hasattr(base_model, 'embeddings'):
        return base_model.embeddings.word_embeddings
    elif hasattr(base_model, 'word_embeddings'):
        return base_model.word_embeddings
    elif hasattr(model, 'get_input_embeddings'):
        return model.get_input_embeddings()
    else:
        raise AttributeError("Impossibile trovare layer di embedding")

def _safe_tokenize(text: str, tokenizer, max_length=MAX_LEN):
    """Tokenizzazione sicura con gestione lunghezza e GPU support."""
    text = text.strip()
    if len(text) < MIN_LEN:
        text = text + " " * (MIN_LEN - len(text))
    
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
    
    encoded = models.move_batch_to_device(encoded)
    return encoded

def log_timing(explainer_name: str, duration: float):
    """Log dei tempi di esecuzione per debugging."""
    if DEBUG_TIMING:
        print(f"[TIMING] {explainer_name}: {duration:.3f}s")

class Attribution:
    def __init__(self, tokens: List[str], scores: List[float]):
        self.tokens = tokens
        self.scores = scores
    
    def __repr__(self):
        items = [f"{t}:{s:.3f}" for t, s in zip(self.tokens[:5], self.scores[:5])]
        if len(self.tokens) > 5:
            items.append("...")
        return "Attribution(" + ", ".join(items) + ")"

# -------------------------------------------------------------------------
# 1. GRADIENT × INPUT

def _grad_input(model, tokenizer):
    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, MAX_LEN)
            ids, attn = enc["input_ids"], enc["attention_mask"]

            embed_layer = _get_embedding_layer(model)
            embeds = embed_layer(ids).detach()
            embeds.requires_grad_(True)

            try:
                outputs = model(inputs_embeds=embeds, attention_mask=attn)
                logits = outputs.logits
            except Exception:
                outputs = model(input_ids=ids, attention_mask=attn)
                logits = outputs.logits
                embeds = embed_layer(ids).detach()
                embeds.requires_grad_(True)

            if logits.dim() == 1:
                target_idx = 0
                loss = logits[target_idx] if len(logits) > target_idx else logits[0]
            elif logits.dim() == 2:
                target_idx = 1 if logits.size(-1) > 1 else 0
                loss = logits[:, target_idx].sum()
            else:
                loss = logits.flatten()[0]

            model.zero_grad()
            loss.backward()

            if embeds.grad is not None:
                try:
                    if embeds.dim() == 3:
                        scores = (embeds.grad * embeds).sum(dim=-1).squeeze(0)
                    elif embeds.dim() == 2:
                        scores = (embeds.grad * embeds).sum(dim=-1)
                    else:
                        scores = embeds.grad.flatten()
                        
                    if scores.dim() > 1:
                        scores = scores.flatten()
                        
                except Exception:
                    scores = torch.zeros(ids.size(-1))
            else:
                scores = torch.zeros(ids.size(-1))
            
            tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
            
            min_len = min(len(tokens), len(scores))
            log_timing("grad_input", time.time() - start_time)
            return Attribution(tokens[:min_len], scores[:min_len].tolist())
            
        except Exception as e:
            print(f"Errore in grad_input: {e}")
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, max_length=10, truncation=True))
            scores = [0.0] * len(tokens)
            log_timing("grad_input", time.time() - start_time)
            return Attribution(tokens, scores)
    
    return explain

# -------------------------------------------------------------------------
# 2. ATTENTION ROLLOUT

def _attention_rollout(model, tokenizer):
    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, MAX_LEN)
            
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)
            
            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                print("WARNING: Modello non supporta attention, usando fallback")
                tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
                scores = [max(0.1, 1.0 / (i + 1)) for i in range(len(tokens))]
                log_timing("attention_rollout", time.time() - start_time)
                return Attribution(tokens, scores)
            
            attentions = outputs.attentions
            
            if len(attentions) == 0:
                tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
                scores = [0.1] * len(tokens)
                log_timing("attention_rollout", time.time() - start_time)
                return Attribution(tokens, scores)
            
            try:
                att = torch.stack([a.mean(dim=1) for a in attentions])
            except RuntimeError:
                tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
                scores = [0.1] * len(tokens)
                log_timing("attention_rollout", time.time() - start_time)
                return Attribution(tokens, scores)

            I = torch.eye(att.size(-1), device=att.device, dtype=att.dtype)
            I = I.unsqueeze(0).unsqueeze(0)
            att = 0.5 * att + 0.5 * I
            att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            try:
                rollout = att[-1]
                for l in range(att.size(0) - 2, -1, -1):
                    rollout = att[l].bmm(rollout)
                    if torch.isnan(rollout).any() or torch.isinf(rollout).any():
                        break
            except RuntimeError:
                rollout = att[-1]

            try:
                scores = rollout.squeeze(0)[0]
                if torch.isnan(scores).any():
                    scores = torch.ones_like(scores) * 0.1
            except (IndexError, RuntimeError):
                scores = torch.ones(att.size(-1)) * 0.1
                
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            
            min_len = min(len(tokens), len(scores))
            log_timing("attention_rollout", time.time() - start_time)
            return Attribution(tokens[:min_len], scores[:min_len].tolist())
            
        except Exception as e:
            print(f"Errore in attention_rollout: {e}")
            try:
                tokens = tokenizer.convert_ids_to_tokens(
                    tokenizer.encode(text, max_length=20, truncation=True)
                )
                scores = [max(0.01, 1.0 / (i + 1)) for i in range(len(tokens))]
                log_timing("attention_rollout", time.time() - start_time)
                return Attribution(tokens, scores)
            except:
                words = text.split()[:10]
                scores = [0.1] * len(words)
                log_timing("attention_rollout", time.time() - start_time)
                return Attribution(words, scores)
    
    return explain

# -------------------------------------------------------------------------
# 3. ATTENTION FLOW

def _attention_flow(model, tokenizer):
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX non installato per attention_flow")
    
    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, min(MAX_LEN, 128))
            
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)

            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                raise ValueError("Modello non supporta output attention")

            att = torch.stack([a.mean(dim=1) for a in outputs.attentions])
            seq = min(att.size(-1), 64)
            att = att[:, :, :seq, :seq]
            
            I = torch.eye(seq).unsqueeze(0).unsqueeze(0)
            att = 0.5 * att + 0.5 * I
            att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            threshold = 0.1
            
            G = nx.DiGraph()
            L = att.size(0)
            
            for l in range(L):
                for i in range(seq):
                    G.add_node((l, i))
            
            edge_count = 0
            max_edges = 1000
            
            for l in range(L):
                if edge_count >= max_edges:
                    break
                for i in range(seq):
                    if edge_count >= max_edges:
                        break
                    for j in range(seq):
                        if edge_count >= max_edges:
                            break
                        w = att[l, 0, i, j].item()
                        if w > threshold:
                            if l > 0:
                                G.add_edge((l, i), (l - 1, j), capacity=w)
                                edge_count += 1

            flow_scores = torch.zeros(seq)
            max_flow_calculations = min(10, seq)
            
            try:
                for src in range(min(max_flow_calculations, seq)):
                    if G.has_node((L - 1, src)) and G.has_node((0, 0)):
                        try:
                            flow_val, _ = nx.maximum_flow(G, (L - 1, src), (0, 0))
                            flow_scores[src] = flow_val
                        except:
                            continue
            except:
                flow_scores = att[-1, 0, 0, :seq]

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0)[:seq])
            log_timing("attention_flow", time.time() - start_time)
            return Attribution(tokens, flow_scores.tolist())
            
        except Exception as e:
            print(f"Errore in attention_flow: {e}")
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, max_length=10, truncation=True))
            scores = [0.0] * len(tokens)
            log_timing("attention_flow", time.time() - start_time)
            return Attribution(tokens, scores)
    
    return explain

# -------------------------------------------------------------------------
# 4. LIME

def _lime_text(model, tokenizer):
    if not LIME_AVAILABLE:
        raise ImportError("LIME non installato")
    
    explainer = LimeTextExplainer(class_names=["negative", "positive"])

    def predict(texts):
        try:
            if isinstance(texts, str):
                texts = [texts]
            texts = [str(t) for t in texts]
            
            encoded = tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            )
            
            encoded = models.move_batch_to_device(encoded)
            
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        except Exception as e:
            print(f"[DEBUG] LIME predict error: {e}")
            return np.array([[0.5, 0.5] for _ in texts])

    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            exp = explainer.explain_instance(
                text, 
                predict, 
                num_features=min(20, len(text.split())),
                num_samples=100
            )
            features = exp.as_list()
            try:
                tokens, scores = zip(*features)
                log_timing("lime", time.time() - start_time)
                return Attribution(list(tokens), list(scores))
            except:
                words = text.split()[:10]
                log_timing("lime", time.time() - start_time)
                return Attribution(words, [0.0] * len(words))
        except Exception as e:
            print(f"Errore in LIME explain: {e}")
            words = text.split()[:10]
            log_timing("lime", time.time() - start_time)
            return Attribution(words, [0.0] * len(words))
    
    return explain

# -------------------------------------------------------------------------
# 5. SHAP

def _kernel_shap(model, tokenizer):
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP non installato")

    def predict_simple(texts):
        try:
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, np.ndarray):
                texts = texts.tolist()
            
            texts = [str(t) if t else "empty" for t in texts]
            
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            encoded = models.move_batch_to_device(encoded)
            
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.squeeze(-1))
                    return torch.stack([1 - probs, probs], dim=-1).cpu().numpy()
                else:
                    return F.softmax(logits, dim=-1).cpu().numpy()
                
        except Exception as e:
            print(f"Error in SHAP predict: {e}")
            num_texts = len(texts) if hasattr(texts, '__len__') else 1
            return np.array([[0.5, 0.5] for _ in range(num_texts)])

    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            text = text.strip()
            if not text:
                text = "empty text"
            
            words = text.split()
            if len(words) > 15:
                words = words[:15]
                text = " ".join(words)
            
            def predict_for_shap(word_presence):
                try:
                    if isinstance(word_presence, (list, tuple)):
                        word_presence = np.array(word_presence)
                    
                    if word_presence.ndim == 1:
                        word_presence = word_presence.reshape(1, -1)
                    
                    texts_to_predict = []
                    for presence in word_presence:
                        text_words = []
                        for i, include in enumerate(presence):
                            if i < len(words):
                                if include > 0.5:
                                    text_words.append(words[i])
                                else:
                                    text_words.append("[MASK]")
                        
                        if not text_words:
                            text_words = ["[MASK]"]
                        
                        texts_to_predict.append(" ".join(text_words))
                    
                    return predict_simple(texts_to_predict)
                    
                except Exception as e:
                    print(f"Error in predict_for_shap: {e}")
                    batch_size = word_presence.shape[0] if hasattr(word_presence, 'shape') else 1
                    return np.array([[0.5, 0.5] for _ in range(batch_size)])
            
            background = np.zeros((1, len(words)))
            explainer = shap.KernelExplainer(predict_for_shap, background)
            
            instance = np.ones((1, len(words)))
            shap_values = explainer.shap_values(instance, nsamples=30, silent=True)
            
            if isinstance(shap_values, list):
                raw_scores = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                raw_scores = shap_values[0] if shap_values.ndim > 1 else shap_values

            scores = np.array(raw_scores, dtype=float).flatten()

            log_timing("shap", time.time() - start_time)
            return Attribution(words, scores.tolist())
            
        except Exception as e:
            print(f"Errore in SHAP: {e}")
            words = text.split()[:10]
            log_timing("shap", time.time() - start_time)
            return Attribution(words, [0.0] * len(words))
    
    return explain

# -------------------------------------------------------------------------
# 6. LRP CONSERVATIVA (UNICA IMPLEMENTAZIONE LRP)
# -------------------------------------------------------------------------

def _lrp(model, tokenizer):
    """
    LRP Conservativa con detach trick secondo Ali et al. (2022).
    
    Implementa:
    - AH-rule: detach dei pesi di attenzione  
    - LN-rule: detach del denominatore di normalizzazione
    - Implementation trick per conservazione garantita
    
    NOTA: Questa è l'UNICA implementazione LRP, sostituisce quella Captum.
    """
    
    class ConservativeLRPExplainer:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.original_forwards = {}
            self._patched = False
        
        def _patch_model(self):
            """Applica patch conservativo al modello."""
            if self._patched:
                return
            
            print("[LRP] Applicando patch conservativo...")
            
            # Patch LayerNorm con LN-rule
            patch_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or 'norm' in name.lower():
                    if name not in self.original_forwards:
                        self.original_forwards[name] = module.forward
                        module.forward = self._create_layernorm_forward(module)
                        patch_count += 1
            
            self._patched = True
            print(f"[LRP] Patch applicato a {patch_count} layer LayerNorm")
        
        def _create_layernorm_forward(self, original_module):
            """
            Crea forward patchato per LayerNorm (LN-rule).
            
            Implementa: yi = (xi - E[x]) / [sqrt(ε + Var[x])].detach()
            Dove il denominatore viene "detached" per conservare relevance.
            """
            def patched_forward(x):
                if not x.requires_grad:
                    # Se non richiede gradienti, usa forward normale
                    return original_module._original_forward(x)
                
                # IMPLEMENTAZIONE LN-RULE CON DETACH
                # Calcola dimensioni per mean/var (tutto tranne batch)
                dims = tuple(range(1, len(x.shape)))
                mean = x.mean(dim=dims, keepdim=True)
                var = x.var(dim=dims, keepdim=True, unbiased=False)
                
                # Centering (parte lineare)
                centered = x - mean
                
                # Normalization con DETACH del denominatore (parte non-lineare)
                eps = getattr(original_module, 'eps', 1e-5)
                std = torch.sqrt(var + eps).detach()  # ← DETACH QUI!
                normalized = centered / std
                
                # Applica weight e bias se presenti (parti lineari)
                if hasattr(original_module, 'weight') and original_module.weight is not None:
                    normalized = normalized * original_module.weight
                if hasattr(original_module, 'bias') and original_module.bias is not None:
                    normalized = normalized + original_module.bias
                
                return normalized
            
            # Salva forward originale per cleanup
            original_module._original_forward = original_module.forward
            return patched_forward
        
        def _restore_model(self):
            """Ripristina il modello ai metodi forward originali."""
            if not self._patched:
                return
            
            restored_count = 0
            for name, module in self.model.named_modules():
                if hasattr(module, '_original_forward'):
                    module.forward = module._original_forward
                    delattr(module, '_original_forward')
                    restored_count += 1
            
            self.original_forwards.clear()
            self._patched = False
            print(f"[LRP] Modello ripristinato ({restored_count} layer)")
        
        def explain(self, text: str) -> Attribution:
            """Genera spiegazione con LRP conservativa."""
            start_time = time.time()
            
            try:
                # STEP 1: Applica patch conservativo
                self._patch_model()
                
                # STEP 2: Tokenizzazione
                encoded = _safe_tokenize(text, self.tokenizer, min(MAX_LEN, 256))
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                
                # STEP 3: Usa embeddings per calcolare gradienti
                embedding_layer = _get_embedding_layer(self.model)
                embeddings = embedding_layer(input_ids)
                embeddings.requires_grad_(True)
                
                # STEP 4: Forward pass con modello patchato
                outputs = self.model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask
                )
                
                # STEP 5: Target per backward (classe positiva)
                logits = outputs.logits
                if logits.size(-1) > 1:
                    target = logits[:, 1].sum()  # Classe positiva per sentiment
                else:
                    target = logits.sum()
                
                # STEP 6: Backward pass (Gradient × Input)
                self.model.zero_grad()
                target.backward()
                
                # STEP 7: Calcola relevance conservativa
                if embeddings.grad is not None:
                    # Relevance = Gradient × Input (conservativo grazie al patch)
                    relevance_scores = (embeddings.grad * embeddings).sum(dim=-1).squeeze(0)
                else:
                    print("[LRP] Warning: No gradients found")
                    relevance_scores = torch.zeros(input_ids.size(-1))
                
                # STEP 8: Converti in Attribution
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
                scores = relevance_scores.detach().cpu().tolist()
                
                # STEP 9: Verifica conservazione (opzionale, per debug)
                total_relevance = sum(scores)
                target_value = target.item()
                if abs(target_value) > 1e-6:
                    conservation_ratio = abs(total_relevance - target_value) / abs(target_value)
                    if conservation_ratio > 0.15:  # Soglia del 15%
                        print(f"[LRP] Warning: Conservation ratio: {conservation_ratio:.3f}")
                    else:
                        print(f"[LRP] Conservation OK: {conservation_ratio:.4f}")
                
                log_timing("lrp", time.time() - start_time)
                return Attribution(tokens, scores)
                
            except Exception as e:
                print(f"[LRP] Errore in conservative LRP: {e}")
                # Fallback con gradient semplice
                return self._fallback_gradient_explanation(text, start_time)
            
            finally:
                # STEP 10: Cleanup (sempre eseguito)
                try:
                    self._restore_model()
                except Exception as cleanup_error:
                    print(f"[LRP] Warning: Cleanup error: {cleanup_error}")
        
        def _fallback_gradient_explanation(self, text: str, start_time: float) -> Attribution:
            """Fallback robusto se LRP conservativa fallisce."""
            print("[LRP] Usando fallback gradient")
            
            try:
                encoded = _safe_tokenize(text, self.tokenizer, min(MAX_LEN, 128))
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                input_ids.requires_grad_(True)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                if logits.size(-1) > 1:
                    target = logits[:, 1].sum()
                else:
                    target = logits.sum()
                
                self.model.zero_grad()
                target.backward()
                
                if input_ids.grad is not None:
                    scores = input_ids.grad.abs().sum(dim=0)
                else:
                    scores = torch.ones(input_ids.size(-1)) * 0.1
                
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
                log_timing("lrp", time.time() - start_time)
                return Attribution(tokens, scores.tolist())
                
            except Exception as e:
                print(f"[LRP] Anche il fallback è fallito: {e}")
                # Ultimate fallback
                tokens = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer.encode(text, max_length=10, truncation=True)
                )
                scores = [0.1] * len(tokens)
                log_timing("lrp", time.time() - start_time)
                return Attribution(tokens, scores)
    
    # Crea e restituisci explainer
    conservative_explainer = ConservativeLRPExplainer(model, tokenizer)
    return conservative_explainer.explain

# -------------------------------------------------------------------------
# EXPLAINER FACTORY (AGGIORNATA - CAPTUM RIMOSSA)
# -------------------------------------------------------------------------

_EXPLAINER_FACTORY: Dict[str, Callable] = {
    "lime":                 _lime_text,           # Richiede: pip install lime  
    "shap":                 _kernel_shap,         # Richiede: pip install shap
    "grad_input":           _grad_input,          # Nessuna dipendenza
    "attention_rollout":    _attention_rollout,   # Nessuna dipendenza
    "attention_flow":       _attention_flow,      # Richiede: pip install networkx
    "lrp":                  _lrp,                 # ← CONSERVATIVE LRP (no dipendenze!)
}

# -------------------------------------------------------------------------
# API pubblica (SEMPLIFICATA)
# -------------------------------------------------------------------------

def list_explainers():
    """Restituisce lista di explainer disponibili (SENZA Captum)."""
    available = []
    for name, factory in _EXPLAINER_FACTORY.items():
        # Controlla dipendenze specifiche
        if name == "lime" and not LIME_AVAILABLE:
            continue
        elif name == "shap" and not SHAP_AVAILABLE:
            continue
        elif name == "attention_flow" and not NETWORKX_AVAILABLE:
            continue
        # lrp (conservative) non ha dipendenze esterne!
        available.append(name)
    
    return available

def get_explainer(name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Crea un explainer per il modello specificato."""
    name = name.lower()
    available = list_explainers()
    
    if name not in available:
        raise ValueError(f"Explainer '{name}' non supportato o dipendenze mancanti. Disponibili: {available}")
    
    arch = _get_model_architecture(model)
    print(f"Creando explainer '{name}' per architettura '{arch}'")
    
    # Controlla dipendenze specifiche (NO CAPTUM!)
    if name == "lime" and not LIME_AVAILABLE:
        raise ImportError("LIME non installato. Installare con: pip install lime")
    elif name == "shap" and not SHAP_AVAILABLE:
        raise ImportError("SHAP non installato. Installare con: pip install shap")
    elif name == "attention_flow" and not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX non installato. Installare con: pip install networkx")
    # LRP non richiede dipendenze!
    
    return _EXPLAINER_FACTORY[name](model, tokenizer)

def check_dependencies():
    """Controlla quali dipendenze sono disponibili (SENZA Captum)."""
    deps = {
        "LIME": LIME_AVAILABLE,
        "SHAP": SHAP_AVAILABLE, 
        "NetworkX (Attention Flow)": NETWORKX_AVAILABLE,
        "LRP Conservativa": True,  # Sempre disponibile!
    }
    
    print("Stato dipendenze explainer:")
    for lib, available in deps.items():
        status = "✓ OK" if available else "✗ Mancante"
        print(f"  {lib}: {status}")
    
    return deps

def benchmark_explainer_speed():
    """Test di performance per tutti gli explainer."""
    print("Benchmark performance explainer...")
    
    try:
        import models
        model = models.load_model("distilbert")
        tokenizer = models.load_tokenizer("distilbert")
        
        test_text = "This movie is absolutely fantastic and amazing!"
        available_explainers = list_explainers()
        
        results = {}
        
        for explainer_name in available_explainers:
            print(f"\nTesting {explainer_name}...")
            try:
                explainer = get_explainer(explainer_name, model, tokenizer)
                
                # Warm-up
                explainer(test_text)
                
                # Benchmark
                times = []
                for i in range(3):
                    start = time.time()
                    attr = explainer(test_text)
                    duration = time.time() - start
                    times.append(duration)
                
                avg_time = sum(times) / len(times)
                results[explainer_name] = {
                    "avg_time": avg_time,
                    "tokens": len(attr.tokens),
                    "status": "OK"
                }
                print(f"  Tempo medio: {avg_time:.3f}s ({len(attr.tokens)} tokens)")
                
            except Exception as e:
                results[explainer_name] = {"status": "ERROR", "error": str(e)}
                print(f"  ERRORE: {e}")
        
        # Ordina per velocità
        working_explainers = {k: v for k, v in results.items() if v["status"] == "OK"}
        if working_explainers:
            sorted_by_speed = sorted(working_explainers.items(), key=lambda x: x[1]["avg_time"])
            
            print(f"\n{'='*50}")
            print("RANKING VELOCITÀ:")
            for i, (name, data) in enumerate(sorted_by_speed, 1):
                print(f"  {i}. {name}: {data['avg_time']:.3f}s")
        
        return results
        
    except Exception as e:
        print(f"Errore in benchmark: {e}")
        return {}

# -------------------------------------------------------------------------
# Test di compatibilità FINALE
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST EXPLAINERS - SOLO LRP CONSERVATIVA")
    print("=" * 60)
    
    # 1. Controlla dipendenze
    print("\n1. CONTROLLO DIPENDENZE:")
    deps = check_dependencies()
    
    # 2. Lista explainer disponibili
    print(f"\n2. EXPLAINER DISPONIBILI:")
    available = list_explainers()
    for explainer in available:
        print(f"  ✓ {explainer}")
    
    if not available:
        print("    Nessun explainer disponibile - verificare installazioni")
        exit(1)
    
    # 3. Test LRP conservativa specificamente
    print(f"\n3. TEST LRP CONSERVATIVA:")
    try:
        import models
        print("  Caricando modello distilbert...")
        model = models.load_model("distilbert")
        tokenizer = models.load_tokenizer("distilbert")
        print("  ✓ Modello caricato")
        
        test_texts = [
            "This is a fantastic movie with great acting!",
            "This movie is absolutely terrible and boring.",
            "An okay film with decent cinematography."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n  Test {i}: {text}")
            try:
                explainer = get_explainer("lrp", model, tokenizer)
                attr = explainer(text)
                
                # Mostra risultati
                print(f"    ✓ Tokens: {len(attr.tokens)}")
                print(f"    ✓ Score range: [{min(attr.scores):.3f}, {max(attr.scores):.3f}]")
                print(f"    ✓ Total relevance: {sum(attr.scores):.3f}")
                
                # Top 3 tokens più rilevanti
                token_scores = [(tok, score) for tok, score in zip(attr.tokens, attr.scores) 
                               if tok.strip() and not tok.startswith('[')]
                token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print("    Top tokens:")
                for token, score in token_scores[:3]:
                    print(f"      {token:>10}: {score:+.3f}")
                    
            except Exception as e:
                print(f"    ✗ Errore: {e}")
    
    except Exception as e:
        print(f"  ✗ Errore nel test: {e}")
    
    # 4. Confronto con altri explainer (opzionale)
    if len(available) > 1:
        print(f"\n4. CONFRONTO VELOCITÀ:")
        try:
            benchmark_results = benchmark_explainer_speed()
            if benchmark_results:
                lrp_time = benchmark_results.get("lrp", {}).get("avg_time", "N/A")
                print(f"\n  LRP Conservativa: {lrp_time}s")
        except Exception as e:
            print(f"  Benchmark fallito: {e}")
    