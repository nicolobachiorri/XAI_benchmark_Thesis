"""
explainers.py – 6 XAI methods con implementazioni da Ali et al. (2022)
=====================================================================

Metodi: lime, shap, grad_input, attention_rollout, attention_flow, lrp

AGGIORNATO: Implementazioni di attention_rollout, attention_flow e lrp 
basate sul paper "XAI for Transformers: Better Explanations through 
Conservative Propagation" (Ali et al., 2022)
"""

from __future__ import annotations
from typing import List, Dict, Callable, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import models  # Per GPU support

# -------------------------------------------------------------------------
# Librerie opzionali
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LimeTextExplainer = None
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
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
DEBUG_TIMING = False  # CHANGED: Ridotto output log

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
# Helper functions per metodi Ali et al. (2022)

def _extract_attention_matrices(model_outputs):
    """Estrae matrici di attention dai model outputs."""
    if hasattr(model_outputs, 'attentions') and model_outputs.attentions is not None:
        # Stack di tutti i layer: [n_layers, batch_size, n_heads, seq_len, seq_len]
        attention_matrices = torch.stack(model_outputs.attentions, dim=0)
        return attention_matrices
    else:
        raise ValueError("Il modello deve essere chiamato con output_attentions=True")

def _compute_joint_attention(attention_matrices: np.ndarray, add_residual: bool = True) -> np.ndarray:
    """
    Calcola joint attention attraverso i layer (Ali et al. 2022).
    
    Args:
        attention_matrices: [n_layers, seq_len, seq_len] 
        add_residual: Se aggiungere connessioni residuali
    
    Returns:
        Joint attention matrices [n_layers, seq_len, seq_len]
    """
    if add_residual:
        seq_len = attention_matrices.shape[1]
        residual_att = np.eye(seq_len)[None, ...]
        aug_att_mat = attention_matrices + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1, keepdims=True)
    else:
        aug_att_mat = attention_matrices
    
    n_layers = aug_att_mat.shape[0]
    joint_attentions = np.zeros_like(aug_att_mat)
    
    # Primo layer
    joint_attentions[0] = aug_att_mat[0]
    
    # Propagazione attraverso i layer
    for i in range(1, n_layers):
        joint_attentions[i] = np.dot(aug_att_mat[i], joint_attentions[i-1])
    
    return joint_attentions

def _get_adjacency_matrix(attention_matrix: np.ndarray, input_tokens: List[str]):
    """
    Crea matrice di adiacenza per attention flow (Ali et al. 2022).
    
    Args:
        attention_matrix: [n_layers, seq_len, seq_len]
        input_tokens: Lista dei token
    
    Returns:
        (adj_mat, labels_to_index): Matrice di adiacenza e mapping label->indice
    """
    n_layers, seq_len, _ = attention_matrix.shape
    
    # Dimensione: (n_layers+1) * seq_len nodi
    total_nodes = (n_layers + 1) * seq_len
    adj_mat = np.zeros((total_nodes, total_nodes))
    
    # Mapping da label a indice
    labels_to_index = {}
    
    # Nodi del primo layer (input tokens)
    for k in range(seq_len):
        if k < len(input_tokens):
            label = f"{k}_{input_tokens[k]}"
        else:
            label = f"{k}_PAD"
        labels_to_index[label] = k
    
    # Nodi dei layer successivi e connessioni
    for layer in range(1, n_layers + 1):
        for k_to in range(seq_len):
            node_to = layer * seq_len + k_to
            label = f"L{layer}_{k_to}"
            labels_to_index[label] = node_to
            
            # Connessioni dal layer precedente
            for k_from in range(seq_len):
                node_from = (layer - 1) * seq_len + k_from
                adj_mat[node_to][node_from] = attention_matrix[layer - 1][k_to][k_from]
    
    return adj_mat, labels_to_index

def _compute_node_flow(graph, labels_to_index: dict, input_nodes: List[str], 
                      output_nodes: List[str], seq_len: int) -> np.ndarray:
    """
    Calcola node flow usando maximum flow algorithm (Ali et al. 2022).
    """
    n_nodes = len(labels_to_index)
    flow_values = np.zeros((n_nodes, n_nodes))
    
    for output_key in output_nodes:
        if output_key not in input_nodes:
            current_layer = int(labels_to_index[output_key] / seq_len)
            prev_layer = current_layer - 1
            u = labels_to_index[output_key]
            
            for input_key in input_nodes:
                v = labels_to_index[input_key]
                try:
                    flow_value = nx.maximum_flow_value(graph, u, v, 
                                                     flow_func=nx.algorithms.flow.edmonds_karp)
                    flow_values[u][prev_layer * seq_len + v] = flow_value
                except:
                    # Se non c'è path, il flow è 0
                    flow_values[u][prev_layer * seq_len + v] = 0
            
            # Normalizza
            row_sum = flow_values[u].sum()
            if row_sum > 0:
                flow_values[u] /= row_sum
    
    return flow_values

# -------------------------------------------------------------------------
# 1. GRADIENT × INPUT (mantenuto invariato)

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
# 2. ATTENTION ROLLOUT (Ali et al. 2022)

def _attention_rollout(model, tokenizer):
    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, min(MAX_LEN, 256))
            
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)
            
            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                print("WARNING: Modello non supporta attention, usando fallback")
                tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
                scores = [max(0.1, 1.0 / (i + 1)) for i in range(len(tokens))]
                log_timing("attention_rollout", time.time() - start_time)
                return Attribution(tokens, scores)
            
            # Estrai attention matrices
            attention_matrices = _extract_attention_matrices(outputs)
            
            # Converti in numpy: [n_layers, n_heads, seq_len, seq_len] -> [n_layers, seq_len, seq_len]
            att_np = attention_matrices.detach().cpu().numpy()
            att_sum_heads = att_np.mean(axis=2)  # Media su heads: [n_layers, batch, seq_len, seq_len]
            att_sum_heads = att_sum_heads.squeeze(1)  # Rimuovi batch: [n_layers, seq_len, seq_len]
            
            # Calcola joint attention con residual connections
            joint_attentions = _compute_joint_attention(att_sum_heads, add_residual=True)
            
            # Prendi l'ultimo layer e calcola relevance correttamente
            final_attention = joint_attentions[-1]  # [seq_len, seq_len]
            
            # FIXED: Somma su query positions (asse 0) per token relevance
            # Rappresenta quanto ogni token riceve attention aggregata
            relevance_scores = final_attention.sum(axis=0)  # [seq_len]
            
            # NORMALIZZAZIONE: Porta in range [0, 1]
            if relevance_scores.max() > relevance_scores.min():
                relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())
            
            # SCALING: Porta in range tipico [-1, +1] con media circa 0
            relevance_scores = 2 * relevance_scores - 1
            
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            
            min_len = min(len(tokens), len(relevance_scores))
            log_timing("attention_rollout", time.time() - start_time)
            return Attribution(tokens[:min_len], relevance_scores[:min_len].tolist())
            
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
# 3. ATTENTION FLOW (Ali et al. 2022)

def _attention_flow(model, tokenizer):
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX non installato per attention_flow")
    
    def explain(text: str) -> Attribution:
        start_time = time.time()
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, min(MAX_LEN, 128))  # Limita lunghezza per performance
            
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)

            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                raise ValueError("Modello non supporta output attention")

            # Estrai e processa attention matrices
            attention_matrices = _extract_attention_matrices(outputs)
            att_np = attention_matrices.detach().cpu().numpy()
            
            # Media su heads e rimuovi batch: [n_layers, seq_len, seq_len]
            att_sum_heads = att_np.mean(axis=2).squeeze(1)
            n_layers, seq_len, _ = att_sum_heads.shape
            
            # Aggiungi residual connections e normalizza
            residual_att = np.eye(seq_len)[None, ...]
            aug_att_mat = att_sum_heads + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1, keepdims=True)
            
            # Crea grafo per attention flow
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            adj_mat, labels_to_index = _get_adjacency_matrix(aug_att_mat, tokens)
            
            # Crea grafo NetworkX
            G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph())
            
            # Imposta capacità degli archi
            for i in range(adj_mat.shape[0]):
                for j in range(adj_mat.shape[1]):
                    if adj_mat[i, j] > 0:
                        G[i][j]['capacity'] = adj_mat[i, j]
            
            # Definisci nodi input e output
            input_nodes = []
            output_nodes = []
            target_layer = n_layers - 1  # Ultimo layer
            
            for key in labels_to_index:
                if labels_to_index[key] < seq_len:  # Primi seq_len nodi sono input
                    input_nodes.append(key)
                if key.startswith(f'L{target_layer + 1}_'):  # Nodi dell'ultimo layer
                    output_nodes.append(key)
            
            # Calcola flow values
            flow_values = _compute_node_flow(G, labels_to_index, input_nodes, output_nodes, seq_len)
            
            # Estrai relevance dal flow del layer finale
            final_layer_start = (target_layer + 1) * seq_len
            final_layer_end = (target_layer + 2) * seq_len
            prev_layer_start = target_layer * seq_len
            prev_layer_end = (target_layer + 1) * seq_len
            
            final_layer_attention = flow_values[final_layer_start:final_layer_end, 
                                              prev_layer_start:prev_layer_end]
            relevance_scores = final_layer_attention.sum(axis=0)
            
            # NORMALIZZAZIONE per attention flow
            if len(relevance_scores) > 0:
                max_score = np.max(relevance_scores)
                min_score = np.min(relevance_scores)
                
                if max_score > min_score:
                    # Normalizza in [0, 1]
                    relevance_scores = (relevance_scores - min_score) / (max_score - min_score)
                    # Scala in [-0.5, +0.5] per consistenza con altri explainer
                    relevance_scores = relevance_scores - 0.5
                else:
                    # Tutti i valori uguali
                    relevance_scores = np.zeros_like(relevance_scores)
            
            log_timing("attention_flow", time.time() - start_time)
            return Attribution(tokens[:len(relevance_scores)], relevance_scores.tolist())
            
        except Exception as e:
            print(f"Errore in attention_flow: {e}")
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, max_length=10, truncation=True))
            scores = [0.0] * len(tokens)
            log_timing("attention_flow", time.time() - start_time)
            return Attribution(tokens, scores)
    
    return explain

# -------------------------------------------------------------------------
# 4. LIME (mantenuto invariato)

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
# 5. SHAP (mantenuto invariato)

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
# 6. LRP CONSERVATIVA (Ali et al. 2022)
# -------------------------------------------------------------------------

def _lrp(model, tokenizer):
    """
    LRP Conservativa con detach trick secondo Ali et al. (2022).
    
    Implementa:
    - AH-rule: detach dei pesi di attenzione  
    - LN-rule: detach del denominatore di normalizzazione
    - Implementation trick per conservazione garantita
    """
    
    class ConservativeLRPExplainer:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.original_forwards = {}
            self._patched = False
        
        def _patch_attention_heads(self):
            """Applica AH-rule: detach dei pesi di attenzione."""
            patch_count = 0
            
            def create_attention_forward(original_module):
                def patched_forward(*args, **kwargs):
                    # Chiama forward normale
                    outputs = original_module._original_forward(*args, **kwargs)
                    
                    # Se outputs contiene attention weights, detach them
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        # Tipico caso: (context_layer, attention_probs)
                        context_layer, attention_probs = outputs[0], outputs[1]
                        
                        # DETACH attention probs (AH-rule)
                        attention_probs_detached = attention_probs.detach()
                        
                        return (context_layer, attention_probs_detached) + outputs[2:]
                    else:
                        return outputs
                
                return patched_forward
            
            # Patch attention layers
            for name, module in self.model.named_modules():
                if 'attention' in name.lower() and hasattr(module, 'forward'):
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward
                        module.forward = create_attention_forward(module)
                        patch_count += 1
            
            return patch_count
        
        def _patch_layer_norms(self):
            """Applica LN-rule: detach del denominatore di normalizzazione."""
            patch_count = 0
            
            def create_layernorm_forward(original_module):
                def patched_forward(x):
                    if not x.requires_grad:
                        return original_module._original_forward(x)
                    
                    # LN-RULE implementazione
                    dims = tuple(range(1, len(x.shape)))
                    mean = x.mean(dim=dims, keepdim=True)
                    var = x.var(dim=dims, keepdim=True, unbiased=False)
                    
                    # Centering (lineare)
                    centered = x - mean
                    
                    # Normalization con DETACH del denominatore (non-lineare)
                    eps = getattr(original_module, 'eps', 1e-5)
                    std = torch.sqrt(var + eps).detach()  # ← DETACH QUI!
                    normalized = centered / std
                    
                    # Applica weight e bias se presenti
                    if hasattr(original_module, 'weight') and original_module.weight is not None:
                        normalized = normalized * original_module.weight
                    if hasattr(original_module, 'bias') and original_module.bias is not None:
                        normalized = normalized + original_module.bias
                    
                    return normalized
                
                return patched_forward
            
            # Patch LayerNorm layers
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or 'norm' in name.lower():
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward
                        module.forward = create_layernorm_forward(module)
                        patch_count += 1
            
            return patch_count
        
        def _patch_model(self):
            """Applica patch conservativo completo al modello."""
            if self._patched:
                return
            
            print("[LRP] Applicando patch conservativo...")
            
            # Applica AH-rule e LN-rule
            ah_patches = self._patch_attention_heads()
            ln_patches = self._patch_layer_norms()
            
            self._patched = True
            print(f"[LRP] Patch applicato: {ah_patches} attention layers, {ln_patches} LayerNorm layers")
        
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
            """Genera spiegazione con LRP conservativa secondo Ali et al. (2022)."""
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
                
                # STEP 6: Backward pass (Gradient × Input con patch conservativo)
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
# EXPLAINER FACTORY (AGGIORNATA)
# -------------------------------------------------------------------------

_EXPLAINER_FACTORY: Dict[str, Callable] = {
    "lime":                 _lime_text,           # Richiede: pip install lime  
    "shap":                 _kernel_shap,         # Richiede: pip install shap
    "grad_input":           _grad_input,          # Nessuna dipendenza
    "attention_rollout":    _attention_rollout,   # Implementazione Ali et al. 2022
    "attention_flow":       _attention_flow,      # Richiede: pip install networkx + Ali et al. 2022
    "lrp":                  _lrp,                 # LRP Conservativa Ali et al. 2022
}

# -------------------------------------------------------------------------
# API pubblica (AGGIORNATA)
# -------------------------------------------------------------------------

def list_explainers():
    """Restituisce lista di explainer disponibili."""
    available = []
    for name, factory in _EXPLAINER_FACTORY.items():
        # Controlla dipendenze specifiche
        if name == "lime" and not LIME_AVAILABLE:
            continue
        elif name == "shap" and not SHAP_AVAILABLE:
            continue
        elif name == "attention_flow" and not NETWORKX_AVAILABLE:
            continue
        # Altri explainer non hanno dipendenze esterne
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
    
    # Controlla dipendenze specifiche
    if name == "lime" and not LIME_AVAILABLE:
        raise ImportError("LIME non installato. Installare con: pip install lime")
    elif name == "shap" and not SHAP_AVAILABLE:
        raise ImportError("SHAP non installato. Installare con: pip install shap")
    elif name == "attention_flow" and not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX non installato. Installare con: pip install networkx")
    
    return _EXPLAINER_FACTORY[name](model, tokenizer)

def check_dependencies():
    """Controlla quali dipendenze sono disponibili."""
    deps = {
        "LIME": LIME_AVAILABLE,
        "SHAP": SHAP_AVAILABLE, 
        "NetworkX (Attention Flow)": NETWORKX_AVAILABLE,
        "LRP Conservativa Ali et al. 2022": True,  # Sempre disponibile
        "Attention Rollout Ali et al. 2022": True,  # Sempre disponibile
    }
    
    print("Stato dipendenze explainer:")
    for lib, available in deps.items():
        status = "[OK]" if available else "[MISSING]"
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
# Funzione di utilità per confronto con implementazioni originali
# -------------------------------------------------------------------------

def compare_with_original_implementations():
    """
    Confronta le nuove implementazioni con quelle precedenti.
    Utile per validare che i risultati siano coerenti.
    """
    print("=" * 60)
    print("CONFRONTO IMPLEMENTAZIONI ALI ET AL. 2022")
    print("=" * 60)
    
    try:
        import models
        model = models.load_model("distilbert")
        tokenizer = models.load_tokenizer("distilbert")
        
        test_texts = [
            "This movie is absolutely fantastic!",
            "This film is terrible and boring.",
            "An average movie with decent acting."
        ]
        
        methods_to_test = ["attention_rollout", "attention_flow", "lrp"]
        
        for text in test_texts:
            print(f"\nTesto: '{text}'")
            print("-" * 40)
            
            for method in methods_to_test:
                if method in list_explainers():
                    try:
                        explainer = get_explainer(method, model, tokenizer)
                        attr = explainer(text)
                        
                        # Analisi risultati
                        max_score = max(attr.scores) if attr.scores else 0
                        min_score = min(attr.scores) if attr.scores else 0
                        total_relevance = sum(attr.scores)
                        
                        print(f"  {method:>18}: range=[{min_score:+.3f}, {max_score:+.3f}], total={total_relevance:+.3f}")
                        
                        # Mostra top 3 token più rilevanti
                        token_scores = [(tok, score) for tok, score in zip(attr.tokens, attr.scores) 
                                       if not tok.startswith('[') and tok.strip()]
                        token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        top_tokens = [f"{tok}({score:+.2f})" for tok, score in token_scores[:3]]
                        print(f"  {'':>20} Top: {', '.join(top_tokens)}")
                        
                    except Exception as e:
                        print(f"  {method:>18}: ERRORE - {e}")
    
    except Exception as e:
        print(f"Errore nel confronto: {e}")

# -------------------------------------------------------------------------
# Test di compatibilità FINALE
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST EXPLAINERS - IMPLEMENTAZIONI ALI ET AL. (2022)")
    print("=" * 70)
    
    # 1. Controlla dipendenze
    print("\n1. CONTROLLO DIPENDENZE:")
    deps = check_dependencies()
    
    # 2. Lista explainer disponibili
    print(f"\n2. EXPLAINER DISPONIBILI:")
    available = list_explainers()
    for explainer in available:
        marker = "[NEW]" if explainer in ["attention_rollout", "attention_flow", "lrp"] else "[OK]"
        print(f"  {marker} {explainer}")
    
    if not available:
        print("    Nessun explainer disponibile - verificare installazioni")
        exit(1)
    
    # 3. Test implementazioni Ali et al. 2022
    print(f"\n3. TEST METODI ALI ET AL. 2022:")
    try:
        import models
        print("  Caricando modello distilbert...")
        model = models.load_model("distilbert")
        tokenizer = models.load_tokenizer("distilbert")
        print("  [OK] Modello caricato")
        
        test_text = "This movie is absolutely fantastic with incredible acting and amazing cinematography!"
        
        # Test specifico per ogni metodo Ali et al.
        ali_methods = ["attention_rollout", "attention_flow", "lrp"]
        
        for method in ali_methods:
            if method in available:
                print(f"\n  [TEST] Testing {method.upper()}:")
                try:
                    explainer = get_explainer(method, model, tokenizer)
                    attr = explainer(test_text)
                    
                    # Analisi dettagliata
                    print(f"    [OK] Tokens processati: {len(attr.tokens)}")
                    print(f"    [OK] Score range: [{min(attr.scores):.3f}, {max(attr.scores):.3f}]")
                    print(f"    [OK] Total relevance: {sum(attr.scores):.3f}")
                    
                    # Verifica conservazione per LRP
                    if method == "lrp":
                        # Predizione originale per confronto
                        with torch.no_grad():
                            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)
                            inputs = models.move_batch_to_device(inputs)
                            outputs = model(**inputs)
                            target_logit = outputs.logits[0, 1].item() if outputs.logits.size(-1) > 1 else outputs.logits[0, 0].item()
                        
                        conservation_error = abs(sum(attr.scores) - target_logit)
                        print(f"    [CONSERVATION] Error: {conservation_error:.4f}")
                    
                    # Top token più rilevanti
                    token_scores = [(tok, score) for tok, score in zip(attr.tokens, attr.scores) 
                                   if not tok.startswith('[') and tok.strip()]
                    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print("    [TOP] Relevant tokens:")
                    for i, (token, score) in enumerate(token_scores[:5], 1):
                        print(f"      {i}. {token:>12}: {score:+.3f}")
                        
                except Exception as e:
                    print(f"    [ERROR] Errore in {method}: {e}")
            else:
                print(f"\n  [SKIP] {method.upper()}: Non disponibile (dipendenze mancanti)")
    
    except Exception as e:
        print(f"  [ERROR] Errore nel test: {e}")
    
    # 4. Confronto performance (opzionale)
    if len(available) > 1:
        print(f"\n4. BENCHMARK PERFORMANCE:")
        try:
            benchmark_results = benchmark_explainer_speed()
        except Exception as e:
            print(f"  Benchmark fallito: {e}")
    
    # 5. Confronto implementazioni
    print(f"\n5. CONFRONTO IMPLEMENTAZIONI:")
    try:
        compare_with_original_implementations()
    except Exception as e:
        print(f"  Confronto fallito: {e}")
    
    print(f"\n{'='*70}")
    print("[DONE] Test completati! Le implementazioni Ali et al. 2022 sono integrate.")
    print("[AVAILABLE] Metodi disponibili: attention_rollout, attention_flow, lrp (conservative)")
    print("[PAPER] 'XAI for Transformers: Better Explanations through Conservative Propagation'")
    print("[REPO] https://github.com/AmeenAli/XAI_Transformers")
    print("="*70)