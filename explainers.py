"""
explainers.py – 6 XAI methods (lime, shap, grad_input, attention_rollout,
attention_flow, lrp) ottimizzati per modelli pre-trained con gestione robusta
dei diversi tipi di architetture (BERT, RoBERTa, DistilBERT, TinyBERT).
"""

from __future__ import annotations
from typing import List, Dict, Callable
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

# -------------------------------------------------------------------------
# Librerie opzionali
try:
    from captum.attr import InputXGradient, LayerLRP
except ImportError:
    InputXGradient = LayerLRP = None

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None

try:
    import shap, numpy as np
except ImportError:
    shap = np = None

try:
    import networkx as nx
except ImportError:
    nx = None

# Parametri globali
MAX_LEN = 512  # Aumentato per modelli più grandi
MIN_LEN = 10   # Lunghezza minima per evitare errori

# -------------------------------------------------------------------------
def _get_model_architecture(model):
    """Identifica l'architettura del modello per gestire differenze specifiche."""
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
    arch = _get_model_architecture(model)
    
    if hasattr(model, 'roberta'):
        return model.roberta
    elif hasattr(model, 'distilbert'):
        return model.distilbert
    elif hasattr(model, 'bert'):
        return model.bert
    else:
        # Fallback: prova ad accedere al primo attributo che sembra un encoder
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

def _forward_pos(model, ids, mask):
    """Forward pass che restituisce la probabilità della classe positiva."""
    logits = model(input_ids=ids, attention_mask=mask).logits
    if logits.size(-1) > 1:
        # Classificazione binaria: prendi classe 1 (positiva)
        return F.softmax(logits, dim=-1)[:, 1]
    else:
        # Output singolo: applica sigmoid
        return torch.sigmoid(logits.squeeze(-1))

def _safe_tokenize(text: str, tokenizer, max_length=MAX_LEN):
    """Tokenizzazione sicura con gestione lunghezza e pulizia testo."""
    # Pulizia base del testo
    text = text.strip()
    if len(text) < MIN_LEN:
        text = text + " " * (MIN_LEN - len(text))  # Padding se troppo corto
    
    # Tokenizzazione
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
    
    return encoded

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
# GRADIENT × INPUT (embedding-level)
def _grad_input(model, tokenizer):
    def explain(text: str) -> Attribution:
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, MAX_LEN)
            ids, attn = enc["input_ids"], enc["attention_mask"]

            # Ottieni embeddings
            embed_layer = _get_embedding_layer(model)
            embeds = embed_layer(ids).detach()
            embeds.requires_grad_(True)

            # Forward pass attraverso il modello
            base_model = _get_base_model(model)
            if hasattr(model, 'classifier'):
                # Modelli con classifier separato
                outputs = base_model(inputs_embeds=embeds, attention_mask=attn)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs[0]
                
                # Pooling (prendi [CLS] token o media)
                if hidden_states.size(1) > 0:
                    pooled = hidden_states[:, 0]  # [CLS] token
                else:
                    pooled = hidden_states.mean(dim=1)
                
                logits = model.classifier(pooled)
            else:
                # Modelli end-to-end
                outputs = model(inputs_embeds=embeds, attention_mask=attn)
                logits = outputs.logits

            # Target: classe positiva
            target_idx = 1 if logits.size(-1) > 1 else 0
            loss = logits[:, target_idx].sum()

            # Backprop
            model.zero_grad()
            loss.backward()

            # Calcola gradient × input
            if embeds.grad is not None:
                scores = (embeds.grad * embeds).sum(dim=-1).squeeze(0)
            else:
                scores = torch.zeros(embeds.size(1))
            
            tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
            return Attribution(tokens, scores.tolist())
            
        except Exception as e:
            print(f"Errore in grad_input: {e}")
            # Fallback: restituisci attribution vuota
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, max_length=10, truncation=True))
            scores = [0.0] * len(tokens)
            return Attribution(tokens, scores)
    
    return explain

# -------------------------------------------------------------------------
# ATTENTION ROLLOUT (Abnar & Zuidema, 2020)
def _attention_rollout(model, tokenizer):
    def explain(text: str) -> Attribution:
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, MAX_LEN)
            
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)
            
            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                raise ValueError("Modello non supporta output attention")
            
            # Media su tutte le teste di attenzione
            att = torch.stack([a.mean(dim=1) for a in outputs.attentions])  # (L,B,seq,seq)

            # Aggiungi identità e normalizza
            I = torch.eye(att.size(-1)).unsqueeze(0).unsqueeze(0)
            att = 0.5 * att + 0.5 * I
            att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            # Rollout dall'ultimo layer
            rollout = att[-1]
            for l in range(att.size(0) - 2, -1, -1):
                rollout = att[l].bmm(rollout)

            # Estrai scores dal primo token (CLS) verso tutti gli altri
            scores = rollout.squeeze(0)[0]
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            return Attribution(tokens, scores.tolist())
            
        except Exception as e:
            print(f"Errore in attention_rollout: {e}")
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, max_length=10, truncation=True))
            scores = [0.0] * len(tokens)
            return Attribution(tokens, scores)
    
    return explain

# -------------------------------------------------------------------------
# ATTENTION FLOW
def _attention_flow(model, tokenizer):
    if nx is None:
        raise ImportError("NetworkX non installato per attention_flow")
    
    def explain(text: str) -> Attribution:
        try:
            model.eval()
            enc = _safe_tokenize(text, tokenizer, MAX_LEN)
            
            with torch.no_grad():
                outputs = model(**enc, output_attentions=True)

            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                raise ValueError("Modello non supporta output attention")

            att = torch.stack([a.mean(dim=1) for a in outputs.attentions])
            seq = att.size(-1)
            I = torch.eye(seq).unsqueeze(0).unsqueeze(0)
            att = 0.5 * att + 0.5 * I
            att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            # Costruisci grafo
            G = nx.DiGraph()
            L = att.size(0)
            for l in range(L):
                for i in range(seq):
                    G.add_node((l, i))
            
            for l in range(L):
                for i in range(seq):
                    for j in range(seq):
                        w = att[l, 0, i, j].item()
                        if w > 1e-6:  # Soglia per evitare pesi troppo piccoli
                            if l > 0:  # Collega con layer precedente
                                G.add_edge((l, i), (l - 1, j), capacity=w)

            # Calcola flow
            flow_scores = torch.zeros(seq)
            try:
                for src in range(seq):
                    if G.has_node((L - 1, src)) and G.has_node((0, 0)):
                        flow_val, _ = nx.maximum_flow(G, (L - 1, src), (0, 0))
                        flow_scores[src] = flow_val
            except:
                # Se maximum_flow fallisce, usa una metrica semplificata
                flow_scores = att[-1, 0, 0, :]

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            return Attribution(tokens, flow_scores.tolist())
            
        except Exception as e:
            print(f"Errore in attention_flow: {e}")
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, max_length=10, truncation=True))
            scores = [0.0] * len(tokens)
            return Attribution(tokens, scores)
    
    return explain

# -------------------------------------------------------------------------
# LIME-Text
def _lime_text(model, tokenizer):
    if LimeTextExplainer is None:
        raise ImportError("LIME non installato")
    
    explainer = LimeTextExplainer(class_names=["negative", "positive"])

    def predict(texts):
        try:
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            )
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        except Exception as e:
            print(f"Errore in LIME predict: {e}")
            # Fallback: probabilità uniformi
            return np.array([[0.5, 0.5] for _ in texts])

    def explain(text: str) -> Attribution:
        try:
            exp = explainer.explain_instance(
                text, 
                predict, 
                num_features=min(20, len(text.split())),
                num_samples=100
            )
            features = exp.as_list()
            if features:
                tokens, scores = zip(*features)
                return Attribution(list(tokens), list(scores))
            else:
                words = text.split()[:10]
                return Attribution(words, [0.0] * len(words))
        except Exception as e:
            print(f"Errore in LIME explain: {e}")
            words = text.split()[:10]
            return Attribution(words, [0.0] * len(words))
    
    return explain

# -------------------------------------------------------------------------
# SHAP KernelExplainer
def _kernel_shap(model, tokenizer):
    if shap is None or np is None:
        raise ImportError("SHAP non installato")

    def predict(texts):
        try:
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            )
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        except Exception as e:
            print(f"Errore in SHAP predict: {e}")
            return np.array([[0.5, 0.5] for _ in texts])

    # Background dataset ridotto
    background = np.array(["This is neutral text."])
    explainer = shap.KernelExplainer(predict, background)

    def explain(text: str) -> Attribution:
        try:
            shap_values = explainer.shap_values([text], nsamples=50)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                scores = shap_values[1][0]  # Classe positiva
            else:
                scores = shap_values[0][0]
            
            tokens = text.split()
            # Assicurati che scores e tokens abbiano la stessa lunghezza
            min_len = min(len(tokens), len(scores))
            return Attribution(tokens[:min_len], scores[:min_len].tolist())
        except Exception as e:
            print(f"Errore in SHAP explain: {e}")
            words = text.split()[:10]
            return Attribution(words, [0.0] * len(words))
    
    return explain

# -------------------------------------------------------------------------
# LRP (Layer-wise Relevance Propagation)
def _lrp(model, tokenizer):
    if LayerLRP is None:
        raise ImportError("Captum non installato per LRP")
    
    def explain(text: str) -> Attribution:
        try:
            # Identifica il layer appropriato per LRP
            base_model = _get_base_model(model)
            if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
                target_layer = base_model.encoder.layer[-1]
            else:
                # Fallback: usa il modello stesso
                target_layer = base_model
            
            lrp = LayerLRP(model, target_layer)
            
            enc = _safe_tokenize(text, tokenizer, MAX_LEN)
            ids, attn = enc["input_ids"], enc["attention_mask"]
            
            attrs = lrp.attribute(ids, additional_forward_args=(attn,))
            scores = attrs.sum(dim=-1).squeeze(0)
            tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
            return Attribution(tokens, scores.tolist())
            
        except Exception as e:
            print(f"Errore in LRP: {e}")
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, max_length=10, truncation=True))
            scores = [0.0] * len(tokens)
            return Attribution(tokens, scores)
    
    return explain

# -------------------------------------------------------------------------
_EXPLAINER_FACTORY: Dict[str, Callable] = {
    "lime":                 _lime_text,
    "shap":                 _kernel_shap,
    "grad_input":           _grad_input,
    "attention_rollout":    _attention_rollout,
    "attention_flow":       _attention_flow,
    "lrp":                  _lrp,
}

# -------------------------------------------------------------------------
# API pubblica
def list_explainers():
    """Restituisce lista di explainer disponibili."""
    return list(_EXPLAINER_FACTORY.keys())

def get_explainer(name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Crea un explainer per il modello specificato.
    
    Args:
        name: Nome dell'explainer
        model: Modello pre-trained
        tokenizer: Tokenizer corrispondente
    
    Returns:
        Callable che prende un testo e restituisce Attribution
    """
    name = name.lower()
    if name not in _EXPLAINER_FACTORY:
        available = ", ".join(_EXPLAINER_FACTORY.keys())
        raise ValueError(f"Explainer '{name}' non supportato. Disponibili: {available}")
    
    # Verifica compatibilità
    arch = _get_model_architecture(model)
    print(f"Creando explainer '{name}' per architettura '{arch}'")
    
    return _EXPLAINER_FACTORY[name](model, tokenizer)

# -------------------------------------------------------------------------
# Test di compatibilità
if __name__ == "__main__":
    print("Test explainers...")
    print(f"Explainer disponibili: {list_explainers()}")
    
    # Test importazioni opzionali
    optional_libs = {
        "LIME": LimeTextExplainer is not None,
        "SHAP": shap is not None,
        "Captum": LayerLRP is not None,
        "NetworkX": nx is not None,
    }
    
    for lib, available in optional_libs.items():
        status = "OK" if available else "Non installato"
        print(f"{lib}: {status}")