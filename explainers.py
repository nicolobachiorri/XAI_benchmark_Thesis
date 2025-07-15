"""
explainers.py – 6 metodi XAI per modelli Transformer di sentiment analysis
===========================================================================

Explainer supportati (stringhe chiave → classe/func):
* **integrated_gradients**  – Captum `IntegratedGradients`
* **gradient_shap**         – Captum `GradientShap`
* **grad_input**            – Captum `GradientAttribution` (gradient × input)
* **attention_rollout**     – media pesata delle attention heads (Attn Roll‑out)
* **lime**                  – LIME Text (parole come features)
* **kernel_shap**           – SHAP KernelExplainer (testo → proba)

Dipendenze:
    pip install captum lime shap

Esempio:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from explainers import get_explainer

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.eval()

    explainer = get_explainer("integrated_gradients", model, tokenizer)
    attributions = explainer("I loved the movie, it was great!")
    print(list(zip(attributions.tokens, attributions.scores)))
"""

# ==== 1. Librerie ====
from __future__ import annotations
from typing import List, Callable, Dict, Union
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

# Captum per metodi gradient‑based
try:
    from captum.attr import (
        IntegratedGradients,
        GradientShap,
        GradientAttribution,  # base class per gradient × input
    )
except ImportError:  # fallback per chi non ha captum installato
    IntegratedGradients = GradientShap = GradientAttribution = None  # type: ignore

# LIME & SHAP sono opzionali
try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None  # type: ignore

try:
    import shap  # noqa: F401
except ImportError:
    shap = None  # type: ignore

# ==== 2. Helper: forward function compatibile Captum ====

def _forward_func(model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Ritorna le logit della classe positiva (indice 1) – Captum expects tensor output."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape (B, num_labels)
    # per Captum serve 1D o 2D; prendiamo probabilità/logit della classe 1
    class_idx = 1 if logits.size(-1) > 1 else 0
    return logits[:, class_idx]


# ==== 3. Wrapper di attributions per uniformità ====
class Attribution:
    """Contiene token list e score list allineate."""

    def __init__(self, tokens: List[str], scores: List[float]):
        self.tokens = tokens
        self.scores = scores

    def __repr__(self):
        pairs = ", ".join(f"{t}:{s:.3f}" for t, s in zip(self.tokens, self.scores))
        return f"Attribution([{pairs}])"


# ==== 4. Implementazioni dei 6 explainers ====

def _captum_explainer(cls, model, tokenizer):
    """Factory generica per metodi Captum (IG, GradientShap, Grad × Input)."""

    def explain(text: str) -> Attribution:
        model.eval()
        encoding = tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # baseline (zeros) per IntegratedGradients/GradientShap
        baseline = torch.zeros_like(input_ids)

        attr_alg = cls(model, _forward_func) if cls is not GradientAttribution else cls()
        if isinstance(attr_alg, GradientAttribution):
            # gradient × input: attr = grad * input
            input_embeds = model.get_input_embeddings()(input_ids)
            input_embeds.requires_grad_(True)
            input_embeds.retain_grad()          

            model.zero_grad()   
            logits = _forward_func(model, input_ids, attention_mask)
            logits.backward(torch.ones_like(logits))
            attributions = input_embeds.grad * input_embeds
            scores = attributions.sum(dim=-1).squeeze(0)  # shape: seq_len
        else:
            attributions, _ = attr_alg.attribute(
                inputs=input_ids,
                baselines=baseline,
                additional_forward_args=(attention_mask,),
                return_convergence_delta=False,
            )
            scores = attributions.sum(dim=-1).squeeze(0)

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        return Attribution(tokens, scores.tolist())

    return explain


def _attention_rollout(model, tokenizer):
    """Media pesata delle attention matrices lungo tutti i layer e heads."""

    def explain(text: str) -> Attribution:
        model.eval()
        encoding = tokenizer(text, return_tensors="pt", output_attentions=True)
        with torch.no_grad():
            outputs = model(**encoding)
            attn = outputs.attentions  # tuple(layer) each (B, heads, seq, seq)

        # Rollout: media su heads, poi prodotto cumulativo layer‑wise (simplified)
        attn_avg = torch.stack([a.mean(dim=1) for a in attn])  # shape (L, B, seq, seq)
        joint_attn = attn_avg[0]
        for i in range(1, attn_avg.size(0)):
            joint_attn = attn_avg[i].bmm(joint_attn)
        scores = joint_attn.squeeze(0)[0]  # attention from [CLS] → altri token
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0))
        return Attribution(tokens, scores.tolist())

    return explain


def _lime_text(model, tokenizer):
    if LimeTextExplainer is None:
        raise ImportError("LIME non installato. `pip install lime`")

    class_names = ["neg", "pos"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts: List[str]):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def explain(text: str) -> Attribution:
        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        # exp.as_list() -> List[(word, score)]
        tokens, scores = zip(*exp.as_list())
        return Attribution(list(tokens), list(scores))

    return explain


def _kernel_shap(model, tokenizer):
    if shap is None:
        raise ImportError("SHAP non installato. `pip install shap`")

    import numpy as np
    from shap import KernelExplainer

    def predict_proba(texts: List[str]):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    # usare un background di 1 frase neutra
    explainer = KernelExplainer(predict_proba, np.array(["."]))

    def explain(text: str) -> Attribution:
        shap_values = explainer.shap_values([text], nsamples=100)
        # shap_values è lista per ogni classe; usiamo classe positiva (1)
        token_scores = shap_values[1][0]
        tokens = text.split()  # shap tokenizza per parola
        return Attribution(tokens, token_scores.tolist())

    return explain

# ==== 5. Factory pubblica ====
_EXPLAINER_FACTORY: Dict[str, Callable[[PreTrainedModel, PreTrainedTokenizer], Callable[[str], Attribution]]] = {
    "integrated_gradients": lambda m, t: _captum_explainer(IntegratedGradients, m, t),
    "gradient_shap":        lambda m, t: _captum_explainer(GradientShap, m, t),
    "grad_input":           lambda m, t: _captum_explainer(GradientAttribution, m, t),
    "attention_rollout":    _attention_rollout,
    "lime":                 _lime_text,
    "kernel_shap":          _kernel_shap,
}


def list_explainers() -> List[str]:
    """Ritorna la lista di stringhe valide per `get_explainer`."""
    return list(_EXPLAINER_FACTORY.keys())


def get_explainer(name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Restituisce una funzione `explain(text) -> Attribution`."""
    name = name.lower()
    if name not in _EXPLAINER_FACTORY:
        raise ValueError(f"Explainer '{name}' non supportato. Usa uno di {list_explainers()}")
    if IntegratedGradients is None and name in {"integrated_gradients", "gradient_shap", "grad_input"}:
        raise ImportError("Captum non installato. `pip install captum`")
    return _EXPLAINER_FACTORY[name](model, tokenizer)
