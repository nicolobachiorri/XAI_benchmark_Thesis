"""
metrics.py – Metriche del paper XAI (senza Human-Agreement)
===========================================================

Implementa **tre metriche automatiche** tratte da "Evaluating the effectiveness
of XAI techniques for encoder-based language models".

1. **Robustness** – stabilità delle saliency sotto perturbazione del testo
2. **Consistency** – allineamento delle saliency fra due modelli gemelli
3. **Contrastivity** – diversità delle saliency fra classi opposte

La metrica di *Human-reasoning Agreement* non è implementata perché richiede
annotazioni manuali.

Ottimizzato per modelli pre-trained con gestione robusta di errori e
compatibilità con diverse architetture (BERT, RoBERTa, DistilBERT, TinyBERT).

Dipendenze:
    transformers, torch, numpy, scipy, tqdm
"""

# ==== 1. Librerie ====
from __future__ import annotations
from typing import List, Callable, Sequence, Tuple, Optional
from explainers import Attribution 
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.stats import spearmanr, entropy  # Spearman ρ, KL divergence
from tqdm import tqdm

# ==== 2. Parametri configurabili ====
DEFAULT_PERTURBATION_RATIO = 0.15
MIN_SHARED_TOKENS = 2  # Minimo numero di token condivisi per calcolare correlazione
RANDOM_SEED = 42

# ==== 3. Helper per gestione probabilità ====

def _get_positive_prob(model: PreTrainedModel, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> float:
    """Calcola la probabilità della classe positiva in modo robusto."""
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits
            
            if logits.size(-1) > 1:
                # Classificazione binaria: applica softmax e prendi classe 1
                probs = F.softmax(logits, dim=-1)
                return probs[:, 1].item()
            else:
                # Output singolo: applica sigmoid
                return torch.sigmoid(logits.squeeze(-1)).item()
    except Exception as e:
        print(f"Errore in _get_positive_prob: {e}")
        return 0.5  # Fallback: probabilità neutra

# ==== 4. Funzioni di perturbazione ====

def _random_mask(text: str, ratio: float = DEFAULT_PERTURBATION_RATIO, mask_token: str = "[MASK]") -> str:
    """Maschera ~ratio% delle parole con un token speciale."""
    if not text.strip():
        return text
    
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    
    n_to_mask = max(1, int(len(tokens) * ratio))
    n_to_mask = min(n_to_mask, len(tokens) - 1)  # Lascia almeno una parola
    
    try:
        idx_to_mask = random.sample(range(len(tokens)), n_to_mask)
        for i in idx_to_mask:
            tokens[i] = mask_token
        return " ".join(tokens)
    except Exception as e:
        print(f"Errore in _random_mask: {e}")
        return text

def _random_delete(text: str, ratio: float = DEFAULT_PERTURBATION_RATIO) -> str:
    """Elimina ~ratio% delle parole casualmente."""
    if not text.strip():
        return text
    
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    
    n_to_delete = max(1, int(len(tokens) * ratio))
    n_to_delete = min(n_to_delete, len(tokens) - 1)  # Lascia almeno una parola
    
    try:
        idx_to_keep = random.sample(range(len(tokens)), len(tokens) - n_to_delete)
        idx_to_keep.sort()
        return " ".join(tokens[i] for i in idx_to_keep)
    except Exception as e:
        print(f"Errore in _random_delete: {e}")
        return text

def _random_substitute(text: str, ratio: float = DEFAULT_PERTURBATION_RATIO) -> str:
    """Sostituisce ~ratio% delle parole con sinonimi semplici."""
    if not text.strip():
        return text
    
    # Dizionario semplice di sostituzioni
    substitutions = {
        "good": "great", "bad": "terrible", "nice": "pleasant", "awful": "horrible",
        "love": "like", "hate": "dislike", "amazing": "wonderful", "terrible": "bad",
        "excellent": "good", "poor": "bad", "fantastic": "great", "boring": "dull",
        "interesting": "engaging", "stupid": "foolish", "smart": "clever", "funny": "amusing"
    }
    
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    
    n_to_substitute = max(1, int(len(tokens) * ratio))
    n_to_substitute = min(n_to_substitute, len(tokens))
    
    try:
        idx_to_substitute = random.sample(range(len(tokens)), n_to_substitute)
        for i in idx_to_substitute:
            word = tokens[i].lower()
            if word in substitutions:
                tokens[i] = substitutions[word]
        return " ".join(tokens)
    except Exception as e:
        print(f"Errore in _random_substitute: {e}")
        return text

# Lista delle funzioni di perturbazione disponibili
PERTURBATION_FUNCTIONS = [_random_mask, _random_delete, _random_substitute]

# ==== 5. Robustness ====

def compute_robustness(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    text: str,
    perturb_fn: Optional[Callable[[str], str]] = None,
    n_perturbations: int = 3,
) -> float:
    """
    Calcola la robustness come Mean Average Difference (MAD) fra saliency
    originali e perturbate.
    
    Args:
        model: Modello pre-trained
        tokenizer: Tokenizer corrispondente
        explainer: Funzione che genera Attribution
        text: Testo da analizzare
        perturb_fn: Funzione di perturbazione (None = usa tutte)
        n_perturbations: Numero di perturbazioni per funzione
    
    Returns:
        float: MAD score (più basso = più robusto)
    """
    try:
        # Calcola attribution originale
        orig_attr = explainer(text)
        if not orig_attr.tokens or not orig_attr.scores:
            return 0.0
        
        # Scegli funzioni di perturbazione
        if perturb_fn is None:
            perturb_functions = PERTURBATION_FUNCTIONS
        else:
            perturb_functions = [perturb_fn]
        
        all_diffs = []
        
        for perturb_func in perturb_functions:
            for _ in range(n_perturbations):
                try:
                    # Genera testo perturbato
                    pert_text = perturb_func(text)
                    if pert_text == text:  # Skip se perturbazione non ha effetto
                        continue
                    
                    # Calcola attribution perturbata
                    pert_attr = explainer(pert_text)
                    if not pert_attr.tokens or not pert_attr.scores:
                        continue
                    
                    # Allinea token comuni
                    score_diffs = []
                    for tok, score in zip(orig_attr.tokens, orig_attr.scores):
                        if tok in pert_attr.tokens:
                            j = pert_attr.tokens.index(tok)
                            diff = abs(score - pert_attr.scores[j])
                            score_diffs.append(diff)
                    
                    if score_diffs:
                        all_diffs.extend(score_diffs)
                        
                except Exception as e:
                    print(f"Errore in perturbazione: {e}")
                    continue
        
        return float(np.mean(all_diffs)) if all_diffs else 0.0
        
    except Exception as e:
        print(f"Errore in compute_robustness: {e}")
        return 0.0

# ==== 6. Consistency ====

def compute_consistency(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer_a: PreTrainedTokenizer,
    tokenizer_b: PreTrainedTokenizer,
    explainer_a: Callable[[str], Attribution],
    explainer_b: Callable[[str], Attribution],
    text: str,
) -> float:
    """
    Calcola la consistency come correlazione di Spearman tra le saliency
    di due modelli per token comuni.
    
    Args:
        model_a, model_b: Due modelli da confrontare
        tokenizer_a, tokenizer_b: Tokenizer corrispondenti
        explainer_a, explainer_b: Explainer per ogni modello
        text: Testo da analizzare
    
    Returns:
        float: Correlazione di Spearman (-1 a 1, più alto = più consistente)
    """
    try:
        # Calcola attribution per entrambi i modelli
        attr_a = explainer_a(text)
        attr_b = explainer_b(text)
        
        if not attr_a.tokens or not attr_b.tokens:
            return 0.0
        
        # Allinea token comuni preservando ordine in attr_a
        shared_scores_a, shared_scores_b = [], []
        
        for tok, score_a in zip(attr_a.tokens, attr_a.scores):
            if tok in attr_b.tokens:
                idx = attr_b.tokens.index(tok)
                shared_scores_a.append(score_a)
                shared_scores_b.append(attr_b.scores[idx])
        
        # Calcola correlazione se ci sono abbastanza token condivisi
        if len(shared_scores_a) < MIN_SHARED_TOKENS:
            return 0.0
        
        # Gestisci caso di varianza zero
        if np.var(shared_scores_a) == 0 or np.var(shared_scores_b) == 0:
            return 1.0 if np.array_equal(shared_scores_a, shared_scores_b) else 0.0
        
        rho, p_value = spearmanr(shared_scores_a, shared_scores_b)
        
        # Gestisci NaN (può succedere con dati costanti)
        if np.isnan(rho):
            return 0.0
        
        return float(rho)
        
    except Exception as e:
        print(f"Errore in compute_consistency: {e}")
        return 0.0

# ==== 7. Contrastivity ====

def _normalize_scores(scores: Sequence[float]) -> np.ndarray:
    """Normalizza i punteggi per formare una distribuzione di probabilità."""
    arr = np.array(scores, dtype=float)
    
    # Sposta a valori non-negativi
    arr = arr - np.min(arr)
    
    # Normalizza
    total = np.sum(arr)
    if total == 0:
        return np.ones(len(arr)) / len(arr)  # Distribuzione uniforme
    
    return arr / total

def compute_contrastivity(
    positive_attrs: List[Attribution],
    negative_attrs: List[Attribution],
    use_jensen_shannon: bool = False,
) -> float:
    """
    Calcola la contrastivity come divergenza KL tra distribuzioni medie
    di importanza delle due classi.
    
    Args:
        positive_attrs: Lista di Attribution per esempi positivi
        negative_attrs: Lista di Attribution per esempi negativi
        use_jensen_shannon: Se True usa Jensen-Shannon invece di KL
    
    Returns:
        float: Divergenza (più alto = feature più diverse tra classi)
    """
    try:
        if not positive_attrs or not negative_attrs:
            return 0.0
        
        # Raccogli tutti i token e i loro punteggi
        token_scores_pos = {}
        token_scores_neg = {}
        
        def accumulate_scores(score_dict: dict, attr: Attribution):
            for tok, score in zip(attr.tokens, attr.scores):
                if tok.strip() and tok not in ['[CLS]', '[SEP]', '[PAD]']:
                    score_dict[tok] = score_dict.get(tok, 0.0) + score
        
        # Accumula punteggi per classe
        for attr in positive_attrs:
            accumulate_scores(token_scores_pos, attr)
        for attr in negative_attrs:
            accumulate_scores(token_scores_neg, attr)
        
        # Crea vocabolario unificato
        vocab = set(token_scores_pos.keys()) | set(token_scores_neg.keys())
        if len(vocab) < 2:
            return 0.0
        
        vocab = sorted(vocab)  # Ordine consistente
        
        # Calcola distribuzioni medie
        pos_scores = [token_scores_pos.get(tok, 0.0) for tok in vocab]
        neg_scores = [token_scores_neg.get(tok, 0.0) for tok in vocab]
        
        # Normalizza per creare distribuzioni di probabilità
        p = _normalize_scores(pos_scores)
        q = _normalize_scores(neg_scores)
        
        # Calcola divergenza
        if use_jensen_shannon:
            # Jensen-Shannon divergence (simmetrica)
            m = 0.5 * (p + q)
            js_div = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
            return float(js_div)
        else:
            # KL divergence standard
            # Aggiungi epsilon per evitare log(0)
            epsilon = 1e-10
            q_smooth = q + epsilon
            kl_div = entropy(p, q_smooth, base=2)
            return float(kl_div)
            
    except Exception as e:
        print(f"Errore in compute_contrastivity: {e}")
        return 0.0

# ==== 8. Funzioni batch per valutazione su dataset ====

def evaluate_robustness_over_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    sample_size: Optional[int] = None,
    show_progress: bool = True,
) -> float:
    """
    Valuta robustness su un dataset di testi.
    
    Args:
        model, tokenizer, explainer: Componenti del modello
        texts: Lista di testi da valutare
        sample_size: Numero di esempi da campionare (None = tutti)
        show_progress: Se mostrare barra di progresso
    
    Returns:
        float: Media delle robustness individuali
    """
    try:
        # Campiona se necessario
        if sample_size and sample_size < len(texts):
            texts = random.sample(texts, sample_size)
        
        # Calcola robustness per ogni testo
        robustness_scores = []
        iterator = tqdm(texts, desc="Robustness") if show_progress else texts
        
        for text in iterator:
            try:
                if text.strip():  # Skip testi vuoti
                    score = compute_robustness(model, tokenizer, explainer, text)
                    robustness_scores.append(score)
            except Exception as e:
                print(f"Errore su testo: {e}")
                continue
        
        return float(np.mean(robustness_scores)) if robustness_scores else 0.0
        
    except Exception as e:
        print(f"Errore in evaluate_robustness_over_dataset: {e}")
        return 0.0

def evaluate_consistency_over_dataset(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer_a: PreTrainedTokenizer,
    tokenizer_b: PreTrainedTokenizer,
    explainer_a: Callable[[str], Attribution],
    explainer_b: Callable[[str], Attribution],
    texts: List[str],
    sample_size: Optional[int] = None,
    show_progress: bool = True,
) -> float:
    """
    Valuta consistency tra due modelli su un dataset.
    
    Returns:
        float: Media delle correlazioni di Spearman
    """
    try:
        # Campiona se necessario
        if sample_size and sample_size < len(texts):
            texts = random.sample(texts, sample_size)
        
        # Calcola consistency per ogni testo
        consistency_scores = []
        iterator = tqdm(texts, desc="Consistency") if show_progress else texts
        
        for text in iterator:
            try:
                if text.strip():  # Skip testi vuoti
                    score = compute_consistency(
                        model_a, model_b, tokenizer_a, tokenizer_b,
                        explainer_a, explainer_b, text
                    )
                    consistency_scores.append(score)
            except Exception as e:
                print(f"Errore su testo: {e}")
                continue
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
        
    except Exception as e:
        print(f"Errore in evaluate_consistency_over_dataset: {e}")
        return 0.0

def evaluate_contrastivity_over_dataset(
    positive_attrs: List[Attribution],
    negative_attrs: List[Attribution],
    use_jensen_shannon: bool = False,
) -> float:
    """
    Valuta contrastivity su liste di attribution.
    
    Returns:
        float: Divergenza KL o Jensen-Shannon
    """
    return compute_contrastivity(positive_attrs, negative_attrs, use_jensen_shannon)

# ==== 9. Utility per debugging ====

def print_metric_summary(
    robustness_score: float,
    consistency_score: float,
    contrastivity_score: float,
):
    """Stampa un riassunto delle metriche calcolate."""
    print("\n=== SUMMARY METRICHE XAI ===")
    print(f"Robustness:    {robustness_score:.4f} (piu basso = piu robusto)")
    print(f"Consistency:   {consistency_score:.4f} (piu alto = piu consistente)")
    print(f"Contrastivity: {contrastivity_score:.4f} (piu alto = piu contrastivo)")
    print("==============================\n")

# ==== 10. Test di compatibilità ====
if __name__ == "__main__":
    print("Test funzioni di perturbazione...")
    
    test_text = "This movie was absolutely fantastic and I loved every minute of it!"
    
    print(f"Originale: {test_text}")
    print(f"Masked: {_random_mask(test_text)}")
    print(f"Deleted: {_random_delete(test_text)}")
    print(f"Substituted: {_random_substitute(test_text)}")
    
    print("\nTest normalizzazione...")
    test_scores = [0.5, -0.2, 0.8, 0.1, -0.1]
    normalized = _normalize_scores(test_scores)
    print(f"Originali: {test_scores}")
    print(f"Normalizzati: {normalized}")
    print(f"Somma: {np.sum(normalized)}")
    
    print("\nTest metrics completato!")