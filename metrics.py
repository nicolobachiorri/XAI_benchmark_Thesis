"""
metrics.py – Metriche del paper XAI (con Consistency corretta)
============================================================

Implementa **tre metriche automatiche** tratte da "Evaluating the effectiveness
of XAI techniques for encoder-based language models".

1. **Robustness** – stabilità delle saliency sotto perturbazione del testo
2. **Consistency** – stabilità dell'explainer con inference seed diversi (CORRETTA)
3. **Contrastivity** – diversità delle saliency fra classi opposte

La metrica di *Human-reasoning Agreement* non è implementata perché richiede
annotazioni manuali.

AGGIORNAMENTO: Consistency ora usa inference seed invece di modelli diversi.
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
RANDOM_STATE = 42

# Parametri per consistency con inference seed
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]

# ==== 3. Helper per gestione probabilità ====

def _get_positive_prob(model: PreTrainedModel, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> float:
    """Calcola la probabilità della classe positiva in modo robusto."""
    try:
        # FIX: Assicura che input siano su GPU
        import models
        input_ids = input_ids.to(models.DEVICE)
        attn_mask = attn_mask.to(models.DEVICE)
        
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

# ==== 6. Consistency (CORRETTA CON INFERENCE SEED) ====

def compute_consistency_inference_seed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = True
) -> float:
    """
    Calcola consistency usando inference seed diversi (approccio corretto).
    
    Invece di confrontare modelli diversi, confronta lo stesso modello+explainer
    con seed diversi per l'inferenza, attivando dropout per stocasticità.
    
    Args:
        model: Modello pre-trained
        tokenizer: Tokenizer corrispondente
        explainer: Funzione explainer
        texts: Lista di testi da analizzare
        seeds: Lista di seed per inferenza stocastica
        show_progress: Se mostrare progress
    
    Returns:
        float: Correlazione di Spearman media tra tutte le coppie di seed
    """
    
    if show_progress:
        print(f"Computing consistency with {len(seeds)} inference seeds...")
    
    # ATTIVA training mode per dropout stocastico
    original_mode = model.training
    model.train()
    
    # DISATTIVA gradient computation (non vogliamo fine-tuning)
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
    
    if show_progress:
        print(f"  Model set to training mode (dropout active)")
        # Mostra se ci sono layer dropout
        dropout_layers = [name for name, module in model.named_modules() 
                         if isinstance(module, torch.nn.Dropout)]
        if dropout_layers:
            print(f"  Found {len(dropout_layers)} dropout layers")
        else:
            print(f"  WARNING: No dropout layers found - consistency might be perfect")
    
    try:
        # Genera explanations per ogni inference seed
        explanations_by_seed = {}
        
        for seed in seeds:
            if show_progress:
                print(f"  Processing inference seed {seed}...")
            
            explanations = []
            text_iterator = tqdm(texts, desc=f"Seed {seed}", leave=False) if show_progress else texts
            
            for text in text_iterator:
                try:
                    # Imposta seed prima di ogni explanation per dropout stocastico
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    
                    attr = explainer(text)
                    explanations.append(attr)
                except Exception as e:
                    if show_progress:
                        print(f"    Error on text: {e}")
                    # Aggiungi attribution vuota
                    explanations.append(Attribution([], []))
            
            explanations_by_seed[seed] = explanations
        
        # Calcola consistency tra tutte le coppie di seed
        consistency_scores = []
        
        for i, seed_a in enumerate(seeds):
            for j, seed_b in enumerate(seeds[i+1:], i+1):
                if show_progress:
                    print(f"  Comparing seeds {seed_a} vs {seed_b}...")
                
                correlation = _compute_spearman_correlation_explanations(
                    explanations_by_seed[seed_a],
                    explanations_by_seed[seed_b]
                )
                consistency_scores.append(correlation)
        
        # Media di tutte le correlazioni
        final_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0
        
        if show_progress:
            print(f"  Final consistency: {final_consistency:.4f}")
            if final_consistency > 0.99:
                print(f"  NOTE: Very high consistency - model might not have significant dropout")
        
        return final_consistency
        
    finally:
        # Ripristina stato originale del modello
        model.train(original_mode)
        for name, param in model.named_parameters():
            param.requires_grad_(original_requires_grad[name])

def _compute_spearman_correlation_explanations(
    explanations_a: List[Attribution],
    explanations_b: List[Attribution]
) -> float:
    """
    Calcola correlazione di Spearman tra due liste di explanations.
    """
    correlations = []
    
    for attr_a, attr_b in zip(explanations_a, explanations_b):
        # Skip se una delle attribution è vuota
        if not attr_a.tokens or not attr_b.tokens:
            continue
        
        # Allinea token comuni
        shared_scores_a, shared_scores_b = [], []
        
        # Usa set per lookup più veloce
        tokens_b_set = set(attr_b.tokens)
        token_to_idx_b = {tok: i for i, tok in enumerate(attr_b.tokens)}
        
        for token, score_a in zip(attr_a.tokens, attr_a.scores):
            if token in tokens_b_set:
                idx = token_to_idx_b[token]
                shared_scores_a.append(score_a)
                shared_scores_b.append(attr_b.scores[idx])
        
        # Calcola correlazione se ci sono abbastanza token condivisi
        if len(shared_scores_a) >= MIN_SHARED_TOKENS:
            # Converti in numpy per calcolo più veloce
            arr_a = np.array(shared_scores_a)
            arr_b = np.array(shared_scores_b)
            
            # Gestisci varianza zero
            if np.var(arr_a) == 0 or np.var(arr_b) == 0:
                correlation = 1.0 if np.array_equal(arr_a, arr_b) else 0.0
            else:
                try:
                    rho, _ = spearmanr(arr_a, arr_b)
                    correlation = float(rho) if not np.isnan(rho) else 0.0
                except Exception:
                    correlation = 0.0
            
            correlations.append(correlation)
    
    return np.mean(correlations) if correlations else 0.0

# ==== 7. Consistency (Solo Inference Seed) ====

def compute_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = False
) -> float:
    """
    Calcola consistency usando inference seed diversi.
    
    Args:
        model: Modello da testare
        tokenizer: Tokenizer del modello
        explainer: Explainer da testare
        texts: Testi da analizzare
        seeds: Seed per inferenza stocastica
        show_progress: Se mostrare progress
    
    Returns:
        float: Consistency score (correlazione di Spearman media)
    """
    return compute_consistency_inference_seed(
        model=model,
        tokenizer=tokenizer,
        explainer=explainer,
        texts=texts,
        seeds=seeds,
        show_progress=show_progress
    )

# ==== 8. Contrastivity ====

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



# Vocabolario: ["great", "terrible", "movie", "film"]
# Positive: great=0.8, terrible=0.0, movie=0.2, film=0.1
# Negative: great=0.0, terrible=0.9, movie=0.2, film=0.1

# Dopo normalizzazione:
# p = [0.73, 0.0, 0.18, 0.09]  # Classe positiva
# q = [0.0, 0.75, 0.17, 0.08]  # Classe negativa

# KL Divergence:
# KL = 0.73 * log2(0.73/0.00001) + 0.0 * log2(...) + 0.18 * log2(0.18/0.17) + 0.09 * log2(0.09/0.08)
# KL = 0.73 * 16.49 + 0 + 0.18 * 0.08 + 0.09 * 0.17
# KL = 12.04 + 0 + 0.014 + 0.015 = 12.07

# Contrastivity = 12.07 (ALTO = molto contrastivo)

# ==== 9. Funzioni batch per valutazione su dataset ====

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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    sample_size: Optional[int] = None,
    show_progress: bool = True,
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS
) -> float:
    """
    Valuta consistency su dataset usando inference seed.
    
    Args:
        model: Modello da testare
        tokenizer: Tokenizer del modello
        explainer: Explainer da testare
        texts: Lista di testi
        sample_size: Numero di esempi (None = tutti)
        show_progress: Se mostrare progress
        seeds: Seed per inferenza stocastica
    
    Returns:
        float: Consistency score media
    """
    return compute_consistency_inference_seed(
        model=model,
        tokenizer=tokenizer,
        explainer=explainer,
        texts=texts[:sample_size] if sample_size else texts,
        seeds=seeds,
        show_progress=show_progress
    )

# ==== 10. Utility per debugging ====

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

def test_inference_seed_effect():
    """Test per verificare che l'inference seed abbia effetto."""
    print("Testing inference seed effect...")
    
    try:
        import models
        model = models.load_model("distilbert")
        tokenizer = models.load_tokenizer("distilbert")
        
        test_text = "This movie is great!"
        encoded = tokenizer(test_text, return_tensors="pt", max_length=50, truncation=True)
        
        # Test in eval mode
        model.eval()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        output1 = model(**encoded).logits
        
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        output2 = model(**encoded).logits
        
        eval_identical = torch.allclose(output1, output2, atol=1e-6)
        print(f"  Eval mode - Identical outputs: {eval_identical}")
        
        # Test in training mode
        model.train()
        for param in model.parameters():
            param.requires_grad_(False)
        
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        output1 = model(**encoded).logits
        
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        output2 = model(**encoded).logits
        
        train_identical = torch.allclose(output1, output2, atol=1e-6)
        print(f"  Training mode - Identical outputs: {train_identical}")
        
        if eval_identical and not train_identical:
            print("  Inference seed is working correctly")
            return True
        elif eval_identical and train_identical:
            print("  WARNING: Dropout might not be significant enough")
            return False
        else:
            print("  Unexpected behavior")
            return False
            
    except Exception as e:
        print(f"  Test failed: {e}")
        return False

# ==== 11. Test di compatibilità ====
if __name__ == "__main__":
    print("Testing metrics...")
    
    # Test funzioni di perturbazione
    test_text = "This movie was absolutely fantastic and I loved every minute of it!"
    
    print(f"Originale: {test_text}")
    print(f"Masked: {_random_mask(test_text)}")
    print(f"Deleted: {_random_delete(test_text)}")
    print(f"Substituted: {_random_substitute(test_text)}")
    
    # Test normalizzazione
    print("\nTesting normalization...")
    test_scores = [0.5, -0.2, 0.8, 0.1, -0.1]
    normalized = _normalize_scores(test_scores)
    print(f"Originali: {test_scores}")
    print(f"Normalizzati: {normalized}")
    print(f"Somma: {np.sum(normalized)}")
    
    # Test inference seed effect
    print("\nTesting inference seed effect...")
    test_inference_seed_effect()
    
    print("\nMetrics test completato!")
    print("\nNOTE: Consistency ora usa solo inference seed approach:")
    print("1. compute_consistency(model, tokenizer, explainer, texts, seeds)")
    print("2. evaluate_consistency_over_dataset(model, tokenizer, explainer, texts)")
    print("3. Il risultato misura la stabilità dell'explainer")
    print("4. Valori attesi: 0.5-0.9 (più alto = più stabile)")
    print("5. Se consistency > 0.99, il dropout potrebbe essere troppo basso")