"""
evaluate.py – Pipeline di valutazione XAI semplificata
=====================================================

Scopo: permettere di valutare rapidamente un modello + un explainer su una 
metrica scelta, usando le funzioni già definite in:
    • models.py (aggiornato con nuovi modelli pre-trained)
    • dataset.py (ottimizzato per IMDB)
    • explainers.py (6 explainer supportati)
    • metrics.py (robustness, consistency, contrastivity)

Uso CLI (esempi):
    python evaluate.py --model distilbert --explainer grad_input --metric robustness
    python evaluate.py --model bert-base --explainer attention_rollout --metric contrastivity

Per «consistency» servono due modelli: --model-a, --model-b
    python evaluate.py --model-a distilbert --model-b bert-base \
                      --explainer grad_input --metric consistency

Parametri facili da cambiare in testa al file.
"""

# ==== 1. Librerie ====
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np

import models
import dataset
import explainers
import metrics

# ==== 2. Parametri globali ====
DEFAULT_SAMPLE_SIZE = 500  # numero di esempi su cui valutare (None = tutto test)
RANDOM_STATE = 42
MAX_SAMPLE_SIZE = 2000  # Limite massimo per evitare tempi troppo lunghi

# Seed per riproducibilità
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ==== 3. Helper caricamento dati ====

def _load_test_texts_labels(sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE) -> Tuple[List[str], List[int]]:
    """
    Carica testi e labels dal dataset di test.
    
    Args:
        sample_size: Numero di esempi da campionare (None = tutti)
    
    Returns:
        Tuple[List[str], List[int]]: (testi, labels)
    """
    try:
        # Usa il dataset globale già caricato
        df = dataset.test_df
        
        if sample_size and sample_size < len(df):
            # Campionamento stratificato per mantenere bilanciamento classi
            df_pos = df[df['label'] == 1]
            df_neg = df[df['label'] == 0]
            
            n_pos = min(sample_size // 2, len(df_pos))
            n_neg = min(sample_size - n_pos, len(df_neg))
            
            sampled_pos = df_pos.sample(n=n_pos, random_state=RANDOM_STATE)
            sampled_neg = df_neg.sample(n=n_neg, random_state=RANDOM_STATE)
            
            df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=RANDOM_STATE)
        
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        print(f"Caricati {len(texts)} esempi dal dataset test")
        print(f"Distribuzione classi: {np.bincount(labels)}")
        
        return texts, labels
        
    except Exception as e:
        print(f"Errore caricamento dati: {e}")
        return [], []

def _validate_model_explainer_compatibility(model_key: str, explainer_name: str) -> bool:
    """
    Verifica che il modello e l'explainer siano compatibili.
    
    Args:
        model_key: Chiave del modello
        explainer_name: Nome dell'explainer
    
    Returns:
        bool: True se compatibili
    """
    try:
        # Carica modello e tokenizer
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        
        # Verifica che l'explainer si possa creare
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        
        # Test rapido su testo di esempio
        test_text = "This is a test sentence for compatibility check."
        attr = explainer(test_text)
        
        if not attr.tokens or not attr.scores:
            print(f"WARN: Explainer {explainer_name} ha restituito attribution vuota")
            return False
        
        return True
        
    except Exception as e:
        print(f"Errore compatibilità {model_key} + {explainer_name}: {e}")
        return False

# ==== 4. Funzioni di valutazione ====

def _eval_robustness(model_key: str, explainer_name: str, sample_size: Optional[int]):
    """
    Valuta robustness di un modello+explainer.
    
    Args:
        model_key: Chiave del modello
        explainer_name: Nome dell'explainer
        sample_size: Numero di esempi da valutare
    """
    print(f"\n=== ROBUSTNESS EVALUATION ===")
    print(f"Modello: {model_key}")
    print(f"Explainer: {explainer_name}")
    print(f"Sample size: {sample_size}")
    
    try:
        # Verifica compatibilità
        if not _validate_model_explainer_compatibility(model_key, explainer_name):
            print("ERRORE: Modello e explainer non sono compatibili")
            return
        
        # Carica componenti
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        
        # Carica dati
        texts, _ = _load_test_texts_labels(sample_size)
        if not texts:
            print("ERRORE: Nessun testo caricato")
            return
        
        # Calcola robustness
        print("Calcolando robustness...")
        score = metrics.evaluate_robustness_over_dataset(
            model, tokenizer, explainer, texts, show_progress=True
        )
        
        print(f"\nRISULTATO:")
        print(f"Robustness ({model_key}, {explainer_name}): {score:.4f}")
        print("(Piu basso = piu robusto)")
        
    except Exception as e:
        print(f"ERRORE in robustness evaluation: {e}")

def _eval_contrastivity(model_key: str, explainer_name: str, sample_size: Optional[int]):
    """
    Valuta contrastivity di un modello+explainer.
    
    Args:
        model_key: Chiave del modello
        explainer_name: Nome dell'explainer
        sample_size: Numero di esempi da valutare
    """
    print(f"\n=== CONTRASTIVITY EVALUATION ===")
    print(f"Modello: {model_key}")
    print(f"Explainer: {explainer_name}")
    print(f"Sample size: {sample_size}")
    
    try:
        # Verifica compatibilità
        if not _validate_model_explainer_compatibility(model_key, explainer_name):
            print("ERRORE: Modello e explainer non sono compatibili")
            return
        
        # Carica componenti
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        
        # Carica dati
        texts, labels = _load_test_texts_labels(sample_size)
        if not texts:
            print("ERRORE: Nessun testo caricato")
            return
        
        # Separa per classe e calcola attribution
        print("Generando attribution per classe positiva...")
        pos_texts = [t for t, l in zip(texts, labels) if l == 1]
        pos_attrs = []
        for text in pos_texts:
            try:
                attr = explainer(text)
                if attr.tokens and attr.scores:
                    pos_attrs.append(attr)
            except Exception as e:
                print(f"Errore su testo positivo: {e}")
                continue
        
        print("Generando attribution per classe negativa...")
        neg_texts = [t for t, l in zip(texts, labels) if l == 0]
        neg_attrs = []
        for text in neg_texts:
            try:
                attr = explainer(text)
                if attr.tokens and attr.scores:
                    neg_attrs.append(attr)
            except Exception as e:
                print(f"Errore su testo negativo: {e}")
                continue
        
        print(f"Attribution generate: {len(pos_attrs)} positive, {len(neg_attrs)} negative")
        
        if not pos_attrs or not neg_attrs:
            print("ERRORE: Attribution insufficienti per calcolare contrastivity")
            return
        
        # Calcola contrastivity
        print("Calcolando contrastivity...")
        score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
        
        print(f"\nRISULTATO:")
        print(f"Contrastivity ({model_key}, {explainer_name}): {score:.4f}")
        print("(Piu alto = piu contrastivo)")
        
    except Exception as e:
        print(f"ERRORE in contrastivity evaluation: {e}")

def _eval_consistency(model_a: str, model_b: str, explainer_name: str, sample_size: Optional[int]):
    """
    Valuta consistency tra due modelli con stesso explainer.
    
    Args:
        model_a: Primo modello
        model_b: Secondo modello
        explainer_name: Nome dell'explainer
        sample_size: Numero di esempi da valutare
    """
    print(f"\n=== CONSISTENCY EVALUATION ===")
    print(f"Modello A: {model_a}")
    print(f"Modello B: {model_b}")
    print(f"Explainer: {explainer_name}")
    print(f"Sample size: {sample_size}")
    
    try:
        # Verifica compatibilità per entrambi i modelli
        if not _validate_model_explainer_compatibility(model_a, explainer_name):
            print(f"ERRORE: Modello {model_a} e explainer {explainer_name} non sono compatibili")
            return
        
        if not _validate_model_explainer_compatibility(model_b, explainer_name):
            print(f"ERRORE: Modello {model_b} e explainer {explainer_name} non sono compatibili")
            return
        
        # Carica componenti per modello A
        model1 = models.load_model(model_a)
        tokenizer1 = models.load_tokenizer(model_a)
        explainer1 = explainers.get_explainer(explainer_name, model1, tokenizer1)
        
        # Carica componenti per modello B
        model2 = models.load_model(model_b)
        tokenizer2 = models.load_tokenizer(model_b)
        explainer2 = explainers.get_explainer(explainer_name, model2, tokenizer2)
        
        # Carica dati
        texts, _ = _load_test_texts_labels(sample_size)
        if not texts:
            print("ERRORE: Nessun testo caricato")
            return
        
        # Calcola consistency
        print("Calcolando consistency...")
        score = metrics.evaluate_consistency_over_dataset(
            model1, model2, tokenizer1, tokenizer2, 
            explainer1, explainer2, texts, show_progress=True
        )
        
        print(f"\nRISULTATO:")
        print(f"Consistency ({model_a} vs {model_b}, {explainer_name}): {score:.4f}")
        print("(Piu alto = piu consistente)")
        
    except Exception as e:
        print(f"ERRORE in consistency evaluation: {e}")

# ==== 5. Funzioni utility ====

def _list_available_models():
    """Lista modelli disponibili."""
    print("\nModelli disponibili:")
    for key, name in models.MODELS.items():
        print(f"  {key}: {name}")

def _list_available_explainers():
    """Lista explainer disponibili."""
    print("\nExplainer disponibili:")
    for explainer in explainers.list_explainers():
        print(f"  {explainer}")

def _validate_sample_size(sample_size: Optional[int]) -> Optional[int]:
    """
    Valida e ajusta sample_size.
    
    Args:
        sample_size: Dimensione campione richiesta
    
    Returns:
        Optional[int]: Dimensione campione validata
    """
    if sample_size is None:
        return None
    
    if sample_size <= 0:
        print("WARN: Sample size deve essere > 0, usando default")
        return DEFAULT_SAMPLE_SIZE
    
    if sample_size > MAX_SAMPLE_SIZE:
        print(f"WARN: Sample size {sample_size} troppo grande, limitato a {MAX_SAMPLE_SIZE}")
        return MAX_SAMPLE_SIZE
    
    return sample_size

# ==== 6. CLI principale ====

def main():
    parser = argparse.ArgumentParser(description="Valutazione XAI semplice")

    # Parametri base
    parser.add_argument("--metric", required=True, 
                       choices=["robustness", "contrastivity", "consistency"], 
                       help="Nome metrica da valutare")
    parser.add_argument("--explainer", required=True, 
                       choices=explainers.list_explainers(),
                       help="Nome explainer")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_SIZE, 
                       help="Numero di esempi da valutare (None=tutti)")

    # Modelli (mutually exclusive per robustness/contrastivity vs consistency)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=models.MODELS.keys(),
                      help="Modello unico (per robustness/contrastivity)")
    group.add_argument("--model-a", choices=models.MODELS.keys(),
                      help="Modello A (per consistency)")

    parser.add_argument("--model-b", choices=models.MODELS.keys(),
                       help="Modello B (solo per consistency)")

    # Opzioni utility
    parser.add_argument("--list-models", action="store_true",
                       help="Lista modelli disponibili")
    parser.add_argument("--list-explainers", action="store_true",
                       help="Lista explainer disponibili")

    args = parser.parse_args()

    # Gestisci opzioni utility
    if args.list_models:
        _list_available_models()
        return
    
    if args.list_explainers:
        _list_available_explainers()
        return

    # Valida sample size
    sample_size = _validate_sample_size(args.sample)

    # Valida combinazioni di argomenti
    if args.metric == "consistency":
        if not (args.model_a and args.model_b):
            parser.error("--metric consistency richiede --model-a e --model-b")
        if args.model_a == args.model_b:
            parser.error("--model-a e --model-b devono essere diversi")
        _eval_consistency(args.model_a, args.model_b, args.explainer, sample_size)
    else:
        if not args.model:
            parser.error("--metric robustness/contrastivity richiede --model")
        if args.metric == "robustness":
            _eval_robustness(args.model, args.explainer, sample_size)
        else:  # contrastivity
            _eval_contrastivity(args.model, args.explainer, sample_size)

if __name__ == "__main__":
    # Importa pandas qui per evitare errori se non usato
    import pandas as pd
    main()
