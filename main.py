"""
main.py – CLI unificata per il benchmark XAI
===========================================

Due comandi principali:
* **explain** – spiega una singola frase
* **evaluate** – valuta modello + explainer su una metrica (integrato)

AGGIORNAMENTO: Eliminato evaluate.py, tutto integrato qui.
Consistency ora usa inference seed invece di modelli diversi.

Esempi:
```bash
# Valutare robustness
python main.py evaluate --metric robustness --model distilbert --explainer lime

# Valutare consistency (con inference seed)
python main.py evaluate --metric consistency --model distilbert --explainer lime

# Spiegare una frase
python main.py explain --model distilbert --explainer lime --text "Great movie!"
```
"""

import argparse
import sys
from typing import List, Optional

import models
import dataset
import explainers
import metrics
from utils import set_seed, Timer

# Seed fisso
set_seed(42)

# Parametri per consistency
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]
DEFAULT_SAMPLE_SIZE = 500

def cmd_explain(args):
    """Comando explain: spiega una frase."""
    print(f"Spiegazione con {args.model} + {args.explainer}")
    print(f"Testo: '{args.text}'")
    
    try:
        # Carica modello
        model = models.load_model(args.model)
        tokenizer = models.load_tokenizer(args.model)
        explainer = explainers.get_explainer(args.explainer, model, tokenizer)
        
        # Genera spiegazione
        with Timer("Spiegazione"):
            attr = explainer(args.text)
        
        # Mostra risultati
        print("\nImportanza token:")
        for token, score in zip(attr.tokens, attr.scores):
            # Solo token "parole"
            if token.isalpha():
                print(f"{token:>15} {score:+.3f}")

            
    except Exception as e:
        print(f"Errore: {e}")
        sys.exit(1)

def get_test_data(sample_size: Optional[int] = None) -> tuple:
    """Carica dati di test con campionamento stratificato."""
    texts = dataset.test_df["text"].tolist()
    labels = dataset.test_df["label"].tolist()
    
    if sample_size and sample_size < len(texts):
        # Campionamento stratificato bilanciato
        pos_indices = [i for i, l in enumerate(labels) if l == 1]
        neg_indices = [i for i, l in enumerate(labels) if l == 0]
        
        n_pos = min(sample_size // 2, len(pos_indices))
        n_neg = min(sample_size - n_pos, len(neg_indices))
        
        import random
        selected_pos = random.sample(pos_indices, n_pos)
        selected_neg = random.sample(neg_indices, n_neg)
        
        selected_indices = selected_pos + selected_neg
        random.shuffle(selected_indices)
        
        texts = [texts[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
    
    return texts, labels

def evaluate_robustness(model_key: str, explainer_name: str, sample_size: int) -> float:
    """Valuta robustness di un modello+explainer."""
    print(f"Valutando robustness...")
    
    try:
        # Carica modello
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        
        # Carica dati
        texts, _ = get_test_data(sample_size)
        
        # Calcola robustness
        score = metrics.evaluate_robustness_over_dataset(
            model, tokenizer, explainer, texts, show_progress=True
        )
        
        return score
        
    except Exception as e:
        print(f"Errore in robustness: {e}")
        return float('nan')

def evaluate_contrastivity(model_key: str, explainer_name: str, sample_size: int) -> float:
    """Valuta contrastivity di un modello+explainer."""
    print(f"Valutando contrastivity...")
    
    try:
        # Carica modello
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        
        # Carica dati
        texts, labels = get_test_data(sample_size)
        
        # Separa per classe
        pos_texts = [t for t, l in zip(texts, labels) if l == 1]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0]
        
        # Limita per velocità
        pos_texts = pos_texts[:min(100, len(pos_texts))]
        neg_texts = neg_texts[:min(100, len(neg_texts))]
        
        print(f"Generando attribution per {len(pos_texts)} testi positivi...")
        pos_attrs = []
        for text in pos_texts:
            try:
                attr = explainer(text)
                if attr.tokens and attr.scores:
                    pos_attrs.append(attr)
            except Exception as e:
                print(f"Errore su testo positivo: {e}")
                continue
        
        print(f"Generando attribution per {len(neg_texts)} testi negativi...")
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
            print("Errore: Attribution insufficienti per calcolare contrastivity")
            return 0.0
        
        # Calcola contrastivity
        score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
        
        return score
        
    except Exception as e:
        print(f"Errore in contrastivity: {e}")
        return float('nan')

def evaluate_consistency(model_key: str, explainer_name: str, sample_size: int, 
                        seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS) -> float:
    """Valuta consistency di un modello+explainer con inference seed."""
    print(f"Valutando consistency con inference seed...")
    print(f"Seed: {seeds}")
    
    try:
        # Carica modello
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        
        # Carica dati (limita per consistency)
        texts, _ = get_test_data(min(50, sample_size))
        
        # Calcola consistency usando inference seed
        score = metrics.evaluate_consistency_over_dataset(
            model=model,
            tokenizer=tokenizer,
            explainer=explainer,
            texts=texts,
            seeds=seeds,
            show_progress=True
        )
        
        return score
        
    except Exception as e:
        print(f"Errore in consistency: {e}")
        return float('nan')

def cmd_evaluate(args):
    """Comando evaluate: valuta una singola combinazione modello+explainer+metrica."""
    print(f"Valutazione: {args.model} + {args.explainer} + {args.metric}")
    print(f"Sample size: {args.sample}")
    
    # Validazione parametri
    if args.model not in models.MODELS:
        print(f"Errore: Modello '{args.model}' non disponibile")
        print(f"Modelli disponibili: {list(models.MODELS.keys())}")
        sys.exit(1)
    
    if args.explainer not in explainers.list_explainers():
        print(f"Errore: Explainer '{args.explainer}' non disponibile")
        print(f"Explainer disponibili: {explainers.list_explainers()}")
        sys.exit(1)
    
    # Parsing seed per consistency
    seeds = args.seeds if args.seeds else DEFAULT_CONSISTENCY_SEEDS
    
    # Valutazione
    with Timer(f"Valutazione {args.metric}"):
        if args.metric == "robustness":
            score = evaluate_robustness(args.model, args.explainer, args.sample)
        elif args.metric == "contrastivity":
            score = evaluate_contrastivity(args.model, args.explainer, args.sample)
        elif args.metric == "consistency":
            score = evaluate_consistency(args.model, args.explainer, args.sample, seeds)
        else:
            print(f"Errore: Metrica '{args.metric}' non supportata")
            sys.exit(1)
    
    # Risultati
    print(f"\n{'='*50}")
    print(f"RISULTATO:")
    print(f"Modello: {args.model}")
    print(f"Explainer: {args.explainer}")
    print(f"Metrica: {args.metric}")
    print(f"Score: {score:.4f}")
    
    # Interpretazione
    if args.metric == "robustness":
        print("(Più basso = più robusto)")
        if score < 0.05:
            print("Interpretazione: Molto robusto")
        elif score < 0.1:
            print("Interpretazione: Robusto")
        elif score < 0.2:
            print("Interpretazione: Moderatamente robusto")
        else:
            print("Interpretazione: Poco robusto")
    
    elif args.metric == "consistency":
        print("(Più alto = più consistente)")
        if score > 0.9:
            print("Interpretazione: Molto consistente")
        elif score > 0.8:
            print("Interpretazione: Consistente")
        elif score > 0.6:
            print("Interpretazione: Moderatamente consistente")
        else:
            print("Interpretazione: Poco consistente")
    
    elif args.metric == "contrastivity":
        print("(Più alto = più contrastivo)")
        if score > 5.0:
            print("Interpretazione: Molto contrastivo")
        elif score > 2.0:
            print("Interpretazione: Contrastivo")
        elif score > 1.0:
            print("Interpretazione: Moderatamente contrastivo")
        else:
            print("Interpretazione: Poco contrastivo")
    
    print(f"{'='*50}")

def list_available_resources():
    """Lista risorse disponibili."""
    print("\nModelli disponibili:")
    for key, name in models.MODELS.items():
        print(f"  {key}: {name}")
    
    print("\nExplainer disponibili:")
    for explainer in explainers.list_explainers():
        print(f"  {explainer}")
    
    print("\nMetriche disponibili:")
    for metric in ["robustness", "consistency", "contrastivity"]:
        print(f"  {metric}")

def main():
    parser = argparse.ArgumentParser(description="XAI Benchmark CLI Unificata")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Explain ----
    p_explain = subparsers.add_parser("explain", help="Spiega una frase")
    p_explain.add_argument("--model", required=True, choices=models.MODELS.keys())
    p_explain.add_argument("--explainer", required=True, choices=explainers.list_explainers())
    p_explain.add_argument("--text", required=True, help="Testo da spiegare")

    # ---- Evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Valuta modello+explainer")
    p_eval.add_argument("--metric", required=True, 
                       choices=["robustness", "contrastivity", "consistency"])
    p_eval.add_argument("--model", required=True, choices=models.MODELS.keys())
    p_eval.add_argument("--explainer", required=True, choices=explainers.list_explainers())
    p_eval.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_SIZE,
                       help="Numero di esempi da valutare")
    p_eval.add_argument("--seeds", nargs="+", type=int, default=None,
                       help="Seed per consistency (default: 42 123 456 789)")

    # ---- List ----
    p_list = subparsers.add_parser("list", help="Lista risorse disponibili")

    args = parser.parse_args()

    # Esegui comando
    if args.command == "explain":
        cmd_explain(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "list":
        list_available_resources()

if __name__ == "__main__":
    main()

