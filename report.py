"""
report.py – Genera tabelle per le metriche XAI
==============================================

Crea tabelle con risultati di Robustness, Consistency e Contrastivity
per tutti i modelli e explainer.

Uso:
    python report.py                    # Tutte le metriche, 500 esempi
    python report.py --sample 300       # 300 esempi
    python report.py --metric robustness # Solo robustness
    python report.py --csv              # Output CSV invece di markdown
"""

import argparse
from collections import defaultdict

import pandas as pd

import models
import dataset
import explainers
import metrics
from utils import set_seed, Timer

# Configurazione
EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "contrastivity", "consistency"]

set_seed(42)

def get_test_data(sample_size=None):
    """Carica dati di test."""
    texts = dataset.test_df["text"].tolist()
    labels = dataset.test_df["label"].tolist()
    
    if sample_size and sample_size < len(texts):
        # Campionamento stratificato
        pos_texts = [t for t, l in zip(texts, labels) if l == 1][:sample_size//2]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0][:sample_size//2]
        texts = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    
    return texts, labels

def eval_robustness(model_key, explainer_name, sample_size):
    """Valuta robustness."""
    model = models.load_model(model_key)
    tokenizer = models.load_tokenizer(model_key)
    explainer = explainers.get_explainer(explainer_name, model, tokenizer)
    texts, _ = get_test_data(sample_size)
    return metrics.evaluate_robustness_over_dataset(model, tokenizer, explainer, texts)

def eval_contrastivity(model_key, explainer_name, sample_size):
    """Valuta contrastivity."""
    model = models.load_model(model_key)
    tokenizer = models.load_tokenizer(model_key)
    explainer = explainers.get_explainer(explainer_name, model, tokenizer)
    texts, labels = get_test_data(sample_size)
    
    # Separa per classe
    pos_attrs = []
    neg_attrs = []
    
    for text, label in zip(texts, labels):
        attr = explainer(text)
        if label == 1:
            pos_attrs.append(attr)
        else:
            neg_attrs.append(attr)
    
    return metrics.compute_contrastivity(pos_attrs, neg_attrs)

def eval_consistency(model_a, model_b, explainer_name, sample_size):
    """Valuta consistency tra due modelli."""
    # Carica modelli
    m1 = models.load_model(model_a)
    m2 = models.load_model(model_b)
    t1 = models.load_tokenizer(model_a)
    t2 = models.load_tokenizer(model_b)
    e1 = explainers.get_explainer(explainer_name, m1, t1)
    e2 = explainers.get_explainer(explainer_name, m2, t2)
    
    texts, _ = get_test_data(sample_size)
    return metrics.evaluate_consistency_over_dataset(m1, m2, t1, t2, e1, e2, texts)

def build_robustness_table(sample_size):
    """Tabella robustness: explainer x modelli."""
    print("Calcolando robustness...")
    results = defaultdict(dict)
    
    for explainer in EXPLAINERS:
        for model_key in models.MODELS.keys():
            try:
                with Timer(f"{explainer} + {model_key}"):
                    score = eval_robustness(model_key, explainer, sample_size)
                results[explainer][model_key] = score
            except Exception as e:
                print(f"Errore {explainer}+{model_key}: {e}")
                results[explainer][model_key] = float('nan')
    
    return pd.DataFrame(results).T

def build_contrastivity_table(sample_size):
    """Tabella contrastivity: explainer x modelli."""
    print("Calcolando contrastivity...")
    results = defaultdict(dict)
    
    for explainer in EXPLAINERS:
        for model_key in models.MODELS.keys():
            try:
                with Timer(f"{explainer} + {model_key}"):
                    score = eval_contrastivity(model_key, explainer, sample_size)
                results[explainer][model_key] = score
            except Exception as e:
                print(f"Errore {explainer}+{model_key}: {e}")
                results[explainer][model_key] = float('nan')
    
    return pd.DataFrame(results).T

def build_consistency_table(sample_size):
    """Tabella consistency: explainer x coppie modelli."""
    print("Calcolando consistency...")
    results = defaultdict(dict)
    
    model_list = list(models.MODELS.keys())
    
    for explainer in EXPLAINERS:
        for i, model_a in enumerate(model_list):
            for model_b in model_list[i+1:]:  # Solo coppie uniche
                pair_name = f"{model_a}_vs_{model_b}"
                try:
                    with Timer(f"{explainer} + {pair_name}"):
                        score = eval_consistency(model_a, model_b, explainer, sample_size)
                    results[explainer][pair_name] = score
                except Exception as e:
                    print(f"Errore {explainer}+{pair_name}: {e}")
                    results[explainer][pair_name] = float('nan')
    
    return pd.DataFrame(results).T

def main():
    parser = argparse.ArgumentParser(description="Genera tabelle XAI")
    parser.add_argument("--metric", choices=METRICS + ["all"], default="all")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    args = parser.parse_args()

    print(f"Generando tabelle con {args.sample} esempi...")
    
    # Scegli metriche
    if args.metric == "all":
        metrics_to_run = METRICS
    else:
        metrics_to_run = [args.metric]

    # Genera tabelle
    for metric in metrics_to_run:
        print(f"\n=== {metric.upper()} ===")
        
        if metric == "robustness":
            df = build_robustness_table(args.sample)
        elif metric == "contrastivity":
            df = build_contrastivity_table(args.sample)
        elif metric == "consistency":
            df = build_consistency_table(args.sample)
        
        # Output
        if args.csv:
            filename = f"{metric}_table.csv"
            df.to_csv(filename)
            print(f"Salvato: {filename}")
        else:
            print(df.to_markdown(floatfmt=".4f"))

if __name__ == "__main__":
    main()