"""
report.py – Genera tabelle per le metriche XAI (AGGIORNATO)
==========================================================

Crea tabelle con risultati di Robustness, Consistency e Contrastivity
per tutti i modelli e explainer.

AGGIORNAMENTO: 
- Consistency ora usa inference seed invece di modelli diversi
- Aggiunto attention_flow tra gli explainer disponibili
- Fix syntax error con global declaration

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

# Configurazione GLOBALE
EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "contrastivity", "consistency"]

# Variabile globale per consistency seeds (dichiarata all'inizio)
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]

set_seed(42)

def get_test_data(sample_size=None):
    """Carica dati di test con campionamento stratificato."""
    texts = dataset.test_df["text"].tolist()
    labels = dataset.test_df["label"].tolist()
    
    if sample_size and sample_size < len(texts):
        # Campionamento stratificato
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

def eval_robustness(model_key, explainer_name, sample_size):
    """Valuta robustness."""
    print(f"  Computing robustness for {model_key} + {explainer_name}")
    
    try:
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        texts, _ = get_test_data(sample_size)
        
        score = metrics.evaluate_robustness_over_dataset(
            model, tokenizer, explainer, texts, show_progress=False
        )
        
        print(f"    Robustness: {score:.4f}")
        return score
        
    except Exception as e:
        print(f"    Error: {e}")
        return float('nan')

def eval_contrastivity(model_key, explainer_name, sample_size):
    """Valuta contrastivity."""
    print(f"  Computing contrastivity for {model_key} + {explainer_name}")
    
    try:
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        texts, labels = get_test_data(sample_size)
        
        # Separa per classe
        pos_texts = [t for t, l in zip(texts, labels) if l == 1]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0]
        
        # Limita numero di testi per velocità
        pos_texts = pos_texts[:min(50, len(pos_texts))]
        neg_texts = neg_texts[:min(50, len(neg_texts))]
        
        # Genera attribution
        pos_attrs = []
        for text in pos_texts:
            try:
                attr = explainer(text)
                pos_attrs.append(attr)
            except Exception:
                continue
        
        neg_attrs = []
        for text in neg_texts:
            try:
                attr = explainer(text)
                neg_attrs.append(attr)
            except Exception:
                continue
        
        if len(pos_attrs) == 0 or len(neg_attrs) == 0:
            print("    Warning: No valid attributions generated")
            return 0.0
        
        score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
        print(f"    Contrastivity: {score:.4f}")
        return score
        
    except Exception as e:
        print(f"    Error: {e}")
        return float('nan')

def eval_consistency(model_key, explainer_name, sample_size, seeds=None):
    """Valuta consistency con inference seed approach."""
    global DEFAULT_CONSISTENCY_SEEDS
    if seeds is None:
        seeds = DEFAULT_CONSISTENCY_SEEDS
        
    print(f"  Computing consistency for {model_key} + {explainer_name}")
    
    try:
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        texts, _ = get_test_data(min(50, sample_size))  # Limita per consistency
        
        # Usa la nuova funzione di consistency
        score = metrics.evaluate_consistency_over_dataset(
            model=model,
            tokenizer=tokenizer,
            explainer=explainer,
            texts=texts,
            seeds=seeds,
            show_progress=False
        )
        
        print(f"    Consistency: {score:.4f}")
        return score
        
    except Exception as e:
        print(f"    Error: {e}")
        return float('nan')

def build_robustness_table(sample_size):
    """Tabella robustness: explainer x modelli."""
    print("Calcolando robustness...")
    results = defaultdict(dict)
    
    # Filtra explainer disponibili
    available_explainers = [exp for exp in EXPLAINERS if exp in explainers.list_explainers()]
    print(f"Explainer disponibili: {available_explainers}")
    
    for explainer in available_explainers:
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
    
    # Filtra explainer disponibili
    available_explainers = [exp for exp in EXPLAINERS if exp in explainers.list_explainers()]
    print(f"Explainer disponibili: {available_explainers}")
    
    for explainer in available_explainers:
        for model_key in models.MODELS.keys():
            try:
                with Timer(f"{explainer} + {model_key}"):
                    score = eval_contrastivity(model_key, explainer, sample_size)
                results[explainer][model_key] = score
            except Exception as e:
                print(f"Errore {explainer}+{model_key}: {e}")
                results[explainer][model_key] = float('nan')
    
    return pd.DataFrame(results).T

def build_consistency_table(sample_size, seeds=None):
    """Tabella consistency: explainer x modelli (con inference seed)."""
    global DEFAULT_CONSISTENCY_SEEDS
    if seeds is None:
        seeds = DEFAULT_CONSISTENCY_SEEDS
        
    print("Calcolando consistency con inference seed...")
    print(f"Usando seeds: {seeds}")
    results = defaultdict(dict)
    
    # Filtra explainer disponibili
    available_explainers = [exp for exp in EXPLAINERS if exp in explainers.list_explainers()]
    print(f"Explainer disponibili: {available_explainers}")
    
    for explainer in available_explainers:
        for model_key in models.MODELS.keys():
            try:
                with Timer(f"{explainer} + {model_key}"):
                    score = eval_consistency(model_key, explainer, sample_size, seeds)
                results[explainer][model_key] = score
            except Exception as e:
                print(f"Errore {explainer}+{model_key}: {e}")
                results[explainer][model_key] = float('nan')
    
    return pd.DataFrame(results).T

def print_table_summary(df, metric_name):
    """Stampa riassunto della tabella."""
    print(f"\n=== {metric_name.upper()} SUMMARY ===")
    
    # Statistiche per explainer
    print("Per explainer:")
    for explainer in df.index:
        values = df.loc[explainer].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {explainer:>15}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Statistiche per modello
    print("Per modello:")
    for model in df.columns:
        values = df[model].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {model:>15}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Migliori combinazioni
    print("Top 3 combinazioni:")
    flat_data = []
    for explainer in df.index:
        for model in df.columns:
            value = df.loc[explainer, model]
            if not pd.isna(value):
                flat_data.append((explainer, model, value))
    
    # Ordina per metrica (robustness: più basso è meglio, altri: più alto è meglio)
    if metric_name == "robustness":
        flat_data.sort(key=lambda x: x[2])  # Ascendente
        print("  (Più basso = meglio)")
    else:
        flat_data.sort(key=lambda x: x[2], reverse=True)  # Discendente
        print("  (Più alto = meglio)")
    
    for i, (explainer, model, value) in enumerate(flat_data[:3]):
        print(f"  {i+1}. {explainer} + {model}: {value:.4f}")

def main():
    global DEFAULT_CONSISTENCY_SEEDS
    
    parser = argparse.ArgumentParser(description="Genera tabelle XAI")
    parser.add_argument("--metric", choices=METRICS + ["all"], default="all")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                       help="Seed per consistency (default: 42 123 456 789)")
    args = parser.parse_args()

    print(f"Generando tabelle con {args.sample} esempi...")
    
    # Aggiorna seed per consistency se forniti
    if args.seeds:
        DEFAULT_CONSISTENCY_SEEDS = args.seeds
        print(f"Seed per consistency aggiornati: {DEFAULT_CONSISTENCY_SEEDS}")
    else:
        print(f"Seed per consistency (default): {DEFAULT_CONSISTENCY_SEEDS}")
    
    # Controlla explainer disponibili
    available_explainers = explainers.list_explainers()
    configured_explainers = [exp for exp in EXPLAINERS if exp in available_explainers]
    missing_explainers = [exp for exp in EXPLAINERS if exp not in available_explainers]
    
    print(f"\nExplainer configurati: {EXPLAINERS}")
    print(f"Explainer disponibili: {configured_explainers}")
    if missing_explainers:
        print(f"Explainer non disponibili (dipendenze mancanti): {missing_explainers}")
    
    if not configured_explainers:
        print("ERRORE: Nessun explainer disponibile! Verificare installazione dipendenze.")
        return
    
    # Scegli metriche
    if args.metric == "all":
        metrics_to_run = METRICS
    else:
        metrics_to_run = [args.metric]

    # Genera tabelle
    for metric in metrics_to_run:
        print(f"\n{'='*60}")
        print(f"GENERANDO TABELLA: {metric.upper()}")
        print(f"{'='*60}")
        
        if metric == "robustness":
            df = build_robustness_table(args.sample)
        elif metric == "contrastivity":
            df = build_contrastivity_table(args.sample)
        elif metric == "consistency":
            df = build_consistency_table(args.sample, args.seeds)
        
        # Output
        if args.csv:
            filename = f"{metric}_table.csv"
            df.to_csv(filename)
            print(f"Salvato: {filename}")
        else:
            print("\nTabella:")
            print(df.to_markdown(floatfmt=".4f"))
        
        # Stampa riassunto
        print_table_summary(df, metric)

if __name__ == "__main__":
    main()