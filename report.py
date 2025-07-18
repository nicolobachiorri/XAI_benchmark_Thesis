"""
report.py – Report XAI OTTIMIZZATO PER MEMORIA
============================================================

STRATEGIA:
1. Carica 1 modello alla volta
2. Calcola TUTTE le metriche per TUTTI gli explainer per quel modello
3. Salva risultati intermedi in JSON/CSV
4. Pulisce memoria completamente
5. Passa al modello successivo
6. Assembla tabelle finali dai risultati salvati

VANTAGGI:
- Uso memoria costante (non cresce con numero modelli)
- Risultati intermedi salvati (recovery in caso di crash)
- Batch processing per efficienza
- Progress tracking dettagliato
- Tabelle finali complete

Uso:
    python report.py --sample 500
    python report.py --metric robustness --batch-size 32
    python report.py --resume  # Riprende da risultati esistenti
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

import models
import dataset
import explainers
import metrics
from utils import set_seed, Timer

# Configurazione
EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "contrastivity", "consistency"]
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]
DEFAULT_BATCH_SIZE = 16

# Directory per risultati intermedi
RESULTS_DIR = Path("results_intermediate")
RESULTS_DIR.mkdir(exist_ok=True)

set_seed(42)

def get_memory_usage():
    """Ottieni uso memoria GPU/CPU."""
    memory_info = {"cpu_gb": 0.0, "gpu_gb": 0.0}
    
    try:
        import psutil
        memory_info["cpu_gb"] = psutil.virtual_memory().used / (1024**3)
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        memory_info["gpu_gb"] = torch.cuda.memory_allocated() / (1024**3)
    
    return memory_info

def clear_memory():
    """Pulisce memoria aggressivamente."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def save_intermediate_results(model_key: str, results: Dict[str, Dict[str, float]]):
    """Salva risultati intermedi per un modello."""
    filename = RESULTS_DIR / f"results_{model_key}.json"
    
    # Converti NaN in null per JSON
    cleaned_results = {}
    for metric_name, explainer_results in results.items():
        cleaned_results[metric_name] = {}
        for explainer_name, score in explainer_results.items():
            if pd.isna(score):
                cleaned_results[metric_name][explainer_name] = None
            else:
                cleaned_results[metric_name][explainer_name] = float(score)
    
    with open(filename, 'w') as f:
        json.dump({
            "model_key": model_key,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": cleaned_results
        }, f, indent=2)
    
    print(f"  [SAVE] Risultati salvati: {filename}")

def load_intermediate_results(model_key: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Carica risultati intermedi se esistono."""
    filename = RESULTS_DIR / f"results_{model_key}.json"
    
    if not filename.exists():
        return None
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Converti null in NaN
        results = {}
        for metric_name, explainer_results in data["results"].items():
            results[metric_name] = {}
            for explainer_name, score in explainer_results.items():
                if score is None:
                    results[metric_name][explainer_name] = float('nan')
                else:
                    results[metric_name][explainer_name] = score
        
        print(f"  [LOAD] Risultati caricati: {filename}")
        return results
        
    except Exception as e:
        print(f"  [ERROR] Errore caricamento {filename}: {e}")
        return None

def get_test_data_cached(sample_size: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Carica dati di test con cache."""
    cache_file = RESULTS_DIR / f"test_data_{sample_size or 'all'}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"  [DATA] Caricati {len(data['texts'])} testi da cache")
            return data['texts'], data['labels']
        except Exception:
            pass
    
    # Genera dati
    print(f"  [DATA] Generando dati test (sample_size={sample_size})")
    texts = dataset.test_df["text"].tolist()
    labels = dataset.test_df["label"].tolist()
    
    if sample_size and sample_size < len(texts):
        import numpy as np
        np.random.seed(42)
        
        pos_indices = np.where(np.array(labels) == 1)[0]
        neg_indices = np.where(np.array(labels) == 0)[0]
        
        n_pos = min(sample_size // 2, len(pos_indices))
        n_neg = min(sample_size - n_pos, len(neg_indices))
        
        selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
        selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
        
        selected_indices = np.concatenate([selected_pos, selected_neg])
        np.random.shuffle(selected_indices)
        
        texts = [texts[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
    
    # Salva cache
    try:
        with open(cache_file, 'w') as f:
            json.dump({"texts": texts, "labels": labels}, f)
    except Exception:
        pass
    
    print(f"  [DATA] {len(texts)} testi ({sum(labels)} pos, {len(labels)-sum(labels)} neg)")
    return texts, labels

def process_texts_batch(texts: List[str], explainer, batch_size: int = DEFAULT_BATCH_SIZE) -> List:
    """Processa testi in batch ottimizzati."""
    results = []
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    for batch in tqdm(batches, desc="    Batch processing", leave=False):
        batch_results = []
        for text in batch:
            try:
                attr = explainer(text)
                batch_results.append(attr)
            except Exception as e:
                # Log error ma continua
                batch_results.append(None)
        results.extend(batch_results)
        
        # Mini cleanup ogni batch
        if len(results) % (batch_size * 4) == 0:
            gc.collect()
    
    return [r for r in results if r is not None]

def compute_all_metrics_for_model(
    model_key: str, 
    sample_size: int, 
    batch_size: int = DEFAULT_BATCH_SIZE,
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    metrics_to_compute: List[str] = METRICS
) -> Dict[str, Dict[str, float]]:
    """
    Computa TUTTE le metriche per TUTTI gli explainer per UN modello.
    Questa è la funzione chiave dell'ottimizzazione.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING MODEL: {model_key}")
    print(f"{'='*60}")
    
    # Controlla se esistono già risultati
    existing_results = load_intermediate_results(model_key)
    if existing_results:
        print(f"  [SKIP] Risultati già esistenti per {model_key}")
        return existing_results
    
    # Memory baseline
    mem_start = get_memory_usage()
    print(f"  [MEM] Start: CPU {mem_start['cpu_gb']:.1f}GB, GPU {mem_start['gpu_gb']:.1f}GB")
    
    try:
        # STEP 1: Carica modello UNA VOLTA
        print(f"  [LOAD] Caricando modello {model_key}...")
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        
        mem_after_model = get_memory_usage()
        print(f"  [MEM] After model: CPU {mem_after_model['cpu_gb']:.1f}GB, GPU {mem_after_model['gpu_gb']:.1f}GB")
        
        # STEP 2: Carica dati UNA VOLTA
        texts, labels = get_test_data_cached(sample_size)
        
        # Prepara dati per contrastivity
        pos_texts = [t for t, l in zip(texts, labels) if l == 1][:50]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0][:50]
        consistency_texts = texts[:min(50, len(texts))]
        
        # STEP 3: Per ogni explainer disponibile, computa tutte le metriche
        available_explainers = [exp for exp in EXPLAINERS if exp in explainers.list_explainers()]
        print(f"  [EXPLAINERS] Disponibili: {available_explainers}")
        
        results = {metric: {} for metric in metrics_to_compute}
        
        for explainer_name in available_explainers:
            print(f"\n  --- EXPLAINER: {explainer_name} ---")
            
            try:
                # Crea explainer
                explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                
                # ROBUSTNESS
                if "robustness" in metrics_to_compute:
                    print(f"    Computing robustness...")
                    start_time = time.time()
                    try:
                        score = metrics.evaluate_robustness_over_dataset(
                            model, tokenizer, explainer, texts, show_progress=False
                        )
                        results["robustness"][explainer_name] = score
                        print(f"    → Robustness: {score:.4f} ({time.time()-start_time:.1f}s)")
                    except Exception as e:
                        print(f"    → Robustness FAILED: {e}")
                        results["robustness"][explainer_name] = float('nan')
                
                # CONTRASTIVITY
                if "contrastivity" in metrics_to_compute:
                    print(f"    Computing contrastivity...")
                    start_time = time.time()
                    try:
                        # Genera attribution in batch
                        pos_attrs = process_texts_batch(pos_texts, explainer, batch_size)
                        neg_attrs = process_texts_batch(neg_texts, explainer, batch_size)
                        
                        if len(pos_attrs) > 0 and len(neg_attrs) > 0:
                            score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
                            results["contrastivity"][explainer_name] = score
                            print(f"    → Contrastivity: {score:.4f} ({time.time()-start_time:.1f}s)")
                        else:
                            print(f"    → Contrastivity: No valid attributions")
                            results["contrastivity"][explainer_name] = 0.0
                    except Exception as e:
                        print(f"    → Contrastivity FAILED: {e}")
                        results["contrastivity"][explainer_name] = float('nan')
                
                # CONSISTENCY
                if "consistency" in metrics_to_compute:
                    print(f"    Computing consistency...")
                    start_time = time.time()
                    try:
                        score = metrics.evaluate_consistency_over_dataset(
                            model=model,
                            tokenizer=tokenizer,
                            explainer=explainer,
                            texts=consistency_texts,
                            seeds=seeds,
                            show_progress=False
                        )
                        results["consistency"][explainer_name] = score
                        print(f"    → Consistency: {score:.4f} ({time.time()-start_time:.1f}s)")
                    except Exception as e:
                        print(f"    → Consistency FAILED: {e}")
                        results["consistency"][explainer_name] = float('nan')
                
                # Cleanup explainer
                del explainer
                gc.collect()
                
            except Exception as e:
                print(f"    → EXPLAINER FAILED: {e}")
                for metric in metrics_to_compute:
                    results[metric][explainer_name] = float('nan')
        
        # STEP 4: Salva risultati intermedi
        save_intermediate_results(model_key, results)
        
        return results
        
    except Exception as e:
        print(f"  [ERROR] Modello {model_key} fallito: {e}")
        # Ritorna risultati vuoti ma non blocca il processo
        return {metric: {} for metric in metrics_to_compute}
        
    finally:
        # STEP 5: CLEANUP COMPLETO
        print(f"  [CLEANUP] Pulendo memoria per {model_key}...")
        
        # Elimina variabili esplicitamente
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'explainer' in locals():
            del explainer
        if 'texts' in locals():
            del texts, labels, pos_texts, neg_texts, consistency_texts
        
        # Cleanup aggressivo
        clear_memory()
        
        mem_end = get_memory_usage()
        print(f"  [MEM] End: CPU {mem_end['cpu_gb']:.1f}GB, GPU {mem_end['gpu_gb']:.1f}GB")
        print(f"  [CLEANUP] Completato per {model_key}")

def build_final_tables(metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]:
    """Assembla tabelle finali dai risultati intermedi."""
    print(f"\n{'='*60}")
    print("BUILDING FINAL TABLES")
    print(f"{'='*60}")
    
    # Carica tutti i risultati intermedi
    all_results = {}
    for model_key in models.MODELS.keys():
        results = load_intermediate_results(model_key)
        if results:
            all_results[model_key] = results
            print(f"  [LOAD] {model_key}: OK")
        else:
            print(f"  [LOAD] {model_key}: MISSING")
    
    if not all_results:
        print("  [ERROR] Nessun risultato intermedio trovato!")
        return {}
    
    # Assembla tabelle per metrica
    tables = {}
    
    for metric in metrics_to_compute:
        print(f"\n  Building table for {metric}...")
        
        # Struttura: explainer_name -> {model_key: score}
        metric_data = defaultdict(dict)
        
        for model_key, model_results in all_results.items():
            if metric in model_results:
                for explainer_name, score in model_results[metric].items():
                    metric_data[explainer_name][model_key] = score
        
        # Converti in DataFrame
        if metric_data:
            df = pd.DataFrame(metric_data).T  # Transpose per avere explainer come righe
            tables[metric] = df
            print(f"    → {metric}: {df.shape[0]} explainers x {df.shape[1]} models")
        else:
            print(f"    → {metric}: NO DATA")
    
    return tables

def print_table_summary(df: pd.DataFrame, metric_name: str):
    """Stampa riassunto tabella."""
    print(f"\n=== {metric_name.upper()} SUMMARY ===")
    
    if df.empty:
        print("  No data available")
        return
    
    # Statistiche per explainer
    print("Per explainer:")
    for explainer in df.index:
        values = df.loc[explainer].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {explainer:>15}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Top 3 combinazioni
    print("Top 3 combinazioni:")
    flat_data = []
    for explainer in df.index:
        for model in df.columns:
            value = df.loc[explainer, model]
            if not pd.isna(value):
                flat_data.append((explainer, model, value))
    
    if flat_data:
        # Ordina per metrica
        if metric_name == "robustness":
            flat_data.sort(key=lambda x: x[2])  # Più basso è meglio
            print("  (Più basso = meglio)")
        else:
            flat_data.sort(key=lambda x: x[2], reverse=True)  # Più alto è meglio
            print("  (Più alto = meglio)")
        
        for i, (explainer, model, value) in enumerate(flat_data[:3]):
            print(f"  {i+1}. {explainer} + {model}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Report XAI Memory-Optimized")
    parser.add_argument("--metric", choices=METRICS + ["all"], default="all")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    parser.add_argument("--resume", action="store_true", help="Riprendi da risultati esistenti")
    parser.add_argument("--clear-cache", action="store_true", help="Cancella risultati intermedi")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_CONSISTENCY_SEEDS)
    parser.add_argument("--models", nargs="+", choices=list(models.MODELS.keys()), 
                       default=list(models.MODELS.keys()), help="Modelli da processare")
    
    args = parser.parse_args()
    
    print(f"MEMORY-OPTIMIZED XAI REPORT")
    print(f"Sample size: {args.sample}")
    print(f"Batch size: {args.batch_size}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    
    # Clear cache se richiesto
    if args.clear_cache:
        import shutil
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
            RESULTS_DIR.mkdir(exist_ok=True)
        print("Cache cancellata")
    
    # Scegli metriche
    if args.metric == "all":
        metrics_to_compute = METRICS
    else:
        metrics_to_compute = [args.metric]
    
    print(f"Metriche: {metrics_to_compute}")
    
    # FASE 1: Processa ogni modello sequenzialmente
    print(f"\n{'='*60}")
    print("FASE 1: PROCESSING MODELS")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for i, model_key in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Model: {model_key}")
        
        if args.resume:
            existing = load_intermediate_results(model_key)
            if existing:
                print(f"  [SKIP] Già processato")
                continue
        
        try:
            with Timer(f"Model {model_key}"):
                compute_all_metrics_for_model(
                    model_key=model_key,
                    sample_size=args.sample,
                    batch_size=args.batch_size,
                    seeds=args.seeds,
                    metrics_to_compute=metrics_to_compute
                )
        except Exception as e:
            print(f"  [ERROR] Model {model_key} failed: {e}")
            continue
    
    # FASE 2: Assembla tabelle finali
    print(f"\n{'='*60}")
    print("FASE 2: BUILDING FINAL TABLES")
    print(f"{'='*60}")
    
    tables = build_final_tables(metrics_to_compute)
    
    # FASE 3: Output risultati
    print(f"\n{'='*60}")
    print("FASE 3: OUTPUT RESULTS")
    print(f"{'='*60}")
    
    for metric_name, df in tables.items():
        print(f"\n--- {metric_name.upper()} ---")
        
        if args.csv:
            filename = f"{metric_name}_table.csv"
            df.to_csv(filename)
            print(f"Salvato: {filename}")
        else:
            print(df.to_markdown(floatfmt=".4f"))
        
        print_table_summary(df, metric_name)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETATO in {total_time/60:.1f} minuti")
    print(f"Risultati intermedi in: {RESULTS_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()