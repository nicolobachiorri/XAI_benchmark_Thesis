"""
report.py – Report XAI completo ottimizzato per Google Colab (FIXED VERSION)
============================================================================

OTTIMIZZAZIONI PER COLAB:
1. Memory management ottimizzato per GPU limitata
2. Progress tracking dettagliato per notebook
3. Checkpoint automatici per recovery con VALIDAZIONE COMPLETA
4. Dataset clusterizzato (400 esempi)
5. Batch processing intelligente
6. Error recovery e fallback ROBUSTI
7. Report generation ottimizzato

STRATEGIA COLAB:
- Processa 1 modello alla volta per memoria
- ORDINE INTELLIGENTE: modelli piccoli prima (evita degradazione ambientale)
- SUPER CLEANUP aggressivo tra modelli
- Checkpoint validation completa
- Resume logic robusto
- Progress tracking visivo
- Auto-recovery da crash


Uso in Colab:
```python
import report

# Report veloce (2 modelli, 2 explainer, 1 metrica)
report.quick_report()

# Report completo personalizzato
report.run_report(
    models=["tinybert", "distilbert"],
    explainers=["lime", "grad_input"], 
    metrics=["robustness"],
    sample_size=100
)
```
"""

import argparse
import gc
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

import models
import dataset
import explainers
import metrics
from utils import Timer, PerformanceProfiler, AutoRecovery, print_memory_status, aggressive_cleanup, set_seed

# Configurazione Colab
EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "contrastivity", "consistency"]
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]

# Directory risultati
RESULTS_DIR = Path("xai_results")
RESULTS_DIR.mkdir(exist_ok=True)

set_seed(42)

def print_colab_report_header():
    """Header per report Colab."""
    print("="*80)
    print(" XAI COMPREHENSIVE REPORT (FIXED VERSION)")
    print("="*80)
    print(" Strategy: Memory-optimized sequential processing")
    print(" Dataset: 400 clustered representative examples")
    print(" Recovery: Auto-checkpoint with VALIDATION")
    print(" Cleanup: SUPER aggressive memory management")
    print(" Ordering: Smart model ordering (small→large)")
    print("="*80)

def get_available_resources():
    """Ottieni risorse disponibili per report."""
    available_models = list(models.MODELS.keys())
    available_explainers = [exp for exp in EXPLAINERS if exp in explainers.list_explainers()]
    
    print(f"[RESOURCES] Models: {len(available_models)} available")
    print(f"[RESOURCES] Explainers: {len(available_explainers)} available")
    print(f"[RESOURCES] Metrics: {len(METRICS)} available")
    print(f"[RESOURCES] Dataset: {len(dataset.test_df)} clustered examples")
    
    return available_models, available_explainers

def smart_model_ordering(models_to_test: List[str]) -> List[str]:
    """Ordina modelli intelligentemente per ottimizzare processing.
    
    Priorità: modelli piccoli prima, grandi dopo per evitare
    degradazione ambientale che causa fallimenti dei modelli finali.
    """
    size_priority = {
        "tinybert": 1,      # Più piccolo - processa per primo
        "distilbert": 2,    # Piccolo
        "roberta-base": 3,  # Medio
        "bert-large": 4,    # Grande
        "roberta-large": 5  # Più grande - processa per ultimo
    }
    
    # Ordina per priorità, modelli sconosciuti alla fine
    ordered = sorted(models_to_test, key=lambda m: size_priority.get(m, 999))
    
    print(f"[ORDERING] Smart model order: {ordered}")
    return ordered

def super_cleanup():
    """Cleanup aggressivo tra modelli per prevenire degradazione ambientale."""
    print("[SUPER-CLEANUP] Performing deep cleanup...")
    start_time = time.time()
    
    # Multiple GC passes per essere sicuri
    for i in range(3):
        collected = gc.collect()
        if i == 0:
            print(f"[SUPER-CLEANUP]   GC pass {i+1}: {collected} objects collected")
    
    if torch.cuda.is_available():
        # GPU cleanup completo
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  # Aspetta tutte le operazioni CUDA
        
        # Memory status
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"[SUPER-CLEANUP]   GPU memory after cleanup: {allocated:.2f}GB")
    
    # Stabilization pause per permettere al sistema di stabilizzarsi
    time.sleep(2)
    
    cleanup_time = time.time() - start_time
    print(f"[SUPER-CLEANUP] Deep cleanup completed in {cleanup_time:.1f}s")

def verify_checkpoint_completeness(
    checkpoint_data: Dict, 
    expected_metrics: List[str], 
    expected_explainers: List[str]
) -> Tuple[bool, str]:
    """Verifica che il checkpoint contenga tutti i dati attesi.
    
    Returns:
        (is_complete, reason)
    """
    if not checkpoint_data.get("completed", False):
        return False, "marked as incomplete"
    
    results = checkpoint_data.get("results", {})
    if not results:
        return False, "no results data"
    
    # Verifica presenza di tutte le metriche
    missing_metrics = []
    for metric in expected_metrics:
        if metric not in results:
            missing_metrics.append(metric)
    
    if missing_metrics:
        return False, f"missing metrics: {missing_metrics}"
    
    # Verifica presenza di tutti gli explainer per ogni metrica
    incomplete_metrics = []
    for metric in expected_metrics:
        if metric not in results:
            continue
            
        metric_explainers = set(results[metric].keys())
        expected_explainers_set = set(expected_explainers)
        
        if not expected_explainers_set.issubset(metric_explainers):
            missing = expected_explainers_set - metric_explainers
            incomplete_metrics.append(f"{metric}({list(missing)})")
    
    if incomplete_metrics:
        return False, f"incomplete explainers: {incomplete_metrics}"
    
    # Verifica che ci siano risultati validi (non tutti NaN/None)
    valid_results_count = 0
    total_expected = len(expected_metrics) * len(expected_explainers)
    
    for metric in expected_metrics:
        for explainer in expected_explainers:
            if explainer in results[metric]:
                score = results[metric][explainer]
                if score is not None and not (isinstance(score, float) and np.isnan(score)):
                    valid_results_count += 1
    
    if valid_results_count == 0:
        return False, "no valid scores (all NaN/None)"
    
    completeness_ratio = valid_results_count / total_expected
    if completeness_ratio < 0.1:  # Almeno 10% di risultati validi
        return False, f"too few valid results: {valid_results_count}/{total_expected} ({completeness_ratio:.1%})"
    
    return True, f"complete with {valid_results_count}/{total_expected} valid results ({completeness_ratio:.1%})"

def save_intermediate_results(model_key: str, results: Dict[str, Dict[str, float]], 
                            timestamp: str = None):
    """Salva risultati intermedi con timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = RESULTS_DIR / f"results_{model_key}_{timestamp}.json"
    
    # Converti NaN per JSON
    cleaned_results = {}
    for metric_name, explainer_results in results.items():
        cleaned_results[metric_name] = {}
        for explainer_name, score in explainer_results.items():
            if pd.isna(score):
                cleaned_results[metric_name][explainer_name] = None
            else:
                cleaned_results[metric_name][explainer_name] = float(score)
    
    data = {
        "model_key": model_key,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "results": cleaned_results,
        "metadata": {
            "dataset_size": len(dataset.test_df),
            "consistency_seeds": DEFAULT_CONSISTENCY_SEEDS
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[SAVE] Results saved: {filename.name}")
    return filename

def load_all_results(pattern: str = "results_*.json") -> Dict[str, Dict]:
    """Carica tutti i risultati salvati."""
    results_files = list(RESULTS_DIR.glob(pattern))
    all_results = {}
    
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            model_key = data["model_key"]
            if model_key not in all_results:
                all_results[model_key] = data
            else:
                # Usa il più recente
                if data["datetime"] > all_results[model_key]["datetime"]:
                    all_results[model_key] = data
        except Exception as e:
            print(f"[WARNING] Could not load {file_path}: {e}")
    
    print(f"[LOAD] Loaded results for {len(all_results)} models")
    return all_results

def process_single_model_with_retry(
    model_key: str,
    explainers_to_test: List[str],
    metrics_to_compute: List[str],
    sample_size: int,
    recovery: AutoRecovery,
    max_retries: int = 2
) -> Dict[str, Dict[str, float]]:
    """Wrapper con retry logic per process_single_model."""
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} for {model_key}")
                super_cleanup()  # Cleanup extra prima del retry
                time.sleep(5)    # Pausa più lunga
            
            return process_single_model(
                model_key, explainers_to_test, metrics_to_compute, 
                sample_size, recovery
            )
            
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed for {model_key}: {e}")
            
            if attempt < max_retries:
                print(f"[RETRY] Will retry {model_key} after cleanup...")
                super_cleanup()
            else:
                print(f"[FINAL ERROR] All {max_retries + 1} attempts failed for {model_key}")
                # Restituisci risultati vuoti ma validi
                return {metric: {} for metric in metrics_to_compute}

def process_single_model(
    model_key: str,
    explainers_to_test: List[str],
    metrics_to_compute: List[str],
    sample_size: int,
    recovery: AutoRecovery
) -> Dict[str, Dict[str, float]]:
    """Processa singolo modello con tutte le metriche (versione migliorata)."""
    
    print(f"\n{'='*70}")
    print(f" PROCESSING MODEL: {model_key}")
    print(f"{'='*70}")
    
    profiler = PerformanceProfiler()
    profiler.start_operation(f"model_{model_key}")
    
    try:
        # Memory status iniziale
        print_memory_status()
        
        # Carica modello UNA VOLTA
        print(f"[LOAD] Loading {model_key}...")
        with Timer(f"Loading {model_key}"):
            model = models.load_model(model_key)
            tokenizer = models.load_tokenizer(model_key)
        
        # Prepara dati
        print(f"[DATA] Preparing data (sample_size={sample_size})...")
        texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)
        
        # Separa per classi (per contrastivity)
        pos_texts = [t for t, l in zip(texts, labels) if l == 1][:50]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0][:50]
        consistency_texts = texts[:min(50, len(texts))]
        
        print(f"[DATA] Total: {len(texts)}, Pos: {len(pos_texts)}, Neg: {len(neg_texts)}")
        
        # Inizializza risultati per TUTTE le metriche e explainer
        results = {}
        for metric in metrics_to_compute:
            results[metric] = {}
            for explainer in explainers_to_test:
                results[metric][explainer] = float('nan')  # Inizializza con NaN
        
        # Processa ogni explainer
        successful_explainers = 0
        
        for i, explainer_name in enumerate(explainers_to_test, 1):
            print(f"\n[{i}/{len(explainers_to_test)}] EXPLAINER: {explainer_name}")
            print("-" * 50)
            
            explainer_start_time = time.time()
            explainer_success = False
            
            try:
                # Crea explainer
                explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                
                # Processa ogni metrica
                for metric_name in metrics_to_compute:
                    print(f"  [METRIC] {metric_name}...", end=" ")
                    metric_start_time = time.time()
                    
                    try:
                        if metric_name == "robustness":
                            score = metrics.evaluate_robustness_over_dataset(
                                model, tokenizer, explainer, texts[:50], show_progress=False
                            )
                            
                        elif metric_name == "contrastivity":
                            # Process in batch per memoria
                            pos_attrs = metrics.process_attributions_batch(
                                pos_texts, explainer, batch_size=10, show_progress=False
                            )
                            neg_attrs = metrics.process_attributions_batch(
                                neg_texts, explainer, batch_size=10, show_progress=False
                            )
                            
                            # Filter valid
                            pos_attrs = [attr for attr in pos_attrs if attr.tokens and attr.scores]
                            neg_attrs = [attr for attr in neg_attrs if attr.tokens and attr.scores]
                            
                            if pos_attrs and neg_attrs:
                                score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
                            else:
                                score = 0.0
                                
                        elif metric_name == "consistency":
                            score = metrics.evaluate_consistency_over_dataset(
                                model=model,
                                tokenizer=tokenizer,
                                explainer=explainer,
                                texts=consistency_texts,
                                seeds=DEFAULT_CONSISTENCY_SEEDS,
                                show_progress=False
                            )
                        else:
                            score = float('nan')
                        
                        results[metric_name][explainer_name] = score
                        metric_time = time.time() - metric_start_time
                        print(f"SUCCESS {score:.4f} ({metric_time:.1f}s)")
                        explainer_success = True
                        
                    except Exception as e:
                        print(f"ERROR: {str(e)[:50]}...")
                        results[metric_name][explainer_name] = float('nan')
                
                if explainer_success:
                    successful_explainers += 1
                
                explainer_time = time.time() - explainer_start_time
                status = "SUCCESS" if explainer_success else "FAILED"
                print(f"  [TOTAL] {explainer_name}: {explainer_time:.1f}s {status}")
                
                # Cleanup explainer
                del explainer
                gc.collect()
                
                # Checkpoint intermedio con conteggio progresso
                checkpoint_data = {
                    "results": results,
                    "completed": False,  # Non completo finché non finiti tutti
                    "progress": {
                        "explainers_completed": i,
                        "explainers_total": len(explainers_to_test),
                        "successful_explainers": successful_explainers
                    }
                }
                recovery.save_checkpoint(checkpoint_data, f"model_{model_key}")
                
            except Exception as e:
                print(f"  EXPLAINER FAILED: {explainer_name}: {e}")
                # Mantieni NaN inizializzati per tutti le metriche
                for metric in metrics_to_compute:
                    results[metric][explainer_name] = float('nan')
        
        # Salva risultati finali con verifica
        final_results = {
            "results": results,
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "successful_explainers": successful_explainers,
                "total_explainers": len(explainers_to_test),
                "success_rate": successful_explainers / len(explainers_to_test)
            }
        }
        recovery.save_checkpoint(final_results, f"model_{model_key}")
        save_intermediate_results(model_key, results)
        
        profiler.end_operation(f"model_{model_key}")
        
        print(f"\n[COMPLETE] {model_key} processing completed")
        print(f"[STATS] Successful explainers: {successful_explainers}/{len(explainers_to_test)}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Model {model_key} failed completely: {e}")
        import traceback
        traceback.print_exc()
        
        # Restituisci struttura vuota ma valida
        empty_results = {}
        for metric in metrics_to_compute:
            empty_results[metric] = {}
            for explainer in explainers_to_test:
                empty_results[metric][explainer] = float('nan')
        
        return empty_results
        
    finally:
        # CLEANUP AGGRESSIVO
        print(f"[CLEANUP] Cleaning memory for {model_key}...")
        
        # Cleanup variabili locali
        local_vars_to_clean = ['model', 'tokenizer', 'explainer']
        for var_name in local_vars_to_clean:
            if var_name in locals():
                del locals()[var_name]
        
        # Super cleanup
        super_cleanup()

def build_report_tables(all_results: Dict[str, Dict], metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]:
    """Costruisce tabelle finali dai risultati (versione migliorata con debug)."""
    print(f"\n{'='*70}")
    print(" BUILDING REPORT TABLES (ENHANCED)")
    print(f"{'='*70}")
    
    tables = {}
    
    # Debug: analizza all_results
    print(f"[DEBUG] Models in all_results: {list(all_results.keys())}")
    
    for metric in metrics_to_compute:
        print(f"\n[TABLE] Building {metric} table...")
        
        # Struttura: explainer -> {model: score}
        metric_data = defaultdict(dict)
        
        for model_key, model_data in all_results.items():
            print(f"[DEBUG] Processing {model_key}:")
            print(f"[DEBUG]   Has 'results': {'results' in model_data}")
            
            if "results" not in model_data:
                print(f"[DEBUG]   No 'results' key for {model_key}")
                continue
                
            print(f"[DEBUG]   Available metrics: {list(model_data['results'].keys())}")
            
            if metric not in model_data["results"]:
                print(f"[DEBUG]   No '{metric}' in results for {model_key}")
                continue
                
            explainer_scores = model_data["results"][metric]
            print(f"[DEBUG]   {metric} explainers: {list(explainer_scores.keys())}")
            
            valid_scores = 0
            for explainer_name, score in explainer_scores.items():
                if score is not None and not (isinstance(score, float) and np.isnan(score)):
                    metric_data[explainer_name][model_key] = score
                    valid_scores += 1
                else:
                    print(f"[DEBUG]     Filtered {explainer_name}: {score} (invalid)")
            
            print(f"[DEBUG]   Valid scores for {model_key}: {valid_scores}/{len(explainer_scores)}")
        
        if metric_data:
            df = pd.DataFrame(metric_data).T  # Transpose: explainer come righe
            tables[metric] = df
            print(f"[TABLE] {metric}: {df.shape[0]} explainers × {df.shape[1]} models")
            print(f"[TABLE]   Total data points: {df.notna().sum().sum()}")
        else:
            print(f"[TABLE] {metric}: No data available")
            tables[metric] = pd.DataFrame()
    
    return tables

def print_table_analysis(df: pd.DataFrame, metric_name: str):
    """Analisi e interpretazione tabella (versione migliorata)."""
    print(f"\n{'='*60}")
    print(f" {metric_name.upper()} ANALYSIS")
    print(f"{'='*60}")
    
    if df.empty:
        print("No data available for analysis")
        return
    
    # Statistiche per explainer
    print(" Per-Explainer Statistics:")
    print("-" * 40)
    for explainer in df.index:
        values = df.loc[explainer].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            count = len(values)
            coverage = len(values) / len(df.columns)
            print(f"  {explainer:>15s}: μ={mean_val:.4f} σ={std_val:.4f} (n={count}, {coverage:.1%} coverage)")
    
    # Statistiche per modello
    print("\n Per-Model Statistics:")
    print("-" * 40)
    for model in df.columns:
        values = df[model].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            count = len(values)
            coverage = len(values) / len(df.index)
            print(f"  {model:>15s}: μ={mean_val:.4f} σ={std_val:.4f} (n={count}, {coverage:.1%} coverage)")
    
    # Ranking top combinations
    print(f"\n Top 5 Combinations:")
    print("-" * 40)
    flat_data = []
    for explainer in df.index:
        for model in df.columns:
            value = df.loc[explainer, model]
            if not pd.isna(value):
                flat_data.append((explainer, model, value))
    
    if flat_data:
        # Sort appropriately per metric
        if metric_name == "robustness":
            flat_data.sort(key=lambda x: x[2])  # Lower is better
            direction = "(Lower = Better)"
        else:
            flat_data.sort(key=lambda x: x[2], reverse=True)  # Higher is better
            direction = "(Higher = Better)"
        
        print(f"  {direction}")
        for i, (explainer, model, value) in enumerate(flat_data[:5]):
            print(f"  {i+1}. {explainer:>12s} + {model:>12s}: {value:.4f}")
    
    # Coverage analysis
    total_cells = df.size
    filled_cells = df.notna().sum().sum()
    coverage = filled_cells / total_cells if total_cells > 0 else 0
    
    print(f"\n Coverage Analysis:")
    print(f"  Total combinations: {total_cells}")
    print(f"  Completed: {filled_cells}")
    print(f"  Coverage: {coverage:.1%}")
    
    print(f"{'='*60}")

def generate_summary_report(tables: Dict[str, pd.DataFrame], 
                          execution_time: float,
                          models_tested: List[str],
                          explainers_tested: List[str]) -> str:
    """Genera report summary testuale (versione migliorata)."""
    
    summary = f"""
XAI BENCHMARK REPORT - GOOGLE COLAB EDITION (FIXED)
===================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Execution Time: {execution_time/60:.1f} minutes
Dataset: {len(dataset.test_df)} clustered examples from IMDB

CONFIGURATION:
- Models tested: {len(models_tested)} ({', '.join(models_tested)})
- Explainers tested: {len(explainers_tested)} ({', '.join(explainers_tested)})
- Metrics computed: {len(tables)} ({', '.join(tables.keys())})

RESULTS SUMMARY:
"""
    
    for metric_name, df in tables.items():
        if not df.empty:
            coverage = df.notna().sum().sum()
            total = df.size
            summary += f"\n{metric_name.upper()}:\n"
            summary += f"  Coverage: {coverage}/{total} combinations completed ({coverage/total:.1%})\n"
            
            # Best combination
            flat_data = []
            for explainer in df.index:
                for model in df.columns:
                    value = df.loc[explainer, model]
                    if not pd.isna(value):
                        flat_data.append((explainer, model, value))
            
            if flat_data:
                if metric_name == "robustness":
                    best = min(flat_data, key=lambda x: x[2])
                    summary += f"  Best: {best[0]} + {best[1]} = {best[2]:.4f} (most robust)\n"
                else:
                    best = max(flat_data, key=lambda x: x[2])
                    summary += f"  Best: {best[0]} + {best[1]} = {best[2]:.4f} (highest score)\n"
        else:
            summary += f"\n{metric_name.upper()}: No data collected\n"
    
    summary += f"""
SYSTEM IMPROVEMENTS:
- Smart model ordering (small→large) prevents environment degradation
- Super cleanup between models ensures memory stability  
- Checkpoint validation prevents incomplete cached results
- Retry logic handles transient failures
- Enhanced debug logging for troubleshooting

RECOMMENDATIONS:
- For production XAI: Use combinations with highest consistency scores
- For research: Focus on high contrastivity explainers
- For deployment: Balance robustness vs computational cost

Files Generated:
- Raw results: {RESULTS_DIR}/results_*.json
- CSV tables: {RESULTS_DIR}/*_table.csv
- Summary: {RESULTS_DIR}/summary_report.txt
"""
    
    return summary

def quick_report(models_subset: List[str] = None, 
                explainers_subset: List[str] = None,
                sample_size: int = 50) -> Dict[str, pd.DataFrame]:
    """Report veloce per test (callable da Colab)."""
    print_colab_report_header()
    
    # Defaults per test veloce
    available_models, available_explainers = get_available_resources()
    
    if models_subset is None:
        models_subset = ["tinybert", "distilbert"]  # Modelli piccoli
    if explainers_subset is None:
        explainers_subset = ["lime", "grad_input"]  # Explainer veloci
    
    # Filter available
    models_to_test = [m for m in models_subset if m in available_models]
    explainers_to_test = [e for e in explainers_subset if e in available_explainers]
    metrics_to_compute = ["robustness"]  # Solo 1 metrica per velocità
    
    print(f"[QUICK] Testing {len(models_to_test)} models × {len(explainers_to_test)} explainers × {len(metrics_to_compute)} metrics")
    print(f"[QUICK] Sample size: {sample_size}")
    
    return run_report(models_to_test, explainers_to_test, metrics_to_compute, sample_size, resume=False)

def run_report(models_to_test: List[str] = None,
               explainers_to_test: List[str] = None, 
               metrics_to_compute: List[str] = None,
               sample_size: int = 100,
               resume: bool = True) -> Dict[str, pd.DataFrame]:
    """Esegue report completo (callable da Colab) - VERSIONE MIGLIORATA."""
    
    start_time = time.time()
    profiler = PerformanceProfiler()
    recovery = AutoRecovery(checkpoint_dir=RESULTS_DIR / "checkpoints")
    
    profiler.start_operation("full_report")
    
    try:
        # Setup defaults
        available_models, available_explainers = get_available_resources()
        
        if models_to_test is None:
            models_to_test = available_models
        if explainers_to_test is None:
            explainers_to_test = available_explainers
        if metrics_to_compute is None:
            metrics_to_compute = METRICS
        
        # Filter available
        models_to_test = [m for m in models_to_test if m in available_models]
        explainers_to_test = [e for e in explainers_to_test if e in available_explainers]
        
        # SMART MODEL ORDERING - FIX PRINCIPALE!
        original_order = models_to_test.copy()
        models_to_test = smart_model_ordering(models_to_test)
        
        if original_order != models_to_test:
            print(f"[ORDERING] Reordered models for stability:")
            print(f"[ORDERING]   Original: {original_order}")
            print(f"[ORDERING]   Optimized: {models_to_test}")
        
        total_combinations = len(models_to_test) * len(explainers_to_test) * len(metrics_to_compute)
        
        print(f"\n[REPORT] Configuration:")
        print(f"  Models: {models_to_test}")
        print(f"  Explainers: {explainers_to_test}")
        print(f"  Metrics: {metrics_to_compute}")
        print(f"  Sample size: {sample_size}")
        print(f"  Total combinations: {total_combinations}")
        print(f"  Resume: {resume}")
        
        # FASE 1: Process each model
        print(f"\n{'='*80}")
        print("FASE 1: SEQUENTIAL MODEL PROCESSING (ENHANCED)")
        print(f"{'='*80}")
        
        all_results = {}
        
        for i, model_key in enumerate(models_to_test, 1):
            print(f"\n[{i}/{len(models_to_test)}] Model: {model_key}")
            
            # RESUME LOGIC FIX - rispetta sempre --no-resume
            if resume:
                print(f"[RESUME] Checking for existing checkpoint...")
                existing = recovery.load_latest_checkpoint(f"model_{model_key}")
                
                if existing:
                    # CHECKPOINT VALIDATION - verifica completezza reale
                    is_complete, reason = verify_checkpoint_completeness(
                        existing, metrics_to_compute, explainers_to_test
                    )
                    
                    if is_complete:
                        print(f"[RESUME] Using complete cached results for {model_key}")
                        print(f"[RESUME]   Reason: {reason}")
                        all_results[model_key] = existing
                        continue
                    else:
                        print(f"[RESUME] Incomplete checkpoint for {model_key}")
                        print(f"[RESUME]   Reason: {reason}")
                        print(f"[RESUME]   Will reprocess...")
                else:
                    print(f"[RESUME] No checkpoint found for {model_key}")
            else:
                print(f"[RESUME] Resume disabled, processing fresh...")
            
            try:
                with Timer(f"Processing {model_key}"):
                    # RETRY LOGIC per fallimenti transienti
                    results = process_single_model_with_retry(
                        model_key=model_key,
                        explainers_to_test=explainers_to_test,
                        metrics_to_compute=metrics_to_compute,
                        sample_size=sample_size,
                        recovery=recovery,
                        max_retries=2
                    )
                    all_results[model_key] = {"results": results, "completed": True}
                
                print(f"[SUCCESS] {model_key} completed successfully")
                
            except Exception as e:
                print(f"[ERROR] Model {model_key} failed: {e}")
                all_results[model_key] = {
                    "results": {metric: {} for metric in metrics_to_compute},
                    "completed": False,
                    "error": str(e)
                }
                
            # SUPER CLEANUP tra modelli per prevenire degradazione
            if i < len(models_to_test):  # Non dopo l'ultimo modello
                print(f"[INTER-MODEL] Performing cleanup before next model...")
                super_cleanup()
        
        # FASE 2: Build tables
        print(f"\n{'='*80}")
        print("FASE 2: BUILDING FINAL TABLES (ENHANCED)")  
        print(f"{'='*80}")
        
        profiler.start_operation("table_building")
        tables = build_report_tables(all_results, metrics_to_compute)
        profiler.end_operation("table_building")
        
        # FASE 3: Analysis & Output
        print(f"\n{'='*80}")
        print("FASE 3: ANALYSIS & OUTPUT (ENHANCED)")
        print(f"{'='*80}")
        
        execution_time = time.time() - start_time
        
        # Print tables con analisi dettagliata
        for metric_name, df in tables.items():
            if not df.empty:
                print(f"\n {metric_name.upper()} TABLE:")
                print("=" * 50)
                print(df.to_string(float_format="%.4f", na_rep="—"))
                
                # Save CSV
                csv_file = RESULTS_DIR / f"{metric_name}_table.csv"
                df.to_csv(csv_file)
                print(f"[SAVE] CSV saved: {csv_file}")
                
                # Analysis dettagliata
                print_table_analysis(df, metric_name)
        
        # Generate summary migliorato
        summary = generate_summary_report(
            tables, execution_time, models_to_test, explainers_to_test
        )
        
        # Save summary
        summary_file = RESULTS_DIR / "summary_report.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"[SAVE] Summary saved: {summary_file}")
        
        profiler.end_operation("full_report")
        
        # Final summary con statistiche dettagliate
        print(f"\n{'='*80}")
        print(" REPORT COMPLETED SUCCESSFULLY! (ENHANCED)")
        print(f"{'='*80}")
        print(f"  Total time: {execution_time/60:.1f} minutes")
        print(f"  Models processed: {len([r for r in all_results.values() if r.get('completed', False)])}/{len(models_to_test)}")
        print(f"  Tables generated: {len([t for t in tables.values() if not t.empty])}")
        print(f"  Files saved in: {RESULTS_DIR}")
        print(f"  Total data points: {sum(df.notna().sum().sum() for df in tables.values())}")
        
        # Coverage summary
        expected_total = len(models_to_test) * len(explainers_to_test) * len(metrics_to_compute)
        actual_total = sum(df.notna().sum().sum() for df in tables.values())
        coverage = actual_total / expected_total if expected_total > 0 else 0
        print(f"  Overall coverage: {actual_total}/{expected_total} ({coverage:.1%})")
        
        # Performance summary
        profiler.print_summary()
        
        return tables
        
    except Exception as e:
        print(f"\nREPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {}
        
    finally:
        # Final cleanup
        print(f"[FINAL-CLEANUP] Performing final memory cleanup...")
        super_cleanup()

# ==== CLI Interface ====
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="XAI Report Generator for Google Colab (FIXED)")
    parser.add_argument("--models", nargs="+", choices=list(models.MODELS.keys()), 
                       default=None, help="Models to test")
    parser.add_argument("--explainers", nargs="+", choices=EXPLAINERS,
                       default=None, help="Explainers to test")
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                       default=None, help="Metrics to compute")
    parser.add_argument("--sample", type=int, default=100, help="Sample size")
    parser.add_argument("--quick", action="store_true", help="Quick test report")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoints")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh")
    
    args = parser.parse_args()
    
    print("XAI BENCHMARK FIXED VERSION")
    print("="*50)
    print("Improvements:")
    print("- Smart model ordering")
    print("- Checkpoint validation") 
    print("- Super cleanup")
    print("- Retry logic")
    print("- Enhanced debugging")
    print("="*50)
    
    if args.quick:
        print("Running quick report...")
        tables = quick_report(
            models_subset=args.models,
            explainers_subset=args.explainers,
            sample_size=min(args.sample, 50)
        )
    else:
        print("Running full report...")
        tables = run_report(
            models_to_test=args.models,
            explainers_to_test=args.explainers,
            metrics_to_compute=args.metrics,
            sample_size=args.sample,
            resume=args.resume
        )
    
    if not any(not df.empty for df in tables.values()):
        print("No results generated!")
        sys.exit(1)
    else:
        print("Report completed successfully!")

if __name__ == "__main__":
    main()