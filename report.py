"""
report.py – Report XAI completo ottimizzato per Google Colab
==========================================================

OTTIMIZZAZIONI PER COLAB:
1. Memory management ottimizzato per GPU limitata
2. Progress tracking dettagliato per notebook
3. Checkpoint automatici per recovery
4. Dataset clusterizzato (400 esempi)
5. Batch processing intelligente
6. Error recovery e fallback
7. Report generation ottimizzato

STRATEGIA COLAB:
- Processa 1 modello alla volta per memoria
- Salva risultati intermedi automaticamente  
- Cleanup aggressivo tra modelli
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
    print(" XAI COMPREHENSIVE REPORT")
    print("="*80)
    print(" Strategy: Memory-optimized sequential processing")
    print(" Dataset: 400 clustered representative examples")
    print(" Recovery: Auto-checkpoint for crash recovery")
    print(" Cleanup: Aggressive memory management between models")
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
    
    print(f"[SAVE] ✓ Results saved: {filename.name}")
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
    
    print(f"[LOAD] ✓ Loaded results for {len(all_results)} models")
    return all_results

def process_single_model(
    model_key: str,
    explainers_to_test: List[str],
    metrics_to_compute: List[str],
    sample_size: int,
    recovery: AutoRecovery
) -> Dict[str, Dict[str, float]]:
    """Processa singolo modello con tutte le metriche."""
    
    print(f"\n{'='*70}")
    print(f" PROCESSING MODEL: {model_key}")
    print(f"{'='*70}")
    
    # Check se già processato
    existing_results = recovery.load_latest_checkpoint(f"model_{model_key}")
    if existing_results and existing_results.get("completed", False):
        print(f"[SKIP] ✓ {model_key} already completed")
        return existing_results["results"]
    
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
        
        print(f"[DATA] ✓ Total: {len(texts)}, Pos: {len(pos_texts)}, Neg: {len(neg_texts)}")
        
        # Risultati per questo modello
        results = {metric: {} for metric in metrics_to_compute}
        
        # Processa ogni explainer
        for i, explainer_name in enumerate(explainers_to_test, 1):
            print(f"\n[{i}/{len(explainers_to_test)}] 🔍 EXPLAINER: {explainer_name}")
            print("-" * 50)
            
            explainer_start_time = time.time()
            
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
                        print(f"✓ {score:.4f} ({metric_time:.1f}s)")
                        
                    except Exception as e:
                        print(f"✗ ERROR: {str(e)[:50]}...")
                        results[metric_name][explainer_name] = float('nan')
                
                explainer_time = time.time() - explainer_start_time
                print(f"  [TOTAL] {explainer_name}: {explainer_time:.1f}s")
                
                # Cleanup explainer
                del explainer
                gc.collect()
                
                # Checkpoint intermedio
                checkpoint_data = {
                    "results": results,
                    "completed": False,
                    "progress": f"{i}/{len(explainers_to_test)} explainers"
                }
                recovery.save_checkpoint(checkpoint_data, f"model_{model_key}")
                
            except Exception as e:
                print(f"  ✗ EXPLAINER FAILED: {explainer_name}: {e}")
                for metric in metrics_to_compute:
                    results[metric][explainer_name] = float('nan')
        
        # Salva risultati finali
        final_results = {
            "results": results,
            "completed": True,
            "timestamp": datetime.now().isoformat()
        }
        recovery.save_checkpoint(final_results, f"model_{model_key}")
        save_intermediate_results(model_key, results)
        
        profiler.end_operation(f"model_{model_key}")
        
        print(f"\n[COMPLETE] ✓ {model_key} processing completed")
        print(f"[MEMORY] Peak usage tracked")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] ✗ Model {model_key} failed: {e}")
        import traceback
        traceback.print_exc()
        return {metric: {} for metric in metrics_to_compute}
        
    finally:
        # CLEANUP AGGRESSIVO
        print(f"[CLEANUP] Cleaning memory for {model_key}...")
        
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'explainer' in locals():
            del explainer
        
        aggressive_cleanup()
        print_memory_status()

def build_report_tables(all_results: Dict[str, Dict], metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]:
    """Costruisce tabelle finali dal risultati."""
    print(f"\n{'='*70}")
    print(" BUILDING REPORT TABLES")
    print(f"{'='*70}")
    
    tables = {}
    
    for metric in metrics_to_compute:
        print(f"[TABLE] Building {metric} table...")
        
        # Struttura: explainer -> {model: score}
        metric_data = defaultdict(dict)
        
        for model_key, model_data in all_results.items():
            if "results" in model_data and metric in model_data["results"]:
                for explainer_name, score in model_data["results"][metric].items():
                    if score is not None:
                        metric_data[explainer_name][model_key] = score
        
        if metric_data:
            df = pd.DataFrame(metric_data).T  # Transpose: explainer come righe
            tables[metric] = df
            print(f"[TABLE] ✓ {metric}: {df.shape[0]} explainers × {df.shape[1]} models")
        else:
            print(f"[TABLE] ✗ {metric}: No data available")
            tables[metric] = pd.DataFrame()
    
    return tables

def print_table_analysis(df: pd.DataFrame, metric_name: str):
    """Analisi e interpretazione tabella."""
    print(f"\n{'='*60}")
    print(f" {metric_name.upper()} ANALYSIS")
    print(f"{'='*60}")
    
    if df.empty:
        print("❌ No data available for analysis")
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
            print(f"  {explainer:>15s}: μ={mean_val:.4f} σ={std_val:.4f} (n={count})")
    
    # Statistiche per modello
    print("\n Per-Model Statistics:")
    print("-" * 40)
    for model in df.columns:
        values = df[model].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            count = len(values)
            print(f"  {model:>15s}: μ={mean_val:.4f} σ={std_val:.4f} (n={count})")
    
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
    
    print(f"{'='*60}")

def generate_summary_report(tables: Dict[str, pd.DataFrame], 
                          execution_time: float,
                          models_tested: List[str],
                          explainers_tested: List[str]) -> str:
    """Genera report summary testuale."""
    
    summary = f"""
XAI BENCHMARK REPORT - GOOGLE COLAB EDITION
============================================
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
            summary += f"\n{metric_name.upper()}:\n"
            summary += f"  Coverage: {df.notna().sum().sum()}/{df.size} combinations completed\n"
            
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
    
    return run_report(models_to_test, explainers_to_test, metrics_to_compute, sample_size)

def run_report(models_to_test: List[str] = None,
               explainers_to_test: List[str] = None, 
               metrics_to_compute: List[str] = None,
               sample_size: int = 100,
               resume: bool = True) -> Dict[str, pd.DataFrame]:
    """Esegue report completo (callable da Colab)."""
    
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
        print("FASE 1: SEQUENTIAL MODEL PROCESSING")
        print(f"{'='*80}")
        
        all_results = {}
        
        for i, model_key in enumerate(models_to_test, 1):
            print(f"\n[{i}/{len(models_to_test)}] Model: {model_key}")
            
            # Check resume
            if resume:
                existing = recovery.load_latest_checkpoint(f"model_{model_key}")
                if existing and existing.get("completed", False):
                    print(f"[RESUME] ✓ Using cached results for {model_key}")
                    all_results[model_key] = existing
                    continue
            
            try:
                with Timer(f"Processing {model_key}"):
                    results = process_single_model(
                        model_key=model_key,
                        explainers_to_test=explainers_to_test,
                        metrics_to_compute=metrics_to_compute,
                        sample_size=sample_size,
                        recovery=recovery
                    )
                    all_results[model_key] = {"results": results, "completed": True}
                
            except Exception as e:
                print(f"[ERROR] Model {model_key} failed: {e}")
                all_results[model_key] = {
                    "results": {metric: {} for metric in metrics_to_compute},
                    "completed": False,
                    "error": str(e)
                }
                continue
        
        # FASE 2: Build tables
        print(f"\n{'='*80}")
        print("FASE 2: BUILDING FINAL TABLES")  
        print(f"{'='*80}")
        
        profiler.start_operation("table_building")
        tables = build_report_tables(all_results, metrics_to_compute)
        profiler.end_operation("table_building")
        
        # FASE 3: Analysis & Output
        print(f"\n{'='*80}")
        print("FASE 3: ANALYSIS & OUTPUT")
        print(f"{'='*80}")
        
        execution_time = time.time() - start_time
        
        # Print tables
        for metric_name, df in tables.items():
            if not df.empty:
                print(f"\n {metric_name.upper()} TABLE:")
                print("=" * 50)
                print(df.to_string(float_format="%.4f", na_rep="—"))
                
                # Save CSV
                csv_file = RESULTS_DIR / f"{metric_name}_table.csv"
                df.to_csv(csv_file)
                print(f"[SAVE] ✓ CSV saved: {csv_file}")
                
                # Analysis
                print_table_analysis(df, metric_name)
        
        # Generate summary
        summary = generate_summary_report(
            tables, execution_time, models_to_test, explainers_to_test
        )
        
        # Save summary
        summary_file = RESULTS_DIR / "summary_report.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"[SAVE] ✓ Summary saved: {summary_file}")
        
        profiler.end_operation("full_report")
        
        # Final summary
        print(f"\n{'='*80}")
        print(" REPORT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"  Total time: {execution_time/60:.1f} minutes")
        print(f" Tables generated: {len([t for t in tables.values() if not t.empty])}")
        print(f" Files saved in: {RESULTS_DIR}")
        print(f" Total combinations: {sum(df.notna().sum().sum() for df in tables.values())}")
        
        # Performance summary
        profiler.print_summary()
        
        return tables
        
    except Exception as e:
        print(f"\n REPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {}
        
    finally:
        aggressive_cleanup()

# ==== CLI Interface ====
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="XAI Report Generator for Google Colab")
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

if __name__ == "__main__":
    main()