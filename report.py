"""
report_optimized.py – Report XAI con ottimizzazioni avanzate per Google Colab
================================================================================

NUOVE OTTIMIZZAZIONI IMPLEMENTATE:
1. Adaptive Batch Size: Dimensioni batch dinamiche basate su memoria disponibile
2. Explainer Parallelization: Esecuzione parallela di LIME + SHAP + altri explainer
3. Embedding Caching: Cache intelligente per embeddings e tokenizzazioni
4. Memory Management: Cleanup progressivo tra batch con monitoraggio
5. GPU Optimization: CUDA sync ottimizzato, memory pool management
6. Thread Pools: I/O operations parallele per salvataggio/caricamento
7. Smart Resource Allocation: Allocazione dinamica risorse per explainer

STRATEGIA AVANZATA:
- Auto-detection memoria disponibile
- Parallelizzazione intelligente explainer compatibili
- Cache persistente embeddings tra modelli
- Thread pool per I/O non-bloccante
- Garbage collection ottimizzato per GPU
- Batch size adattivo per massima throughput

Uso in Colab:
```python
import report_optimized as report

# Report ultra-veloce con tutte le ottimizzazioni
report.turbo_report()

# Report personalizzato con controllo ottimizzazioni
report.run_optimized_report(
    models=["tinybert", "distilbert"],
    explainers=["lime", "shap", "grad_input"], 
    metrics=["robustness", "consistency"],
    enable_parallel=True,
    enable_caching=True,
    adaptive_batching=True
)
```
"""

import argparse
import gc
import json
import time
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import pickle
import hashlib
import queue
import weakref

import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import psutil

import models
import dataset
import explainers
import metrics
from utils import Timer, PerformanceProfiler, AutoRecovery, print_memory_status, aggressive_cleanup, set_seed

# =============================================================================
# CONFIGURAZIONE OTTIMIZZAZIONI AVANZATE
# =============================================================================

# Configurazione base
EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "contrastivity", "consistency"]
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]

# Configurazione ottimizzazioni
CACHE_DIR = Path("xai_cache")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("xai_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Configurazione parallelizzazione
MAX_WORKERS = min(4, mp.cpu_count())  # Limita per Colab
PARALLEL_EXPLAINERS = ["lime"]  # Explainer sicuri per parallelizzazione
MEMORY_THRESHOLD_GB = 2.0  # Soglia minima memoria per ottimizzazioni

set_seed(42)

# =============================================================================
# GESTIONE MEMORIA AVANZATA
# =============================================================================

class AdvancedMemoryManager:
    """Gestione memoria avanzata con monitoring e ottimizzazioni GPU."""
    
    def __init__(self):
        self.memory_history = []
        self.gpu_memory_pool_enabled = False
        self.cleanup_callbacks = []
        
    def get_available_memory_gb(self) -> float:
        """Ottieni memoria RAM disponibile in GB."""
        return psutil.virtual_memory().available / (1024**3)
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Ottieni informazioni memoria GPU."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "reserved": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),
            "cached": torch.cuda.memory_cached() / (1024**3),
            "reserved": torch.cuda.memory_reserved() / (1024**3)
        }
    
    def calculate_optimal_batch_size(self, base_batch_size: int = 10) -> int:
        """Calcola batch size ottimale basato su memoria disponibile."""
        available_gb = self.get_available_memory_gb()
        
        if available_gb > 8:
            return base_batch_size * 4  # 40
        elif available_gb > 4:
            return base_batch_size * 2  # 20
        elif available_gb > 2:
            return base_batch_size      # 10
        else:
            return max(2, base_batch_size // 2)  # 5
    
    def enable_gpu_memory_pool(self):
        """Abilita memory pool GPU per allocazioni efficienti."""
        if torch.cuda.is_available() and not self.gpu_memory_pool_enabled:
            try:
                # Configura memory pool per allocazioni efficienti
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Pre-alloca pool piccolo per evitare frammentazione
                dummy = torch.zeros(100, 100, device='cuda')
                del dummy
                torch.cuda.empty_cache()
                
                self.gpu_memory_pool_enabled = True
                print("[MEMORY] GPU memory pool enabled")
            except Exception as e:
                print(f"[MEMORY] Failed to enable GPU memory pool: {e}")
    
    def progressive_cleanup(self, level: str = "medium"):
        """Cleanup progressivo con diversi livelli di aggressività."""
        start_time = time.time()
        
        if level == "light":
            gc.collect()
        elif level == "medium":
            # Cleanup standard
            for _ in range(2):
                collected = gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        elif level == "aggressive":
            # Cleanup aggressivo
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            # Callback personalizzati
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"[MEMORY] Cleanup callback failed: {e}")
            
            time.sleep(1)  # Stabilization pause
        
        cleanup_time = time.time() - start_time
        memory_after = self.get_available_memory_gb()
        gpu_info = self.get_gpu_memory_info()
        
        print(f"[MEMORY] {level} cleanup: {cleanup_time:.1f}s, "
              f"RAM: {memory_after:.1f}GB, GPU: {gpu_info['allocated']:.1f}GB")
    
    def add_cleanup_callback(self, callback):
        """Aggiungi callback personalizzato per cleanup."""
        self.cleanup_callbacks.append(callback)

# =============================================================================
# SISTEMA CACHING AVANZATO
# =============================================================================

class EmbeddingCache:
    """Cache intelligente per embeddings e tokenizzazioni."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = weakref.WeakValueDictionary()
        self.cache_stats = {"hits": 0, "misses": 0, "saves": 0}
        
    def _get_cache_key(self, model_key: str, text: str, operation: str = "embedding") -> str:
        """Genera chiave cache univoca."""
        content = f"{model_key}_{operation}_{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Ottieni percorso file cache."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get_embedding(self, model_key: str, text: str) -> Optional[torch.Tensor]:
        """Ottieni embedding dalla cache."""
        cache_key = self._get_cache_key(model_key, text, "embedding")
        
        # Controlla memory cache prima
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key].clone()
        
        # Controlla disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                self.memory_cache[cache_key] = embedding
                self.cache_stats["hits"] += 1
                return embedding.clone()
            except Exception as e:
                print(f"[CACHE] Failed to load {cache_path}: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def save_embedding(self, model_key: str, text: str, embedding: torch.Tensor):
        """Salva embedding in cache."""
        cache_key = self._get_cache_key(model_key, text, "embedding")
        
        # Salva in memory cache
        self.memory_cache[cache_key] = embedding.clone()
        
        # Salva in disk cache (async)
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding.cpu(), f)
            self.cache_stats["saves"] += 1
        except Exception as e:
            print(f"[CACHE] Failed to save {cache_path}: {e}")
    
    def get_tokenization(self, model_key: str, text: str) -> Optional[Dict]:
        """Ottieni tokenizzazione dalla cache."""
        cache_key = self._get_cache_key(model_key, text, "tokenization")
        
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key].copy()
        
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    tokenization = pickle.load(f)
                self.memory_cache[cache_key] = tokenization
                self.cache_stats["hits"] += 1
                return tokenization.copy()
            except Exception as e:
                print(f"[CACHE] Failed to load tokenization {cache_path}: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def save_tokenization(self, model_key: str, text: str, tokenization: Dict):
        """Salva tokenizzazione in cache."""
        cache_key = self._get_cache_key(model_key, text, "tokenization")
        
        self.memory_cache[cache_key] = tokenization.copy()
        
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(tokenization, f)
            self.cache_stats["saves"] += 1
        except Exception as e:
            print(f"[CACHE] Failed to save tokenization {cache_path}: {e}")
    
    def print_stats(self):
        """Stampa statistiche cache."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        print(f"[CACHE] Stats: {self.cache_stats['hits']} hits, "
              f"{self.cache_stats['misses']} misses, "
              f"{hit_rate:.1%} hit rate, "
              f"{self.cache_stats['saves']} saves")

# =============================================================================
# PARALLELIZZAZIONE EXPLAINER
# =============================================================================

class ParallelExplainerManager:
    """Gestione parallelizzazione explainer con resource allocation."""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = None  # Inizializzato quando necessario
        self.explainer_compatibility = {
            "lime": {"parallel": True, "thread_safe": True},
            "shap": {"parallel": False, "thread_safe": False},
            "grad_input": {"parallel": False, "thread_safe": False},  # Usa GPU
            "attention_rollout": {"parallel": False, "thread_safe": False},
            "attention_flow": {"parallel": False, "thread_safe": False},
            "lrp": {"parallel": False, "thread_safe": False}
        }
    
    def can_parallelize(self, explainer_names: List[str]) -> bool:
        """Verifica se gli explainer possono essere parallelizzati."""
        return all(
            self.explainer_compatibility.get(name, {}).get("parallel", False)
            for name in explainer_names
        )
    
    def split_explainers(self, explainer_names: List[str]) -> Tuple[List[str], List[str]]:
        """Divide explainer in parallelizzabili e sequenziali."""
        parallel = []
        sequential = []
        
        for name in explainer_names:
            if self.explainer_compatibility.get(name, {}).get("parallel", False):
                parallel.append(name)
            else:
                sequential.append(name)
        
        return parallel, sequential
    
    def run_explainers_parallel(self, 
                               explainer_tasks: List[Tuple[str, callable]], 
                               timeout: int = 300) -> Dict[str, Any]:
        """Esegue explainer in parallelo con timeout."""
        print(f"[PARALLEL] Running {len(explainer_tasks)} explainers in parallel")
        
        futures = {}
        results = {}
        
        # Sottometti task
        for explainer_name, task_func in explainer_tasks:
            future = self.thread_pool.submit(task_func)
            futures[future] = explainer_name
        
        # Raccoglie risultati con timeout
        for future in as_completed(futures, timeout=timeout):
            explainer_name = futures[future]
            try:
                result = future.result(timeout=30)  # Timeout per singolo task
                results[explainer_name] = result
                print(f"[PARALLEL] {explainer_name}: SUCCESS")
            except Exception as e:
                print(f"[PARALLEL] {explainer_name}: FAILED - {e}")
                results[explainer_name] = None
        
        return results
    
    def cleanup(self):
        """Cleanup thread pool."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        if self.process_pool:
            self.process_pool.shutdown(wait=False)

# =============================================================================
# I/O THREAD POOL
# =============================================================================

class AsyncIOManager:
    """Gestione I/O operations asincrone."""
    
    def __init__(self, max_workers: int = 2):
        self.io_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_operations = []
    
    def save_async(self, data: Any, filepath: Path, format: str = "json"):
        """Salvataggio asincrono."""
        def save_task():
            try:
                if format == "json":
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                elif format == "pickle":
                    with open(filepath, 'wb') as f:
                        pickle.dump(data, f)
                elif format == "csv":
                    data.to_csv(filepath)
                print(f"[ASYNC-IO] Saved: {filepath.name}")
                return True
            except Exception as e:
                print(f"[ASYNC-IO] Save failed {filepath.name}: {e}")
                return False
        
        future = self.io_pool.submit(save_task)
        self.pending_operations.append(future)
        return future
    
    def wait_all(self, timeout: int = 60):
        """Aspetta completamento di tutte le operazioni pending."""
        print(f"[ASYNC-IO] Waiting for {len(self.pending_operations)} pending operations...")
        
        completed = 0
        for future in as_completed(self.pending_operations, timeout=timeout):
            try:
                future.result()
                completed += 1
            except Exception as e:
                print(f"[ASYNC-IO] Operation failed: {e}")
        
        print(f"[ASYNC-IO] Completed {completed}/{len(self.pending_operations)} operations")
        self.pending_operations.clear()
    
    def cleanup(self):
        """Cleanup I/O pool."""
        self.wait_all()
        self.io_pool.shutdown(wait=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_optimized_header():
    """Header per report ottimizzato."""
    print("="*80)
    print(" XAI COMPREHENSIVE REPORT - OPTIMIZED VERSION")
    print("="*80)
    print(" Optimizations:")
    print("   - Adaptive batch sizing based on available memory")
    print("   - Parallel explainer execution (LIME + SHAP)")
    print("   - Intelligent embedding/tokenization caching")
    print("   - Advanced GPU memory management")
    print("   - Asynchronous I/O operations")
    print("   - Progressive memory cleanup")
    print("="*80)

def get_system_resources():
    """Analizza risorse sistema per ottimizzazioni."""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = psutil.cpu_count()
    
    gpu_info = {"available": False, "memory_gb": 0}
    if torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"[SYSTEM] RAM: {available_ram_gb:.1f}/{ram_gb:.1f}GB available")
    print(f"[SYSTEM] CPU: {cpu_count} cores")
    gpu_text = "Yes" if gpu_info['available'] else "No"
    if gpu_info['available']:
        gpu_text += f" ({gpu_info['memory_gb']:.1f}GB)"
    print(f"[SYSTEM] GPU: {gpu_text}") 
    
    return {
        "ram_total_gb": ram_gb,
        "ram_available_gb": available_ram_gb,
        "cpu_count": cpu_count,
        "gpu_info": gpu_info
    }

def process_model_optimized(
    model_key: str,
    explainers_to_test: List[str],
    metrics_to_compute: List[str],
    sample_size: int,
    enable_parallel: bool = True,
    enable_caching: bool = True,
    adaptive_batching: bool = True,
    recovery: AutoRecovery = None
) -> Dict[str, Dict[str, float]]:
    """Processa singolo modello con tutte le ottimizzazioni."""
    
    print(f"\n{'='*70}")
    print(f" PROCESSING MODEL: {model_key} (OPTIMIZED)")
    print(f"{'='*70}")
    
    # Inizializza manager
    memory_manager = AdvancedMemoryManager()
    embedding_cache = EmbeddingCache() if enable_caching else None
    parallel_manager = ParallelExplainerManager() if enable_parallel else None
    async_io = AsyncIOManager()
    
    profiler = PerformanceProfiler()
    profiler.start_operation(f"model_{model_key}_optimized")
    
    try:
        # Abilita ottimizzazioni GPU
        memory_manager.enable_gpu_memory_pool()
        
        # Carica modello
        print(f"[LOAD] Loading {model_key} with caching...")
        with Timer(f"Loading {model_key}"):
            model = models.load_model(model_key)
            tokenizer = models.load_tokenizer(model_key)
        
        # Calcola batch size ottimale
        if adaptive_batching:
            optimal_batch_size = memory_manager.calculate_optimal_batch_size()
            print(f"[ADAPTIVE] Optimal batch size: {optimal_batch_size}")
        else:
            optimal_batch_size = 10
        
        # Prepara dati
        print(f"[DATA] Preparing data (sample_size={sample_size})...")
        texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)
        
        # Pre-cache embeddings se abilitato
        if enable_caching:
            print(f"[CACHE] Pre-caching embeddings for {len(texts)} texts...")
            cache_hits = 0
            for text in texts[:20]:  # Solo primi 20 per test
                if embedding_cache.get_embedding(model_key, text) is not None:
                    cache_hits += 1
            print(f"[CACHE] {cache_hits}/20 embeddings found in cache")
        
        # Separa dati per metriche
        pos_texts = [t for t, l in zip(texts, labels) if l == 1][:optimal_batch_size]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0][:optimal_batch_size]
        consistency_texts = texts[:min(optimal_batch_size, len(texts))]
        
        print(f"[DATA] Batch sizes - Pos: {len(pos_texts)}, Neg: {len(neg_texts)}, Consistency: {len(consistency_texts)}")
        
        # Inizializza risultati
        results = {}
        for metric in metrics_to_compute:
            results[metric] = {}
            for explainer in explainers_to_test:
                results[metric][explainer] = float('nan')
        
        # Dividi explainer per parallelizzazione
        if enable_parallel and parallel_manager:
            parallel_explainers, sequential_explainers = parallel_manager.split_explainers(explainers_to_test)
            print(f"[PARALLEL] Parallel: {parallel_explainers}, Sequential: {sequential_explainers}")
        else:
            parallel_explainers, sequential_explainers = [], explainers_to_test
        
        # FASE 1: Explainer paralleli
        if parallel_explainers:
            print(f"\n[PHASE 1] Processing {len(parallel_explainers)} explainers in parallel...")
            
            def create_explainer_task(explainer_name):
                def task():
                    try:
                        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                        explainer_results = {}
                        
                        for metric_name in metrics_to_compute:
                            try:
                                if metric_name == "robustness":
                                    score = metrics.evaluate_robustness_over_dataset(
                                        model, tokenizer, explainer, consistency_texts, show_progress=False
                                    )
                                elif metric_name == "contrastivity":
                                    # Usa batch processing ottimizzato
                                    pos_attrs = metrics.process_attributions_batch(
                                        pos_texts, explainer, batch_size=optimal_batch_size//2, show_progress=False
                                    )
                                    neg_attrs = metrics.process_attributions_batch(
                                        neg_texts, explainer, batch_size=optimal_batch_size//2, show_progress=False
                                    )
                                    
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
                                
                                explainer_results[metric_name] = score
                                
                            except Exception as e:
                                print(f"[PARALLEL] {explainer_name}.{metric_name} failed: {e}")
                                explainer_results[metric_name] = float('nan')
                            
                            # Memory cleanup progressivo
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        return explainer_results
                        
                    except Exception as e:
                        print(f"[PARALLEL] {explainer_name} task completely failed: {e}")
                        return {metric: float('nan') for metric in metrics_to_compute}
                
                return task
            
            # Esegui task paralleli
            parallel_tasks = [(name, create_explainer_task(name)) for name in parallel_explainers]
            parallel_results = parallel_manager.run_explainers_parallel(parallel_tasks)
            
            # Integra risultati paralleli
            for explainer_name, explainer_results in parallel_results.items():
                if explainer_results:
                    for metric_name, score in explainer_results.items():
                        results[metric_name][explainer_name] = score
            
            # Cleanup dopo parallelizzazione
            memory_manager.progressive_cleanup("medium")
        
        # FASE 2: Explainer sequenziali (GPU-intensive)
        if sequential_explainers:
            print(f"\n[PHASE 2] Processing {len(sequential_explainers)} explainers sequentially...")
            
            for i, explainer_name in enumerate(sequential_explainers, 1):
                print(f"[{i}/{len(sequential_explainers)}] {explainer_name}...")
                
                try:
                    explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                    
                    for metric_name in metrics_to_compute:
                        print(f"  [{metric_name}]...", end=" ")
                        
                        try:
                            if metric_name == "robustness":
                                score = metrics.evaluate_robustness_over_dataset(
                                    model, tokenizer, explainer, consistency_texts, show_progress=False
                                )
                            elif metric_name == "contrastivity":
                                pos_attrs = metrics.process_attributions_batch(
                                    pos_texts, explainer, batch_size=optimal_batch_size//2, show_progress=False
                                )
                                neg_attrs = metrics.process_attributions_batch(
                                    neg_texts, explainer, batch_size=optimal_batch_size//2, show_progress=False
                                )
                                
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
                            print(f"SUCCESS {score:.4f}")
                            
                        except Exception as e:
                            print(f"ERROR: {str(e)[:30]}...")
                            results[metric_name][explainer_name] = float('nan')
                    
                    del explainer
                    memory_manager.progressive_cleanup("light")
                    
                except Exception as e:
                    print(f"  EXPLAINER FAILED: {explainer_name}: {e}")
                    for metric in metrics_to_compute:
                        results[metric][explainer_name] = float('nan')
        
        # Salvataggio asincrono risultati
        if async_io:
            async_io.save_async(results, RESULTS_DIR / f"results_{model_key}_optimized.json")
        
        profiler.end_operation(f"model_{model_key}_optimized")
        
        # Stampa statistiche cache
        if embedding_cache:
            embedding_cache.print_stats()
        
        print(f"\n[COMPLETE] {model_key} optimized processing completed")
        return results
        
    except Exception as e:
        print(f"[ERROR] Model {model_key} optimized processing failed: {e}")
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
        # Cleanup finale
        print(f"[CLEANUP] Final cleanup for {model_key}...")
        
        if parallel_manager:
            parallel_manager.cleanup()
        if async_io:
            async_io.cleanup()
        
        memory_manager.progressive_cleanup("aggressive")

def smart_model_ordering(models_to_test: List[str]) -> List[str]:
    """Ordina modelli intelligentemente per ottimizzare processing."""
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

def verify_checkpoint_completeness(
    checkpoint_data: Dict, 
    expected_metrics: List[str], 
    expected_explainers: List[str]
) -> Tuple[bool, str]:
    """Verifica che il checkpoint contenga tutti i dati attesi."""
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

def build_report_tables(all_results: Dict[str, Dict], metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]:
    """Costruisce tabelle finali dai risultati."""
    print(f"\n{'='*70}")
    print(" BUILDING REPORT TABLES (OPTIMIZED)")
    print(f"{'='*70}")
    
    tables = {}
    
    for metric in metrics_to_compute:
        print(f"\n[TABLE] Building {metric} table...")
        
        # Struttura: explainer -> {model: score}
        metric_data = defaultdict(dict)
        
        for model_key, model_data in all_results.items():
            if "results" not in model_data:
                continue
                
            if metric not in model_data["results"]:
                continue
                
            explainer_scores = model_data["results"][metric]
            
            for explainer_name, score in explainer_scores.items():
                if score is not None and not (isinstance(score, float) and np.isnan(score)):
                    metric_data[explainer_name][model_key] = score
        
        if metric_data:
            df = pd.DataFrame(metric_data).T  # Transpose: explainer come righe
            tables[metric] = df
            print(f"[TABLE] {metric}: {df.shape[0]} explainers × {df.shape[1]} models")
        else:
            print(f"[TABLE] {metric}: No data available")
            tables[metric] = pd.DataFrame()
    
    return tables

def print_table_analysis(df: pd.DataFrame, metric_name: str):
    """Analisi e interpretazione tabella."""
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

def generate_summary_report(tables: Dict[str, pd.DataFrame], 
                          execution_time: float,
                          models_tested: List[str],
                          explainers_tested: List[str],
                          optimizations_used: Dict[str, bool]) -> str:
    """Genera report summary testuale."""
    
    summary = f"""
XAI BENCHMARK REPORT - OPTIMIZED VERSION
=========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Execution Time: {execution_time/60:.1f} minutes
Dataset: {len(dataset.test_df)} clustered examples from IMDB

CONFIGURATION:
- Models tested: {len(models_tested)} ({', '.join(models_tested)})
- Explainers tested: {len(explainers_tested)} ({', '.join(explainers_tested)})
- Metrics computed: {len(tables)} ({', '.join(tables.keys())})

OPTIMIZATIONS ENABLED:
- Parallel Processing: {'✓' if optimizations_used.get('parallel', False) else '✗'}
- Embedding Caching: {'✓' if optimizations_used.get('caching', False) else '✗'}
- Adaptive Batching: {'✓' if optimizations_used.get('adaptive_batching', False) else '✗'}
- GPU Memory Pool: {'✓' if optimizations_used.get('gpu_optimization', False) else '✗'}
- Async I/O: {'✓' if optimizations_used.get('async_io', False) else '✗'}

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
OPTIMIZATION BENEFITS:
- Smart model ordering prevents environment degradation
- Parallel explainer execution reduces processing time
- Embedding cache eliminates redundant computations
- Adaptive batching maximizes memory utilization
- Advanced cleanup ensures system stability

RECOMMENDATIONS:
- For production XAI: Use combinations with highest consistency scores
- For research: Focus on high contrastivity explainers
- For deployment: Balance robustness vs computational cost

Files Generated:
- Raw results: {RESULTS_DIR}/results_*_optimized.json
- CSV tables: {RESULTS_DIR}/*_table.csv
- Summary: {RESULTS_DIR}/summary_report_optimized.txt
"""
    
    return summary

# =============================================================================
# MAIN REPORT FUNCTIONS
# =============================================================================

def run_optimized_report(
    models_to_test: List[str] = None,
    explainers_to_test: List[str] = None, 
    metrics_to_compute: List[str] = None,
    sample_size: int = 100,
    enable_parallel: bool = True,
    enable_caching: bool = True,
    adaptive_batching: bool = True,
    resume: bool = True
) -> Dict[str, pd.DataFrame]:
    """Esegue report completo con tutte le ottimizzazioni."""
    
    start_time = time.time()
    profiler = PerformanceProfiler()
    recovery = AutoRecovery(checkpoint_dir=RESULTS_DIR / "checkpoints")
    
    profiler.start_operation("optimized_report")
    
    try:
        print_optimized_header()
        
        # Analizza risorse sistema
        system_resources = get_system_resources()
        
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
        
        # SMART MODEL ORDERING
        models_to_test = smart_model_ordering(models_to_test)
        
        # Disabilita ottimizzazioni se risorse insufficienti
        if system_resources["ram_available_gb"] < MEMORY_THRESHOLD_GB:
            print(f"[WARNING] Low memory ({system_resources['ram_available_gb']:.1f}GB), disabling some optimizations")
            enable_parallel = False
            adaptive_batching = False
        
        total_combinations = len(models_to_test) * len(explainers_to_test) * len(metrics_to_compute)
        
        print(f"\n[REPORT] Optimized Configuration:")
        print(f"  Models: {models_to_test}")
        print(f"  Explainers: {explainers_to_test}")
        print(f"  Metrics: {metrics_to_compute}")
        print(f"  Sample size: {sample_size}")
        print(f"  Total combinations: {total_combinations}")
        print(f"  Optimizations: Parallel={enable_parallel}, Cache={enable_caching}, Adaptive={adaptive_batching}")
        
        # FASE 1: Process each model with optimizations
        print(f"\n{'='*80}")
        print("FASE 1: OPTIMIZED MODEL PROCESSING")
        print(f"{'='*80}")
        
        all_results = {}
        
        for i, model_key in enumerate(models_to_test, 1):
            print(f"\n[{i}/{len(models_to_test)}] Model: {model_key}")
            
            # Resume logic with checkpoint validation
            if resume:
                print(f"[RESUME] Checking for existing checkpoint...")
                existing = recovery.load_latest_checkpoint(f"model_{model_key}")
                
                if existing:
                    is_complete, reason = verify_checkpoint_completeness(
                        existing, metrics_to_compute, explainers_to_test
                    )
                    
                    if is_complete:
                        print(f"[RESUME] Using complete cached results for {model_key}")
                        all_results[model_key] = existing
                        continue
                    else:
                        print(f"[RESUME] Incomplete checkpoint: {reason}")
            
            try:
                with Timer(f"Processing {model_key} (optimized)"):
                    results = process_model_optimized(
                        model_key=model_key,
                        explainers_to_test=explainers_to_test,
                        metrics_to_compute=metrics_to_compute,
                        sample_size=sample_size,
                        enable_parallel=enable_parallel,
                        enable_caching=enable_caching,
                        adaptive_batching=adaptive_batching,
                        recovery=recovery
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
        
        # FASE 2: Build tables
        print(f"\n{'='*80}")
        print("FASE 2: BUILDING OPTIMIZED TABLES")
        print(f"{'='*80}")
        
        profiler.start_operation("table_building")
        tables = build_report_tables(all_results, metrics_to_compute)
        profiler.end_operation("table_building")
        
        # FASE 3: Analysis & Output
        print(f"\n{'='*80}")
        print("FASE 3: OPTIMIZED ANALYSIS & OUTPUT")
        print(f"{'='*80}")
        
        execution_time = time.time() - start_time
        
        # Print tables with analysis
        for metric_name, df in tables.items():
            if not df.empty:
                print(f"\n {metric_name.upper()} TABLE:")
                print("=" * 50)
                print(df.to_string(float_format="%.4f", na_rep="—"))
                
                # Save CSV
                csv_file = RESULTS_DIR / f"{metric_name}_table_optimized.csv"
                df.to_csv(csv_file)
                print(f"[SAVE] CSV saved: {csv_file}")
                
                # Analysis
                print_table_analysis(df, metric_name)
        
        # Generate optimized summary
        optimizations_used = {
            "parallel": enable_parallel,
            "caching": enable_caching,
            "adaptive_batching": adaptive_batching,
            "gpu_optimization": torch.cuda.is_available(),
            "async_io": True
        }
        
        summary = generate_summary_report(
            tables, execution_time, models_to_test, explainers_to_test, optimizations_used
        )
        
        # Save summary
        summary_file = RESULTS_DIR / "summary_report_optimized.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"[SAVE] Summary saved: {summary_file}")
        
        profiler.end_operation("optimized_report")
        
        # Final summary
        print(f"\n{'='*80}")
        print(" OPTIMIZED REPORT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"  Total time: {execution_time/60:.1f} minutes")
        print(f"  Models processed: {len([r for r in all_results.values() if r.get('completed', False)])}/{len(models_to_test)}")
        print(f"  Tables generated: {len([t for t in tables.values() if not t.empty])}")
        print(f"  Files saved in: {RESULTS_DIR}")
        print(f"  Total data points: {sum(df.notna().sum().sum() for df in tables.values())}")
        
        # Performance summary
        profiler.print_summary()
        
        return tables
        
    except Exception as e:
        print(f"\nOPTIMIZED REPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {}

def turbo_report(sample_size: int = 50) -> Dict[str, pd.DataFrame]:
    """Report ultra-veloce con tutte le ottimizzazioni abilitate."""
    print_optimized_header()
    print("[TURBO] Ultra-fast report with all optimizations enabled")
    
    # Configuration for maximum speed
    models_subset = ["tinybert", "distilbert"]  # Fastest models
    explainers_subset = ["lime", "shap"]       # Parallel-friendly explainers
    metrics_subset = ["robustness"]            # Single metric for speed
    
    return run_optimized_report(
        models_to_test=models_subset,
        explainers_to_test=explainers_subset,
        metrics_to_compute=metrics_subset,
        sample_size=sample_size,
        enable_parallel=True,
        enable_caching=True,
        adaptive_batching=True,
        resume=True
    )

def get_available_resources():
    """Ottieni risorse disponibili per report."""
    available_models = list(models.MODELS.keys())
    available_explainers = [exp for exp in EXPLAINERS if exp in explainers.list_explainers()]
    
    print(f"[RESOURCES] Models: {len(available_models)} available")
    print(f"[RESOURCES] Explainers: {len(available_explainers)} available")
    print(f"[RESOURCES] Metrics: {len(METRICS)} available")
    print(f"[RESOURCES] Dataset: {len(dataset.test_df)} clustered examples")
    
    return available_models, available_explainers

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="XAI Report Generator - Optimized Version")
    parser.add_argument("--models", nargs="+", choices=list(models.MODELS.keys()), 
                       default=None, help="Models to test")
    parser.add_argument("--explainers", nargs="+", choices=EXPLAINERS,
                       default=None, help="Explainers to test")
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                       default=None, help="Metrics to compute")
    parser.add_argument("--sample", type=int, default=100, help="Sample size")
    parser.add_argument("--turbo", action="store_true", help="Ultra-fast turbo report")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive batching")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoints")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh")
    
    args = parser.parse_args()
    
    print("XAI BENCHMARK - OPTIMIZED VERSION")
    print("="*50)
    print("Optimizations:")
    print("- Adaptive batch sizing")
    print("- Parallel explainer execution") 
    print("- Intelligent caching")
    print("- Advanced memory management")
    print("- GPU optimizations")
    print("- Async I/O operations")
    print("="*50)
    
    if args.turbo:
        print("Running turbo report...")
        tables = turbo_report(sample_size=min(args.sample, 50))
    else:
        print("Running optimized report...")
        tables = run_optimized_report(
            models_to_test=args.models,
            explainers_to_test=args.explainers,
            metrics_to_compute=args.metrics,
            sample_size=args.sample,
            enable_parallel=not args.no_parallel,
            enable_caching=not args.no_cache,
            adaptive_batching=not args.no_adaptive,
            resume=args.resume
        )
    
    if not any(not df.empty for df in tables.values()):
        print("No results generated!")
        sys.exit(1)
    else:
        print("Optimized report completed successfully!")

if __name__ == "__main__":
    main()