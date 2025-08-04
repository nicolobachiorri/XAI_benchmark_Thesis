"""
utils.py – Utility ottimizzate per Google Colab
===============================================

AGGIORNAMENTI PER COLAB:
1. Colab-specific environment utilities
2. Enhanced memory monitoring con alerting
3. Progress tracking ottimizzato per notebook
4. File management per Colab filesystem
5. Performance profiling utilities
6. Auto-cleanup e recovery functions
"""

import json
import random
import time
import gc
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, List, TypeVar, Optional, Dict
from datetime import datetime, timedelta

import numpy as np
import torch
from tqdm.auto import tqdm  # Auto-detect notebook vs terminal

T = TypeVar("T")

# ==== Colab Environment Setup ====
def setup_colab_environment():
    """Setup ottimale per Google Colab."""
    print("[SETUP] Configuring Colab environment...")
    
    # Suppress common warnings in Colab
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Environment variables per stabilità
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Non-blocking per performance
    
    # Matplotlib backend per Colab
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        pass
    
    print("[SETUP]  Colab environment configured")

# Setup automatico
setup_colab_environment()

# ==== Seed Management ====
def set_seed(seed: int) -> None:
    """Fissa seed per riproducibilità completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ulteriore controllo per determinismo
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==== Memory Management Avanzato ====
class MemoryMonitor:
    """Monitor memoria RAM e GPU con alerting."""
    
    def __init__(self, alert_threshold_gb: float = 10.0):
        self.alert_threshold = alert_threshold_gb
        self.peak_usage = {"ram": 0.0, "gpu": 0.0}
        
    def get_usage(self) -> Dict[str, float]:
        """Ottieni uso memoria corrente."""
        usage = {}
        
        # RAM
        try:
            import psutil
            ram_info = psutil.virtual_memory()
            usage["ram_used_gb"] = ram_info.used / (1024**3)
            usage["ram_total_gb"] = ram_info.total / (1024**3)
            usage["ram_percent"] = ram_info.percent
        except ImportError:
            usage["ram_used_gb"] = 0.0
            usage["ram_total_gb"] = 0.0
            usage["ram_percent"] = 0.0
        
        # GPU
        if torch.cuda.is_available():
            usage["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            usage["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            usage["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage["gpu_percent"] = (usage["gpu_allocated_gb"] / usage["gpu_total_gb"]) * 100
        else:
            usage["gpu_allocated_gb"] = 0.0
            usage["gpu_reserved_gb"] = 0.0
            usage["gpu_total_gb"] = 0.0
            usage["gpu_percent"] = 0.0
        
        # Update peaks
        self.peak_usage["ram"] = max(self.peak_usage["ram"], usage["ram_used_gb"])
        self.peak_usage["gpu"] = max(self.peak_usage["gpu"], usage["gpu_allocated_gb"])
        
        return usage
    
    def check_memory_alert(self) -> Optional[str]:
        """Controlla se memoria supera soglia."""
        usage = self.get_usage()
        
        if usage["gpu_allocated_gb"] > self.alert_threshold:
            return f" HIGH GPU USAGE: {usage['gpu_allocated_gb']:.1f}GB > {self.alert_threshold}GB"
        elif usage["ram_used_gb"] > self.alert_threshold:
            return f" HIGH RAM USAGE: {usage['ram_used_gb']:.1f}GB > {self.alert_threshold}GB"
        
        return None
    
    def print_status(self):
        """Stampa status memoria dettagliato."""
        usage = self.get_usage()
        
        print(f"\n{'='*50}")
        print("MEMORY STATUS")
        print(f"{'='*50}")
        print(f"RAM:  {usage['ram_used_gb']:.1f}GB / {usage['ram_total_gb']:.1f}GB ({usage['ram_percent']:.1f}%)")
        print(f"GPU:  {usage['gpu_allocated_gb']:.1f}GB / {usage['gpu_total_gb']:.1f}GB ({usage['gpu_percent']:.1f}%)")
        print(f"Peak: RAM {self.peak_usage['ram']:.1f}GB, GPU {self.peak_usage['gpu']:.1f}GB")
        
        alert = self.check_memory_alert()
        if alert:
            print(f"\n{alert}")
        
        print(f"{'='*50}")

# Instance globale
memory_monitor = MemoryMonitor()

def get_memory_usage() -> Dict[str, float]:
    """Shortcut per memoria."""
    return memory_monitor.get_usage()

def print_memory_status():
    """Shortcut per status memoria."""
    memory_monitor.print_status()

def aggressive_cleanup():
    """Cleanup aggressivo per Colab."""
    print("[CLEANUP] Performing aggressive memory cleanup...")
    
    # Garbage collection
    gc.collect()
    
    # GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    
    # Force garbage collection again
    gc.collect()
    
    print("[CLEANUP]  Memory cleanup completed")

# ==== Performance Tracking ====
class Timer:
    """Timer avanzato con statistiche."""
    
    def __init__(self, name: str = "Operation", track_memory: bool = True):
        self.name = name
        self.track_memory = track_memory
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        if self.track_memory:
            self.start_memory = get_memory_usage()
        print(f"[START] {self.name}...")
        return self
        
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        
        print(f"[DONE] {self.name} completed in {format_time(duration)}")
        
        if self.track_memory and self.start_memory:
            end_memory = get_memory_usage()
            ram_delta = end_memory["ram_used_gb"] - self.start_memory["ram_used_gb"]
            gpu_delta = end_memory["gpu_allocated_gb"] - self.start_memory["gpu_allocated_gb"]
            
            if abs(ram_delta) > 0.1 or abs(gpu_delta) > 0.1:
                print(f"[MEMORY] RAM: {ram_delta:+.2f}GB, GPU: {gpu_delta:+.2f}GB")

class PerformanceProfiler:
    """Profiler per tracking performance."""
    
    def __init__(self):
        self.timings = {}
        self.memory_snapshots = []
    
    def start_operation(self, name: str):
        """Inizia tracking operazione."""
        self.timings[name] = {
            "start": time.time(),
            "memory_start": get_memory_usage()
        }
    
    def end_operation(self, name: str):
        """Termina tracking operazione."""
        if name in self.timings:
            self.timings[name]["end"] = time.time()
            self.timings[name]["memory_end"] = get_memory_usage()
            self.timings[name]["duration"] = self.timings[name]["end"] - self.timings[name]["start"]
    
    def get_summary(self) -> Dict:
        """Ottieni summary performance."""
        summary = {}
        for name, data in self.timings.items():
            if "duration" in data:
                summary[name] = {
                    "duration": data["duration"],
                    "memory_delta_ram": data["memory_end"]["ram_used_gb"] - data["memory_start"]["ram_used_gb"],
                    "memory_delta_gpu": data["memory_end"]["gpu_allocated_gb"] - data["memory_start"]["gpu_allocated_gb"]
                }
        return summary
    
    def print_summary(self):
        """Stampa summary performance."""
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        summary = self.get_summary()
        for name, data in summary.items():
            print(f"{name:30s} {format_time(data['duration']):>10s} "
                  f"RAM: {data['memory_delta_ram']:+6.2f}GB "
                  f"GPU: {data['memory_delta_gpu']:+6.2f}GB")
        
        print(f"{'='*60}")

# ==== Data Processing Utilities ====
def chunk_iter(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """Divide iterabile in chunk con progress."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def safe_batch_process(items: List[T], process_func: callable, batch_size: int = 10, 
                      desc: str = "Processing", show_progress: bool = True) -> List:
    """Process items in batch con error handling."""
    results = []
    batches = list(chunk_iter(items, batch_size))
    
    iterator = tqdm(batches, desc=desc, leave=False) if show_progress else batches
    
    for batch in iterator:
        batch_results = []
        for item in batch:
            try:
                result = process_func(item)
                batch_results.append(result)
            except Exception as e:
                print(f"[ERROR] Processing failed for item: {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
        
        # Memory check ogni batch
        alert = memory_monitor.check_memory_alert()
        if alert:
            print(f"\n{alert}")
            aggressive_cleanup()
    
    return results

# ==== File Management per Colab ====
def save_json(path: str, obj: Any, backup: bool = True) -> None:
    """Salva JSON con backup per Colab."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup se file esistente
    if backup and path.exists():
        backup_path = path.with_suffix(f".backup_{int(time.time())}.json")
        path.rename(backup_path)
    
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)  # default=str per datetime, etc.

def load_json(path: str) -> Any:
    """Carica JSON con error handling."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARNING] File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error in {path}: {e}")
        return None

def cleanup_old_files(directory: str, pattern: str = "*.backup_*", days_old: int = 7):
    """Pulisce file vecchi."""
    directory = Path(directory)
    if not directory.exists():
        return
    
    cutoff_time = time.time() - (days_old * 24 * 3600)
    removed_count = 0
    
    for file_path in directory.glob(pattern):
        if file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
            removed_count += 1
    
    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} old backup files")

# ==== Progress Bar Utilities ====
def tqdm_notebook(*args, **kwargs):
    """Progress bar ottimizzata per Colab notebook."""
    # Force notebook display in Colab
    kwargs.setdefault('leave', False)
    kwargs.setdefault('ncols', 100)
    return tqdm(*args, **kwargs)

def progress_callback(current: int, total: int, desc: str = "Progress"):
    """Callback per progress manual."""
    percent = (current / total) * 100
    print(f"\r[{desc}] {current}/{total} ({percent:.1f}%)", end="", flush=True)
    if current == total:
        print()  # New line quando completo

# ==== Time Formatting ====
def format_time(seconds: float) -> str:
    """Formatta tempo in formato leggibile."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h:.0f}h {m:.0f}m"

def format_eta(start_time: float, current: int, total: int) -> str:
    """Calcola e formatta ETA."""
    if current == 0:
        return "unknown"
    
    elapsed = time.time() - start_time
    rate = current / elapsed
    remaining = (total - current) / rate
    
    return format_time(remaining)

# ==== System Info ====
def print_system_info():
    """Stampa info sistema completo per Colab."""
    print(f"\n{'='*60}")
    print("COLAB SYSTEM INFO")
    print(f"{'='*60}")
    
    # Python & Environment
    import sys
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    # GPU Info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU: Not available")
    
    # Memory status
    memory_monitor.print_status()
    
    # Disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage("/")
        disk_free = disk_usage.free / (1024**3)
        disk_total = disk_usage.total / (1024**3)
        print(f"Disk: {disk_free:.1f}GB free / {disk_total:.1f}GB total")
    except Exception:
        pass
    
    print(f"{'='*60}")

# ==== Auto Recovery ====
class AutoRecovery:
    """Auto recovery per operazioni lunghe."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, data: Dict, name: str):
        """Salva checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{name}_{int(time.time())}.json"
        save_json(checkpoint_file, data)
        print(f"[CHECKPOINT] Saved: {checkpoint_file.name}")
    
    def load_latest_checkpoint(self, name_pattern: str) -> Optional[Dict]:
        """Carica checkpoint più recente."""
        checkpoints = list(self.checkpoint_dir.glob(f"{name_pattern}_*.json"))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        data = load_json(latest)
        print(f"[RECOVERY] Loaded: {latest.name}")
        return data

# ==== Test Utilities ====
def run_colab_diagnostics():
    """Esegue diagnostici completi per Colab."""
    print("Running Colab diagnostics...")
    
    # System info
    print_system_info()
    
    # Memory test
    print("\n[TEST] Memory allocation test...")
    try:
        test_tensor = torch.randn(1000, 1000).cuda() if torch.cuda.is_available() else torch.randn(1000, 1000)
        print("[TEST]  Memory allocation successful")
        del test_tensor
        aggressive_cleanup()
    except Exception as e:
        print(f"[TEST]  Memory allocation failed: {e}")
    
    # Performance test
    print("\n[TEST] Performance test...")
    with Timer("Performance test"):
        time.sleep(0.1)  # Simulate work
    
    print("\n Colab diagnostics completed!")

# ==== Initialize ====
if __name__ == "__main__":
    print("Testing Colab utilities...")
    run_colab_diagnostics()
else:
    # Auto-setup when imported
    set_seed(42)
    print("[UTILS]  Colab utilities loaded")