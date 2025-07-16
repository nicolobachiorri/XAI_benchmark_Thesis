"""
utils.py – Utility semplici per il progetto XAI Benchmark
========================================================

Funzioni base riusabili senza troppa complessità.

Funzionalità:
1. set_seed() - fissa seed per riproducibilità
2. chunk_iter() - divide lista in pezzi
3. save_json/load_json() - salva/carica JSON
4. tqdm_wrapper() - progress bar per DataLoader
5. format_time() - formatta secondi in h/m/s
6. check_gpu() - info base GPU
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, List, TypeVar

import numpy as np
import torch
from tqdm import tqdm

T = TypeVar("T")

# ==== 1. Seed per riproducibilità ====
def set_seed(seed: int) -> None:
    """Fissa seed per riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==== 2. Divide lista in chunk ====
def chunk_iter(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """Divide un iterabile in liste di lunghezza size."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# ==== 3. JSON semplice ====
def save_json(path: str, obj: Any) -> None:
    """Salva oggetto in JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Any:
    """Carica oggetto da JSON."""
    with open(path, "r") as f:
        return json.load(f)

# ==== 4. Progress bar ====
def tqdm_wrapper(dataloader, desc: str = "Progress"):
    """Progress bar per DataLoader."""
    return tqdm(dataloader, desc=desc, leave=False)

# ==== 5. Formatta tempo ====
def format_time(seconds: float) -> str:
    """Formatta secondi in formato leggibile."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h:.0f}h {m:.0f}m"

# ==== 6. Info GPU base ====
def check_gpu() -> dict:
    """Info base su GPU."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "count": torch.cuda.device_count(),
        "current": torch.cuda.current_device(),
        "name": torch.cuda.get_device_name(),
    }

# ==== 7. Context manager per timing ====
class Timer:
    """Misura tempo di esecuzione."""
    def __init__(self, name: str = "Operazione"):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        print(f"Inizio: {self.name}")
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start
        print(f"Completato: {self.name} in {format_time(duration)}")

if __name__ == "__main__":
    # Test rapido
    set_seed(42)
    print("GPU:", check_gpu())
    print("Tempo:", format_time(3661))
    
    with Timer("Test"):
        time.sleep(0.1)