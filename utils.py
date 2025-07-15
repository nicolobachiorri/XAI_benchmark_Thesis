"""
utils.py – Utility comuni per il progetto XAI Benchmark
======================================================

File pensato per raccogliere **piccole funzioni riusabili** senza dipendenze
pesanti, in modo da non ripetere codice in più script.

Funzionalità principali
-----------------------
1. **set_seed(seed)** – fissa seed per *random*, *numpy*, *torch* (+ CUDA).
2. **chunk_iter(iterable, size)** – genera mini‑batch di grandezza fissa.
3. **save_json(path, obj) / load_json(path)** – lettura/scrittura JSON semplice
   (gestisce *Path* o stringa). Usa indent=2 per leggere comodamente i file.
4. **tqdm_wrapper(dataloader)** – progress bar già configurata per loop su
   *torch.utils.data.DataLoader*.


"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence, TypeVar

import numpy as np
import torch
from tqdm import tqdm

__all__: List[str] = [
    "set_seed",
    "chunk_iter",
    "save_json",
    "load_json",
    "tqdm_wrapper",
]

T = TypeVar("T")

# ==== 1. Reproducibilità globale ====

def set_seed(seed: int) -> None:
    """Fissa tutti i generatori di random per garantire esperimenti riproducibili."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==== 2. Suddivisione iterabile ====

def chunk_iter(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """Divide un iterabile in liste di lunghezza `size` (ultima più corta)."""
    chunk: List[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ==== 3. I/O JSON semplificato ====

def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def save_json(path: str | Path, obj: Any) -> None:
    """Salva `obj` (serializzabile JSON) su disco con indentazione leggibile."""
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Carica un oggetto da file JSON."""
    p = _as_path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ==== 4. Progress bar DataLoader ====

def tqdm_wrapper(dataloader: torch.utils.data.DataLoader, desc: str | None = None):
    """Ritorna un `tqdm` configurato per iterare sul dataloader."""
    return tqdm(dataloader, total=len(dataloader), desc=desc or "Batches", leave=False)
