"""
XAI Benchmark - Utilities Module
Funzioni di supporto essenziali per il progetto
"""

import os
import random
import shutil
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Union
import logging

logger = logging.getLogger(__name__)


# ================== SEED MANAGEMENT ==================

def set_global_seed(seed: int = 42):
    """Imposta seed globale per riproducibilità"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global seed impostato a {seed}")


# ================== PATH UTILITIES ==================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Assicura che una directory esista"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_filename(filename: str) -> str:
    """Rende un filename sicuro per il filesystem"""
    unsafe_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    return safe_name.strip()


# ================== CACHE UTILITIES ==================

def get_cache_size(cache_dir: Union[str, Path]) -> str:
    """Calcola dimensione della cache in formato leggibile"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return "0 B"
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    
    # Converti in formato leggibile
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.1f} TB"


def clear_cache(cache_dir: Union[str, Path]):
    """Rimuove tutta la cache"""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        cache_path.mkdir(exist_ok=True)
        logger.info("Cache rimossa")


# ================== FILE UTILITIES ==================

def save_json(data: Dict[str, Any], filepath: Union[str, Path]):
    """Salva dizionario in JSON"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Carica dizionario da JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]):
    """Salva oggetto in pickle"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Carica oggetto da pickle"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ================== SYSTEM INFO ==================

def get_system_info() -> Dict[str, Any]:
    """Restituisce informazioni base del sistema"""
    import platform
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


def print_system_info():
    """Stampa informazioni sistema"""
    info = get_system_info()
    
    print("\nSYSTEM INFO:")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['torch_version']}")
    print(f"CUDA: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"GPUs: {info['gpu_count']}")


# ================== TEST ==================

def test_utils():
    """Test delle utility"""
    print("TEST: Utils module")
    
    try:
        set_global_seed(42)
        print("SUCCESS: Seed management")
        
        test_dir = ensure_dir("./test_utils")
        print("SUCCESS: Path utilities")
        
        cache_size = get_cache_size("./")
        print(f"SUCCESS: Cache utilities - {cache_size}")
        
        print_system_info()
        print("SUCCESS: System info")
        
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    test_utils()