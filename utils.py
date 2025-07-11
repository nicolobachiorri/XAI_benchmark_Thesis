import os, random, pickle, torch, numpy as np

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cache_save(obj, path):
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return True
    except:
        return False

def cache_load(path):
    try:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except:
        pass
    return None