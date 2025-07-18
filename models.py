"""
models.py – Gestione modelli pre-trained per sentiment analysis (GPU ENABLED)
============================================================

AGGIORNAMENTO: Forza uso GPU quando disponibile
"""

from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==== Mappa modelli pre-trained ====
MODELS: Dict[str, str] = {
    "distilbert":       "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "roberta-large":    "siebert/sentiment-roberta-large-english", 
    "bert-large":       "assemblyai/bert-large-uncased-sst2",
    "roberta-base":     "AnkitAI/reviews-roberta-base-sentiment-analysis",
    "tinybert":         "Harsha901/tinybert-imdb-sentiment-analysis-model",
}

# ==== GPU Configuration ====
def get_device():
    """Restituisce il device ottimale (GPU se disponibile, altrimenti CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] Using: {torch.cuda.get_device_name()}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return device
    else:
        print("[CPU] CUDA not available, using CPU")
        return torch.device("cpu")

# Device globale
DEVICE = get_device()

def load_tokenizer(model_key: str):
    """Restituisce il tokenizer associato al modello."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato. Disponibili: {list(MODELS.keys())}")
    
    model_name = MODELS[model_key]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # FIX: Assicura che ci sia un pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
            
        return tokenizer
    except Exception as e:
        print(f"Errore caricamento tokenizer {model_key}: {e}")
        raise

def load_model(model_key: str, num_labels: int = 2):
    """Carica il modello pre-addestrato per sentiment analysis SU GPU."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato. Disponibili: {list(MODELS.keys())}")
    
    model_name = MODELS[model_key]
    
    try:
        # Carica modello
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                attn_implementation="eager",
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32  # FP16 su GPU per efficienza
            )
            print(f"[DEBUG] {model_key}: caricato con attn_implementation='eager'")
        except (ValueError, TypeError) as e:
            print(f"[DEBUG] {model_key}: fallback loading (attn_implementation non supportato)")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
            )
        
        # FORZA modello su GPU
        model = model.to(DEVICE)
        model.eval()
        
        # Verifica che sia effettivamente su GPU
        if DEVICE.type == "cuda":
            print(f"[GPU] Model {model_key} loaded on {next(model.parameters()).device}")
            # Mostra uso memoria GPU
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[GPU] Memory allocated: {allocated:.2f}GB")
        
        # Verifica configurazione
        if hasattr(model.config, 'num_labels'):
            actual_labels = model.config.num_labels
            if actual_labels != num_labels:
                print(f"[INFO] {model_key}: ha {actual_labels} labels, richieste {num_labels}. "
                      f"Utilizzando configurazione del modello.")
        
        return model
        
    except Exception as e:
        print(f"Errore caricamento modello {model_key}: {e}")
        raise

def move_batch_to_device(batch):
    """Sposta un batch di dati sul device corretto."""
    if isinstance(batch, dict):
        return {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in batch.items()}
    elif hasattr(batch, 'to'):
        return batch.to(DEVICE)
    else:
        return batch

def move_batch_to_device(batch):
    """Sposta un batch di dati sul device corretto."""
    if isinstance(batch, dict):
        return {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in batch.items()}
    elif hasattr(batch, 'to'):
        return batch.to(DEVICE)
    else:
        return batch

def get_gpu_memory_usage():
    """Restituisce uso memoria GPU in GB."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
        }
    return {"allocated": 0, "cached": 0, "max_allocated": 0}

# ==== Test di compatibilità ====
if __name__ == "__main__":
    """Test veloce per verificare che tutti i modelli si carichino correttamente su GPU."""
    print("Testing model loading with GPU...")
    print("=" * 50)
    
    # Info GPU iniziale
    print(f"Device: {DEVICE}")
    gpu_info = get_gpu_memory_usage()
    print(f"GPU Memory initial: {gpu_info['allocated']:.2f}GB allocated")
    
    for model_key in list(MODELS.keys())[:2]:  # Testa solo i primi 2 per velocità
        try:
            print(f"\nTesting {model_key}...")
            print(f"  Model name: {MODELS[model_key]}")
            
            # Test tokenizer
            print("  Loading tokenizer...", end="")
            tokenizer = load_tokenizer(model_key)
            print(" ✓")
            
            # Test model
            print("  Loading model...", end="")
            model = load_model(model_key)
            print(" ✓")
            
            # Verifica device
            model_device = next(model.parameters()).device
            print(f"  Model device: {model_device}")
            
            # Test inference con GPU
            print("  Testing GPU inference...", end="")
            test_text = "This is a test sentence."
            encoded = tokenizer(test_text, return_tensors="pt", max_length=50, truncation=True)
            
            # IMPORTANTE: Sposta input su GPU
            encoded = move_batch_to_device(encoded)
            
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
            
            print(f" ✓ (logits shape: {logits.shape}, device: {logits.device})")
            
            # Mostra memoria GPU
            gpu_info = get_gpu_memory_usage()
            print(f"  GPU Memory: {gpu_info['allocated']:.2f}GB allocated")
            
            print(f"  SUCCESS: {model_key} on {model_device}")
            
            # Cleanup per test successivo
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  FAILED: {model_key}")
            print(f"    Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Model loading test completed!")
    
    # Memory finale
    gpu_final = get_gpu_memory_usage()
    print(f"Final GPU Memory: {gpu_final['allocated']:.2f}GB allocated")