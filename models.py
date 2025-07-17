"""
models.py – Gestione modelli pre-trained per sentiment analysis
============================================================

Carica modelli Transformer già addestrati da Hugging Face Hub.
Tutti i modelli sono configurati per classificazione binaria (pos/neg).
NON viene effettuato alcun fine-tuning - solo caricamento dei modelli pre-trained.
"""

from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==== Mappa modelli pre-trained ====
MODELS: Dict[str, str] = {
    # Modelli già fine-tuned per sentiment analysis
    "distilbert":       "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "roberta-large":    "siebert/sentiment-roberta-large-english", 
    "bert-large":       "assemblyai/bert-large-uncased-sst2",
    "roberta-base":     "AnkitAI/reviews-roberta-base-sentiment-analysis",
   # "bert-base":        "Asteroid-Destroyer/bert-amazon-sentiment",
    "tinybert":         "Harsha901/tinybert-imdb-sentiment-analysis-model",
}

# ==== Funzioni di caricamento ====

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
    """Carica il modello pre-addestrato per sentiment analysis."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato. Disponibili: {list(MODELS.keys())}")
    
    model_name = MODELS[model_key]
    
    try:
        # FIX: Aggiungi ignore_mismatched_sizes per evitare errori di dimensioni
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # FIX per errori di dimensioni mismatch
        )
        
        # Verifica che il modello abbia il numero corretto di labels
        if hasattr(model.config, 'num_labels') and model.config.num_labels != num_labels:
            print(f"[WARNING] Il modello {model_key} ha {model.config.num_labels} labels, "
                  f"richieste {num_labels}. Utilizzando configurazione del modello.")
                    
        model.eval()  # FIX: Assicura che sia in modalità eval
        return model
        
    except Exception as e:
        print(f"Errore caricamento modello {model_key}: {e}")
        raise


# ==== Test di compatibilità ====
if __name__ == "__main__":
    """Test veloce per verificare che tutti i modelli si carichino correttamente."""
    print("Testing model loading...")
    for model_key in MODELS.keys():
        try:
            print(f"Loading {model_key}...")
            tokenizer = load_tokenizer(model_key)
            model = load_model(model_key)
            print(f"✓ {model_key}: {model.config.num_labels} labels")
        except Exception as e:
            print(f"✗ {model_key}: Error - {e}")
    
    print("Model loading test completed!")