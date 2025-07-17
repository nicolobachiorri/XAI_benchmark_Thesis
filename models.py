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
    #"bert-base":        "nlptown/bert-base-multilingual-uncased-sentiment",
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
        # FIX: Prova prima con attn_implementation="eager" per evitare warning
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                attn_implementation="eager"  # FIX per warning attention
            )
            print(f"[DEBUG] {model_key}: caricato con attn_implementation='eager'")
        except (ValueError, TypeError) as e:
            # Fallback se attn_implementation non è supportato
            print(f"[DEBUG] {model_key}: fallback loading (attn_implementation non supportato)")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        
        # Verifica che il modello abbia il numero corretto di labels
        if hasattr(model.config, 'num_labels'):
            actual_labels = model.config.num_labels
            if actual_labels != num_labels:
                print(f"[INFO] {model_key}: ha {actual_labels} labels, richieste {num_labels}. "
                      f"Utilizzando configurazione del modello.")
        
        # FIX: Assicura che sia in modalità eval
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Errore caricamento modello {model_key}: {e}")
        raise


# ==== Test di compatibilità ====
if __name__ == "__main__":
    """Test veloce per verificare che tutti i modelli si carichino correttamente."""
    print("Testing model loading...")
    print("=" * 50)
    
    for model_key in MODELS.keys():
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
            
            # Info sul modello
            print(f"  Labels: {model.config.num_labels}")
            print(f"  Architecture: {model.__class__.__name__}")
            
            # Test inference veloce
            print("  Testing inference...", end="")
            test_text = "This is a test sentence."
            encoded = tokenizer(test_text, return_tensors="pt", max_length=50, truncation=True)
            outputs = model(**encoded)
            logits = outputs.logits
            print(f" ✓ (logits shape: {logits.shape})")
            
            print(f"  SUCCESS: {model_key}")
            
        except Exception as e:
            print(f"  FAILED: {model_key}")
            print(f"    Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Model loading test completed!")