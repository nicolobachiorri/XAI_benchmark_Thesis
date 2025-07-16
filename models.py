"""
models.py – Gestione modelli e fine‑tuning IMDB
-------------------------------------------------

* 6 modelli pre‑addestrati definiti in `MODELS` (tutti già fine-tuned per sentiment analysis)
* Tutti i modelli sono già ottimizzati per classificazione binaria di sentiment
* Per i modelli già fine-tuned, si effettua solo il caricamento senza ulteriore training
* Opzione `freeze_backbone=True` mantenuta per compatibilità ma non necessaria
  dato che i modelli sono già pronti all'uso

Dipendenze minime: transformers ≥ 4, torch ≥ 2, accelerate.
"""

# ==== 1. Librerie ====
from pathlib import Path
from typing import Dict
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ==== 2. Mappa modelli aggiornata ====
MODELS: Dict[str, str] = {
    # Modelli già fine-tuned per sentiment analysis
    "distilbert":       "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "roberta-large":    "siebert/sentiment-roberta-large-english", 
    "bert-large":       "assemblyai/bert-large-uncased-sst2",
    "roberta-base":     "AnkitAI/reviews-roberta-base-sentiment-analysis",
    "bert-base":        "Asteroid-Destroyer/bert-amazon-sentiment",
    "tinybert":         "Harsha901/tinybert-imdb-sentiment-analysis-model",
}

# ==== 3. Parametri globali ====
OUTPUT_DIR     = Path("checkpoints")
NUM_EPOCHS     = 2
LEARNING_RATE  = 2e-5
BATCH_SIZE     = 16
SEED           = 42
OUTPUT_DIR.mkdir(exist_ok=True)

# ==== 4. Helper di caricamento ====

def load_tokenizer(model_key: str):
    """Restituisce il tokenizer associato al modello."""
    return AutoTokenizer.from_pretrained(MODELS[model_key])


def load_model(model_key: str, num_labels: int = 2, freeze_backbone: bool = False):
    """Carica il modello pre-addestrato per sentiment analysis."""
    model_name = MODELS[model_key]
    
    # Tutti i modelli sono già fine-tuned per sentiment analysis
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Verifica che il modello abbia il numero corretto di labels
    if hasattr(model.config, 'num_labels') and model.config.num_labels != num_labels:
        print(f"[WARNING] Il modello {model_key} ha {model.config.num_labels} labels, "
              f"richieste {num_labels}. Utilizzando configurazione del modello.")
    
    # Congela tutto il backbone se richiesto (per compatibilità)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any(classifier_name in name for classifier_name in ["classifier", "head", "pooler"]):
                param.requires_grad = False
                
    return model

# ==== 5. Funzione di fine‑tuning (modificata per modelli pre-trained) ====

def fine_tune(
    model_key: str,
    train_dataset,
    val_dataset,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    freeze_backbone: bool = False,
):
    """Carica il modello pre-addestrato (opzionalmente esegue ulteriore fine-tuning)."""

    # Caricamento modello + tokenizer
    tokenizer = load_tokenizer(model_key)
    model = load_model(model_key, freeze_backbone=freeze_backbone)

    print(f"[{model_key}] Modello già fine-tuned per sentiment analysis caricato.")
    
    # Se non è richiesto ulteriore training, restituisci il modello così com'è
    if not freeze_backbone and num_epochs == 0:
        print(f"[{model_key}] Nessun ulteriore fine-tuning richiesto.")
        return model

    # Esegui ulteriore fine-tuning se richiesto
    if num_epochs > 0:
        run_name = f"{model_key}{'-frozen' if freeze_backbone else ''}-retrained"
        output_subdir = OUTPUT_DIR / run_name

        # TrainingArguments: tentativo "ricco" + fallback "compatto"
        try:
            args = TrainingArguments(
                output_dir=output_subdir.as_posix(),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=lr,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_steps=50,
                seed=SEED,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                report_to="none",
                run_name=run_name,
            )
        except TypeError as e:
            # log minimal info, poi ricadi su versione base
            print("[WARN] TrainingArguments ridotti, motivo:", e)
            args = TrainingArguments(
                output_dir=output_subdir.as_posix(),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=lr,
                logging_steps=50,
                report_to="none", 
                seed=SEED,
            )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )

        print(f"[{run_name}] avvio ulteriore fine‑tuning ({num_epochs} epoche)…")
        trainer.train()
        trainer.save_model(output_subdir)
        print(f"[{run_name}] checkpoint salvato in {output_subdir}")

        return trainer.model
    
    return model

# ==== 6. Eseguibile standalone (quick test) ====
if __name__ == "__main__":
    """Test veloce per verificare che tutti i modelli si carichino correttamente."""
    import dataset  # importa il tuo dataset.py
    import torch

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