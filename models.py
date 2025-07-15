"""
models.py – Gestione modelli e fine‑tuning IMDB
-------------------------------------------------

* 8 modelli pre‑addestrati definiti in `MODELS`
* Per tutti si aggiunge automaticamente la **linear head** di classificazione
  (via `AutoModelForSequenceClassification` di Hugging Face)
* Fine‑tuning uniforme con `Trainer`; per il modello già fine‑tuned di Siebert
  si effettua solo il caricamento. 
* Opzione `freeze_backbone=True` per sperimentare l’addestramento **solo**
  della testa lineare, come ablation o per risparmiare GPU.

Dipendenze minime: transformers ≥ 4, torch ≥ 2, accelerate.
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

# ==== 2. Mappa modelli ====
MODELS: Dict[str, str] = {
    # BERT‑family
    "tinybert":     "huawei-noah/TinyBERT_General_4L_312D",
    "distilbert":   "distilbert/distilbert-base-uncased",
    "bert-base":    "google-bert/bert-base-uncased",
    "bert-large":   "google-bert/bert-large-uncased",
    # RoBERTa (già fine‑tuned)
    "roberta-large": "siebert/sentiment-roberta-large-english",
    # XLM‑RoBERTa
    "xlm-roberta":   "FacebookAI/xlm-roberta-large",
    # DeBERTa‑family
    "deberta-v2-xl": "microsoft/deberta-v2-xlarge",
    "mdeberta-v3":   "microsoft/mdeberta-v3-base",
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
    """Carica il modello e (opzionalmente) congela il backbone."""
    model_name = MODELS[model_key]

    # Modello già fine‑tuned (lo carichiamo com’è)
    if model_key == "roberta-large":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    # Congela tutto il backbone se richiesto
    if freeze_backbone and model_key != "roberta-large":
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model

# ==== 5. Funzione di fine‑tuning ====

def fine_tune(
    model_key: str,
    train_dataset,
    val_dataset,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    freeze_backbone: bool = False,
):
    """Allena (o carica) il modello indicato e restituisce i pesi allenati."""

    # Caricamento modello + tokenizer
    tokenizer = load_tokenizer(model_key)
    model = load_model(model_key, freeze_backbone=freeze_backbone)

    # Se modello già fine‑tuned e non vogliamo ulteriori passi, usciamo subito
    if model_key == "roberta-large" and not freeze_backbone:
        print(f"[{model_key}] modello già allenato: nessun fine‑tuning eseguito.")
        return model

    run_name = f"{model_key}{'-frozen' if freeze_backbone else ''}"
    output_subdir = OUTPUT_DIR / run_name

    # ---------------------------------------------------------------
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
            seed=SEED,
        )
    # ---------------------------------------------------------------


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print(f"[{run_name}] avvio fine‑tuning ({num_epochs} epoche)…")
    trainer.train()
    trainer.save_model(output_subdir)
    print(f"[{run_name}] checkpoint salvato in {output_subdir}")

    return trainer.model

# ==== 6. Eseguibile standalone (quick test) ====
if __name__ == "__main__":
    """Allenamento veloce su 1-2 batch per verificare che tutto funzioni."""
    import dataset  # importa il tuo dataset.py
    import torch

    # Usa subset piccola per smoke test
    train_small = torch.utils.data.Subset(dataset.IMDBDataset(dataset.train_df), range(128))
    val_small   = torch.utils.data.Subset(dataset.IMDBDataset(dataset.val_df),   range(128))

    _ = fine_tune("tinybert", train_small, val_small, num_epochs=1, freeze_backbone=True)
    print("Smoke test completato ")
