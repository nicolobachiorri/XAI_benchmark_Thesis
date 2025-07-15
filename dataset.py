"""
Simple IMDB Sentiment Dataset Preparation (columns: text, label)
----------------------------------------------------------------

Questo script prepara il dataset IMDB (formato CSV) per il fine‑tuning di un
modello Transformer di sentiment analysis. È pensato per studenti di
statistica alle primissime armi:

* Intero flusso in un singolo file Python, senza classi sofisticate
* Commenti dettagliati e variabili facili da modificare
* Compatibile con colonne **text** (recensione) e **label**
  ("positive"/"negative" *oppure* 1/0)

Autore: Esempio Studente di Statistica
Data: 15 luglio 2025
"""

# ==== 1. Librerie ====
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ==== 2. Parametri Facili da Cambiare ====
DATA_DIR = Path(".")          # cartella che contiene i CSV
TRAIN_FILE = "Train.csv"       # rinomina se necessario
TEST_FILE  = "Test.csv"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
RANDOM_STATE = 42

# ==== 3. Caricamento Dataset ====
train_df = pd.read_csv(DATA_DIR / TRAIN_FILE)
test_df  = pd.read_csv(DATA_DIR / TEST_FILE)

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# ==== 4. Pulizia / Mapping etichette ====
# Ci aspettiamo colonne: text (stringa) e label (string/int)

LABEL_MAP = {"negative": 0, "positive": 1, "0": 0, "1": 1}

def to_int_label(x):
    """Converte label in intero 0/1 se non lo è già."""
    if isinstance(x, (int, float)):
        return int(x)
    try:
        return LABEL_MAP[x.strip().lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Label sconosciuta: {x}")

for df in (train_df, test_df):
    df["label"] = df["label"].apply(to_int_label)

# Suddivisione train/validation 90‑10 stratificata
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df["label"],
)

print(f"Training: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

# ==== 5. Tokenizzazione ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(text_list):
    """Tokenizza una lista di stringhe, restituendo dict pronto per torch."""
    return tokenizer(
        text_list,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

# ==== 6. Torch Dataset ====
class IMDBDataset(Dataset):
    def __init__(self, dataframe):
        encodings = encode_texts(list(dataframe["text"]))
        self.input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_masks = torch.tensor(encodings["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(dataframe["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels":         self.labels[idx],
        }

# ==== 7. DataLoader ====
train_loader = DataLoader(IMDBDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(IMDBDataset(val_df),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(IMDBDataset(test_df),  batch_size=BATCH_SIZE)

# ==== 8. Sanity Check ====
if __name__ == "__main__":
    batch = next(iter(train_loader))
    shapes = {k: v.shape for k, v in batch.items()}
    print("Batch tensor shapes:", shapes)
