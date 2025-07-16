"""
Simple IMDB Sentiment Dataset Preparation (columns: text, label)
----------------------------------------------------------------

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

# Parametri di tokenizzazione ottimizzati per modelli pre-trained
MAX_LENGTH = 512  # Aumentato da 128 per catturare più contesto
BATCH_SIZE = 16
RANDOM_STATE = 42

# ==== 3. Caricamento Dataset ====
try:
    train_df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    test_df  = pd.read_csv(DATA_DIR / TEST_FILE)
    print(f"Dataset caricato: Train {len(train_df)} righe, Test {len(test_df)} righe")
except FileNotFoundError as e:
    print(f"Errore caricamento dataset: {e}")
    print("Assicurati che Train.csv e Test.csv siano nella cartella corrente")
    raise

# Verifica struttura dataset
required_columns = ["text", "label"]
for df_name, df in [("Train", train_df), ("Test", test_df)]:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset {df_name} manca colonne: {missing_cols}")
    print(f"{df_name}: colonne corrette {list(df.columns)}")

# ==== 4. Pulizia / Mapping etichette ====
# Ci aspettiamo colonne: text (stringa) e label (string/int)

LABEL_MAP = {
    "negative": 0, "positive": 1, 
    "neg": 0, "pos": 1,
    "0": 0, "1": 1,
    0: 0, 1: 1
}

def to_int_label(x):
    """Converte label in intero 0/1 se non lo è già."""
    if isinstance(x, (int, float)):
        return int(x)
    try:
        if isinstance(x, str):
            return LABEL_MAP[x.strip().lower()]
        return LABEL_MAP[x]
    except (KeyError, AttributeError):
        raise ValueError(f"Label sconosciuta: {x}. Valori supportati: {list(LABEL_MAP.keys())}")

# Pulizia e conversione labels
for df_name, df in [("Train", train_df), ("Test", test_df)]:
    # Rimuovi righe con testo vuoto o NaN
    initial_len = len(df)
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"].str.strip() != ""]
    if len(df) < initial_len:
        print(f"WARN {df_name}: rimossi {initial_len - len(df)} esempi con testo vuoto")
    
    # Converti labels
    df["label"] = df["label"].apply(to_int_label)
    
    # Verifica distribuzione classi
    label_counts = df["label"].value_counts().sort_index()
    print(f"{df_name} distribuzione: {dict(label_counts)}")

# Aggiorna i dataframe globali dopo pulizia
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Suddivisione train/validation 90‑10 stratificata
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df["label"],
)

print(f"Split completato: Training {len(train_df)}, Validation {len(val_df)}, Test {len(test_df)}")

# ==== 5. Tokenizzazione Flessibile ====
def get_tokenizer_for_model(model_name: str):
    """Carica il tokenizer appropriato per il modello specificato."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Aggiungi pad token se non presente (per alcuni modelli)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"WARN Errore caricamento tokenizer per {model_name}: {e}")
        # Fallback a tokenizer generico
        return AutoTokenizer.from_pretrained("bert-base-uncased")

def encode_texts(text_list, tokenizer, max_length=MAX_LENGTH):
    """Tokenizza una lista di stringhe, restituendo dict pronto per torch."""
    return tokenizer(
        text_list,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

# ==== 6. Torch Dataset Migliorato ====
class IMDBDataset(Dataset):
    def __init__(self, dataframe, tokenizer=None, max_length=MAX_LENGTH):
        """
        Dataset IMDB ottimizzato per modelli pre-trained.
        
        Args:
            dataframe: DataFrame con colonne 'text' e 'label'
            tokenizer: Tokenizer da usare (se None, usa bert-base-uncased)
            max_length: Lunghezza massima sequenze
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.max_length = max_length
        
        # Usa tokenizer fornito o default
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer
            
        # Pre-tokenizza tutto il dataset per efficienza
        print(f"Tokenizzando {len(self.dataframe)} esempi...")
        encodings = encode_texts(list(self.dataframe["text"]), self.tokenizer, max_length)
        
        self.input_ids = encodings["input_ids"]
        self.attention_masks = encodings["attention_mask"]
        self.labels = torch.tensor(self.dataframe["label"].values, dtype=torch.long)
        
        print(f"Dataset tokenizzato: {len(self)} esempi")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels":         self.labels[idx],
        }

# ==== 7. Factory Functions per DataLoader ====
def create_dataloaders(model_name=None, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """
    Crea DataLoader ottimizzati per un modello specifico.
    
    Args:
        model_name: Nome del modello per scegliere il tokenizer appropriato
        batch_size: Dimensione batch
        max_length: Lunghezza massima sequenze
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Carica tokenizer appropriato
    if model_name:
        tokenizer = get_tokenizer_for_model(model_name)
    else:
        tokenizer = None
    
    # Crea datasets
    train_dataset = IMDBDataset(train_df, tokenizer, max_length)
    val_dataset = IMDBDataset(val_df, tokenizer, max_length) 
    test_dataset = IMDBDataset(test_df, tokenizer, max_length)
    
    # Crea dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ==== 8. DataLoader Default (mantenuto per compatibilità) ====
# Usa tokenizer generico per compatibilità con codice esistente
default_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_loader = DataLoader(IMDBDataset(train_df, default_tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(IMDBDataset(val_df, default_tokenizer),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(IMDBDataset(test_df, default_tokenizer),  batch_size=BATCH_SIZE, shuffle=False)

# ==== 9. Sanity Check ====
if __name__ == "__main__":
    print("\n=== SANITY CHECK ===")
    
    # Test caricamento base
    batch = next(iter(train_loader))
    shapes = {k: v.shape for k, v in batch.items()}
    print(f"Batch tensor shapes: {shapes}")
    
    # Test con modelli specifici
    test_models = [
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "siebert/sentiment-roberta-large-english"
    ]
    
    for model_name in test_models:
        try:
            print(f"\nTest con {model_name}:")
            train_dl, val_dl, test_dl = create_dataloaders(model_name, batch_size=4)
            batch = next(iter(train_dl))
            print(f"Batch shapes: {[f'{k}: {v.shape}' for k, v in batch.items()]}")
        except Exception as e:
            print(f"Errore con {model_name}: {e}")
    
    print("\nSanity check completato!")