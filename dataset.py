"""
dataset.py - Dataset IMDB con clustering intelligente per Google Colab (UPDATED)
================================================================================

AGGIORNAMENTI:
1. Clustering KMeans aggiornato: K=80 cluster, 5 osservazioni per cluster
2. Dataset finale: sempre 400 esempi rappresentativi (80 × 5 = 400)
3. DataLoader ottimizzati per Colab (num_workers=0)
4. Sampling stratificato per mantenere balance
5. Cache del dataset clusterizzato
6. Memory-efficient processing

STRATEGIA CLUSTERING AGGIORNATA:
- KMeans con K=80 cluster sui text embeddings
- 5 esempi per cluster (mix pos/neg quando possibile)  
- Risultato: 400 esempi rappresentativi del dataset completo

"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import Tuple, List, Optional
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import models

# ==== Parametri AGGIORNATI ====
DATA_DIR = Path(".")
TEST_FILE = "Test.csv"  # Solo test file
CACHE_DIR = Path("dataset_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Parametri clustering AGGIORNATI
N_CLUSTERS = 80                          # Ridotto da 100 a 80
SAMPLES_PER_CLUSTER = 5                  # Aumentato da 4 a 5
TARGET_DATASET_SIZE = N_CLUSTERS * SAMPLES_PER_CLUSTER  # 80 × 5 = 400

# Parametri tokenizzazione
MAX_LENGTH = 512
BATCH_SIZE = 16
RANDOM_STATE = 42

print(f"[DATASET] UPDATED clustering: {N_CLUSTERS} clusters × {SAMPLES_PER_CLUSTER} samples = {TARGET_DATASET_SIZE} total")

# ==== Funzioni Clustering (aggiornate) ====
def create_text_embeddings(texts: List[str], max_features: int = 5000) -> np.ndarray:
    """Crea embeddings TF-IDF per clustering."""
    print(f"[EMBEDDING] Creating TF-IDF embeddings for {len(texts)} texts...")
    
    # TF-IDF con parametri ottimizzati per più cluster
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=2,            # ignora parole troppo rare
        max_df=0.95,         # ignora parole troppo comuni
        sublinear_tf=True    # scaling logaritmico
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Normalizza per KMeans
    scaler = StandardScaler(with_mean=False)  # sparse matrix compatible
    embeddings = scaler.fit_transform(tfidf_matrix)
    
    print(f"[EMBEDDING] ✓ Shape: {embeddings.shape}")
    return embeddings.toarray(), vectorizer

def perform_clustering(embeddings: np.ndarray, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """Applica KMeans clustering con parametri aggiornati."""
    print(f"[CLUSTERING] KMeans with {n_clusters} clusters (updated from 100)...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
        init='k-means++',    # Inizializzazione migliorata per più cluster
        algorithm='lloyd'    # Algoritmo standard per stabilità
    )
    
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Verifica distribuzione cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"[CLUSTERING] ✓ {len(unique_labels)} clusters created")
    print(f"[CLUSTERING]   Cluster sizes: min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}")
    
    return cluster_labels

def sample_from_clusters(df: pd.DataFrame, cluster_labels: np.ndarray, 
                        samples_per_cluster: int = SAMPLES_PER_CLUSTER) -> pd.DataFrame:
    """Campiona esempi rappresentativi da ogni cluster (aggiornato per 5 samples)."""
    print(f"[SAMPLING] Extracting {samples_per_cluster} samples per cluster (updated)...")
    
    sampled_indices = []
    cluster_stats = {"empty": 0, "insufficient": 0, "full": 0}
    
    for cluster_id in range(N_CLUSTERS):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            cluster_stats["empty"] += 1
            print(f"[WARNING] Cluster {cluster_id} is empty")
            continue
            
        # Separa per classe
        cluster_df = df.iloc[cluster_indices]
        pos_indices = cluster_indices[cluster_df['label'] == 1]
        neg_indices = cluster_indices[cluster_df['label'] == 0]
        
        # Sampling stratificato bilanciato (aggiornato per 5 samples)
        selected = []
        
        # Strategia per 5 samples: 3-2 o 2-3 split quando possibile
        n_pos_target = min(3, len(pos_indices))  # Preferisce 3 positivi
        n_neg_target = min(samples_per_cluster - n_pos_target, len(neg_indices))
        
        # Aggiusta se non abbastanza negativi
        if n_neg_target < (samples_per_cluster - n_pos_target) and len(pos_indices) > n_pos_target:
            additional_pos = min(samples_per_cluster - n_neg_target - n_pos_target, 
                               len(pos_indices) - n_pos_target)
            n_pos_target += additional_pos
        
        # Campiona positivi
        if n_pos_target > 0 and len(pos_indices) > 0:
            selected_pos = np.random.choice(pos_indices, 
                                          min(n_pos_target, len(pos_indices)), 
                                          replace=False)
            selected.extend(selected_pos)
        
        # Campiona negativi
        if n_neg_target > 0 and len(neg_indices) > 0:
            selected_neg = np.random.choice(neg_indices, 
                                          min(n_neg_target, len(neg_indices)), 
                                          replace=False)
            selected.extend(selected_neg)
            
        # Se ancora non abbastanza, prendi random dal cluster
        remaining_needed = samples_per_cluster - len(selected)
        if remaining_needed > 0:
            remaining_indices = [idx for idx in cluster_indices if idx not in selected]
            if remaining_indices:
                additional = np.random.choice(remaining_indices, 
                                           min(remaining_needed, len(remaining_indices)), 
                                           replace=False)
                selected.extend(additional)
        
        if len(selected) == samples_per_cluster:
            cluster_stats["full"] += 1
        elif len(selected) > 0:
            cluster_stats["insufficient"] += 1
        
        sampled_indices.extend(selected)
    
    # Crea dataset finale
    sampled_df = df.iloc[sampled_indices].copy().reset_index(drop=True)
    
    # Statistiche dettagliate
    total_samples = len(sampled_df)
    pos_samples = (sampled_df['label'] == 1).sum()
    neg_samples = total_samples - pos_samples
    
    print(f"[SAMPLING] ✓ Final dataset: {total_samples} samples")
    print(f"[SAMPLING] ✓ Distribution: {pos_samples} positive, {neg_samples} negative")
    print(f"[SAMPLING] ✓ Balance: {pos_samples/total_samples:.1%} positive")
    print(f"[SAMPLING]   Cluster stats: {cluster_stats['full']} full, {cluster_stats['insufficient']} partial, {cluster_stats['empty']} empty")
    
    return sampled_df

def create_clustered_dataset(df: pd.DataFrame, cache_file: Path) -> pd.DataFrame:
    """Crea dataset clusterizzato con parametri aggiornati."""
    print(f"[CLUSTER] Creating clustered dataset from {len(df)} examples...")
    print(f"[CLUSTER] Target: {N_CLUSTERS} clusters × {SAMPLES_PER_CLUSTER} samples = {TARGET_DATASET_SIZE} total")
    
    # Set seed per riproducibilità
    np.random.seed(RANDOM_STATE)
    
    # Step 1: Crea embeddings
    texts = df['text'].tolist()
    embeddings, vectorizer = create_text_embeddings(texts)
    
    # Step 2: Clustering aggiornato
    cluster_labels = perform_clustering(embeddings)
    
    # Step 3: Sampling aggiornato
    clustered_df = sample_from_clusters(df, cluster_labels)
    
    # Step 4: Cache con metadati aggiornati
    cache_data = {
        'dataframe': clustered_df,
        'cluster_labels': cluster_labels,
        'vectorizer': vectorizer,
        'metadata': {
            'original_size': len(df),
            'clustered_size': len(clustered_df),
            'n_clusters': N_CLUSTERS,
            'samples_per_cluster': SAMPLES_PER_CLUSTER,
            'version': '2.0_updated',  # Version marker
            'clustering_strategy': f'{N_CLUSTERS}x{SAMPLES_PER_CLUSTER}'
        }
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"[CACHE] ✓ Saved to {cache_file}")
    
    return clustered_df

def load_clustered_dataset(cache_file: Path) -> Optional[pd.DataFrame]:
    """Carica dataset clusterizzato da cache (con verifica versione)."""
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        df = cache_data['dataframe']
        metadata = cache_data['metadata']
        
        # Verifica se cache usa nuovi parametri
        cache_clusters = metadata.get('n_clusters', 100)  # Default old value
        cache_samples = metadata.get('samples_per_cluster', 4)  # Default old value
        
        if cache_clusters != N_CLUSTERS or cache_samples != SAMPLES_PER_CLUSTER:
            print(f"[CACHE] Cache mismatch: {cache_clusters}x{cache_samples} vs current {N_CLUSTERS}x{SAMPLES_PER_CLUSTER}")
            print(f"[CACHE] Will regenerate with updated parameters")
            return None
        
        print(f"[CACHE] ✓ Loaded {len(df)} samples from cache")
        print(f"[CACHE] ✓ Parameters: {cache_clusters} clusters × {cache_samples} samples")
        print(f"[CACHE] ✓ Original: {metadata['original_size']} → Clustered: {metadata['clustered_size']}")
        
        return df
    except Exception as e:
        print(f"[CACHE] ✗ Cache loading failed: {e}")
        return None

# ==== Caricamento Dataset ====
try:
    # Carica solo test dataset (per XAI evaluation)
    test_df_original = pd.read_csv(DATA_DIR / TEST_FILE)
    print(f"[LOAD] ✓ Test dataset: {len(test_df_original)} examples")
except FileNotFoundError as e:
    print(f"[ERROR] Dataset loading failed: {e}")
    print("Assicurati che Test.csv sia nella cartella corrente")
    raise

# Verifica struttura
required_columns = ["text", "label"]
missing_cols = [col for col in required_columns if col not in test_df_original.columns]
if missing_cols:
    raise ValueError(f"Dataset Test manca colonne: {missing_cols}")

# ==== Clustering del Test Set (AGGIORNATO) ====
cache_file = CACHE_DIR / f"clustered_test_{N_CLUSTERS}c_{SAMPLES_PER_CLUSTER}s_v2.pkl"  # v2 per nuova versione

print(f"\n{'='*60}")
print("DATASET CLUSTERING (UPDATED)")
print(f"{'='*60}")
print(f"Strategy: {N_CLUSTERS} clusters × {SAMPLES_PER_CLUSTER} samples = {TARGET_DATASET_SIZE} total")

# Prova a caricare da cache
test_df = load_clustered_dataset(cache_file)

if test_df is None:
    print("[CLUSTER] Cache not found or outdated, creating new clustered dataset...")
    test_df = create_clustered_dataset(test_df_original, cache_file)
else:
    print("[CLUSTER] Using cached clustered dataset")

print(f"\n[FINAL] Working dataset: {len(test_df)} examples")
print(f"[FINAL] Reduction: {len(test_df_original)} → {len(test_df)} ({len(test_df)/len(test_df_original):.1%})")

# ==== Pulizia Dataset ====
LABEL_MAP = {
    "negative": 0, "positive": 1, 
    "neg": 0, "pos": 1,
    "0": 0, "1": 1,
    0: 0, 1: 1
}

def to_int_label(x):
    """Converte label in intero 0/1."""
    if isinstance(x, (int, float)):
        return int(x)
    try:
        if isinstance(x, str):
            return LABEL_MAP[x.strip().lower()]
        return LABEL_MAP[x]
    except (KeyError, AttributeError):
        raise ValueError(f"Label sconosciuta: {x}")

# Pulizia
# Rimuovi righe vuote
initial_len = len(test_df)
test_df.dropna(subset=["text"], inplace=True)
test_df = test_df[test_df["text"].str.strip() != ""]

# Converti labels
test_df["label"] = test_df["label"].apply(to_int_label)

# Log
if len(test_df) < initial_len:
    print(f"[CLEAN] Test: removed {initial_len - len(test_df)} empty examples")

label_counts = test_df["label"].value_counts().sort_index()
print(f"[CLEAN] Test distribution: {dict(label_counts)}")

# Reset index
test_df = test_df.reset_index(drop=True)

# ==== Tokenizzazione ottimizzata Colab ====
def get_tokenizer_for_model(model_name: str):
    """Carica tokenizer per modello."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        return tokenizer
    except Exception as e:
        print(f"[TOKENIZER] Error for {model_name}: {e}")
        # Fallback
        return AutoTokenizer.from_pretrained("bert-base-uncased")

def encode_texts(text_list, tokenizer, max_length=MAX_LENGTH):
    """Tokenizza lista di testi."""
    return tokenizer(
        text_list,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

# ==== Dataset Class Ottimizzata ====
class IMDBDataset(Dataset):
    def __init__(self, dataframe, tokenizer=None, max_length=MAX_LENGTH):
        """Dataset IMDB ottimizzato per Colab."""
        self.dataframe = dataframe.reset_index(drop=True)
        self.max_length = max_length
        
        # Tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer
            
        print(f"[DATASET] Tokenizing {len(self.dataframe)} examples...")
        
        # Pre-tokenizza tutto per efficienza
        encodings = encode_texts(list(self.dataframe["text"]), self.tokenizer, max_length)
        
        self.input_ids = encodings["input_ids"]
        self.attention_masks = encodings["attention_mask"]
        self.labels = torch.tensor(self.dataframe["label"].values, dtype=torch.long)
        
        print(f"[DATASET] ✓ {len(self)} examples ready")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

# ==== DataLoader Factory per Colab ====
def create_dataloaders(model_name=None, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """Crea DataLoader per test dataset (per XAI evaluation)."""
    # Tokenizer
    if model_name:
        tokenizer = get_tokenizer_for_model(model_name)
    else:
        tokenizer = None
    
    # Solo test dataset
    test_dataset = IMDBDataset(test_df, tokenizer, max_length)
    
    # DataLoader OTTIMIZZATO PER COLAB
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,      # IMPORTANTE: 0 per Colab  
        pin_memory=False
    )
    
    return test_loader

# ==== Default DataLoader ====
default_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
test_loader = DataLoader(
    IMDBDataset(test_df, default_tokenizer), 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=0
)

# ==== Utility Functions ====
def get_clustered_sample(sample_size: Optional[int] = None, stratified: bool = True) -> Tuple[List[str], List[int]]:
    """Restituisce sample del dataset clusterizzato."""
    if sample_size is None or sample_size >= len(test_df):
        texts = test_df["text"].tolist()
        labels = test_df["label"].tolist()
    else:
        if stratified:
            # Sampling stratificato
            pos_df = test_df[test_df["label"] == 1]
            neg_df = test_df[test_df["label"] == 0]
            
            n_pos = min(sample_size // 2, len(pos_df))
            n_neg = min(sample_size - n_pos, len(neg_df))
            
            pos_sample = pos_df.sample(n_pos, random_state=RANDOM_STATE)
            neg_sample = neg_df.sample(n_neg, random_state=RANDOM_STATE)
            
            sample_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=RANDOM_STATE)
        else:
            sample_df = test_df.sample(sample_size, random_state=RANDOM_STATE)
        
        texts = sample_df["text"].tolist()
        labels = sample_df["label"].tolist()
    
    return texts, labels

def print_dataset_info():
    """Stampa info dataset aggiornato."""
    print(f"\n{'='*60}")
    print("DATASET INFO (UPDATED)")
    print(f"{'='*60}")
    print(f"Test size: {len(test_df)} (clustered from {len(test_df_original)})")
    print(f"Clustering: {N_CLUSTERS} clusters × {SAMPLES_PER_CLUSTER} samples")
    print(f"Strategy change: 100×4 → {N_CLUSTERS}×{SAMPLES_PER_CLUSTER} (more granular)")
    print(f"Reduction ratio: {len(test_df)/len(test_df_original):.1%}")
    
    pos = (test_df["label"] == 1).sum()
    neg = len(test_df) - pos
    print(f"Test: {pos} pos ({pos/len(test_df):.1%}), {neg} neg ({neg/len(test_df):.1%})")

# ==== Test ====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATASET TEST (UPDATED VERSION)")
    print("="*60)
    
    print_dataset_info()
    
    # Test DataLoader
    print(f"\nTesting DataLoader...")
    batch = next(iter(test_loader))
    shapes = {k: v.shape for k, v in batch.items()}
    print(f"Batch shapes: {shapes}")
    
    # Test sampling
    print(f"\nTesting sampling...")
    texts, labels = get_clustered_sample(10)
    print(f"Sample: {len(texts)} texts, {sum(labels)} positive")
    
    # Verifica bilanciamento cluster sampling
    print(f"\nCluster balance verification...")
    sample_texts, sample_labels = get_clustered_sample(80)  # 1 per cluster
    pos_ratio = sum(sample_labels) / len(sample_labels)
    print(f"Cluster sampling balance: {pos_ratio:.1%} positive")
    
    print(f"\n✓ Dataset ready for Colab XAI benchmark! (UPDATED)")
    print(f"Working with {len(test_df)} representative examples ({N_CLUSTERS}×{SAMPLES_PER_CLUSTER}) instead of {len(test_df_original)}")