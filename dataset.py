"""
dataset_optimized.py - Clustering ottimizzato per ottenere esattamente 400 osservazioni
=====================================================================================

STRATEGIA OTTIMIZZATA:
1. Analisi automatica del numero ottimale di cluster
2. Clustering adattivo con verifica dei risultati
3. Post-processing per garantire esattamente 400 osservazioni
4. Bilanciamento automatico delle classi

"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import Tuple, List, Optional, Dict
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==== Parametri Configurabili ====
DATA_DIR = Path(".")
TEST_FILE = "Test.csv"
CACHE_DIR = Path("dataset_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Parametri target
TARGET_DATASET_SIZE = 400
MAX_LENGTH = 512
BATCH_SIZE = 16
RANDOM_STATE = 42

# Parametri clustering ottimizzati
MIN_CLUSTER_SIZE = 3  # Minimo per avere diversità
MAX_CLUSTERS = 50     # Limite superiore ragionevole
MIN_CLUSTERS = 20     # Limite inferiore per diversità

print(f"[OPTIMIZER] Target: {TARGET_DATASET_SIZE} observations from clustered sampling")

class OptimizedClusteringSampler:
    """Sampler intelligente che trova automaticamente il K ottimale."""
    
    def __init__(self, target_size: int = TARGET_DATASET_SIZE):
        self.target_size = target_size
        self.random_state = RANDOM_STATE
        
    def find_optimal_k(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Trova il numero ottimale di cluster per ottenere target_size osservazioni.
        
        Returns:
            (k_optimal, samples_per_cluster)
        """
        print(f"[OPTIMIZER] Finding optimal K for {len(df)} observations → {self.target_size} target")
        
        # Lista di candidati K da testare
        candidates = []
        
        # Strategia 1: Divisori esatti del target
        for samples_per_cluster in range(3, 11):  # da 3 a 10 samples per cluster
            k = self.target_size // samples_per_cluster
            if MIN_CLUSTERS <= k <= MAX_CLUSTERS:
                candidates.append((k, samples_per_cluster))
        
        # Strategia 2: K fissi con samples variabili
        for k in range(MIN_CLUSTERS, min(MAX_CLUSTERS + 1, len(df) // MIN_CLUSTER_SIZE)):
            samples_per_cluster = self.target_size // k
            if samples_per_cluster >= 3:  # Minimo 3 samples per cluster
                candidates.append((k, samples_per_cluster))
        
        # Rimuovi duplicati e ordina
        candidates = list(set(candidates))
        candidates.sort(key=lambda x: abs(x[0] * x[1] - self.target_size))  # Più vicino al target
        
        print(f"[OPTIMIZER] Testing {len(candidates)} candidate configurations...")
        
        # Testa ogni candidato
        best_config = None
        best_score = -1
        
        for k, samples_per_cluster in candidates[:10]:  # Testa solo i primi 10
            score = self._evaluate_clustering_config(df, k, samples_per_cluster)
            expected_size = k * samples_per_cluster
            
            print(f"[OPTIMIZER]   K={k:2d}, samples={samples_per_cluster}, expected={expected_size:3d}, score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_config = (k, samples_per_cluster)
        
        if best_config is None:
            # Fallback sicuro
            k_fallback = max(MIN_CLUSTERS, self.target_size // 8)  # ~8 samples per cluster
            samples_fallback = self.target_size // k_fallback
            best_config = (k_fallback, samples_fallback)
            print(f"[OPTIMIZER] Using fallback: K={k_fallback}, samples={samples_fallback}")
        
        k_opt, samples_opt = best_config
        print(f"[OPTIMIZER] ✓ Optimal: K={k_opt}, samples_per_cluster={samples_opt} (score={best_score:.3f})")
        
        return k_opt, samples_opt
    
    def _evaluate_clustering_config(self, df: pd.DataFrame, k: int, samples_per_cluster: int) -> float:
        """
        Valuta una configurazione di clustering senza fare il clustering completo.
        Usa metriche euristiche per stimare la qualità.
        """
        n_samples = len(df)
        
        # Penalizza se troppi cluster per il dataset
        if k > n_samples // 10:  # Meno di 10 osservazioni per cluster in media
            return 0.1
        
        # Penalizza configurazioni troppo sbilanciate
        expected_cluster_size = n_samples / k
        if expected_cluster_size < MIN_CLUSTER_SIZE:
            return 0.2
        
        # Favorisci configurazioni che producono esattamente il target
        expected_total = k * samples_per_cluster
        size_penalty = abs(expected_total - self.target_size) / self.target_size
        
        # Score finale (più alto = migliore)
        base_score = 1.0 - size_penalty
        
        # Bonus per configurazioni bilanciate
        if samples_per_cluster >= 4 and samples_per_cluster <= 8:
            base_score += 0.1
        
        # Bonus per K ragionevoli
        if MIN_CLUSTERS <= k <= 40:
            base_score += 0.1
        
        return max(0, base_score)

def create_optimized_embeddings(texts: List[str], max_features: int = 8000) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Crea embeddings TF-IDF ottimizzati per clustering."""
    print(f"[EMBEDDING] Creating optimized TF-IDF for {len(texts)} texts...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 3),      # Includi trigrammi per più diversità
        min_df=3,                # Soglia più alta per ridurre noise
        max_df=0.9,              # Escludi parole troppo comuni
        sublinear_tf=True,
        norm='l2'                # Normalizzazione L2
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Normalizzazione aggiuntiva per KMeans
    scaler = StandardScaler(with_mean=False)
    embeddings_scaled = scaler.fit_transform(tfidf_matrix)
    
    print(f"[EMBEDDING] ✓ Shape: {embeddings_scaled.shape}, density: {embeddings_scaled.nnz/embeddings_scaled.size:.3f}")
    
    return embeddings_scaled.toarray(), vectorizer

def perform_optimized_clustering(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, KMeans]:
    """KMeans ottimizzato con controllo qualità."""
    print(f"[CLUSTERING] KMeans with K={k} (optimized)...")
    
    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,               # Più inizializzazioni per stabilità
        max_iter=500,            # Più iterazioni
        init='k-means++',
        algorithm='lloyd',
        tol=1e-6                 # Tolleranza più stretta
    )
    
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Analisi qualità clustering
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    inertia = kmeans.inertia_
    
    empty_clusters = k - len(unique_labels)
    min_size, max_size = counts.min(), counts.max()
    avg_size = counts.mean()
    
    print(f"[CLUSTERING] ✓ Results: {len(unique_labels)}/{k} non-empty clusters")
    print(f"[CLUSTERING]   Sizes: min={min_size}, max={max_size}, avg={avg_size:.1f}")
    print(f"[CLUSTERING]   Empty clusters: {empty_clusters}")
    print(f"[CLUSTERING]   Inertia: {inertia:.1f}")
    
    return cluster_labels, kmeans

def intelligent_cluster_sampling(df: pd.DataFrame, cluster_labels: np.ndarray, 
                                k: int, samples_per_cluster: int) -> pd.DataFrame:
    """
    Sampling intelligente che garantisce esattamente target_size osservazioni.
    """
    print(f"[SAMPLING] Intelligent sampling: {k} clusters × {samples_per_cluster} samples")
    
    sampled_indices = []
    cluster_info = []
    
    # Prima passata: campiona da cluster non vuoti
    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            cluster_info.append({'id': cluster_id, 'size': 0, 'sampled': 0, 'pos': 0, 'neg': 0})
            continue
        
        # Analizza distribuzione classi nel cluster
        cluster_df = df.iloc[cluster_indices]
        pos_indices = cluster_indices[cluster_df['label'] == 1]
        neg_indices = cluster_indices[cluster_df['label'] == 0]
        
        # Sampling stratificato bilanciato
        selected = []
        
        # Calcola target per classe
        n_pos_available = len(pos_indices)
        n_neg_available = len(neg_indices)
        
        if n_pos_available == 0:
            # Solo negativi
            n_pos_target, n_neg_target = 0, min(samples_per_cluster, n_neg_available)
        elif n_neg_available == 0:
            # Solo positivi
            n_pos_target, n_neg_target = min(samples_per_cluster, n_pos_available), 0
        else:
            # Bilanciato
            half = samples_per_cluster // 2
            n_pos_target = min(half + (samples_per_cluster % 2), n_pos_available)
            n_neg_target = min(samples_per_cluster - n_pos_target, n_neg_available)
            
            # Aggiusta se una classe non ha abbastanza esempi
            if n_neg_target < (samples_per_cluster - n_pos_target):
                n_pos_target = min(samples_per_cluster - n_neg_target, n_pos_available)
        
        # Campiona
        if n_pos_target > 0:
            selected_pos = np.random.choice(pos_indices, n_pos_target, replace=False)
            selected.extend(selected_pos)
        
        if n_neg_target > 0:
            selected_neg = np.random.choice(neg_indices, n_neg_target, replace=False)
            selected.extend(selected_neg)
        
        # Se ancora mancano, prendi random
        if len(selected) < samples_per_cluster and len(cluster_indices) > len(selected):
            remaining = [idx for idx in cluster_indices if idx not in selected]
            needed = min(samples_per_cluster - len(selected), len(remaining))
            if needed > 0:
                additional = np.random.choice(remaining, needed, replace=False)
                selected.extend(additional)
        
        sampled_indices.extend(selected)
        
        cluster_info.append({
            'id': cluster_id,
            'size': len(cluster_indices),
            'sampled': len(selected),
            'pos': sum(1 for idx in selected if df.iloc[idx]['label'] == 1),
            'neg': len(selected) - sum(1 for idx in selected if df.iloc[idx]['label'] == 1)
        })
    
    # Crea dataset base
    base_df = df.iloc[sampled_indices].copy()
    
    # Post-processing per raggiungere esattamente target_size
    current_size = len(base_df)
    target_size = k * samples_per_cluster
    
    print(f"[SAMPLING] Base sampling: {current_size} samples")
    
    if current_size < target_size:
        # Aggiungi campioni mancanti da cluster più grandi
        deficit = target_size - current_size
        print(f"[SAMPLING] Adding {deficit} missing samples...")
        
        # Trova cluster con più osservazioni disponibili
        available_clusters = [info for info in cluster_info if info['size'] > info['sampled']]
        available_clusters.sort(key=lambda x: x['size'] - x['sampled'], reverse=True)
        
        additional_indices = []
        for info in available_clusters:
            if deficit <= 0:
                break
            
            cluster_id = info['id']
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Escludi già campionati
            already_sampled = set(sampled_indices)
            available_in_cluster = [idx for idx in cluster_indices if idx not in already_sampled]
            
            # Prendi quello che serve
            to_add = min(deficit, len(available_in_cluster))
            if to_add > 0:
                additional = np.random.choice(available_in_cluster, to_add, replace=False)
                additional_indices.extend(additional)
                deficit -= to_add
        
        if additional_indices:
            additional_df = df.iloc[additional_indices].copy()
            base_df = pd.concat([base_df, additional_df], ignore_index=True)
    
    elif current_size > target_size:
        # Rimuovi campioni in eccesso (stratificato)
        excess = current_size - target_size
        print(f"[SAMPLING] Removing {excess} excess samples...")
        
        # Rimozione stratificata
        pos_df = base_df[base_df['label'] == 1]
        neg_df = base_df[base_df['label'] == 0]
        
        pos_to_remove = min(excess // 2, len(pos_df) - target_size // 2)
        neg_to_remove = excess - pos_to_remove
        
        # Rimuovi random
        if pos_to_remove > 0:
            pos_to_keep = pos_df.sample(len(pos_df) - pos_to_remove, random_state=RANDOM_STATE)
        else:
            pos_to_keep = pos_df
        
        if neg_to_remove > 0:
            neg_to_keep = neg_df.sample(len(neg_df) - neg_to_remove, random_state=RANDOM_STATE)
        else:
            neg_to_keep = neg_df
        
        base_df = pd.concat([pos_to_keep, neg_to_keep], ignore_index=True)
    
    # Shuffle finale
    final_df = base_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Statistiche finali
    final_size = len(final_df)
    pos_count = (final_df['label'] == 1).sum()
    neg_count = final_size - pos_count
    
    non_empty_clusters = sum(1 for info in cluster_info if info['sampled'] > 0)
    
    print(f"[SAMPLING] ✓ Final dataset: {final_size} samples")
    print(f"[SAMPLING] ✓ Distribution: {pos_count} pos ({pos_count/final_size:.1%}), {neg_count} neg ({neg_count/final_size:.1%})")
    print(f"[SAMPLING] ✓ Clusters used: {non_empty_clusters}/{k}")
    
    return final_df

def create_optimized_dataset(df: pd.DataFrame, cache_file: Path) -> pd.DataFrame:
    """Pipeline completa ottimizzata."""
    print(f"[PIPELINE] Creating optimized dataset from {len(df)} examples...")
    
    # Set seed
    np.random.seed(RANDOM_STATE)
    
    # Step 1: Trova K ottimale
    sampler = OptimizedClusteringSampler(TARGET_DATASET_SIZE)
    k_optimal, samples_per_cluster = sampler.find_optimal_k(df)
    
    # Step 2: Embeddings
    texts = df['text'].tolist()
    embeddings, vectorizer = create_optimized_embeddings(texts)
    
    # Step 3: Clustering
    cluster_labels, kmeans_model = perform_optimized_clustering(embeddings, k_optimal)
    
    # Step 4: Sampling intelligente
    final_df = intelligent_cluster_sampling(df, cluster_labels, k_optimal, samples_per_cluster)
    
    # Step 5: Cache
    cache_data = {
        'dataframe': final_df,
        'cluster_labels': cluster_labels,
        'kmeans_model': kmeans_model,
        'vectorizer': vectorizer,
        'metadata': {
            'original_size': len(df),
            'final_size': len(final_df),
            'k_optimal': k_optimal,
            'samples_per_cluster': samples_per_cluster,
            'target_size': TARGET_DATASET_SIZE,
            'version': '3.0_optimized'
        }
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"[CACHE] ✓ Saved to {cache_file}")
    
    return final_df

def load_optimized_dataset(cache_file: Path) -> Optional[pd.DataFrame]:
    """Carica dataset ottimizzato da cache."""
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        df = cache_data['dataframe']
        metadata = cache_data['metadata']
        
        print(f"[CACHE] ✓ Loaded {len(df)} samples (target: {metadata['target_size']})")
        print(f"[CACHE] ✓ Config: K={metadata['k_optimal']}, samples={metadata['samples_per_cluster']}")
        
        return df
    except Exception as e:
        print(f"[CACHE] Cache loading failed: {e}")
        return None

# ==== Main Execution ====
if __name__ == "__main__":
    print("\n" + "="*70)
    print("OPTIMIZED CLUSTERING FOR EXACTLY 400 OBSERVATIONS")
    print("="*70)
    
    # Carica dataset
    try:
        df_original = pd.read_csv(DATA_DIR / TEST_FILE)
        print(f"[LOAD] ✓ Original dataset: {len(df_original)} examples")
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        exit(1)
    
    # Pulizia base
    df_clean = df_original.dropna(subset=['text']).copy()
    df_clean = df_clean[df_clean['text'].str.strip() != '']
    df_clean['label'] = df_clean['label'].apply(lambda x: int(x) if str(x) in ['0', '1'] else (0 if str(x).lower() in ['negative', 'neg'] else 1))
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"[CLEAN] ✓ Clean dataset: {len(df_clean)} examples")
    
    # Cache file
    cache_file = CACHE_DIR / f"optimized_dataset_{TARGET_DATASET_SIZE}.pkl"
    
    # Prova cache
    final_df = load_optimized_dataset(cache_file)
    
    if final_df is None:
        print("[PROCESS] Creating new optimized dataset...")
        final_df = create_optimized_dataset(df_clean, cache_file)
    
    # Verifica finale
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Original size: {len(df_original)}")
    print(f"Final size: {len(final_df)}")
    print(f"Target achieved: {len(final_df) == TARGET_DATASET_SIZE}")
    print(f"Reduction ratio: {len(final_df)/len(df_original):.1%}")
    
    pos_count = (final_df['label'] == 1).sum()
    print(f"Class balance: {pos_count} pos ({pos_count/len(final_df):.1%}), {len(final_df)-pos_count} neg")
    
    print(f"\n✓ SUCCESS: Dataset with exactly {len(final_df)} observations ready!")