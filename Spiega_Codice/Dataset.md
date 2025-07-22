# Spiegazione del file `dataset.py`

## Panoramica

Il file `dataset.py` implementa un sistema  di **clustering e campionamento** per ridurre il dataset IMDB da 5000 esempi a soli **400 esempi rappresentativi**, mantenendo la diversità e il bilanciamento del dataset originale.

## Obiettivo

**Problema**: Il dataset IMDB completo ha troppi esempi per essere processato efficientemente su Google Colab
**Soluzione**: Clustering K-Means + campionamento stratificato per ottenere un subset rappresentativo

## Strategia di Clustering

### Parametri Principali
```python
N_CLUSTERS = 80                    # Numero di cluster K-Means
SAMPLES_PER_CLUSTER = 5           # Esempi per cluster
TARGET_DATASET_SIZE = 80 × 5 = 400 # Dataset finale
```

### Pipeline di Processing
```
Dataset Originale (5000+ esempi)
         ↓
    TF-IDF Embeddings
         ↓
    K-Means (80 clusters)
         ↓
  Campionamento (5 per cluster)
         ↓
    Dataset Finale (400 esempi)
```

---

## Componenti Principali

### 1. **Creazione Embeddings TF-IDF**
```python
def create_text_embeddings(texts: List[str]) -> np.ndarray
```

**Cosa fa:**
- Converte testi in vettori numerici usando TF-IDF
- Parametri ottimizzati: unigrams + bigrams, stop words rimossi
- Normalizzazione per K-Means

**Perché TF-IDF:**
- Veloce da computare (vs BERT embeddings)
- Cattura semantica basica sufficiente per clustering
- Scalabile su migliaia di documenti

### 2. **Clustering K-Means**
```python
def perform_clustering(embeddings: np.ndarray) -> np.ndarray
```

**Cosa fa:**
- Applica K-Means con K=80 cluster
- Ogni cluster raggruppa testi semanticamente simili
- Restituisce label di cluster per ogni testo

**Parametri chiave:**
- `n_clusters=80`: Granularità ottimale tra diversità e rappresentatività  
- `init='k-means++'`: Inizializzazione intelligente
- `random_state=42`: Riproducibilità

### 3. **Campionamento Strategico**
```python
def sample_from_clusters(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame
```

**Strategia di Sampling:**

Per ogni cluster:
1. **Separazione per classe**: Divide esempi positivi e negativi
2. **Bilanciamento 3-2**: Preferisce 3 esempi di una classe, 2 dell'altra
3. **Fallback intelligente**: Se una classe manca, completa con l'altra
4. **Random finale**: Riempie posti rimanenti con esempi casuali del cluster

**Esempio pratico:**
```
Cluster 15: 12 esempi (7 positivi, 5 negativi)
Campionamento: 3 positivi + 2 negativi = 5 esempi finali
```

---

## Vantaggi del Clustering

### **Diversità Massimizzata**
- 80 cluster catturano 80 "topic" diversi del dataset
- Ogni cluster rappresenta un tema semantico distinto
- Evita ridondanza di esempi troppo simili

### **Bilanciamento Preservato** 
- Campionamento stratificato per classe (pos/neg)
- Mantiene proporzioni originali del dataset
- Ogni cluster contribuisce equamente

### **Efficienza Computazionale**
- 400 vs 5000+ esempi = 12x più veloce
- Caricamento e processing molto più rapidi
- Ideale per ambiente Colab con risorse limitate

### **Rappresentatività Garantita**
- Cluster coprono tutto lo spazio semantico
- Esempi rappresentativi di ogni area tematica
- Qualità delle XAI evaluation preservata

---

## Sistema di Caching

### **Cache Intelligente**
```python
cache_file = CACHE_DIR / f"clustered_test_{N_CLUSTERS}c_{SAMPLES_PER_CLUSTER}s_v2.pkl"
```

**Funzionalità:**
- **Prima esecuzione**: Clustering completo (2-3 minuti)
- **Esecuzioni successive**: Caricamento istantaneo da cache
- **Auto-invalidazione**: Se parametri cambiano, rigenera automaticamente
- **Versioning**: Cache diverse per configurazioni diverse

### **Verifica Compatibilità**
```python
if cache_clusters != N_CLUSTERS or cache_samples != SAMPLES_PER_CLUSTER:
    print(f"[CACHE] Will regenerate with updated parameters")
    return None
```

---

## Workflow Completo

### **Fase 1: Inizializzazione**
```python
# Carica dataset originale
test_df_original = pd.read_csv("Test.csv")  # 5000+ esempi

# Pulisce dati (rimuove vuoti, normalizza label)
test_df_original = clean_dataset(test_df_original)
```

### **Fase 2: Clustering (se cache mancante)**
```python
# 1. Embeddings TF-IDF
embeddings = create_text_embeddings(texts)  # (5000, 5000) matrix

# 2. K-Means clustering  
cluster_labels = perform_clustering(embeddings)  # Array di 80 cluster ID

# 3. Campionamento strategico
test_df = sample_from_clusters(df, cluster_labels)  # 400 esempi finali
```

### **Fase 3: Output**
```python
# Dataset finale pronto per uso
print(f"Dataset: {len(test_df)} esempi")  # 400
print(f"Riduzione: {len(test_df)/len(test_df_original):.1%}")  # ~8%
```

---

## Statistiche e Validazione

### **Logging Dettagliato**
Il sistema stampa statistiche complete:

```
[CLUSTERING] ✓ 80 clusters created
[CLUSTERING]   Cluster sizes: min=45, max=78, avg=62.5
[SAMPLING] ✓ Final dataset: 400 samples
[SAMPLING] ✓ Distribution: 201 positive, 199 negative
[SAMPLING] ✓ Balance: 50.2% positive
[SAMPLING]   Cluster stats: 78 full, 2 partial, 0 empty
```

### **Metriche di Qualità**
- **Coverage**: % cluster con samples completi
- **Balance**: Distribuzione pos/neg preservata  
- **Reduction**: Fattore di riduzione dataset
- **Representativeness**: Diversità semantica mantenuta

---

## Configurazione e Personalizzazione

### **Parametri Modificabili**
```python
N_CLUSTERS = 80           # Più cluster = più diversità
SAMPLES_PER_CLUSTER = 5   # Più samples = più rappresentatività  
MAX_FEATURES = 5000       # Dimensioni vocabulary TF-IDF
RANDOM_STATE = 42         # Seed per riproducibilità
```

### **Trade-offs**
- **Più cluster** → Maggiore diversità, ma cluster più piccoli
- **Più samples/cluster** → Dataset più grande, ma più rappresentativo
- **Più features TF-IDF** → Embeddings più ricchi, ma più lenti

### **Strategie Alternative**
Il sistema è modulare e permette di cambiare:
- **Algoritmo clustering**: DBSCAN, Hierarchical, ecc.
- **Embeddings**: BERT, Sentence-Transformers, ecc.  
- **Sampling**: Uniform, weighted, active learning, ecc.

---

## Integrazione nel Pipeline XAI

### **Dataset Class**
```python
class IMDBDataset(Dataset):
    # Pre-tokenizza tutto il dataset per efficienza
    # Ottimizzato per DataLoader PyTorch
    # Compatible con tutti i modelli transformer
```

### **Utility Functions**
```python
# Campionamento on-demand per esperimenti
texts, labels = get_clustered_sample(sample_size=100)

# DataLoader per modelli specifici
loader = create_dataloaders(model_name="distilbert")

# Info e statistiche
print_dataset_info()
```

### **Workflow Tipico**
```python
import dataset

# Dataset già pronto all'import
print(f"Esempi disponibili: {len(dataset.test_df)}")

# Sampling per esperimenti
texts, labels = dataset.get_clustered_sample(50)

# Uso diretto con modelli
for explainer_name in ["lime", "shap"]:
    for text in texts[:5]:
        explanation = explainer(text)
```

---

## Benefici per XAI Evaluation

### **Velocità**
- **12x più veloce** vs dataset completo
- **Clustering una tantum**, poi instant loading
- **Batch processing** ottimizzato

### **Qualità**
- **Diversità preservata** tramite clustering semantico
- **Bilanciamento mantenuto** con sampling stratificato  
- **Rappresentatività garantita** da 80 topic distinti

### **Scalabilità**
- **Memoria efficiente** per Colab
- **Parallelizzabile** per esperimenti multipli
- **Estendibile** a dataset più grandi

### **Riproducibilità** 
- **Seed fisso** per risultati consistenti
- **Cache persistente** per esperimenti ripetuti
- **Versioning** per tracking configurazioni

---

## Conclusioni

Il sistema di clustering implementato in `dataset.py` risolve elegantemente il problema della scalabilità per XAI evaluation:

1. **Riduce drasticamente** la dimensione del dataset (5000+ → 400)
2. **Preserva la qualità** attraverso clustering semantico intelligente  
3. **Mantiene l'efficienza** con caching e ottimizzazioni Colab
4. **Garantisce riproducibilità** con seed e versioning

Questo approccio permette di eseguire esperimenti XAI completi in pochi minuti invece che ore, senza compromettere la validità scientifica dei risultati.