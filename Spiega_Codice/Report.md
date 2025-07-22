# Spiegazione del file `report.py`

## Panoramica

Il file `report.py` implementa un sistema completo per generare report di valutazione XAI, che testa combinazioni di modelli, explainer e metriche in modo ottimizzato per Google Colab. Il sistema è stato semplificato per rimuovere la complessità della parallelizzazione explainer mantenendo le ottimizzazioni più impattanti.

## Obiettivo

**Problema**: Valutare sistematicamente la qualità degli explainer su diversi modelli richiede migliaia di computazioni costose
**Soluzione**: Sistema ottimizzato con caching, memory management e processing intelligente per eseguire evaluation complete in tempi ragionevoli su Colab

## Architettura Semplificata

### Strategia di Processing
```
Modelli × Explainer × Metriche = Report Completo

Esempio:
5 modelli × 6 explainer × 3 metriche = 90 combinazioni totali
```

### Pipeline di Execution
```
Smart Model Ordering (piccoli → grandi)
         ↓
For each model:
    ↓
    Load Model + Tokenizer
    ↓
    Adaptive Batch Size Calculation
    ↓
    Data Preparation (stratified sampling)
    ↓
    For each explainer:
        ↓
        Create Explainer Instance
        ↓
        For each metric:
            ↓
            Robustness/Contrastivity/Consistency
            ↓
            Store Results
        ↓
        Memory Cleanup
    ↓
    Save Model Results + Checkpoint
         ↓
Build Tables & Analysis
         ↓
Generate Final Report
```

---

## Componenti Principali

### 1. **Advanced Memory Manager**
```python
class AdvancedMemoryManager:
    def calculate_optimal_batch_size(self, base_batch_size: int = 10) -> int
    def enable_gpu_memory_pool(self)
    def progressive_cleanup(self, level: str = "medium")
```

**Funzionalità:**
- **Monitoring memoria**: Traccia RAM e GPU usage in tempo reale
- **Batch size adattivo**: Calcola dimensioni ottimali basate su memoria disponibile
- **GPU memory pool**: Pre-alloca memoria GPU per evitare frammentazione
- **Cleanup progressivo**: 3 livelli di pulizia (light/medium/aggressive)

**Logica adaptive batching:**
```python
if available_gb > 8:    return batch_size * 4  # 40 esempi
elif available_gb > 4:  return batch_size * 2  # 20 esempi  
elif available_gb > 2:  return batch_size      # 10 esempi
else:                   return batch_size // 2  # 5 esempi
```

### 2. **Embedding Cache System**
```python
class EmbeddingCache:
    def get_embedding(self, model_key: str, text: str) -> Optional[torch.Tensor]
    def save_embedding(self, model_key: str, text: str, embedding: torch.Tensor)
```

**Architettura Cache:**
- **Memory cache**: WeakValueDictionary per accesso istantaneo
- **Disk cache**: Persistenza tra sessioni con pickle
- **Model-specific**: Chiavi separate per ogni modello
- **Hash-based**: MD5 di model+text per chiavi univoche

**Cache Key Generation:**
```python
def _get_cache_key(self, model_key: str, text: str, operation: str) -> str:
    content = f"{model_key}_{operation}_{text}"
    return hashlib.md5(content.encode()).hexdigest()
```

**Benefici:**
- **First run**: Computing completo + salvataggio cache
- **Subsequent runs**: Loading istantaneo = 5-10x speedup

### 3. **Async I/O Manager**
```python
class AsyncIOManager:
    def save_async(self, data: Any, filepath: Path, format: str = "json")
    def wait_all(self, timeout: int = 60)
```

**Funzionalità:**
- **Background saving**: Salva risultati mentre continua processing
- **Multiple formats**: JSON, pickle, CSV support
- **Thread pool**: 2 worker per I/O non-bloccante
- **Graceful shutdown**: Aspetta completamento operazioni pending

---

## Flusso di Processing Dettagliato

### **Model Processing Loop**
```python
def process_model_simplified(
    model_key: str,
    explainers_to_test: List[str],
    metrics_to_compute: List[str],
    sample_size: int,
    enable_caching: bool = True,
    adaptive_batching: bool = True
) -> Dict[str, Dict[str, float]]
```

**Step 1: Setup & Optimization**
```python
# Inizializza manager
memory_manager = AdvancedMemoryManager()
embedding_cache = EmbeddingCache()
async_io = AsyncIOManager()

# Abilita GPU optimizations
memory_manager.enable_gpu_memory_pool()

# Calcola batch size ottimale
optimal_batch_size = memory_manager.calculate_optimal_batch_size()
```

**Step 2: Model Loading**
```python
model = models.load_model(model_key)
tokenizer = models.load_tokenizer(model_key)
```

**Step 3: Data Preparation**
```python
texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)

# Separa per metriche specifiche
pos_texts = [t for t, l in zip(texts, labels) if l == 1][:optimal_batch_size]
neg_texts = [t for t, l in zip(texts, labels) if l == 0][:optimal_batch_size]  
consistency_texts = texts[:min(optimal_batch_size, len(texts))]
```

**Step 4: Explainer Loop Sequenziale**
```python
for explainer_name in explainers_to_test:
    # Crea explainer
    explainer = explainers.get_explainer(explainer_name, model, tokenizer)
    
    # Loop metriche
    for metric_name in metrics_to_compute:
        if metric_name == "robustness":
            score = metrics.evaluate_robustness_over_dataset(...)
        elif metric_name == "contrastivity":
            score = metrics.compute_contrastivity(...)
        elif metric_name == "consistency":  
            score = metrics.evaluate_consistency_over_dataset(...)
        
        results[metric_name][explainer_name] = score
    
    # Cleanup per explainer
    del explainer
    memory_manager.progressive_cleanup("light")
```

### **Metric Evaluation Deep Dive**

#### **Robustness Evaluation**
```python
score = metrics.evaluate_robustness_over_dataset(
    model, tokenizer, explainer, consistency_texts, show_progress=False
)
```
**Cosa fa:**
- Genera perturbazioni del testo (mask, delete, substitute)
- Calcola attributions su testo originale e perturbato
- Misura differenze nelle attributions
- Ritorna stabilità media (lower = more robust)

#### **Contrastivity Evaluation**  
```python
# Process in batch per memoria
pos_attrs = metrics.process_attributions_batch(
    pos_texts, explainer, batch_size=optimal_batch_size//2
)
neg_attrs = metrics.process_attributions_batch(
    neg_texts, explainer, batch_size=optimal_batch_size//2  
)

score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
```
**Cosa fa:**
- Genera attributions per testi positivi e negativi
- Accumula token importance scores per classe
- Calcola divergenza KL tra distribuzioni
- Ritorna discriminabilità tra classi (higher = more contrastive)

#### **Consistency Evaluation**
```python
score = metrics.evaluate_consistency_over_dataset(
    model=model,
    tokenizer=tokenizer, 
    explainer=explainer,
    texts=consistency_texts,
    seeds=DEFAULT_CONSISTENCY_SEEDS,
    show_progress=False
)
```
**Cosa fa:**
- Attiva dropout nel modello per variabilità stocastica
- Genera explanations con seed diversi [42, 123, 456, 789]
- Calcola correlazioni Spearman tra explanations
- Ritorna consistenza media (higher = more consistent)

---

## Sistema di Checkpoint e Recovery

### **Auto Recovery System**
```python
recovery = AutoRecovery(checkpoint_dir=RESULTS_DIR / "checkpoints")
```

**Funzionalità:**
- **Checkpoint automatici**: Salva stato dopo ogni explainer
- **Resume intelligente**: Riprende da dove si era interrotto
- **Validation**: Verifica completezza checkpoint prima del resume
- **Timestamping**: Checkpoint con timestamp per tracking

### **Checkpoint Validation**
```python
def verify_checkpoint_completeness(
    checkpoint_data: Dict,
    expected_metrics: List[str], 
    expected_explainers: List[str]
) -> Tuple[bool, str]
```

**Controlli:**
- **Completeness flag**: Checkpoint marcato come completo
- **Missing metrics**: Tutte le metriche richieste presenti
- **Missing explainers**: Tutti gli explainer processati
- **Valid results**: Almeno 10% risultati non-NaN

---

## Table Building e Analysis

### **Results Aggregation**
```python
def build_report_tables(all_results: Dict, metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]
```

**Struttura Output:**
```
Per ogni metrica → DataFrame:
                tinybert  distilbert  roberta-base  bert-large
lime              0.1234      0.2345        0.3456     0.4567
shap              0.2345      0.3456        0.4567     0.5678  
grad_input        0.3456      0.4567        0.5678     0.6789
attention_*       0.4567      0.5678        0.6789     0.7890
lrp               0.5678      0.6789        0.7890     0.8901
```

### **Statistical Analysis**
```python
def print_table_analysis(df: pd.DataFrame, metric_name: str)
```

**Analisi Computate:**
- **Per-Explainer Statistics**: Media, std, coverage per riga
- **Per-Model Statistics**: Media, std, coverage per colonna  
- **Top 5 Combinations**: Migliori coppie explainer-model
- **Coverage Analysis**: Percentuale celle completate con successo

**Interpretazione Automatica:**
```python
if metric_name == "robustness":
    # Lower is better
    direction = "(Lower = Better)"
    best = min(flat_data, key=lambda x: x[2])
else:
    # Higher is better
    direction = "(Higher = Better)"  
    best = max(flat_data, key=lambda x: x[2])
```

---

## Output Generation

### **File Output Generati**
```python
# CSV Tables (una per metrica)
robustness_table_simplified.csv
contrastivity_table_simplified.csv  
consistency_table_simplified.csv

# JSON Results (uno per modello)
results_tinybert_simplified.json
results_distilbert_simplified.json
...

# Summary Report
summary_report_simplified.txt
```

### **Summary Report Structure**
```
XAI BENCHMARK REPORT - SIMPLIFIED VERSION
==========================================
Generated: 2024-01-15 10:30:45
Execution Time: 25.3 minutes
Dataset: 400 clustered examples from IMDB

CONFIGURATION:
- Models tested: 5 (tinybert, distilbert, roberta-base, bert-large, roberta-large)
- Explainers tested: 6 (lime, shap, grad_input, attention_rollout, attention_flow, lrp)
- Metrics computed: 3 (robustness, contrastivity, consistency)

RESULTS SUMMARY:
ROBUSTNESS:
  Coverage: 28/30 combinations completed (93.3%)
  Best: lime + tinybert = 0.0234 (most robust)

SIMPLIFIED ARCHITECTURE BENEFITS:
- Removed explainer parallelization complexity
- More reliable sequential processing
- Focus on high-impact optimizations
```

---

## Smart Optimizations

### **Smart Model Ordering**
```python
def smart_model_ordering(models_to_test: List[str]) -> List[str]:
    size_priority = {
        "tinybert": 1,      # Processa per primo
        "distilbert": 2,    
        "roberta-base": 3,  
        "bert-large": 4,    
        "roberta-large": 5  # Processa per ultimo
    }
```

**Perché funziona:**
- **Environment degradation**: Modelli grandi degradano ambiente Colab
- **Memory fragmentation**: Processing piccoli→grandi evita frammentazione
- **Failure prevention**: Se crash, almeno modelli piccoli sono completati

### **Memory Cleanup Strategy**
```python
# Dopo ogni metrica
torch.cuda.empty_cache()

# Dopo ogni explainer  
memory_manager.progressive_cleanup("light")

# Dopo ogni modello
memory_manager.progressive_cleanup("aggressive")
```

**Livelli di Cleanup:**
- **Light**: `gc.collect()` basic
- **Medium**: Multiple GC + GPU cache clear
- **Aggressive**: Full GC + GPU IPC collect + peak stats reset + pause

---

## Usage Patterns

### **Quick Testing**
```python
import report

# Test velocissimo (2-3 minuti)
tables = report.turbo_report()
```

### **Customized Report**
```python
tables = report.run_simplified_report(
    models_to_test=["tinybert", "distilbert"],
    explainers_to_test=["lime", "shap", "grad_input"],
    metrics_to_compute=["robustness", "consistency"],
    sample_size=100,
    enable_caching=True,      # Importante per speedup
    adaptive_batching=True,   # Importante per memoria
    resume=True              # Importante per recovery
)
```

### **CLI Interface**
```bash
python report.py --models tinybert distilbert --explainers lime shap --sample 50 --turbo
```

---

### **Vantaggi della semplificazione:**
1. **Affidabilità**: Nessun thread-safety issue
2. **Debugging**: Flusso sequenziale chiaro
3. **Manutenibilità**: Codice 30% più breve
4. **Performance**: Focus su ottimizzazioni ad alto impatto
5. **Stabilità**: Processing più prevedibile

### **Ottimizzazioni mantenute (quelle che contano):**
- **Adaptive batching**: 1.5-2x speedup memoria
- **Embedding caching**: 5-10x speedup per testi ripetuti  
- **Memory management**: Previene crash OOM
- **GPU optimization**: Allocazioni più efficienti
- **Async I/O**: Processing non bloccante

---

## Performance Characteristics

### **Memory Usage:**
```
Peak GPU: 8-12GB (dipende da modello)
Peak RAM: 4-8GB (dipende da batch size)
Cache size: 100MB-1GB (dipende da dataset)
```

### **Speedup vs Naive Implementation:**
- **First run**: 3-4x speedup (memory + batching)
- **Cached runs**: 8-12x speedup (cache + memory + batching)

---

