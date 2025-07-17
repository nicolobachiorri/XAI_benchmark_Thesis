# XAI Benchmark per Modelli di Sentiment Analysis

Questo progetto implementa un benchmark per valutare tecniche di Explainable AI (XAI) su modelli Transformer pre-addestrati per sentiment analysis. Il benchmark confronta 6 modelli diversi con 6 metodi di spiegazione usando 3 metriche automatiche.

## Struttura del Progetto

```
xai-benchmark/
├── README.md                 # Documentazione progetto
├── requirements.txt          # Dipendenze Python
├── Train.csv                 # Dataset IMDB training
├── Test.csv                  # Dataset IMDB test
│
├── main.py                   # CLI principale
├── evaluate.py               # Valutazione modelli+explainer
├── report.py                 # Generazione tabelle aggregate
│
├── models.py                 # Gestione modelli pre-trained
├── dataset.py                # Caricamento dataset IMDB
├── explainers.py             # 6 metodi XAI
├── metrics.py                # 3 metriche di valutazione
└── utils.py                  # Funzioni utility
```

## Modelli Supportati

Il progetto utilizza 6 modelli Transformer già pre-addestrati per sentiment analysis:

- **distilbert**: DistilBERT fine-tuned su SST-2 (91.3% accuracy)
- **roberta-large**: RoBERTa-large valutato su 15 dataset diversi
- **bert-large**: BERT-large fine-tuned su SST-2
- **roberta-base**: RoBERTa-base fine-tuned su Amazon Reviews
- **bert-base**: BERT-base fine-tuned su Amazon sentiment
- **tinybert**: TinyBERT fine-tuned su IMDB (subset 5K)

Tutti i modelli sono caricati da Hugging Face Hub senza training aggiuntivo.

## Explainer XAI

Il progetto implementa 6 metodi di spiegazione:

**Gradient-based:**
- **grad_input**: Gradient × Input a livello di embedding
- **lrp**: Layer-wise Relevance Propagation

**Attention-based:**
- **attention_rollout**: Rollout dell'attenzione tra layer
- **attention_flow**: Maximum flow nell'attenzione

**Perturbation-based:**
- **lime**: Local Interpretable Model-agnostic Explanations
- **shap**: SHapley Additive exPlanations (Kernel)

## Metriche di Valutazione

Il benchmark implementa 3 metriche automatiche:

### Robustness
- **Cosa misura**: Stabilità delle spiegazioni sotto perturbazioni del testo
- **Calcolo**: Mean Average Difference (MAD) tra spiegazioni originali e perturbate
- **Interpretazione**: Più basso = più robusto

### Contrastivity
- **Cosa misura**: Diversità delle spiegazioni tra classi opposte
- **Calcolo**: Divergenza KL tra distribuzioni di importanza positive/negative
- **Interpretazione**: Più alto = più contrastivo


### Consistency
- **Cosa misura**: Accordo tra spiegazioni di modelli diversi
- **Calcolo**: Correlazione di Spearman tra importanze di token comuni
- **Interpretazione**: Più alto = più consistente
- **Nota**: Implementazione adattata da [Mersha et al., 2025 , "Evaluating the effectiveness of XAI techniques"] . Nel paper originale si confrontano modelli identici con seed diversi (senza fare fine-tuning non so come si fa), ma nel nostro caso confrontiamo modelli diversi pre-addestrati per testare la generalizzabilità delle spiegazioni XAI. 

## Installazione

```bash
git clone https://github.com/username/xai-benchmark.git
cd xai-benchmark
pip install -r requirements.txt
```

## Preparazione Dataset

Assicurati di avere i file `Train.csv` e `Test.csv` nella directory principale:

```csv
text,label
"This movie was great!",1
"Terrible film, waste of time.",0
```

## Utilizzo

### Spiegazione singola frase
```bash
python main.py explain --model distilbert --explainer lime --text "This movie is amazing!"
```

### Valutazione singola metrica
```bash
python main.py evaluate --metric robustness --model distilbert --explainer grad_input --sample 500
```

### Confronto consistency tra modelli
```bash
python main.py evaluate --metric consistency --model-a distilbert --model-b bert-base --explainer lime --sample 300
```

### Generazione tabelle complete
```bash
# Tutte le metriche
python report.py --sample 500

# Singola metrica
python report.py --metric robustness --sample 300

# Output CSV
python report.py --csv --sample 500
```

## Output del Benchmark

### Struttura Tabelle

**Robustness e Contrastivity:**
```
Explainer     | distilbert | bert-base | roberta-large | ...
--------------|------------|-----------|---------------|----
lime          | 0.1234     | 0.1456    | 0.1123        | ...
shap          | 0.1345     | 0.1567    | 0.1234        | ...
grad_input    | 0.1456     | 0.1678    | 0.1345        | ...
```

**Consistency:**
```
Explainer     | distilbert_vs_bert-base | bert-base_vs_roberta-large | ...
--------------|-------------------------|----------------------------|----
lime          | 0.7234                  | 0.6456                     | ...
shap          | 0.7345                  | 0.6567                     | ...
grad_input    | 0.7456                  | 0.6678                     | ...
```

## Parametri Configurabili

### dataset.py
- `MAX_LENGTH = 512`: Lunghezza massima sequenze
- `BATCH_SIZE = 16`: Dimensione batch

### metrics.py
- `DEFAULT_PERTURBATION_RATIO = 0.15`: Percentuale parole da perturbare
- `MIN_SHARED_TOKENS = 2`: Minimo token condivisi per correlazione

### explainers.py
- `MAX_LEN = 512`: Lunghezza massima per explainer

## Troubleshooting

### Errori Comuni

**"Model not found"**
```bash
# Verifica modelli disponibili
python main.py evaluate --help
```

**"CUDA out of memory"**
- Riduci `--sample` (es. da 500 a 100)
- Riduci `BATCH_SIZE` in `dataset.py`

**"Explainer failed"**
```bash
# Verifica explainer disponibili
python -c "import explainers; print(explainers.list_explainers())"
```

### Performance

- **Test rapido**: `--sample 50` (2-3 minuti)
- **Risultati affidabili**: `--sample 500` (30-60 minuti)
- **Memoria**: Modelli large richiedono 8GB+ RAM

## Setup Google Colab

```python
# Cella 1: Setup
!git clone https://github.com/username/xai-benchmark.git
%cd xai-benchmark
!pip install -r requirements.txt

# Cella 2: Test rapido
!python main.py explain --model distilbert --explainer lime --text "Great movie!"

# Cella 3: Tabella veloce
!python report.py --sample 20
```

## Estensioni Possibili

1. **Nuovi modelli**: Aggiungi in `models.py`
2. **Nuovi explainer**: Implementa in `explainers.py`
3. **Nuove metriche**: Estendi `metrics.py`
4. **Visualizzazione**: Aggiungi grafici e heatmap

