# XAI Benchmark per Modelli di Sentiment Analysis

Questo progetto implementa un framework per valutare tecniche di Explainable AI (XAI) su modelli Transformer pre-addestrati per sentiment analysis. Il benchmark confronta **5 modelli diversi** con **5 metodi di spiegazione** usando **3 metriche automatiche**.

## Panoramica del Framework

```
Dataset IMDB → 5 Modelli Pre-trained → 5 Explainer XAI → 3 Metriche Automatiche → Report Comparative
```

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
├── test_everything.py        # Test progressivi completi
│
├── models.py                 # Gestione modelli pre-trained
├── dataset.py                # Caricamento dataset IMDB
├── explainers.py             # 6 metodi XAI
├── metrics.py                # 3 metriche di valutazione
└── utils.py                  # Funzioni utility
```

## Modelli Supportati

- **TinyBERT**: Versione distillata (~14.5M parametri, fine-tuned su IMDB)
- **DistilBERT**: DistilBERT ottimizzato (~66M parametri, 91.3% accuracy su SST-2)
- **RoBERTa-base**: Fine-tuned su Amazon Reviews (~125M parametri)
- **BERT-large**: Fine-tuned su SST-2 (~340M parametri)
- **RoBERTa-large**: Multi-dataset (~355M parametri)

## Metodi XAI Implementati

- **`grad_input`**: Gradient × Input a livello di embedding
- **`lrp`**: Layer-wise Relevance Propagation
- **`attention_rollout`**: Rollout dell'attenzione tra layer
- **`lime`**: Local Interpretable Model-agnostic Explanations
- **`shap`**: SHapley Additive exPlanations (Kernel)

## Metriche di Valutazione

Implementa **3 metriche automatiche** basate su Mersha et al. (2025):

### 1.  Robustness

Cosa misura: Stabilità delle spiegazioni sotto perturbazioni del testo
Metodo: Mean Average Difference (MAD) tra spiegazioni originali e perturbate
Interpretazione: Più basso = più robusto
Formula: Differenze element-wise tra attribution su testo originale e perturbato

### 2. Consistency

Cosa misura: Accordo tra spiegazioni di modelli diversi sullo stesso testo
Metodo: Correlazione di Spearman tra importanze di token comuni
Interpretazione: Più alto = più consistente
Note: Adattata per confrontare modelli diversi pre-addestrati (invece di modelli identici con seed diversi)

### 3. Contrastivity

Cosa misura: Diversità delle spiegazioni tra classi opposte (positivo vs negativo)
Metodo: Divergenza KL tra distribuzioni di importanza per le due classi
Interpretazione: Più alto = più contrastivo
Utilità: Verifica che l'XAI evidenzi feature diverse per predizioni opposte

## Installazione

```bash
git clone https://github.com/username/xai-benchmark.git
cd xai-benchmark
pip install -r requirements.txt
```

## Utilizzo

### Test Sistema
```bash
python test_everything.py --level 5
```

### Spiegazione Singola
```bash
python main.py explain --model distilbert --explainer lime --text "This movie is amazing!"
```

### Valutazione Metrica
```bash
python main.py evaluate --metric robustness --model distilbert --explainer grad_input --sample 500
```

### Report Completo
```bash
python report.py --sample 50 --csv
```

## Configurazione

Parametri principali in `dataset.py`, `metrics.py`, `evaluate.py`:
- `MAX_LENGTH = 512` - Lunghezza massima sequenze
- `DEFAULT_SAMPLE_SIZE = 500` - Esempi per valutazione
- `DEFAULT_PERTURBATION_RATIO = 0.15` - Percentuale perturbazioni
