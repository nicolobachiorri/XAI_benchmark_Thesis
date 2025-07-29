# XAI Benchmark Framework

Framework completo per la valutazione di metodi di eXplainable AI su modelli di sentiment analysis, ottimizzato per Google Colab con avanzate ottimizzazioni delle prestazioni.

## Introduzione

Questo framework valuta la qualità delle spiegazioni generate da diversi metodi XAI su modelli pre-addestrati per l'analisi del sentiment. Il sistema utilizza **clustering intelligente** per ridurre il dataset IMDB da 25.000 a 400 esempi rappresentativi (80 cluster × 5 campioni), mantenendo la diversità e il bilanciamento dei dati.

Il framework è specificamente progettato per Google Colab con gestione avanzata della memoria GPU, caching intelligente e backup automatico su Google Drive.

## Explainers Supportati

| Metodo | Descrizione | Dipendenze |
|--------|-------------|------------|
| **LIME** | Local Interpretable Model-agnostic Explanations | lime |
| **SHAP** | SHapley Additive exPlanations | shap |
| **Gradient×Input** | Attribution basata su gradienti | - |
| **Attention Rollout** | Analisi del flusso di attention | - |
| **Attention Flow** | Analisi di attention basata su grafi | networkx |
| **LRP** | Layer-wise Relevance Propagation | - |

Le dipendenze mancanti vengono installate automaticamente durante l'esecuzione.

## Metriche di Valutazione

### Robustness
Misura la stabilità delle spiegazioni sotto perturbazioni dell'input (masking, cancellazione, sostituzione).
- **Range**: 0.0 - 1.0 (più basso = più robusto)
- **Buon punteggio**: < 0.1

### Consistency
Misura la stabilità delle spiegazioni con diversi seed di inferenza (comportamento stocastico).
- **Range**: 0.0 - 1.0 (più alto = più consistente)  
- **Buon punteggio**: > 0.8

### Contrastivity
Misura quanto bene le spiegazioni discriminano tra sentiment positivi e negativi.
- **Range**: 0.0 - ∞ (più alto = migliore discriminazione)
- **Buon punteggio**: > 2.0

## Struttura del Progetto

```
├── main.py           # Interfaccia principale e funzioni core
├── report.py         # Generazione report completo
├── models.py         # Gestione modelli pre-addestrati
├── dataset.py        # Dataset IMDB con clustering
├── explainers.py     # Implementazione metodi XAI
├── metrics.py        # Metriche di valutazione
├── utils.py          # Utility per Colab e ottimizzazioni
├── requirements.txt  # Dipendenze Python
└── Test.csv          # Dataset IMDB test (richiesto)
```

## Modelli Disponibili

| Modello | Descrizione | Dimensione |
|---------|-------------|------------|
| `tinybert` | BERT leggero per IMDB | Piccolo |
| `distilbert` | DistilBERT per SST-2 | Piccolo |
| `roberta-base` | RoBERTa base per sentiment | Medio |
| `bert-large` | BERT Large per SST-2 | Grande |
| `roberta-large` | RoBERTa Large per sentiment | Grande |

## Ottimizzazioni Implementate

### Gestione Memoria Avanzata
- **Adaptive Batch Sizing**: Dimensioni batch dinamiche basate su memoria disponibile
- **GPU Memory Pool**: Pool di memoria pre-allocato per evitare frammentazione
- **Progressive Cleanup**: Pulizia memoria automatica tra batch
- **Memory Monitoring**: Monitoraggio continuo con alerting

### Caching Intelligente
- **Embedding Cache**: Cache per embeddings e tokenizzazioni
- **Checkpoint System**: Salvataggio automatico dei progressi
- **Resume Capability**: Ripresa automatica da checkpoint esistenti

### Ottimizzazioni Dataset
- **Clustering K-means**: 80 cluster con 5 campioni rappresentativi
- **Stratified Sampling**: Mantenimento bilanciamento positivo/negativo
- **Memory-Efficient Loading**: Caricamento ottimizzato per Colab

### I/O Asincrono
- **Background Saving**: Salvataggio file in background
- **Thread Pool I/O**: Operazioni I/O parallele
- **Google Drive Backup**: Backup automatico risultati

### Processamento Sequenziale
- **Simplified Architecture**: Architettura semplificata senza parallelizzazione explainer
- **Error Recovery**: Gestione errori robusta con fallback
- **Resource Allocation**: Allocazione dinamica risorse

## Uso Rapido

```bash
# Report completo veloce
!python report.py --sample 100
```

I risultati vengono salvati automaticamente in CSV e JSON con backup su Google Drive.