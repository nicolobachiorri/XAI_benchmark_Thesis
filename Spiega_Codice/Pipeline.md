# Pipeline Completa XAI Report

## Pipeline di Processing Completa

```
Dataset IMDB Originale (5000+ esempi)
         ↓
    TF-IDF Embeddings
         ↓
    K-Means (80 clusters)
         ↓
  Campionamento (5 per cluster)
         ↓
    Dataset Clusterizzato (400 esempi)
         ↓
   Cache Dataset & Verifica Qualità
         ↓
    Caricamento Modelli Pre-trained
    [tinybert, distilbert, roberta-base, bert-large, roberta-large]
         ↓
    Smart Model Ordering (piccoli → grandi)
         ↓
    Inizializzazione Ottimizzazioni
    [Memory Manager, Embedding Cache, Parallel Manager, Async I/O]
         ↓
  ┌────────────────────────────────────────┐
  │         LOOP MODELLI (5 iterazioni)    │
  │                                        │
  │  Modello Corrente (es. distilbert)     │
  │          ↓                             │
  │  Caricamento Modello + Tokenizer       │
  │          ↓                             │
  │  Adaptive Batch Size Calculation       │
  │  (basato su memoria disponibile)       │
  │          ↓                             │
  │  Preparazione Dati Stratificata        │
  │  [pos_texts, neg_texts, consistency]   │
  │          ↓                             │
  │  Split Explainer                       │
  │  [Parallel: lime] [Sequential: shap,   │
  │   grad_input, attention_*, lrp]        │
  │          ↓                             │
  │  ┌─────────────────────────────────┐   │
  │  │    FASE EXPLAINER PARALLELI     │   │
  │  │                                 │   │
  │  │  ThreadPool Execution (LIME)    │   │
  │  │           ↓                     │   │
  │  │  ┌─────────────────────────┐    │   │
  │  │  │    LOOP METRICHE        │    │   │
  │  │  │                         │    │   │
  │  │  │  Robustness Evaluation  │    │   │
  │  │  │  [perturbation analysis]│    │   │
  │  │  │          ↓              │    │   │
  │  │  │ Contrastivity Analysis  │    │   │
  │  │  │ [pos vs neg attribution]│    │   │
  │  │  │          ↓              │    │   │
  │  │  │  Consistency Evaluation │    │   │
  │  │  │[inference seed variance]│    │   │
  │  │  └─────────────────────────┘    │   │
  │  │           ↓                     │   │
  │  │  Aggregazione Risultati Parallel│   │
  │  └─────────────────────────────────┘   │
  │          ↓                             │
  │  Memory Cleanup (medium)               │
  │          ↓                             │
  │  ┌─────────────────────────────────┐   │
  │  │   FASE EXPLAINER SEQUENZIALI    │   │
  │  │                                 │   │
  │  │  Per ogni explainer:            │   │
  │  │  [shap, grad_input, attention_*,│   │
  │  │   lrp]                          │   │
  │  │           ↓                     │   │
  │  │  Caricamento Explainer          │   │
  │  │           ↓                     │   │
  │  │  ┌─────────────────────────┐    │   │
  │  │  │    LOOP METRICHE        │    │   │
  │  │  │                         │    │   │
  │  │  │  Robustness:            │    │   │
  │  │  │  evaluate_robustness_   │    │   │
  │  │  │  over_dataset()         │    │   │
  │  │  │          ↓              │    │   │
  │  │  │  Contrastivity:         │    │   │
  │  │  │  process_attributions_  │    │   │
  │  │  │  batch() → compute_     │    │   │
  │  │  │  contrastivity()        │    │   │
  │  │  │          ↓              │    │   │
  │  │  │  Consistency:           │    │   │
  │  │  │  evaluate_consistency_  │    │   │
  │  │  │  over_dataset()         │    │   │
  │  │  │  [con inference seeds]  │    │   │
  │  │  └─────────────────────────┘    │   │
  │  │           ↓                     │   │
  │  │  Memory Cleanup (light)         │   │
  │  └─────────────────────────────────┘   │
  │          ↓                             │
  │  Checkpoint Save                       │
  │  (results_modelname_timestamp.json)    │
  │          ↓                             │
  │  Async I/O Save                        │
  │          ↓                             │
  │  Memory Cleanup (aggressive)           │
  └────────────────────────────────────────┘
         ↓
    Aggregazione Risultati Finali
    (all_results dictionary)
         ↓
  ┌─────────────────────────────────────────┐
  │         COSTRUZIONE TABELLE             │
  │                                         │
  │  build_report_tables()                  │
  │          ↓                              │
  │  Per ogni metrica:                      │
  │  ┌─────────────────────────────────┐    │
  │  │    ROBUSTNESS TABLE             │    │
  │  │                                 │    │
  │  │  Struttura: explainer × model   │    │
  │  │  DataFrame con score robustness │    │
  │  └─────────────────────────────────┘    │
  │  ┌─────────────────────────────────┐    │
  │  │    CONTRASTIVITY TABLE          │    │
  │  │                                 │    │
  │  │  Struttura: explainer × model   │    │
  │  │  DataFrame con score contrast   │    │
  │  └─────────────────────────────────┘    │
  │  ┌─────────────────────────────────┐    │
  │  │    CONSISTENCY TABLE            │    │
  │  │                                 │    │
  │  │  Struttura: explainer × model   │    │
  │  │  DataFrame con score consistency│    │
  │  └─────────────────────────────────┘    │
  └─────────────────────────────────────────┘
         ↓
    Analisi Statistica per Tabella
    [print_table_analysis()]
         ↓
    ┌─────────────────────────────────────┐
    │        RANKING E INSIGHTS           │
    │                                     │
    │  Per-Explainer Statistics           │
    │  [media, std, coverage per row]     │
    │          ↓                          │
    │  Per-Model Statistics               │
    │  [media, std, coverage per column]  │
    │          ↓                          │
    │  Top 5 Combinations                 │
    │  [migliori coppie explainer-model]  │
    │          ↓                          │
    │  Coverage Analysis                  │
    │  [% celle completate con successo]  │
    └─────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────┐
    │         OUTPUT GENERAZIONE          │
    │                                     │
    │  CSV Export                         │
    │  [robustness_table.csv,             │
    │   contrastivity_table.csv,          │
    │   consistency_table.csv]            │
    │          ↓                          │
    │  JSON Results Export                │
    │  [results_model_timestamp.json      │
    │   per ogni modello]                 │
    │          ↓                          │
    │  Summary Report Testuale            │
    │  [generate_summary_report()]        │
    │          ↓                          │
    │  Performance Profiling              │
    │  [timing, memory usage per fase]    │
    └─────────────────────────────────────┘
         ↓
    Report Finale Completato
    ┌─────────────────────────────────────┐
    │          FILES GENERATI             │
    │                                     │
    │   3 CSV Tables (metriche)           │
    │   Summary Report (analisi)          │
    │   5 JSON Files (risultati raw)      │
    │   Performance Log (profiling)       │
    └─────────────────────────────────────┘
```

## Dettaglio Flussi Paralleli e Sequenziali

### Flusso Explainer Paralleli (Thread Pool)
```
Input: [lime] explainer list
         ↓
ThreadPoolExecutor (max_workers=4)
         ↓
┌─────────────────────────────────────────┐
│          TASK PARALLELO LIME            │
│                                         │
│  create_explainer_task(lime)            │
│           ↓                             │
│  explainer = get_explainer(lime, ...)   │
│           ↓                             │
│  ┌─────────────────────────────────┐    │
│  │      LOOP METRICHE PARALLELO    │    │
│  │                                 │    │
│  │  robustness = evaluate_...()    │    │
│  │  contrastivity = compute_...()  │    │
│  │  consistency = evaluate_...()   │    │
│  │  torch.cuda.empty_cache()       │    │
│  └─────────────────────────────────┘    │
│           ↓                             │
│  return {robustness: X, contrastivity:  │
│          Y, consistency: Z}             │
└─────────────────────────────────────────┘
         ↓
as_completed() collector → risultati aggregati
```

### Flusso Explainer Sequenziali (GPU-bound)
```
Input: [shap, grad_input, attention_rollout, attention_flow, lrp]
         ↓
For explainer in sequential_explainers:
         ↓
┌─────────────────────────────────────────┐
│         PROCESSING SEQUENZIALE          │
│                                         │
│  explainer = get_explainer(name, ...)   │
│           ↓                             │
│  for metric_name in metrics:            │
│           ↓                             │
│  ┌─────────────────────────────────┐    │
│  │     METRIC EVALUATION           │    │
│  │                                 │    │
│  │  if metric == "robustness":     │    │
│  │    score = evaluate_robustness_ │    │
│  │           over_dataset()        │    │
│  │  elif metric == "contrastivity":│    │
│  │    pos_attrs = process_...()    │    │
│  │    neg_attrs = process_...()    │    │
│  │    score = compute_contrast...()│    │
│  │  elif metric == "consistency":  │    │
│  │    score = evaluate_consistency_│    │
│  │           over_dataset()        │    │
│  └─────────────────────────────────┘    │
│           ↓                             │
│  results[metric][explainer] = score     │
│           ↓                             │
│  memory_cleanup(light)                  │
└─────────────────────────────────────────┘
```

## Flusso Dati attraverso le Metriche

### Robustness Evaluation Flow
```
Input: texts (consistency_texts)
         ↓
evaluate_robustness_over_dataset()
         ↓
For each text:
  ┌─────────────────────────────────────┐
  │        PERTURBATION LOOP            │
  │                                     │
  | original_attribution = explainer(text)  │
  │           ↓                         │
  │  for perturbation_func in [mask,    │
  │       delete, substitute]:          │
  │           ↓                         │
  │  perturbed_text = perturbation_func(text) │
  │           ↓                         │
  │  perturbed_attribution = explainer(│
  │                    perturbed_text)  │
  │           ↓                         │
  │  compute_difference(original,       │
  │                    perturbed)       │
  └─────────────────────────────────────┘
         ↓
aggregate_differences() → robustness_score
```

### Contrastivity Evaluation Flow
```
Input: pos_texts, neg_texts
         ↓
process_attributions_batch(pos_texts, explainer)
         ↓                    ↓
pos_attributions        neg_attributions
         ↓                    ↓
         └────────┬───────────┘
                  ↓
        compute_contrastivity()
                  ↓
        ┌─────────────────────────┐
        │   TOKEN ACCUMULATION    │
        │                         │
        │  For attr in pos_attrs: │
        │    token_scores_pos[tok]│
        │           += score      │
        │  For attr in neg_attrs: │
        │    token_scores_neg[tok]│
        │           += score      │
        └─────────────────────────┘
                  ↓
        vocab = union(pos_tokens, neg_tokens)
                  ↓
        pos_distribution = normalize(pos_scores)
        neg_distribution = normalize(neg_scores)
                  ↓
        KL_divergence(pos_dist, neg_dist) → contrastivity_score
```

### Consistency Evaluation Flow
```
Input: texts, seeds [42, 123, 456, 789]
         ↓
evaluate_consistency_over_dataset()
         ↓
model.train()  # Abilita dropout per variabilità
         ↓
For each seed:
  ┌─────────────────────────────────────┐
  │        SEED-SPECIFIC PROCESSING     │
  │                                     │
  │  set_seed(seed)                     │
  │           ↓                         │
  │  explanations = []                  │
  │  for text in texts:                 │
  │           ↓                         │
  │  attribution = explainer(text)      │
  │  # con dropout stocastico attivo    │
  │           ↓                         │
  │  explanations.append(attribution)   │
  └─────────────────────────────────────┘
         ↓
explanations_by_seed[seed] = explanations
         ↓
compute_pairwise_correlations(explanations_by_seed)
         ↓
For seed_pair in combinations(seeds, 2):
  ┌─────────────────────────────────────┐
  │     CORRELATION COMPUTATION         │
  │                                     │
  │  explanations_a = by_seed[seed_a]   │
  │  explanations_b = by_seed[seed_b]   │
  │           ↓                         │
  │  for (attr_a, attr_b) in zip(...):  │
  │           ↓                         │
  │  shared_tokens = intersect(tokens)  │
  │           ↓                         │
  │  spearmanr(shared_scores_a,         │
  │           shared_scores_b)          │
  └─────────────────────────────────────┘
         ↓
mean(all_correlations) → consistency_score
```

## Ottimizzazioni Integrate nel Flusso

### Memory Management Integration
```
Ogni 50 esempi → clear_memory_if_needed()
         ↓
Dopo ogni explainer → progressive_cleanup(light)
         ↓
Dopo explainer paralleli → progressive_cleanup(medium)  
         ↓
Dopo ogni modello → progressive_cleanup(aggressive)
```

### Caching Integration  
```
Primo utilizzo embedding → miss → compute → save
         ↓
Utilizzi successivi → hit → load da cache → speedup 5-10x
```

### Adaptive Batching Integration
```
available_memory > 8GB → batch_size = 40
4GB < available_memory ≤ 8GB → batch_size = 20  
2GB < available_memory ≤ 4GB → batch_size = 10
available_memory ≤ 2GB → batch_size = 5
```

Questa pipeline completa mostra il flusso end-to-end dall'input del dataset raw fino ai risultati finali, includendo tutte le ottimizzazioni, parallelizzazioni, e strategie di gestione memoria implementate nel sistema.