# Spiegazione Euristica delle 3 Metriche XAI

## Overview delle Metriche

Le 3 metriche valutano **diversi aspetti della qualità** degli explainer:

| Metrica | Cosa Misura | Range | Migliore |
|---------|-------------|-------|----------|
| **Robustness** | Stabilità sotto perturbazioni | [0, +∞] | **Più basso** |
| **Consistency** | Stabilità tra seed diversi | [-1, +1] | **Più alto** |
| **Contrastivity** | Differenza tra classi opposte | [0, +∞] | **Più alto** |

---

## 1. ROBUSTNESS - "Quanto è Stabile l'Explainer?"

### Idea Intuitiva:
> *"Se cambio leggermente il testo, l'explanation dovrebbe rimanere simile"*

### Come Funziona:

```python
def compute_robustness(model, tokenizer, explainer, text):
    # 1. Genera explanation originale
    original_explanation = explainer("This movie is fantastic!")
    # → tokens: ["This", "movie", "is", "fantastic", "!"]
    # → scores: [0.1,   0.2,    0.05, 0.8,        0.1]
    
    all_differences = []
    
    # 2. Applica 3 tipi di perturbazioni
    for perturbation_function in [mask, delete, substitute]:
        for attempt in [1, 2]:  # 2 tentativi per tipo
            
            # 3. Perturba il testo
            perturbed_text = perturbation_function("This movie is fantastic!")
            # Esempi:
            # - Mask: "This [MASK] is fantastic!"
            # - Delete: "This is fantastic!" 
            # - Substitute: "This movie is great!"
            
            # 4. Genera explanation perturbata
            perturbed_explanation = explainer(perturbed_text)
            
            # 5. Calcola differenze per token condivisi
            for token, original_score in original_explanation:
                if token in perturbed_explanation:
                    perturbed_score = perturbed_explanation[token]
                    difference = abs(original_score - perturbed_score)
                    all_differences.append(difference)
    
    # 6. Media di tutte le differenze
    robustness = mean(all_differences)
    return robustness  # Più basso = più robusto
```

### Esempio Pratico:

```
Testo originale: "This movie is fantastic!"
Scores originali: [0.1, 0.2, 0.05, 0.8, 0.1]

Perturbazione 1: "This [MASK] is fantastic!"
Scores perturbati: [0.12, -, 0.04, 0.75, 0.09]
Differenze: |0.1-0.12|=0.02, |0.05-0.04|=0.01, |0.8-0.75|=0.05, |0.1-0.09|=0.01

Perturbazione 2: "This movie is great!"
Scores perturbati: [0.11, 0.18, 0.06, 0.72, 0.13]
Differenze: |0.1-0.11|=0.01, |0.2-0.18|=0.02, |0.05-0.06|=0.01, |0.8-0.72|=0.08

Media finale: (0.02+0.01+0.05+0.01+0.01+0.02+0.01+0.08)/8 = 0.026

ROBUSTNESS = 0.026 → Molto robusto!
```

### Interpretazione:
- **0.000-0.050**: Molto robusto
- **0.050-0.100**: Robusto  
- **0.100-0.200**: Moderato
- **>0.200**: Poco robusto

---

## 2. CONSISTENCY - "È Consistente tra Seed Diversi?"

### Idea Intuitiva:
> *"Se uso seed diversi (dropout randomness), l'explanation dovrebbe essere simile"*

### Come Funziona:

```python
def compute_consistency(model, tokenizer, explainer, texts, seeds=[42, 123, 456, 789]):
    # 1. Metti modello in training mode (attiva dropout)
    model.train()
    
    explanations_by_seed = {}
    
    # 2. Per ogni seed, genera explanations
    for seed in seeds:
        set_random_seed(seed)  # Cambia comportamento dropout
        
        explanations = []
        for text in texts:
            explanation = explainer(text)
            explanations.append(explanation)
        
        explanations_by_seed[seed] = explanations
    
    # 3. Calcola correlazioni tra tutte le coppie di seed
    correlations = []
    for seed_a, seed_b in combinations(seeds):
        correlation = spearman_correlation(
            explanations_by_seed[seed_a], 
            explanations_by_seed[seed_b]
        )
        correlations.append(correlation)
    
    # 4. Media delle correlazioni
    consistency = mean(correlations)
    return consistency  # Più alto = più consistente
```

### Esempio Pratico:

```
Testo: "Great movie!"

Seed 42:  explanation = [("Great", 0.8), ("movie", 0.2)]
Seed 123: explanation = [("Great", 0.75), ("movie", 0.25)]
Seed 456: explanation = [("Great", 0.82), ("movie", 0.18)]
Seed 789: explanation = [("Great", 0.78), ("movie", 0.22)]

Correlazioni:
- Seed 42 vs 123: spearman([0.8, 0.2], [0.75, 0.25]) = 0.95
- Seed 42 vs 456: spearman([0.8, 0.2], [0.82, 0.18]) = 0.98  
- Seed 42 vs 789: spearman([0.8, 0.2], [0.78, 0.22]) = 0.97
- Seed 123 vs 456: spearman([0.75, 0.25], [0.82, 0.18]) = 0.93
- Seed 123 vs 789: spearman([0.75, 0.25], [0.78, 0.22]) = 0.99
- Seed 456 vs 789: spearman([0.82, 0.18], [0.78, 0.22]) = 0.96

CONSISTENCY = (0.95+0.98+0.97+0.93+0.99+0.96)/6 = 0.96 → Molto consistente!
```

### Interpretazione:
- **0.9-1.0**: Molto consistente
- **0.7-0.9**: Consistente
- **0.5-0.7**: Moderato  
- **<0.5**: Inconsistente

---

## 3. CONTRASTIVITY - "Distingue Bene le Classi Opposte?"

### Idea Intuitiva:
> *"Le explanations per review positive e negative dovrebbero essere molto diverse"*

### Come Funziona:

```python
def compute_contrastivity(positive_attrs, negative_attrs):
    # 1. Accumula scores per token attraverso tutti gli esempi
    positive_tokens = {}  # token -> score_totale
    negative_tokens = {}
    
    # Per esempi positivi
    for explanation in positive_attrs:
        for token, score in explanation:
            positive_tokens[token] = positive_tokens.get(token, 0) + score
    
    # Per esempi negativi  
    for explanation in negative_attrs:
        for token, score in explanation:
            negative_tokens[token] = negative_tokens.get(token, 0) + score
    
    # 2. Crea vocabolario unificato
    all_tokens = set(positive_tokens.keys()) | set(negative_tokens.keys())
    
    # 3. Crea distribuzioni di probabilità
    pos_distribution = []
    neg_distribution = []
    
    for token in all_tokens:
        pos_score = positive_tokens.get(token, 0.0)
        neg_score = negative_tokens.get(token, 0.0)
        
        pos_distribution.append(pos_score)
        neg_distribution.append(neg_score)
    
    # 4. Normalizza a distribuzioni di probabilità
    pos_distribution = normalize(pos_distribution)
    neg_distribution = normalize(neg_distribution)
    
    # 5. Calcola divergenza KL tra le distribuzioni
    contrastivity = KL_divergence(pos_distribution, neg_distribution)
    return contrastivity  # Più alto = più contrastivo
```

### Esempio Pratico:

```
Review Positive: "Great fantastic movie!"
→ explanation: [("Great", 0.8), ("fantastic", 0.9), ("movie", 0.1)]

Review Negative: "Terrible awful movie!"  
→ explanation: [("Terrible", 0.7), ("awful", 0.8), ("movie", 0.1)]

Accumulo scores:
positive_tokens = {"Great": 0.8, "fantastic": 0.9, "movie": 0.1}
negative_tokens = {"Terrible": 0.7, "awful": 0.8, "movie": 0.1}

Vocabolario: ["Great", "fantastic", "movie", "Terrible", "awful"]

Distribuzioni:
positive: [0.8, 0.9, 0.1, 0.0, 0.0] → normalizzata: [0.44, 0.50, 0.06, 0.00, 0.00]
negative: [0.0, 0.0, 0.1, 0.7, 0.8] → normalizzata: [0.00, 0.00, 0.06, 0.44, 0.50]

KL-divergence = molto alta perché distribuzioni molto diverse!
CONTRASTIVITY = 4.2 → Molto contrastivo!
```

### Interpretazione:
- **>5.0**: Molto contrastivo
- **2.0-5.0**: Contrastivo
- **1.0-2.0**: Moderato
- **<1.0**: Poco contrastivo

---

## Riassunto Pratico

### Cosa Misura Ogni Metrica:

1. **ROBUSTNESS** → *"L'explainer è affidabile?"*
   - Testo leggermente diverso → Explanation simile = ROBUSTO
   - Testo leggermente diverso → Explanation diversa = FRAGILE

2. **CONSISTENCY** → *"L'explainer è deterministico?"*  
   - Stesso testo, seed diversi → Explanation simile = CONSISTENTE
   - Stesso testo, seed diversi → Explanation diversa = RANDOM

3. **CONTRASTIVITY** → *"L'explainer distingue le classi?"*
   - Review positive vs negative → Explanation diverse = CONTRASTIVO
   - Review positive vs negative → Explanation simili = INDISTINGUIBILE

### Interpretazione dei Risultati:

Un **explainer ideale** dovrebbe avere:
- **Robustness BASSO** (stabile sotto perturbazioni)
- **Consistency ALTO** (deterministico)  
- **Contrastivity ALTO** (distingue le classi)

### Trade-offs Tipici:
- **LIME**: Alta contrastivity, bassa consistency
- **Attention**: Alta consistency, bassa contrastivity
- **SHAP**: Bilanciato su tutte le metriche