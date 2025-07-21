# Spiegazione Euristica degli Explainer XAI

## Overview degli Explainer

Gli explainer XAI rispondono alla domanda: **"Perché il modello ha fatto questa predizione?"**

Ogni explainer usa un **approccio diverso** per identificare quali parole sono più importanti:

| Explainer | Approccio | Tipo | Velocità | Accuratezza |
|-----------|-----------|------|----------|-------------|
| **LIME** | Perturbazione locale | Model-agnostic | Media | Alta |
| **SHAP** | Valori di Shapley | Model-agnostic | Lenta | Molto alta |
| **Grad×Input** | Gradienti × Input | Gradient-based | Veloce | Media |
| **Attention Rollout** | Propagazione attention | Attention-based | Veloce | Media |
| **Attention Flow** | Flusso su grafo | Attention-based | Lenta | Alta |
| **LRP** | Propagazione rilevanza | Gradient-based | Media | Alta |

---

## 1. LIME - "Local Interpretable Model-agnostic Explanations"

### Idea Intuitiva:
> *"Imparo un modello semplice che approssima il comportamento del modello complesso vicino a questo specifico esempio"*

### Come Funziona:

```python
def lime_explain(text, model):
    # 1. Prendi il testo originale
    original_text = "This movie is fantastic!"
    original_prediction = model(original_text)  # 0.85 (positivo)
    
    # 2. Genera variazioni rimuovendo parole
    variations = [
        "movie is fantastic!",        # rimuovi "This"
        "This is fantastic!",         # rimuovi "movie"  
        "This movie fantastic!",      # rimuovi "is"
        "This movie is",              # rimuovi "fantastic!"
        "This movie",                 # rimuovi "is fantastic!"
        # ... e molte altre combinazioni
    ]
    
    # 3. Ottieni predizioni per tutte le variazioni
    predictions = []
    for variation in variations:
        pred = model(variation)
        predictions.append(pred)
    
    # 4. Addestra modello lineare semplice
    # X = presenza/assenza di ogni parola (0/1)
    # Y = predizioni del modello
    linear_model = LinearRegression()
    linear_model.fit(X, predictions)
    
    # 5. I coefficienti = importanza delle parole
    importance = linear_model.coefficients
    return importance
```

### Esempio Pratico:

```
Testo: "This movie is fantastic!"
Predizione originale: 0.85 (positivo)

Variazioni testate:
"movie is fantastic!" → 0.82 (drop di 0.03)
"This is fantastic!" → 0.78 (drop di 0.07)  
"This movie fantastic!" → 0.84 (drop di 0.01)
"This movie is" → 0.15 (drop di 0.70)

Modello lineare impara:
- "This": coefficiente +0.03
- "movie": coefficiente +0.07
- "is": coefficiente +0.01  
- "fantastic!": coefficiente +0.70

LIME Result: [("fantastic!", 0.70), ("movie", 0.07), ("This", 0.03), ("is", 0.01)]
```

### Vantaggi:
- **Model-agnostic**: Funziona con qualsiasi modello
- **Intuitivo**: Facile da capire
- **Locale**: Si concentra sull'esempio specifico

### Svantaggi:
- **Lento**: Deve generare molte variazioni
- **Instabile**: Risultati possono variare tra run
- **Approssimato**: Modello lineare è semplificazione

---

## 2. SHAP - "SHapley Additive exPlanations"

### Idea Intuitiva:
> *"Quanto contribuisce ogni parola al risultato finale? Calcoliamo il 'fair share' di ogni parola usando la teoria dei giochi"*

### Come Funziona:

```python
def shap_explain(text, model):
    words = text.split()  # ["This", "movie", "is", "fantastic!"]
    
    # 1. Calcola predizione con tutte le parole
    full_prediction = model("This movie is fantastic!")  # 0.85
    
    # 2. Calcola predizione senza parole (baseline)
    empty_prediction = model("")  # 0.50 (neutro)
    
    # 3. Per ogni parola, calcola contributo marginale
    # considerando TUTTE le possibili coalizioni
    
    shapley_values = []
    for target_word in words:
        marginal_contributions = []
        
        # Considera tutte le coalizioni senza target_word
        for coalition in all_subsets(words - target_word):
            # Predizione senza target_word
            text_without = " ".join(coalition)
            pred_without = model(text_without)
            
            # Predizione con target_word
            text_with = " ".join(coalition + [target_word])
            pred_with = model(text_with)
            
            # Contributo marginale
            marginal = pred_with - pred_without
            marginal_contributions.append(marginal)
        
        # Shapley value = media pesata dei contributi marginali
        shapley_value = weighted_average(marginal_contributions)
        shapley_values.append(shapley_value)
    
    return shapley_values
```

### Esempio Pratico:

```
Testo: "This movie is fantastic!"

Coalizioni per calcolare valore di "fantastic!":
∅ → "fantastic!" : 0.20 - 0.50 = -0.30
{"This"} → {"This", "fantastic!"}: 0.65 - 0.55 = +0.10
{"movie"} → {"movie", "fantastic!"}: 0.75 - 0.60 = +0.15
{"This", "movie"} → {"This", "movie", "fantastic!"}: 0.85 - 0.65 = +0.20
...

Media pesata dei contributi marginali:
SHAP("fantastic!") = 0.45

Risultato finale:
SHAP values: [("fantastic!", 0.45), ("movie", 0.20), ("is", 0.05), ("This", 0.15)]
Verifica: 0.45 + 0.20 + 0.05 + 0.15 = 0.85 ✓
```

### Vantaggi:
- **Teoricamente fondato**: Basato su teoria dei giochi
- **Additivo**: I valori sommano alla predizione finale
- **Fair**: Ogni parola riceve il suo giusto contributo

### Svantaggi:
- **Molto lento**: Crescita esponenziale con numero di parole
- **Computazionalmente intensivo**: Molte combinazioni da testare

---

## 3. Gradient × Input - "Quanto influenza ogni parola sui gradienti?"

### Idea Intuitiva:
> *"Moltiplico quanto il modello 'presta attenzione' a ogni parola (gradiente) per quanto quella parola è 'forte' (valore input)"*

### Come Funziona:

```python
def grad_input_explain(text, model):
    # 1. Tokenizza e ottieni embeddings
    tokens = tokenize(text)  # ["This", "movie", "is", "fantastic!"]
    embeddings = get_embeddings(tokens)  # [emb1, emb2, emb3, emb4]
    embeddings.requires_grad = True
    
    # 2. Forward pass
    prediction = model(embeddings)  # 0.85
    
    # 3. Backward pass per ottenere gradienti
    prediction.backward()
    gradients = embeddings.grad  # [grad1, grad2, grad3, grad4]
    
    # 4. Moltiplica gradienti × input
    attributions = gradients * embeddings  # element-wise
    
    # 5. Somma lungo dimensione embedding
    scores = attributions.sum(dim=-1)  # [score1, score2, score3, score4]
    
    return scores
```

### Esempio Pratico:

```
Testo: "This movie is fantastic!"

Embeddings (semplificati):
"This": [0.1, 0.2, 0.3]
"movie": [0.4, 0.5, 0.6]  
"is": [0.1, 0.1, 0.1]
"fantastic!": [0.8, 0.9, 0.7]

Gradienti dopo backward:
"This": [0.2, 0.1, 0.3]
"movie": [0.5, 0.6, 0.4]
"is": [0.1, 0.2, 0.1]  
"fantastic!": [1.2, 1.5, 1.3]

Gradient × Input:
"This": [0.1×0.2, 0.2×0.1, 0.3×0.3] = [0.02, 0.02, 0.09] → sum = 0.13
"movie": [0.4×0.5, 0.5×0.6, 0.6×0.4] = [0.20, 0.30, 0.24] → sum = 0.74
"is": [0.1×0.1, 0.1×0.2, 0.1×0.1] = [0.01, 0.02, 0.01] → sum = 0.04
"fantastic!": [0.8×1.2, 0.9×1.5, 0.7×1.3] = [0.96, 1.35, 0.91] → sum = 3.22

Result: [("fantastic!", 3.22), ("movie", 0.74), ("This", 0.13), ("is", 0.04)]
```

### Vantaggi:
- **Veloce**: Solo una forward e backward pass
- **Diretto**: Usa informazioni native del modello
- **Efficiente**: Pochi calcoli extra

### Svantaggi:
- **Rumoroso**: Gradienti possono essere instabili
- **Saturation**: Problemi con neuroni saturi
- **Bias**: Favorisce input con valori grandi

---

## 4. Attention Rollout - "Segui il Flusso di Attention"

### Idea Intuitiva:
> *"L'attention ci dice dove il modello 'guarda'. Propaghiamo questo sguardo attraverso tutti i layer per vedere l'effetto finale"*

### Come Funziona:

```python
def attention_rollout_explain(text, model):
    # 1. Ottieni attention weights da tutti i layer
    attentions = model(text, output_attentions=True).attentions
    # attentions[0] = attention del layer 1
    # attentions[1] = attention del layer 2, etc.
    
    # 2. Media attention heads per ogni layer
    layer_attentions = []
    for layer_att in attentions:
        avg_attention = layer_att.mean(dim=1)  # Media sui heads
        layer_attentions.append(avg_attention)
    
    # 3. Aggiungi identità per preservare informazione residuale
    for i, att in enumerate(layer_attentions):
        identity = eye(att.size(-1))
        layer_attentions[i] = 0.5 * att + 0.5 * identity
    
    # 4. Propaga attention attraverso i layer
    rolled_attention = layer_attentions[0]
    for layer_att in layer_attentions[1:]:
        rolled_attention = layer_att @ rolled_attention
    
    # 5. Somma attention verso ogni token
    final_scores = rolled_attention.sum(dim=0)
    
    return final_scores
```

### Esempio Pratico:

```
Testo: "This movie is fantastic!"

Layer 1 attention (dopo media heads + identità):
    This  movie   is  fant
This [0.6,  0.2,  0.1, 0.1]
movie[0.1,  0.6,  0.2, 0.1]  
is   [0.1,  0.2,  0.6, 0.1]
fant [0.2,  0.1,  0.1, 0.6]

Layer 2 attention:
    This  movie   is  fant
This [0.7,  0.1,  0.1, 0.1]
movie[0.2,  0.5,  0.1, 0.2]
is   [0.1,  0.1,  0.7, 0.1]  
fant [0.3,  0.1,  0.1, 0.5]

Rollout (Layer2 @ Layer1):
Risultato finale mostra quanto attention arriva a ogni token
dopo propagazione attraverso tutti i layer.

Final scores: [("fantastic!", 0.45), ("This", 0.25), ("movie", 0.20), ("is", 0.10)]
```

### Vantaggi:
- **Interpretabile**: Basato su attention che è già interpretabile
- **Veloce**: Usa solo attention weights esistenti
- **Multi-layer**: Considera interazioni tra layer

### Svantaggi:
- **Solo Transformers**: Funziona solo con modelli attention-based
- **Assumption**: Assume che attention = importanza (non sempre vero)
- **Averaging**: Perdita informazioni mediando attention heads

---

## 5. Attention Flow - "Flusso Massimo su Grafo di Attention"

### Idea Intuitiva:
> *"Creo un grafo dove l'attention sono le 'strade' e calcolo quanto 'traffico' può fluire dall'output verso ogni parola input"*

### Come Funziona:

```python
def attention_flow_explain(text, model):
    # 1. Ottieni attention da tutti i layer
    attentions = model(text, output_attentions=True).attentions
    
    # 2. Costruisci grafo con nodi per ogni layer×posizione
    graph = NetworkXGraph()
    
    # 3. Aggiungi nodi
    for layer in range(num_layers):
        for pos in range(sequence_length):
            node_id = layer * sequence_length + pos
            graph.add_node(node_id)
    
    # 4. Aggiungi archi basati su attention weights
    for layer in range(num_layers - 1):
        for i in range(sequence_length):
            for j in range(sequence_length):
                source = (layer + 1) * sequence_length + i
                target = layer * sequence_length + j
                capacity = attention[layer][i][j]
                graph.add_edge(source, target, capacity=capacity)
    
    # 5. Calcola flusso massimo dall'output verso ogni token input
    scores = []
    for token_pos in range(sequence_length):
        source = (num_layers - 1) * sequence_length + 0  # Output
        target = 0 * sequence_length + token_pos  # Token input
        
        max_flow = networkx.maximum_flow_value(graph, source, target)
        scores.append(max_flow)
    
    return scores
```

### Esempio Pratico:

```
Testo: "This movie is fantastic!" (3 layer Transformer)

Grafo nodi:
Layer 0: [This₀, movie₁, is₂, fant₃]
Layer 1: [This₄, movie₅, is₆, fant₇]  
Layer 2: [This₈, movie₉, is₁₀, fant₁₁]

Archi con capacità = attention weights:
Layer 1 → Layer 0:
This₄ → This₀ (capacità: 0.6)
This₄ → movie₁ (capacità: 0.2)
movie₅ → movie₁ (capacità: 0.7)
...

Flusso massimo:
Output (fant₁₁) → This₀: 0.12
Output (fant₁₁) → movie₁: 0.25  
Output (fant₁₁) → is₂: 0.08
Output (fant₁₁) → fant₃: 0.55

Result: [("fantastic!", 0.55), ("movie", 0.25), ("This", 0.12), ("is", 0.08)]
```

### Vantaggi:
- **Teoricamente robusto**: Basato su teoria dei grafi
- **Path-aware**: Considera tutti i percorsi possibili
- **Quantitativo**: Misura precisa del "flusso di informazione"

### Svantaggi:
- **Molto lento**: Algoritmi di flusso massimo sono costosi
- **Complesso**: Difficile da debuggare e ottimizzare
- **Assumptions**: Assume che attention = capacità di flusso

---

## 6. LRP - "Layer-wise Relevance Propagation"

### Idea Intuitiva:
> *"Distribuisco la 'responsabilità' della predizione finale verso gli input, layer per layer, seguendo regole di conservazione"*

### Come Funziona:

```python
def lrp_explain(text, model):
    # 1. Forward pass normale
    embeddings = get_embeddings(text)
    prediction = model(embeddings)  # 0.85
    
    # 2. Inizia con rilevanza = predizione finale
    final_relevance = prediction  # 0.85
    
    # 3. Propaga rilevanza indietro layer per layer
    # Regola ε: R_i = Σ_j (a_i * w_ij) / (Σ_k (a_k * w_kj) + ε) * R_j
    
    current_relevance = final_relevance
    
    # Per ogni layer (dall'ultimo al primo)
    for layer in reversed(model.layers):
        input_relevance = []
        
        for input_neuron in layer.inputs:
            # Calcola quanto questo input contribuisce a ogni output
            contributions = []
            for output_neuron in layer.outputs:
                activation = input_neuron.value
                weight = layer.weight[input_neuron][output_neuron]
                
                # Contributo = (input × weight) / (somma tutti input × weight)
                numerator = activation * weight
                denominator = sum(inp.value * layer.weight[inp][output_neuron] 
                                for inp in layer.inputs) + epsilon
                
                contribution = numerator / denominator
                relevance_from_output = contribution * current_relevance[output_neuron]
                contributions.append(relevance_from_output)
            
            input_relevance.append(sum(contributions))
        
        current_relevance = input_relevance
    
    # 4. Rilevanza finale per ogni token di input
    return current_relevance
```

### Esempio Pratico:

```
Testo: "This movie is fantastic!"
Predizione finale: 0.85

Layer 3 (output): R = [0.85]

Layer 2: Distribuisci 0.85 tra neuroni nascosti
R = [0.12, 0.23, 0.08, 0.42]

Layer 1: Distribuisci rilevanze verso embeddings
R = [0.05, 0.18, 0.02, 0.60]

Embeddings: Somma rilevanza per ogni token
"This": 0.05
"movie": 0.18
"is": 0.02  
"fantastic!": 0.60

Verifica conservazione: 0.05 + 0.18 + 0.02 + 0.60 = 0.85 ✓

Result: [("fantastic!", 0.60), ("movie", 0.18), ("This", 0.05), ("is", 0.02)]
```

### Vantaggi:
- **Conservativo**: La rilevanza totale si conserva
- **Layer-wise**: Traccia il flusso attraverso ogni layer
- **Principled**: Regole matematiche ben definite

### Svantaggi:
- **Complesso**: Implementazione non triviale
- **Model-specific**: Regole diverse per layer diversi
- **Instabile**: Divisioni per zero e problemi numerici

---





