"""
HumanReasoning.py - Modulo per generazione e valutazione Human Reasoning Agreement
==================================================================================

FUNZIONALITÀ:
1. Genera ground truth usando LLM (OpenRouter/Kimi-K2)
2. Calcola metrica Human Reasoning Agreement (MAP)
3. Integrazione seamless con main.py e report.py
4. Cache automatica e gestione errori

UTILIZZO:
```python
import HumanReasoning as hr

# 1. Genera ground truth (una volta sola)
hr.generate_ground_truth(api_key="your_key")

# 2. Valuta explainer
score = hr.evaluate_human_reasoning(model, tokenizer, explainer)

# 3. Integra con framework esistente
# (automaticamente riconosciuto da main.py e report.py)
```
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import re
import ast
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm
import dataset
import explainers

# ==== Configurazione ====
HR_CACHE_FILE = "hr_ground_truth_400.csv"
HR_CACHE_DIR = Path("hr_cache")
HR_CACHE_DIR.mkdir(exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "moonshotai/kimi-k2:free"
DEFAULT_TOP_K = 10
RATE_LIMIT_DELAY = 1.2  # Secondi tra chiamate API

print("[HR] Human Reasoning module loaded")

# ==== Classe Generator ====
class HRGenerator:
    """Generatore Human Reasoning Ground Truth usando OpenRouter API"""
    
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Per tracking
            "X-Title": "XAI Human Reasoning Benchmark"
        }
        
    def create_prompt(self, text: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Crea prompt ottimizzato per il LLM"""
        return f"""Analizza questo testo per sentiment analysis. Identifica le {top_k} parole più importanti per determinare se il sentiment è positivo o negativo.

TESTO: "{text}"

Restituisci SOLO una lista Python ordinata dalla parola più importante alla meno importante:
["parola1", "parola2", "parola3", ...]

IMPORTANTE: 
- Solo parole singole (no frasi)
- Solo parole presenti nel testo
- Ordinate per importanza decrescente
- Formato esatto: ["word1", "word2", ...]

Lista:"""

    def query_llm(self, prompt: str, max_retries: int = 3) -> List[str]:
        """Interroga OpenRouter API con retry logic"""
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Bassa per consistenza
            "max_tokens": 150,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    OPENROUTER_URL, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse word list
                words = self._extract_word_list(content)
                
                if words:  # Success
                    return words
                else:
                    print(f"[HR] Attempt {attempt+1}: Could not parse response")
                    
            except requests.exceptions.RequestException as e:
                print(f"[HR] Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                print(f"[HR] Unexpected error: {e}")
                
        return []  # Failed after all retries
    
    def _extract_word_list(self, content: str) -> List[str]:
        """Estrai lista di parole dal response LLM"""
        try:
            # Pattern 1: Lista Python standard ["word1", "word2"]
            pattern1 = r'\[(.*?)\]'
            match = re.search(pattern1, content, re.DOTALL)
            
            if match:
                list_content = match.group(1)
                
                # Pulisci e split
                words = []
                for item in list_content.split(','):
                    word = item.strip().strip('"\'').lower()
                    if word and word.isalpha() and len(word) > 1:
                        words.append(word)
                
                return words[:DEFAULT_TOP_K]  # Limita a top_k
            
            # Pattern 2: Lista semplice separata da virgole
            if ',' in content and '[' not in content:
                words = []
                for word in content.split(','):
                    word = word.strip().strip('"-').lower()
                    if word and word.isalpha() and len(word) > 1:
                        words.append(word)
                return words[:DEFAULT_TOP_K]
            
            return []
            
        except Exception as e:
            print(f"[HR] Parse error: {e}")
            return []

    def generate_dataset(self, sample_size: int = 400, 
                        save_intermediate: bool = True) -> pd.DataFrame:
        """Genera HR ground truth dataset completo"""
        
        print(f"[HR] Generating Human Reasoning ground truth...")
        print(f"[HR] Model: {self.model}")
        print(f"[HR] Sample size: {sample_size}")
        print(f"[HR] Estimated time: ~{sample_size * RATE_LIMIT_DELAY / 60:.1f} minutes")
        
        # Ottieni dati dal dataset ottimizzato
        texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)
        
        hr_results = []
        successful = 0
        failed = 0
        
        # Progress bar
        for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc="HR Generation")):
            
            # Crea prompt
            prompt = self.create_prompt(text)
            
            # Query LLM
            hr_words = self.query_llm(prompt)
            
            if hr_words:
                successful += 1
                status = "success"
            else:
                failed += 1
                status = "failed"
                hr_words = []  # Lista vuota per fallimenti
            
            hr_results.append({
                'idx': i,
                'text': text,
                'label': label,
                'hr_ranking': hr_words,
                'hr_count': len(hr_words),
                'status': status
            })
            
            # Salvataggio intermedio ogni 50 esempi
            if save_intermediate and (i + 1) % 50 == 0:
                temp_df = pd.DataFrame(hr_results)
                temp_file = HR_CACHE_DIR / f"hr_temp_{i+1}.csv"
                temp_df.to_csv(temp_file, index=False)
                print(f"[HR] Intermediate save: {temp_file}")
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
        
        # Crea DataFrame finale
        df = pd.DataFrame(hr_results)
        
        # Statistiche finali
        print(f"\n[HR] Generation completed!")
        print(f"[HR] Successful: {successful}/{len(texts)} ({successful/len(texts):.1%})")
        print(f"[HR] Failed: {failed}/{len(texts)} ({failed/len(texts):.1%})")
        print(f"[HR] Average words per example: {df['hr_count'].mean():.1f}")
        
        # Salva dataset finale
        output_file = HR_CACHE_DIR / HR_CACHE_FILE
        df.to_csv(output_file, index=False)
        print(f"[HR] Final dataset saved: {output_file}")
        
        return df

# ==== Funzioni di Valutazione ====
def load_ground_truth(filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Carica HR ground truth dataset"""
    if filepath is None:
        filepath = HR_CACHE_DIR / HR_CACHE_FILE
    
    try:
        df = pd.read_csv(filepath)
        
        # Parse hr_ranking da stringa a lista
        def parse_ranking(x):
            if pd.isna(x) or x == '[]':
                return []
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                return x if isinstance(x, list) else []
            except:
                return []
        
        df['hr_ranking'] = df['hr_ranking'].apply(parse_ranking)
        
        print(f"[HR] Loaded ground truth: {len(df)} examples")
        print(f"[HR] Valid rankings: {(df['hr_count'] > 0).sum()}")
        
        return df
        
    except FileNotFoundError:
        print(f"[HR] Ground truth file not found: {filepath}")
        print(f"[HR] Generate it first using generate_ground_truth()")
        return None
    except Exception as e:
        print(f"[HR] Error loading ground truth: {e}")
        return None

def compute_human_reasoning_score(xai_tokens: List[str], xai_scores: List[float], 
                                hr_ranking: List[str]) -> float:
    """
    Calcola Human Reasoning Agreement usando Mean Average Precision (MAP)
    
    Args:
        xai_tokens: Lista token dall'explainer
        xai_scores: Lista score corrispondenti  
        hr_ranking: Lista parole ordinate da LLM (più importante → meno importante)
    
    Returns:
        float: Average Precision score (0-1, higher is better)
    """
    if not hr_ranking or not xai_tokens or not xai_scores:
        return 0.0
    
    # Ordina XAI tokens per score (decrescente)
    xai_ranking = [token.lower() for token, score in 
                  sorted(zip(xai_tokens, xai_scores), key=lambda x: x[1], reverse=True)]
    
    # Normalizza HR ranking
    hr_set = set(word.lower() for word in hr_ranking)
    
    # Calcola Average Precision
    relevant_count = 0
    precision_sum = 0
    
    for k, xai_word in enumerate(xai_ranking, 1):
        if xai_word in hr_set:  # rel(k) = 1
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k
    
    # AP = sum(P(k) * rel(k)) / number_of_relevant_documents
    ap = precision_sum / len(hr_ranking) if hr_ranking else 0.0
    return ap

def evaluate_human_reasoning(model, tokenizer, explainer, 
                           hr_dataset: Optional[pd.DataFrame] = None,
                           show_progress: bool = True) -> float:
    """
    Valuta Human Reasoning Agreement su dataset completo
    
    Args:
        model: Modello PyTorch
        tokenizer: Tokenizer del modello
        explainer: Funzione explainer
        hr_dataset: Ground truth dataset (se None, carica automaticamente)
        show_progress: Mostra progress bar
    
    Returns:
        float: Mean Average Precision across dataset (0-1, higher is better)
    """
    
    # Carica ground truth se non fornito
    if hr_dataset is None:
        hr_dataset = load_ground_truth()
        if hr_dataset is None:
            return 0.0
    
    # Filtra solo esempi con HR valido
    valid_dataset = hr_dataset[hr_dataset['hr_count'] > 0].copy()
    
    if len(valid_dataset) == 0:
        print("[HR] No valid HR examples found")
        return 0.0
    
    print(f"[HR] Evaluating on {len(valid_dataset)} valid examples...")
    
    hr_scores = []
    failed_explanations = 0
    
    iterator = tqdm(valid_dataset.iterrows(), total=len(valid_dataset), 
                   desc="HR Evaluation", leave=False) if show_progress else valid_dataset.iterrows()
    
    for idx, row in iterator:
        try:
            text = row['text']
            hr_ranking = row['hr_ranking']
            
            # Genera XAI explanation
            attr = explainer(text)
            
            # Verifica validità explanation
            if not attr.tokens or not attr.scores:
                failed_explanations += 1
                continue
            
            # Calcola HR score
            hr_score = compute_human_reasoning_score(
                attr.tokens, attr.scores, hr_ranking
            )
            hr_scores.append(hr_score)
            
        except Exception as e:
            print(f"[HR] Error processing example {idx}: {e}")
            failed_explanations += 1
            continue
    
    # Calcola MAP finale
    mean_ap = float(np.mean(hr_scores)) if hr_scores else 0.0
    
    print(f"[HR] Evaluation completed:")
    print(f"[HR]   Processed: {len(hr_scores)}/{len(valid_dataset)} examples")
    print(f"[HR]   Failed explanations: {failed_explanations}")
    print(f"[HR]   Mean Average Precision: {mean_ap:.4f}")
    
    return mean_ap

# ==== API Pubbliche ====
def generate_ground_truth(api_key: str, sample_size: int = 400, 
                         model: str = DEFAULT_MODEL) -> pd.DataFrame:
    """
    Genera HR ground truth dataset (funzione pubblica)
    
    Args:
        api_key: OpenRouter API key
        sample_size: Numero di esempi da processare
        model: Modello LLM da usare
    
    Returns:
        pd.DataFrame: Dataset con ground truth HR
    """
    generator = HRGenerator(api_key, model)
    return generator.generate_dataset(sample_size)

def is_available() -> bool:
    """Controlla se HR ground truth è disponibile"""
    return (HR_CACHE_DIR / HR_CACHE_FILE).exists()

def get_info() -> Dict:
    """Ottieni informazioni su HR ground truth disponibile"""
    if not is_available():
        return {"available": False}
    
    df = load_ground_truth()
    if df is None:
        return {"available": False}
    
    return {
        "available": True,
        "total_examples": len(df),
        "valid_examples": (df['hr_count'] > 0).sum(),
        "success_rate": (df['status'] == 'success').mean() if 'status' in df.columns else None,
        "avg_words_per_example": df['hr_count'].mean(),
        "file_path": str(HR_CACHE_DIR / HR_CACHE_FILE)
    }

# ==== Test e Debug ====
def test_single_example(api_key: str, text: str = None) -> Dict:
    """Test su singolo esempio per debug"""
    if text is None:
        # Usa esempio dal dataset
        texts, _ = dataset.get_clustered_sample(1)
        text = texts[0]
    
    print(f"[HR-TEST] Testing with: '{text[:100]}...'")
    
    generator = HRGenerator(api_key)
    prompt = generator.create_prompt(text)
    words = generator.query_llm(prompt)
    
    return {
        "text": text,
        "prompt": prompt,
        "hr_words": words,
        "success": len(words) > 0
    }

# ==== Main Execution ====
if __name__ == "__main__":
    print("\n" + "="*70)
    print("HUMAN REASONING MODULE - STANDALONE TEST")
    print("="*70)
    
    # Controlla disponibilità
    info = get_info()
    print(f"HR Ground Truth Available: {info['available']}")
    
    if info['available']:
        print(f"Examples: {info['total_examples']} total, {info['valid_examples']} valid")
        print(f"Average words per example: {info['avg_words_per_example']:.1f}")
    else:
        print("Generate ground truth first with: generate_ground_truth(api_key)")
    
    print("\nUsage:")
    print("import HumanReasoning as hr")
    print("hr.generate_ground_truth(api_key='your_key')  # Once")
    print("score = hr.evaluate_human_reasoning(model, tokenizer, explainer)")