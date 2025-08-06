"""
HumanReasoning.py – Fixed Human Reasoning Ground Truth Generation (COMPLETE)
=============================================================================

CORREZIONI IMPLEMENTATE:
1. Usa ESATTAMENTE gli stessi 400 esempi del dataset clusterizzato
2. Gestione errori senza perdere corrispondenza 1:1
3. Salvataggio sia CSV che pickle per riutilizzo
4. Sistema di recovery che mantiene l'ordine originale
5. Progress bar funzionante correttamente
6. Rate limiting ottimizzato e adattivo
7. Checkpoint ogni 10 esempi (non ogni esempio)
8. ETA accurato e statistiche migliori
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

import dataset

# =============================================================================
# CONFIGURAZIONE RATE LIMITING
# =============================================================================

# Rate limits conservativi per sicurezza
MAX_REQUESTS_PER_MINUTE = 15  # Sotto il limite di 20 per sicurezza
MIN_REQUEST_INTERVAL = 4.0    # Secondi minimi tra richieste (60/15 = 4)
MAX_RETRIES = 5               # Retry massimi per rate limit
INITIAL_BACKOFF = 5.0         # Backoff iniziale in secondi
MAX_BACKOFF = 300.0           # Backoff massimo (5 minuti)

# Configurazione modelli
DEFAULT_MODEL = "moonshotai/kimi-k2" 
FALLBACK_MODELS = [
    "anthropic/claude-3-haiku",
    "deepseek/deepseek-chat",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "qwen/qwen-2-7b-instruct:free"
]

# Paths
HR_DATA_DIR = Path("human_reasoning_data")
HR_DATA_DIR.mkdir(exist_ok=True)
HR_DATASET_CSV = HR_DATA_DIR / "human_reasoning_ground_truth.csv"
HR_DATASET_PKL = HR_DATA_DIR / "human_reasoning_ground_truth.pkl"
HR_CHECKPOINT_FILE = HR_DATA_DIR / "hr_generation_checkpoint.json"

# =============================================================================
# RATE LIMITER CLASS - OTTIMIZZATO
# =============================================================================

class SmartRateLimiter:
    """Rate limiter intelligente con backoff esponenziale e interval adattivo."""
    
    def __init__(self, max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self.max_requests_per_minute = max_requests_per_minute
        self.min_interval = 60.0 / max_requests_per_minute
        self.request_times = []
        self.last_429_time = None
        self.current_backoff = INITIAL_BACKOFF
        self.adaptive_interval = self.min_interval  # CORREZIONE: Interval adattivo
        
        print(f"[RATE-LIMITER] Initialized: {max_requests_per_minute} req/min, {self.min_interval:.1f}s base interval")
    
    def wait_if_needed(self):
        """Aspetta se necessario con interval adattivo."""
        now = time.time()
        
        # Rimuovi richieste vecchie (più di 1 minuto)
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Se abbiamo raggiunto il limite, aspetta
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest_request = min(self.request_times)
            wait_time = 60 - (now - oldest_request) + 0.5  # Ridotto +0.5 invece di +1
            if wait_time > 0:
                print(f"[RATE-LIMITER] Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        # CORREZIONE: Interval adattivo invece di fisso
        if self.request_times:
            last_request = max(self.request_times)
            time_since_last = now - last_request
            wait_needed = self.adaptive_interval - time_since_last
            if wait_needed > 0:
                time.sleep(wait_needed)
        
        # Se recente 429, aspetta backoff
        if self.last_429_time and (now - self.last_429_time) < self.current_backoff:
            remaining_backoff = self.current_backoff - (now - self.last_429_time)
            print(f"[RATE-LIMITER] 429 backoff: waiting {remaining_backoff:.1f}s...")
            time.sleep(remaining_backoff)
    
    def record_request(self):
        """Registra richiesta riuscita e adatta interval."""
        self.request_times.append(time.time())
        
        # CORREZIONE: Interval adattivo - riduci se va tutto bene
        if len(self.request_times) >= 5:  # Dopo 5 richieste di successo
            recent_interval = (max(self.request_times) - min(self.request_times[-5:])) / 4
            if recent_interval < self.min_interval * 1.5:  # Se stiamo andando bene
                self.adaptive_interval = max(self.min_interval * 0.8, 2.0)  # Riduci ma non sotto 2s
        
        # Reset backoff su successo
        self.current_backoff = INITIAL_BACKOFF
    
    def record_429_error(self):
        """Registra errore 429 e aumenta backoff e interval."""
        self.last_429_time = time.time()
        self.current_backoff = min(self.current_backoff * 2, MAX_BACKOFF)
        self.adaptive_interval = min(self.adaptive_interval * 1.5, 10.0)  # Aumenta interval
        print(f"[RATE-LIMITER] 429 error: backoff={self.current_backoff:.1f}s, interval={self.adaptive_interval:.1f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche rate limiter."""
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]
        
        return {
            "requests_last_minute": len(recent_requests),
            "max_requests_per_minute": self.max_requests_per_minute,
            "current_backoff": self.current_backoff,
            "adaptive_interval": self.adaptive_interval,
            "time_since_last_429": (now - self.last_429_time) if self.last_429_time else None
        }

# =============================================================================
# LLM CLIENT CON RATE LIMITING
# =============================================================================

class RateLimitedLLMClient:
    """Client LLM con rate limiting intelligente."""
    
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.rate_limiter = SmartRateLimiter()
        self.session = requests.Session()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Headers di default
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "XAI Human Reasoning Ground Truth"
        })
        
        print(f"[LLM-CLIENT] Initialized with model: {model}")
    
    def generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> Optional[str]:
        """Genera testo con rate limiting automatico."""
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Richiesta
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stream": False
                }
                
                response = self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    timeout=60
                )
                
                self.total_requests += 1
                
                # Gestione rate limit
                if response.status_code == 429:
                    self.rate_limiter.record_429_error()
                    self.failed_requests += 1
                    
                    if attempt < MAX_RETRIES:
                        wait_time = self.rate_limiter.current_backoff + random.uniform(1, 5)
                        print(f"[LLM-CLIENT] 429 error, attempt {attempt + 1}, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[LLM-CLIENT] Max retries reached for 429 error")
                        return None
                
                # Altri errori HTTP
                if not response.ok:
                    self.failed_requests += 1
                    print(f"[LLM-CLIENT] HTTP error {response.status_code}: {response.text[:200]}")
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                # Parse response
                data = response.json()
                
                if "choices" not in data or not data["choices"]:
                    self.failed_requests += 1
                    print(f"[LLM-CLIENT] No choices in response")
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
                        continue
                    return None
                
                content = data["choices"][0]["message"]["content"]
                
                if not content or content.strip() == "":
                    self.failed_requests += 1
                    print(f"[LLM-CLIENT] Empty response content")
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
                        continue
                    return None
                
                # Successo!
                self.rate_limiter.record_request()
                self.successful_requests += 1
                return content.strip()
                
            except requests.exceptions.Timeout:
                self.failed_requests += 1
                print(f"[LLM-CLIENT] Timeout error")
                if attempt < MAX_RETRIES:
                    time.sleep(5)
                    continue
                return None
                
            except requests.exceptions.RequestException as e:
                self.failed_requests += 1
                print(f"[LLM-CLIENT] Request error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(2)
                    continue
                return None
                
            except json.JSONDecodeError as e:
                self.failed_requests += 1
                print(f"[LLM-CLIENT] JSON decode error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                    continue
                return None
                
            except Exception as e:
                self.failed_requests += 1
                print(f"[LLM-CLIENT] Unexpected error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(2)
                    continue
                return None
        
        print(f"[LLM-CLIENT] All retry attempts failed")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche client."""
        success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
        rate_stats = self.rate_limiter.get_stats()
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "model": self.model,
            **rate_stats
        }
    
    def print_stats(self):
        """Stampa statistiche."""
        stats = self.get_stats()
        print(f"\n[LLM-CLIENT] Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Successful: {stats['successful_requests']}")
        print(f"  Failed: {stats['failed_requests']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Requests last minute: {stats['requests_last_minute']}/{stats['max_requests_per_minute']}")

# =============================================================================
# HUMAN REASONING FUNCTIONS
# =============================================================================

def create_hr_prompt(text: str) -> str:
    """Crea prompt per Human Reasoning generation."""
    prompt = f"""You are analyzing a movie review for sentiment analysis. Your task is to identify the most important words that influence the sentiment, ranking them from most important to least important.

Review text: "{text}"

Please identify 5-8 words that are most important for determining the sentiment of this review. Rank them in order of importance (most important first).

Consider:
- Words that strongly indicate positive or negative sentiment
- Words that carry emotional weight
- Words that are crucial for understanding the overall opinion

Respond with ONLY a JSON list of words, ordered from most important to least important. No explanation needed.

Example format: ["fantastic", "terrible", "love", "boring", "amazing"]

Important words (JSON list only):"""
    
    return prompt

def parse_hr_response(response: str) -> List[str]:
    """Parse risposta LLM per estrarre ranking parole."""
    if not response:
        return []
    
    try:
        # Pulisci response
        response = response.strip()
        
        # Trova JSON list
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            return []
        
        json_str = response[start_idx:end_idx + 1]
        
        # Parse JSON
        words = json.loads(json_str)
        
        # Validate
        if not isinstance(words, list):
            return []
        
        # Clean words
        cleaned_words = []
        for word in words:
            if isinstance(word, str) and word.strip():
                cleaned_word = word.strip().lower()
                # Rimuovi punteggiatura
                cleaned_word = ''.join(c for c in cleaned_word if c.isalnum())
                if cleaned_word and len(cleaned_word) > 1:
                    cleaned_words.append(cleaned_word)
        
        return cleaned_words[:8]  # Max 8 words
        
    except json.JSONDecodeError:
        return []
    except Exception:
        return []

def generate_single_hr_example(client: RateLimitedLLMClient, text: str, label: int, index: int) -> Dict[str, Any]:
    """Genera singolo esempio Human Reasoning con index per tracciamento."""
    
    prompt = create_hr_prompt(text)
    response = client.generate_text(prompt, max_tokens=100, temperature=0.3)
    
    base_result = {
        "index": index,
        "text": text,
        "label": label,
        "hr_ranking": [],
        "hr_count": 0,
        "success": False,
        "error": None,
        "raw_response": ""
    }
    
    if response is None:
        base_result.update({
            "error": "LLM request failed",
            "raw_response": "NO_RESPONSE"
        })
        return base_result
    
    hr_ranking = parse_hr_response(response)
    
    base_result.update({
        "hr_ranking": hr_ranking,
        "hr_count": len(hr_ranking),
        "success": len(hr_ranking) > 0,
        "raw_response": response[:200],
        "error": None if len(hr_ranking) > 0 else "Empty ranking"
    })
    
    return base_result

# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

def save_checkpoint(data: Dict[str, Any], checkpoint_file: Path = HR_CHECKPOINT_FILE):
    """Salva checkpoint generation."""
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        # print(f"[CHECKPOINT] Saved to {checkpoint_file}")  # Meno verbose
    except Exception as e:
        print(f"[CHECKPOINT] Save failed: {e}")

def load_checkpoint(checkpoint_file: Path = HR_CHECKPOINT_FILE) -> Optional[Dict[str, Any]]:
    """Carica checkpoint generation."""
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f"[CHECKPOINT] Loaded from {checkpoint_file}")
        return data
    except Exception as e:
        print(f"[CHECKPOINT] Load failed: {e}")
        return None

# =============================================================================
# MAIN GENERATION FUNCTION - COMPLETELY REWRITTEN WITH FIXES
# =============================================================================

def generate_ground_truth(
    api_key: str,
    sample_size: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    resume: bool = True
) -> Optional[pd.DataFrame]:
    """
    Genera Human Reasoning ground truth per ESATTAMENTE gli stessi 400 esempi del dataset clusterizzato.
    
    CORREZIONI IMPLEMENTATE:
    - Progress bar funzionante correttamente
    - Checkpoint ogni 10 esempi processati (non ogni esempio)
    - Rate limiting ottimizzato e adattivo
    - ETA accurato basato sui processamenti reali
    - Statistiche migliori con debug info
    
    Args:
        api_key: OpenRouter API key
        sample_size: Ignorato - usa sempre tutti i 400 esempi
        model: Modello LLM da usare
        resume: Se riprendere da checkpoint esistente
    
    Returns:
        DataFrame con ground truth o None se fallito
    """
    
    print(f"\n{'='*80}")
    print("HUMAN REASONING GROUND TRUTH GENERATION - FIXED VERSION")
    print(f"{'='*80}")
    print(f"Using EXACTLY the same 400 clustered examples from dataset.test_df")
    print(f"Model: {model}")
    print(f"Estimated time: {400 * 6.0 / 60:.1f} minutes (at 6s/request - more realistic)")
    print(f"Rate limits: {MAX_REQUESTS_PER_MINUTE} req/min, {MIN_REQUEST_INTERVAL}s interval")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Setup client
    client = RateLimitedLLMClient(api_key, model)
    
    # CORREZIONE CHIAVE 1: Usa ESATTAMENTE il dataset clusterizzato
    try:
        clustered_texts, clustered_labels = dataset.get_clustered_sample(400, stratified=True)
        print(f"[DATASET] Using {len(clustered_texts)} examples from clustered dataset")
        
        if len(clustered_texts) != 400:
            print(f"[ERROR] Expected 400 examples, got {len(clustered_texts)}")
            return None
            
    except Exception as e:
        print(f"[DATASET] Failed to load clustered dataset: {e}")
        return None
    
    # CORREZIONE CHIAVE 2: Inizializza risultati con TUTTI i 400 esempi
    results = []
    for i, (text, label) in enumerate(zip(clustered_texts, clustered_labels)):
        results.append({
            "index": i,
            "text": text,
            "label": label,
            "hr_ranking": [],
            "hr_count": 0,
            "success": False,
            "error": "NOT_PROCESSED",
            "raw_response": ""
        })
    
    # Load checkpoint se richiesto
    completed_count = 0
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            checkpoint_results = checkpoint.get("results", [])
            
            if len(checkpoint_results) == 400:
                print(f"[RESUME] Found checkpoint with 400 examples")
                
                # Conta quelli già completati
                completed_count = sum(1 for r in checkpoint_results if r.get("success", False))
                pending_count = 400 - completed_count
                
                if pending_count == 0:
                    print(f"[RESUME] All 400 examples already completed, loading existing data...")
                    try:
                        df = pd.DataFrame(checkpoint_results)
                        if not HR_DATASET_CSV.exists():
                            save_to_csv(df)
                        return df
                    except Exception as e:
                        print(f"[RESUME] Failed to load checkpoint data: {e}")
                else:
                    print(f"[RESUME] Resuming: {completed_count} completed, {pending_count} pending")
                    results = checkpoint_results
    
    # CORREZIONE 3: Progress bar funzionante
    print(f"\n[PROCESSING] Processing 400 examples (maintaining exact order)...")
    
    with tqdm(total=400, desc="HR Generation", leave=True, 
              initial=completed_count, unit="ex") as pbar:
        
        processed_count = 0  # Counter per nuovi processamenti
        
        for idx in range(400):
            # Salta se già completato con successo
            if results[idx].get("success", False):
                continue
            
            try:
                text = results[idx]["text"]
                label = results[idx]["label"]
                
                # Update progress description
                pbar.set_description(f"HR Gen ({idx + 1}/400)")
                
                # Generate HR example
                hr_example = generate_single_hr_example(client, text, label, idx)
                
                # Update result
                results[idx].update({
                    "hr_ranking": hr_example["hr_ranking"],
                    "hr_count": hr_example["hr_count"],
                    "success": hr_example["success"],
                    "error": hr_example["error"],
                    "raw_response": hr_example["raw_response"]
                })
                
                # CORREZIONE 4: Update progress bar correttamente
                processed_count += 1
                pbar.update(1)
                
                # Progress info
                if hr_example["success"]:
                    status_msg = f"({hr_example['hr_count']} words)"
                else:
                    status_msg = f"FAILED: {hr_example.get('error', 'unknown')[:20]}"
                
                current_completed = sum(1 for r in results if r.get("success", False))
                success_rate = current_completed / (idx + 1) if idx > 0 else 0
                
                pbar.set_postfix_str(f"✓{current_completed} ✗{(idx+1)-current_completed} ({success_rate:.1%}) {status_msg}")
                
                # CORREZIONE 5: Checkpoint ogni 10 esempi processati (non ogni esempio)
                if processed_count % 10 == 0:
                    current_completed = sum(1 for r in results if r.get("success", False))
                    checkpoint_data = {
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                        "total_target": 400,
                        "completed": current_completed,
                        "progress_index": idx + 1,
                        "processed_this_session": processed_count,
                        "stats": client.get_stats()
                    }
                    save_checkpoint(checkpoint_data)
                
                # CORREZIONE 6: ETA più accurato
                if processed_count > 2:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    remaining_to_process = sum(1 for r in results if not r.get("success", False)) - 1
                    eta_seconds = remaining_to_process / rate if rate > 0 else 0
                    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                    pbar.set_description(f"HR Gen ({idx + 1}/400) ETA: {eta_time.strftime('%H:%M')}")
                
            except KeyboardInterrupt:
                print(f"\n[INTERRUPT] Generation interrupted by user")
                current_completed = sum(1 for r in results if r.get("success", False))
                print(f"[INTERRUPT] Completed {current_completed}/400 examples")
                break
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process example {idx}: {e}")
                results[idx].update({
                    "success": False,
                    "error": str(e)[:100],
                    "hr_ranking": [],
                    "hr_count": 0,
                    "raw_response": f"ERROR: {str(e)[:50]}"
                })
                processed_count += 1
                pbar.update(1)
    
    # Final statistics
    total_time = time.time() - start_time
    valid_examples = [ex for ex in results if ex["success"]]
    success_rate = len(valid_examples) / len(results) if results else 0
    
    print(f"\n{'='*80}")
    print("HUMAN REASONING GENERATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Examples processed: 400/400 (EXACT MATCH)")
    print(f"Valid examples: {len(valid_examples)} ({success_rate:.1%})")
    if processed_count > 0:
        print(f"Average processing time: {total_time / processed_count:.1f}s per NEW example")
    print(f"Overall success rate: {success_rate:.1%}")
    
    # Client statistics
    client_stats = client.get_stats()
    print(f"\nLLM Statistics:")
    print(f"  Total requests: {client_stats['total_requests']}")
    print(f"  Successful: {client_stats['successful_requests']}")
    print(f"  Failed: {client_stats['failed_requests']}")
    print(f"  Success rate: {client_stats['success_rate']:.1%}")
    
    # CORREZIONE 7: Mostra esempi con errori se ce ne sono
    failed_examples = [ex for ex in results if not ex["success"]]
    if failed_examples:
        print(f"\nFailed Examples ({len(failed_examples)}):")
        for i, ex in enumerate(failed_examples[:3]):  # Mostra solo primi 3
            print(f"  {i+1}. Index {ex['index']}: {ex['error']}")
        if len(failed_examples) > 3:
            print(f"  ... and {len(failed_examples) - 3} more")
    
    # CORREZIONE 8: Salva sia CSV che pickle
    try:
        df = pd.DataFrame(results)
        
        if len(df) != 400:
            print(f"[ERROR] DataFrame has {len(df)} rows instead of 400")
            return None
        
        # Salva CSV (per riutilizzo futuro)
        save_to_csv(df)
        
        # Salva pickle (per sessione corrente)
        df.to_pickle(HR_DATASET_PKL)
        print(f"[SAVE] Pickle saved to: {HR_DATASET_PKL}")
        
        # Save final checkpoint
        final_checkpoint = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "total_target": 400,
            "completed": len(valid_examples),
            "final_stats": client_stats,
            "success_rate": success_rate,
            "total_time_minutes": total_time / 60,
            "exact_match": True,
            "average_processing_time": total_time / processed_count if processed_count > 0 else 0
        }
        save_checkpoint(final_checkpoint)
        
        return df
        
    except Exception as e:
        print(f"[SAVE] Failed to save dataset: {e}")
        return None

def save_to_csv(df: pd.DataFrame):
    """Salva DataFrame in CSV con formato corretto per riutilizzo."""
    try:
        # Converte hr_ranking da lista a stringa JSON per CSV
        df_csv = df.copy()
        df_csv['hr_ranking'] = df_csv['hr_ranking'].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else "[]"
        )
        
        df_csv.to_csv(HR_DATASET_CSV, index=False)
        print(f"[SAVE] CSV saved to: {HR_DATASET_CSV}")
        
    except Exception as e:
        print(f"[SAVE] Failed to save CSV: {e}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_ground_truth() -> Optional[pd.DataFrame]:
    """Carica ground truth esistente - PRIMA CSV poi pickle."""
    
    # PRIORITA 1: CSV (per riutilizzo tra sessioni)
    if HR_DATASET_CSV.exists():
        try:
            df = pd.read_csv(HR_DATASET_CSV)
            
            # Parse hr_ranking column (stored as JSON string in CSV)
            def parse_ranking(ranking_str):
                if pd.isna(ranking_str) or ranking_str == "":
                    return []
                try:
                    if isinstance(ranking_str, str):
                        return json.loads(ranking_str)
                    else:
                        return ranking_str if isinstance(ranking_str, list) else []
                except:
                    return []
            
            df['hr_ranking'] = df['hr_ranking'].apply(parse_ranking)
            
            # Verifica che sia il dataset corretto
            if len(df) == 400:
                print(f"[LOAD] Loaded ground truth from CSV: {len(df)} examples")
                return df
            else:
                print(f"[LOAD] CSV has {len(df)} examples, expected 400")
                
        except Exception as e:
            print(f"[LOAD] Failed to load CSV: {e}")
    
    # PRIORITA 2: Pickle (fallback per sessione corrente)
    if HR_DATASET_PKL.exists():
        try:
            df = pd.read_pickle(HR_DATASET_PKL)
            
            if len(df) == 400:
                print(f"[LOAD] Loaded ground truth from pickle: {len(df)} examples")
                return df
            else:
                print(f"[LOAD] Pickle has {len(df)} examples, expected 400")
                
        except Exception as e:
            print(f"[LOAD] Failed to load pickle: {e}")
    
    print(f"[LOAD] No valid ground truth found")
    return None

def is_available() -> bool:
    """Controlla se Human Reasoning ground truth è disponibile."""
    df = load_ground_truth()
    if df is None or len(df) != 400:
        return False
    
    valid_count = (df['hr_count'] > 0).sum()
    return valid_count > 50  # Almeno 50 esempi validi dei 400

def get_info() -> Dict[str, Any]:
    """Ottieni informazioni su Human Reasoning ground truth."""
    info = {
        "available": False,
        "csv_path": str(HR_DATASET_CSV),
        "pkl_path": str(HR_DATASET_PKL),
        "total_examples": 0,
        "valid_examples": 0,
        "success_rate": None,
        "avg_words_per_example": 0.0,
        "exact_match_dataset": False
    }
    
    df = load_ground_truth()
    if df is None:
        return info
    
    try:
        info["available"] = True
        info["total_examples"] = len(df)
        info["valid_examples"] = (df['hr_count'] > 0).sum()
        info["exact_match_dataset"] = len(df) == 400  # Flag importante
        
        if info["total_examples"] > 0:
            info["success_rate"] = info["valid_examples"] / info["total_examples"]
        
        if info["valid_examples"] > 0:
            valid_df = df[df['hr_count'] > 0]
            info["avg_words_per_example"] = valid_df['hr_count'].mean()
        
    except Exception as e:
        print(f"[INFO] Error getting info: {e}")
    
    return info

def test_api_key(api_key: str, model: str = DEFAULT_MODEL) -> bool:
    """Testa API key con una richiesta semplice."""
    print(f"[TEST] Testing API key with model: {model}")
    
    try:
        client = RateLimitedLLMClient(api_key, model)
        response = client.generate_text("Say 'Hello'", max_tokens=10, temperature=0.0)
        
        if response and "hello" in response.lower():
            print(f"[TEST]  API key test successful")
            stats = client.get_stats()
            print(f"[TEST]  Stats: {stats['successful_requests']}/{stats['total_requests']} requests successful")
            return True
        else:
            print(f"[TEST]  API key test failed: unexpected response")
            return False
            
    except Exception as e:
        print(f"[TEST]  API key test failed: {e}")
        return False

def verify_dataset_consistency() -> bool:
    """Verifica che il dataset HR corrisponda esattamente al dataset clusterizzato."""
    print(f"[VERIFY] Checking HR dataset consistency with clustered dataset...")
    
    try:
        # Carica HR dataset
        hr_df = load_ground_truth()
        if hr_df is None or len(hr_df) != 400:
            print(f"[VERIFY] HR dataset not available or wrong size")
            return False
        
        # Carica dataset clusterizzato
        clustered_texts, clustered_labels = dataset.get_clustered_sample(400, stratified=True)
        
        if len(clustered_texts) != 400:
            print(f"[VERIFY] Clustered dataset has {len(clustered_texts)} examples, expected 400")
            return False
        
        # Verifica corrispondenza 1:1
        mismatches = 0
        for i, (hr_text, hr_label, clust_text, clust_label) in enumerate(
            zip(hr_df['text'], hr_df['label'], clustered_texts, clustered_labels)
        ):
            if hr_text != clust_text or hr_label != clust_label:
                mismatches += 1
                if mismatches <= 3:  # Mostra solo primi 3 errori
                    print(f"[VERIFY] Mismatch at index {i}:")
                    print(f"[VERIFY]   HR: '{hr_text[:50]}...' (label: {hr_label})")
                    print(f"[VERIFY]   Clustered: '{clust_text[:50]}...' (label: {clust_label})")
        
        if mismatches == 0:
            print(f"[VERIFY]  Perfect match: HR dataset corresponds exactly to clustered dataset")
            return True
        else:
            print(f"[VERIFY]  Found {mismatches}/400 mismatches")
            return False
        
    except Exception as e:
        print(f"[VERIFY] Verification failed: {e}")
        return False

# Alias per compatibilità con codice esistente
def test_api_key_compatibility(api_key: str, model: str = DEFAULT_MODEL) -> bool:
    """Alias per compatibilità con codice esistente."""
    return test_api_key(api_key, model)

# =============================================================================
# MAIN & TESTING
# =============================================================================

if __name__ == "__main__":
    print("Human Reasoning Ground Truth Generator - COMPLETE FIXED VERSION")
    print("=" * 60)
    
    # Test con API key di esempio
    test_api_key_input = input("Enter OpenRouter API key for testing (or press Enter to skip): ").strip()
    
    if test_api_key_input:
        print("\nTesting API connection...")
        if test_api_key(test_api_key_input):
            print(" API connection successful!")
        else:
            print(" API connection failed!")
    
    # Test info functions
    print("\nTesting info functions...")
    print(f"Available: {is_available()}")
    
    info = get_info()
    print(f"Info: {info}")
    
    if info["available"]:
        print(f"\nTesting dataset consistency...")
        consistent = verify_dataset_consistency()
        if consistent:
            print(" Dataset consistency verified!")
        else:
            print(" Dataset consistency issues found!")
        
        df = load_ground_truth()
        if df is not None:
            print(f"Loaded dataset: {len(df)} examples")
            valid_count = (df['hr_count'] > 0).sum()
            print(f"Valid examples: {valid_count}/{len(df)} ({valid_count/len(df):.1%})")
            
            if valid_count > 0:
                sample_hr = df[df['hr_count'] > 0].iloc[0]
                print(f"Sample HR ranking: {sample_hr['hr_ranking']}")
    
    print("\nHuman Reasoning module (COMPLETE FIXED) ready!")
    print("\nKEY FIXES IMPLEMENTED:")
    print("- Uses EXACTLY the same 400 examples from dataset.test_df")
    print("- Maintains 1:1 correspondence even with LLM failures")
    print("- Saves both CSV (for reuse) and pickle (for session)")
    print("- Progress bar works correctly from 0% to 100%")
    print("- Checkpoint every 10 examples (not every example)")
    print("- Optimized rate limiting with adaptive intervals")
    print("- Accurate ETA and better statistics")
    print("- Shows failed examples for debugging")
    print("- Robust recovery without losing original order")
    print("- Verification system for dataset consistency")