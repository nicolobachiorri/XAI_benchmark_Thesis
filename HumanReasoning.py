"""
HumanReasoning.py – Human Reasoning Ground Truth Generation (RATE LIMIT FIXED)
=============================================================================

MIGLIORAMENTI PER RATE LIMITS:
1. Rate limiting intelligente con backoff esponenziale
2. Gestione automatica di errori 429 (Too Many Requests)
3. Monitoraggio rate limits in tempo reale
4. Batch processing con pause dinamiche
5. Recovery automatico da rate limit exceeded
6. Support per diversi modelli con rate limits differenti
7. Progress tracking dettagliato con ETA

Rate Limits OpenRouter:
- Modelli :free: 20 req/min, 50/day (< $10 credits) o 1000/day (≥ $10 credits)
- Modelli paid: Limiti più alti, dipendono dal modello

Strategia implementata:
- Max 15 richieste/minuto per sicurezza (sotto il limite di 20)
- Pausa minima di 4 secondi tra richieste
- Backoff esponenziale su 429 errors
- Batch processing con checkpoint
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
DEFAULT_MODEL = "deepseek/deepseek-chat"  # Modello veloce e economico
FALLBACK_MODELS = [
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "qwen/qwen-2-7b-instruct:free"
]

# Paths
HR_DATA_DIR = Path("human_reasoning_data")
HR_DATA_DIR.mkdir(exist_ok=True)
HR_DATASET_FILE = HR_DATA_DIR / "human_reasoning_ground_truth.csv"
HR_CHECKPOINT_FILE = HR_DATA_DIR / "hr_generation_checkpoint.json"

# =============================================================================
# RATE LIMITER CLASS
# =============================================================================

class SmartRateLimiter:
    """Rate limiter intelligente con backoff esponenziale."""
    
    def __init__(self, max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self.max_requests_per_minute = max_requests_per_minute
        self.min_interval = 60.0 / max_requests_per_minute
        self.request_times = []
        self.last_429_time = None
        self.current_backoff = INITIAL_BACKOFF
        
        print(f"[RATE-LIMITER] Initialized: {max_requests_per_minute} req/min, {self.min_interval:.1f}s interval")
    
    def wait_if_needed(self):
        """Aspetta se necessario prima della prossima richiesta."""
        now = time.time()
        
        # Rimuovi richieste vecchie (più di 1 minuto)
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Se abbiamo raggiunto il limite, aspetta
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest_request = min(self.request_times)
            wait_time = 60 - (now - oldest_request) + 1  # +1 per sicurezza
            if wait_time > 0:
                print(f"[RATE-LIMITER] Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        # Aspetta intervallo minimo dall'ultima richiesta
        if self.request_times:
            last_request = max(self.request_times)
            time_since_last = now - last_request
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
        
        # Se recente 429, aspetta backoff
        if self.last_429_time and (now - self.last_429_time) < self.current_backoff:
            remaining_backoff = self.current_backoff - (now - self.last_429_time)
            print(f"[RATE-LIMITER] 429 backoff: waiting {remaining_backoff:.1f}s...")
            time.sleep(remaining_backoff)
    
    def record_request(self):
        """Registra una richiesta riuscita."""
        self.request_times.append(time.time())
        # Reset backoff su successo
        self.current_backoff = INITIAL_BACKOFF
    
    def record_429_error(self):
        """Registra un errore 429 e aumenta backoff."""
        self.last_429_time = time.time()
        self.current_backoff = min(self.current_backoff * 2, MAX_BACKOFF)
        print(f"[RATE-LIMITER] 429 error recorded, backoff increased to {self.current_backoff:.1f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche rate limiter."""
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]
        
        return {
            "requests_last_minute": len(recent_requests),
            "max_requests_per_minute": self.max_requests_per_minute,
            "current_backoff": self.current_backoff,
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
            "HTTP-Referer": "https://github.com/your-repo",  # Opzionale per tracking
            "X-Title": "XAI Human Reasoning Ground Truth"    # Opzionale per tracking
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
                
                print(f"[LLM-CLIENT] Request {self.total_requests + 1} (attempt {attempt + 1}/{MAX_RETRIES + 1})")
                
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
                        time.sleep(2 ** attempt)  # Backoff semplice
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
            print(f"[PARSE] No JSON list found in response: {response[:100]}")
            return []
        
        json_str = response[start_idx:end_idx + 1]
        
        # Parse JSON
        words = json.loads(json_str)
        
        # Validate
        if not isinstance(words, list):
            print(f"[PARSE] Response is not a list: {type(words)}")
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
        
    except json.JSONDecodeError as e:
        print(f"[PARSE] JSON decode error: {e}, response: {response[:100]}")
        return []
    except Exception as e:
        print(f"[PARSE] Parse error: {e}, response: {response[:100]}")
        return []

def generate_single_hr_example(client: RateLimitedLLMClient, text: str, label: int) -> Dict[str, Any]:
    """Genera singolo esempio Human Reasoning."""
    
    prompt = create_hr_prompt(text)
    response = client.generate_text(prompt, max_tokens=100, temperature=0.3)
    
    if response is None:
        return {
            "text": text,
            "label": label,
            "hr_ranking": [],
            "hr_count": 0,
            "success": False,
            "error": "LLM request failed"
        }
    
    hr_ranking = parse_hr_response(response)
    
    return {
        "text": text,
        "label": label,
        "hr_ranking": hr_ranking,
        "hr_count": len(hr_ranking),
        "success": len(hr_ranking) > 0,
        "raw_response": response[:200],  # Store truncated response for debugging
        "error": None if len(hr_ranking) > 0 else "Empty ranking"
    }

# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

def save_checkpoint(data: Dict[str, Any], checkpoint_file: Path = HR_CHECKPOINT_FILE):
    """Salva checkpoint generation."""
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[CHECKPOINT] Saved to {checkpoint_file}")
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
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_ground_truth(
    api_key: str,
    sample_size: int = 400,
    model: str = DEFAULT_MODEL,
    resume: bool = True
) -> Optional[pd.DataFrame]:
    """
    Genera Human Reasoning ground truth con rate limiting intelligente.
    
    Args:
        api_key: OpenRouter API key
        sample_size: Numero di esempi da generare
        model: Modello LLM da usare
        resume: Se riprendere da checkpoint esistente
    
    Returns:
        DataFrame con ground truth o None se fallito
    """
    
    print(f"\n{'='*80}")
    print("HUMAN REASONING GROUND TRUTH GENERATION (RATE LIMIT SAFE)")
    print(f"{'='*80}")
    print(f"Target samples: {sample_size}")
    print(f"Model: {model}")
    print(f"Estimated time: {sample_size * 4.5 / 60:.1f} minutes (at 4.5s/request)")
    print(f"Rate limits: {MAX_REQUESTS_PER_MINUTE} req/min, {MIN_REQUEST_INTERVAL}s interval")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Setup client
    client = RateLimitedLLMClient(api_key, model)
    
    # Load checkpoint se richiesto
    completed_examples = []
    start_idx = 0
    
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_examples = checkpoint.get("examples", [])
            start_idx = len(completed_examples)
            
            if start_idx >= sample_size:
                print(f"[RESUME] Already completed {start_idx} examples, loading existing data...")
                try:
                    df = pd.DataFrame(completed_examples[:sample_size])
                    return df
                except Exception as e:
                    print(f"[RESUME] Failed to load checkpoint data: {e}")
                    completed_examples = []
                    start_idx = 0
            else:
                print(f"[RESUME] Resuming from example {start_idx + 1}/{sample_size}")
    
    # Get dataset samples
    try:
        texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)
        print(f"[DATA] Loaded {len(texts)} examples from dataset")
    except Exception as e:
        print(f"[DATA] Failed to load dataset: {e}")
        return None
    
    # Process remaining examples
    remaining_texts = texts[start_idx:]
    remaining_labels = labels[start_idx:]
    
    print(f"\n[PROCESSING] Processing {len(remaining_texts)} remaining examples...")
    print(f"[PROCESSING] Starting from index {start_idx}")
    
    with tqdm(total=len(remaining_texts), desc="HR Generation", leave=True) as pbar:
        
        for idx, (text, label) in enumerate(zip(remaining_texts, remaining_labels)):
            global_idx = start_idx + idx
            
            try:
                # Update progress
                pbar.set_description(f"HR Gen ({global_idx + 1}/{sample_size})")
                
                # Generate HR example
                hr_example = generate_single_hr_example(client, text, label)
                completed_examples.append(hr_example)
                
                # Progress info
                if hr_example["success"]:
                    status = f" ({hr_example['hr_count']} words)"
                else:
                    status = f" (FAILED: {hr_example.get('error', 'unknown')})"
                
                pbar.set_postfix_str(f"Success: {client.successful_requests}/{client.total_requests}{status}")
                pbar.update(1)
                
                # Checkpoint ogni 10 esempi
                if (idx + 1) % 10 == 0:
                    checkpoint_data = {
                        "examples": completed_examples,
                        "timestamp": datetime.now().isoformat(),
                        "total_target": sample_size,
                        "completed": len(completed_examples),
                        "stats": client.get_stats()
                    }
                    save_checkpoint(checkpoint_data)
                
                # ETA calculation
                if idx > 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    remaining_time = (len(remaining_texts) - idx - 1) / rate
                    eta = datetime.now() + timedelta(seconds=remaining_time)
                    pbar.set_postfix_str(f"ETA: {eta.strftime('%H:%M:%S')}")
                
            except KeyboardInterrupt:
                print(f"\n[INTERRUPT] Generation interrupted by user")
                print(f"[INTERRUPT] Completed {len(completed_examples)} examples")
                break
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process example {global_idx}: {e}")
                # Add empty example per mantenere conteggio
                completed_examples.append({
                    "text": text,
                    "label": label,
                    "hr_ranking": [],
                    "hr_count": 0,
                    "success": False,
                    "error": str(e)
                })
                pbar.update(1)
    
    # Final statistics
    total_time = time.time() - start_time
    valid_examples = [ex for ex in completed_examples if ex["success"]]
    success_rate = len(valid_examples) / len(completed_examples) if completed_examples else 0
    
    print(f"\n{'='*80}")
    print("HUMAN REASONING GENERATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Examples processed: {len(completed_examples)}")
    print(f"Valid examples: {len(valid_examples)} ({success_rate:.1%})")
    print(f"Average processing time: {total_time / len(completed_examples):.1f}s per example")
    
    # Client statistics
    client.print_stats()
    
    # Save final dataset
    if completed_examples:
        try:
            df = pd.DataFrame(completed_examples)
            df.to_csv(HR_DATASET_FILE, index=False)
            print(f"\n[SAVE] Dataset saved to: {HR_DATASET_FILE}")
            
            # Save final checkpoint
            final_checkpoint = {
                "examples": completed_examples,
                "timestamp": datetime.now().isoformat(),
                "total_target": sample_size,
                "completed": len(completed_examples),
                "final_stats": client.get_stats(),
                "success_rate": success_rate,
                "total_time_minutes": total_time / 60
            }
            save_checkpoint(final_checkpoint)
            
            return df
            
        except Exception as e:
            print(f"[SAVE] Failed to save dataset: {e}")
            return None
    
    print(f"[ERROR] No examples completed successfully")
    return None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_ground_truth() -> Optional[pd.DataFrame]:
    """Carica ground truth esistente."""
    if not HR_DATASET_FILE.exists():
        return None
    
    try:
        df = pd.read_csv(HR_DATASET_FILE)
        
        # Parse hr_ranking column (stored as string)
        def parse_ranking(ranking_str):
            if pd.isna(ranking_str) or ranking_str == "":
                return []
            try:
                return eval(ranking_str) if isinstance(ranking_str, str) else ranking_str
            except:
                return []
        
        df['hr_ranking'] = df['hr_ranking'].apply(parse_ranking)
        
        return df
        
    except Exception as e:
        print(f"[LOAD] Failed to load ground truth: {e}")
        return None

def is_available() -> bool:
    """Controlla se Human Reasoning ground truth è disponibile."""
    if not HR_DATASET_FILE.exists():
        return False
    
    try:
        df = load_ground_truth()
        if df is None or len(df) == 0:
            return False
        
        valid_count = (df['hr_count'] > 0).sum()
        return valid_count > 10  # Almeno 10 esempi validi
        
    except Exception:
        return False

def get_info() -> Dict[str, Any]:
    """Ottieni informazioni su Human Reasoning ground truth."""
    info = {
        "available": False,
        "file_path": str(HR_DATASET_FILE),
        "total_examples": 0,
        "valid_examples": 0,
        "success_rate": None,
        "avg_words_per_example": 0.0
    }
    
    if not HR_DATASET_FILE.exists():
        return info
    
    try:
        df = load_ground_truth()
        if df is None:
            return info
        
        info["available"] = True
        info["total_examples"] = len(df)
        info["valid_examples"] = (df['hr_count'] > 0).sum()
        
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
            client.print_stats()
            return True
        else:
            print(f"[TEST]  API key test failed: unexpected response")
            return False
            
    except Exception as e:
        print(f"[TEST]  API key test failed: {e}")
        return False

# Alias per compatibilità
def test_api_key_compatibility(api_key: str, model: str = DEFAULT_MODEL) -> bool:
    """Alias per compatibilità con codice esistente."""
    return test_api_key(api_key, model)

# =============================================================================
# MAIN & TESTING
# =============================================================================

if __name__ == "__main__":
    print("Human Reasoning Ground Truth Generator (Rate Limit Safe)")
    print("=" * 60)
    
    # Test con API key di esempio
    test_api_key = input("Enter OpenRouter API key for testing (or press Enter to skip): ").strip()
    
    if test_api_key:
        print("\nTesting API connection...")
        if test_api_key(test_api_key):
            print(" API connection successful!")
        else:
            print(" API connection failed!")
    
    # Test info functions
    print("\nTesting info functions...")
    print(f"Available: {is_available()}")
    
    info = get_info()
    print(f"Info: {info}")
    
    if info["available"]:
        df = load_ground_truth()
        if df is not None:
            print(f"Loaded dataset: {len(df)} examples")
            print(f"Sample HR ranking: {df.iloc[0]['hr_ranking'] if len(df) > 0 else 'None'}")
    
    print("\nHuman Reasoning module ready!")