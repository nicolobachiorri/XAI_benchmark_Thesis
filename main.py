"""
main.py – CLI unificata per XAI benchmark (GOOGLE COLAB OPTIMIZED)
=================================================================

OTTIMIZZAZIONI PER COLAB:
1. Error handling robusto per environment instabile
2. Memory monitoring integrato
3. Progress tracking ottimizzato per notebook
4. Auto-cleanup tra operazioni
5. Fallback intelligenti per modelli/explainer
6. Dataset clusterizzato (400 esempi)
7. **HUMAN REASONING INTEGRATION**

Comandi principali:
* **explain** – spiega una singola frase
* **evaluate** – valuta modello + explainer su una metrica (incluso Human Reasoning)
* **test** – test veloce sistema completo
* **hr-status** – controlla status Human Reasoning ground truth
* **hr-generate** – genera Human Reasoning ground truth

Esempi Colab:
```python
# In Colab, usa direttamente:
import main

# Test veloce
main.quick_test()

# Explain
main.explain_text("Great movie!", "distilbert", "lime")

# Evaluate (incluso Human Reasoning)
main.evaluate_combination("distilbert", "lime", "human_reasoning")

# Check HR status
main.check_hr_status()

# Generate HR ground truth (richiede API key)
main.generate_hr_ground_truth("your_openrouter_api_key")
```
"""

import argparse
import sys
import time
from typing import List, Optional, Dict, Any
import torch

import models
import dataset
import explainers
import metrics
import HumanReasoning as hr
from utils import Timer, print_memory_status, aggressive_cleanup, set_seed, PerformanceProfiler

# Setup iniziale
set_seed(42)

# Parametri per Colab
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]
DEFAULT_SAMPLE_SIZE = 100  # Ridotto per Colab

def print_colab_header():
    """Header informativo per Colab con Human Reasoning."""
    print("="*70)
    print(" XAI BENCHMARK + HUMAN REASONING")
    print("="*70)
    print(" Dataset: 400 clustered examples from IMDB")
    print(" Models: Pre-trained sentiment analysis")
    print(" Explainers: LIME, SHAP, Gradients, Attention")
    print(" Metrics: Robustness, Consistency, Contrastivity, Human Reasoning")
    print(" Human Reasoning: LLM-generated importance rankings")
    print("="*70)

def check_dependencies():
    """Verifica dipendenze e disponibilità (incluso Human Reasoning)."""
    print("\n[CHECK] Verifying dependencies...")
    
    # Models
    available_models = list(models.MODELS.keys())
    print(f"[CHECK]  {len(available_models)} models available: {available_models}")
    
    # Explainers  
    available_explainers = explainers.list_explainers()
    print(f"[CHECK]  {len(available_explainers)} explainers available: {available_explainers}")
    
    # Dataset
    print(f"[CHECK]  Dataset: {len(dataset.test_df)} clustered examples")
    
    # Human Reasoning
    hr_info = hr.get_info()
    if hr_info['available']:
        print(f"[CHECK]  Human Reasoning: {hr_info['valid_examples']}/{hr_info['total_examples']} valid examples")
    else:
        print(f"[CHECK]  Human Reasoning: Not available (generate first)")
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[CHECK]  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print(f"[CHECK]  GPU: Not available (using CPU)")
    
    return available_models, available_explainers

def safe_load_model_explainer(model_key: str, explainer_name: str):
    """Carica modello ed explainer con error handling."""
    try:
        print(f"[LOAD] Loading {model_key}...")
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        
        print(f"[LOAD] Creating {explainer_name} explainer...")
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        
        print(f"[LOAD]  {model_key} + {explainer_name} ready")
        return model, tokenizer, explainer
        
    except Exception as e:
        print(f"[ERROR] Failed to load {model_key} + {explainer_name}: {e}")
        print(f"[SUGGESTION] Try smaller model like 'tinybert' or different explainer")
        return None, None, None

def explain_text(text: str, model_key: str, explainer_name: str, show_prediction: bool = True):
    """Explain singolo testo (funzione chiamabile da Colab)."""
    print_colab_header()
    print(f"\n[EXPLAIN] Text: '{text}'")
    print(f"[EXPLAIN] Model: {model_key}, Explainer: {explainer_name}")
    
    with Timer("Explanation", track_memory=True):
        # Load components
        model, tokenizer, explainer = safe_load_model_explainer(model_key, explainer_name)
        if model is None:
            return None
        
        try:
            # Model prediction se richiesto
            if show_prediction:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = models.move_batch_to_device(inputs)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).squeeze()
                    pred_label = torch.argmax(probs).item()
                    pred_score = probs[pred_label].item()
                    label_name = "positive" if pred_label == 1 else "negative"
                
                print(f"\n[PREDICTION] {label_name} (confidence: {pred_score:.3f})")
            
            # Generate explanation
            print(f"\n[EXPLANATION] Generating attribution...")
            attr = explainer(text)
            
            if not attr.tokens or not attr.scores:
                print(f"[ERROR] Empty attribution generated")
                return None
            
            # Display results
            print(f"\n[RESULTS] Token importance ({len(attr.tokens)} tokens):")
            print("-" * 50)
            
            # Sort by absolute importance
            token_scores = list(zip(attr.tokens, attr.scores))
            token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (token, score) in enumerate(token_scores[:10]):  # Top 10
                if token.isalpha() and len(token) > 1:  # Solo parole significative
                    direction = "→" if score > 0 else "←"
                    print(f"{i+1:2d}. {token:>15s} {direction} {score:+.3f}")
            
            print("-" * 50)
            print(f"[DONE] Explanation completed!")
            
            return attr
            
        except Exception as e:
            print(f"[ERROR] Explanation failed: {e}")
            return None
        
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            aggressive_cleanup()

def evaluate_combination(model_key: str, explainer_name: str, metric_name: str, 
                        sample_size: int = DEFAULT_SAMPLE_SIZE) -> Optional[float]:
    """Valuta combinazione modello+explainer+metrica (chiamabile da Colab, incluso Human Reasoning)."""
    print_colab_header()
    print(f"\n[EVALUATE] {model_key} + {explainer_name} + {metric_name}")
    print(f"[EVALUATE] Sample size: {sample_size}")
    
    profiler = PerformanceProfiler()
    profiler.start_operation("total_evaluation")
    
    try:
        # Load components
        profiler.start_operation("model_loading")
        model, tokenizer, explainer = safe_load_model_explainer(model_key, explainer_name)
        profiler.end_operation("model_loading")
        
        if model is None:
            return None
        
        # Get data
        profiler.start_operation("data_preparation")
        texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)
        print(f"[DATA] Using {len(texts)} examples ({sum(labels)} positive)")
        profiler.end_operation("data_preparation")
        
        # Evaluate metric
        profiler.start_operation(f"metric_{metric_name}")
        score = None
        
        if metric_name == "robustness":
            print(f"\n[ROBUSTNESS] Evaluating stability under perturbations...")
            score = metrics.evaluate_robustness_over_dataset(
                model, tokenizer, explainer, texts, show_progress=True
            )
            
        elif metric_name == "contrastivity":
            print(f"\n[CONTRASTIVITY] Evaluating class discrimination...")
            pos_texts = [t for t, l in zip(texts, labels) if l == 1][:50]
            neg_texts = [t for t, l in zip(texts, labels) if l == 0][:50]
            
            print(f"[CONTRASTIVITY] Processing {len(pos_texts)} positive texts...")
            pos_attrs = metrics.process_attributions_batch(pos_texts, explainer, batch_size=10)
            
            print(f"[CONTRASTIVITY] Processing {len(neg_texts)} negative texts...")
            neg_attrs = metrics.process_attributions_batch(neg_texts, explainer, batch_size=10)
            
            # Filter valid attributions
            pos_attrs = [attr for attr in pos_attrs if attr.tokens and attr.scores]
            neg_attrs = [attr for attr in neg_attrs if attr.tokens and attr.scores]
            
            if pos_attrs and neg_attrs:
                score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
                print(f"[CONTRASTIVITY] Used {len(pos_attrs)} pos + {len(neg_attrs)} neg attributions")
            else:
                print(f"[CONTRASTIVITY] Insufficient valid attributions")
                score = 0.0
                
        elif metric_name == "consistency":
            print(f"\n[CONSISTENCY] Evaluating inference seed stability...")
            consistency_texts = texts[:min(50, len(texts))]  # Limit for speed
            score = metrics.evaluate_consistency_over_dataset(
                model=model,
                tokenizer=tokenizer,
                explainer=explainer,
                texts=consistency_texts,
                seeds=DEFAULT_CONSISTENCY_SEEDS,
                show_progress=True
            )
            
        elif metric_name == "human_reasoning":
            print(f"\n[HUMAN REASONING] Evaluating agreement with human-like reasoning...")
            
            # Check HR availability
            if not hr.is_available():
                print(f"[ERROR] Human Reasoning ground truth not available.")
                print(f"[SUGGESTION] Generate it first using: main.generate_hr_ground_truth('your_api_key')")
                return None
            
            # Load HR dataset
            hr_dataset = hr.load_ground_truth()
            if hr_dataset is None:
                print(f"[ERROR] Failed to load Human Reasoning dataset")
                return None
            
            # Evaluate
            score = metrics.evaluate_human_reasoning_over_dataset(
                model=model,
                tokenizer=tokenizer,
                explainer=explainer,
                hr_dataset=hr_dataset,
                show_progress=True
            )
            
        else:
            print(f"[ERROR] Unknown metric: {metric_name}")
            print(f"[AVAILABLE] Available metrics: robustness, contrastivity, consistency, human_reasoning")
            return None
        
        profiler.end_operation(f"metric_{metric_name}")
        profiler.end_operation("total_evaluation")
        
        # Results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Model:     {model_key}")
        print(f"Explainer: {explainer_name}")
        print(f"Metric:    {metric_name}")
        print(f"Score:     {score:.4f}")
        
        # Interpretation
        if metric_name == "robustness":
            print(f"Interpretation: {'Very Robust' if score < 0.05 else 'Robust' if score < 0.1 else 'Moderate' if score < 0.2 else 'Poor'}")
            print(f"(Lower is better)")
        elif metric_name == "consistency":
            print(f"Interpretation: {'Very Consistent' if score > 0.9 else 'Consistent' if score > 0.8 else 'Moderate' if score > 0.6 else 'Poor'}")
            print(f"(Higher is better)")
        elif metric_name == "contrastivity":
            print(f"Interpretation: {'Very Contrastive' if score > 5.0 else 'Contrastive' if score > 2.0 else 'Moderate' if score > 1.0 else 'Poor'}")
            print(f"(Higher is better)")
        elif metric_name == "human_reasoning":
            print(f"Interpretation: {'Excellent Agreement' if score > 0.8 else 'Good Agreement' if score > 0.6 else 'Moderate Agreement' if score > 0.4 else 'Poor Agreement'}")
            print(f"(Higher is better - Mean Average Precision)")
        
        print(f"{'='*60}")
        
        # Performance summary
        profiler.print_summary()
        
        return score
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        aggressive_cleanup()
        print_memory_status()

def check_hr_status():
    """Controlla status Human Reasoning ground truth (funzione comoda per Colab)."""
    print_colab_header()
    print(f"\n[HR STATUS] Checking Human Reasoning availability...")
    
    hr_info = hr.get_info()
    
    print(f"\n{'='*60}")
    print("HUMAN REASONING STATUS")
    print(f"{'='*60}")
    
    if hr_info['available']:
        print(f" Status: Available")
        print(f"  Total examples: {hr_info['total_examples']}")
        print(f"  Valid examples: {hr_info['valid_examples']}")
        if hr_info['success_rate'] is not None:
            print(f"  Success rate: {hr_info['success_rate']:.1%}")
        print(f"  Avg words per example: {hr_info['avg_words_per_example']:.1f}")
        print(f"  File path: {hr_info['file_path']}")
        
        print(f"\n Human Reasoning is ready for evaluation!")
        print(f"  Use: main.evaluate_combination(model, explainer, 'human_reasoning')")
    else:
        print(f" Status: Not Available")
        print(f"\n  To generate Human Reasoning ground truth:")
        print(f"  1. Get OpenRouter API key from: https://openrouter.ai/")
        print(f"  2. Run: main.generate_hr_ground_truth('your_api_key')")
        print(f"  3. Wait ~8-10 minutes for 400 examples to be processed")
    
    print(f"{'='*60}")
    
    return hr_info['available']

def generate_hr_ground_truth(api_key: str, sample_size: int = 400):
    """Genera Human Reasoning ground truth (funzione comoda per Colab)."""
    print_colab_header()
    print(f"\n[HR GENERATE] Generating Human Reasoning ground truth...")
    print(f"[HR GENERATE] API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else 'provided'}")
    print(f"[HR GENERATE] Sample size: {sample_size}")
    
    if not api_key or len(api_key) < 10:
        print(f"[ERROR] Invalid API key provided")
        print(f"[INSTRUCTION] Get your OpenRouter API key from: https://openrouter.ai/")
        return False
    
    try:
        print(f"\n{'='*60}")
        print("HUMAN REASONING GENERATION")
        print(f"{'='*60}")
        print(f"This will take approximately {sample_size * 1.2 / 60:.1f} minutes...")
        print(f"The process will:")
        print(f"  1. Sample {sample_size} texts from clustered dataset")
        print(f"  2. Send each to LLM (Kimi-K2) for importance ranking")
        print(f"  3. Parse and validate responses")
        print(f"  4. Save ground truth dataset")
        print(f"{'='*60}")
        
        # Start generation
        with Timer(f"HR Generation ({sample_size} examples)", track_memory=True):
            hr_dataset = hr.generate_ground_truth(
                api_key=api_key,
                sample_size=sample_size
            )
        
        # Verify results
        if hr_dataset is not None and len(hr_dataset) > 0:
            valid_count = (hr_dataset['hr_count'] > 0).sum()
            success_rate = valid_count / len(hr_dataset)
            
            print(f"\n{'='*60}")
            print("GENERATION COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"  Total examples processed: {len(hr_dataset)}")
            print(f"  Valid rankings generated: {valid_count}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Average words per example: {hr_dataset['hr_count'].mean():.1f}")
            
            if success_rate > 0.7:
                print(f"\n Human Reasoning ground truth is ready!")
                print(f"  You can now evaluate explainers using 'human_reasoning' metric")
                print(f"  Example: main.evaluate_combination('tinybert', 'lime', 'human_reasoning')")
                return True
            else:
                print(f"\n Low success rate ({success_rate:.1%})")
                print(f"  Some examples may have failed due to API issues")
                print(f"  Ground truth is available but may be incomplete")
                return True
        else:
            print(f"\n Generation failed!")
            print(f"  No valid data was generated")
            return False
    
    except Exception as e:
        print(f"\n[ERROR] Human Reasoning generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test(model_key: str = "tinybert", explainer_name: str = "lime"):
    """Test veloce del sistema (funzione comoda per Colab) - ora include Human Reasoning."""
    print_colab_header()
    print(f"\n[QUICK TEST] Testing {model_key} + {explainer_name}")
    
    # Check dependencies
    available_models, available_explainers = check_dependencies()
    
    if model_key not in available_models:
        model_key = available_models[0]
        print(f"[FALLBACK] Using {model_key} instead")
    
    if explainer_name not in available_explainers:
        explainer_name = available_explainers[0]
        print(f"[FALLBACK] Using {explainer_name} instead")
    
    # Test explanation
    print(f"\n[TEST 1] Single explanation...")
    result = explain_text("This movie is absolutely fantastic!", model_key, explainer_name)
    
    if result:
        print(f"[TEST 1]  Explanation successful")
        
        # Test evaluation (standard metric)
        print(f"\n[TEST 2] Quick evaluation (robustness)...")
        score = evaluate_combination(model_key, explainer_name, "robustness", sample_size=10)
        
        if score is not None:
            print(f"[TEST 2]  Evaluation successful (score: {score:.4f})")
            
            # Test Human Reasoning availability
            print(f"\n[TEST 3] Human Reasoning status...")
            hr_available = check_hr_status()
            
            if hr_available:
                print(f"[TEST 3]  Human Reasoning available")
                
                # Optional: Test HR evaluation if available
                print(f"\n[TEST 4] Human Reasoning evaluation (optional)...")
                try:
                    hr_score = evaluate_combination(model_key, explainer_name, "human_reasoning", sample_size=5)
                    if hr_score is not None:
                        print(f"[TEST 4]  Human Reasoning evaluation successful (score: {hr_score:.4f})")
                    else:
                        print(f"[TEST 4]  Human Reasoning evaluation failed (but system is functional)")
                except Exception:
                    print(f"[TEST 4]  Human Reasoning evaluation skipped (but system is functional)")
            else:
                print(f"[TEST 3]  Human Reasoning not available (can be generated later)")
            
            print(f"\n QUICK TEST PASSED! System is ready for full experiments.")
            print(f"\nNext steps:")
            print(f"  • For full evaluation: main.evaluate_combination(model, explainer, metric)")
            if not hr_available:
                print(f"  • For Human Reasoning: main.generate_hr_ground_truth('your_api_key')")
            print(f"  • Available metrics: robustness, consistency, contrastivity, human_reasoning")
            
            return True
        else:
            print(f"[TEST 2]  Evaluation failed")
    else:
        print(f"[TEST 1]  Explanation failed")
    
    print(f"\n QUICK TEST FAILED! Check error messages above.")
    return False

# ==== CLI Functions (per compatibilità) ====
def cmd_explain(args):
    """CLI explain command."""
    result = explain_text(args.text, args.model, args.explainer)
    if result is None:
        sys.exit(1)

def cmd_evaluate(args):
    """CLI evaluate command."""
    score = evaluate_combination(args.model, args.explainer, args.metric, args.sample)
    if score is None:
        sys.exit(1)

def cmd_test(args):
    """CLI test command."""
    success = quick_test()
    if not success:
        sys.exit(1)

def cmd_hr_status(args):
    """CLI HR status command."""
    available = check_hr_status()
    if not available:
        sys.exit(1)

def cmd_hr_generate(args):
    """CLI HR generate command."""
    if not args.api_key:
        print("[ERROR] API key required for HR generation")
        print("Get your key from: https://openrouter.ai/")
        sys.exit(1)
    
    success = generate_hr_ground_truth(args.api_key, args.sample)
    if not success:
        sys.exit(1)

def list_available_resources():
    """Lista risorse disponibili."""
    print_colab_header()
    check_dependencies()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="XAI Benchmark CLI per Google Colab + Human Reasoning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Explain ----
    p_explain = subparsers.add_parser("explain", help="Spiega una frase")
    p_explain.add_argument("--model", required=True, choices=models.MODELS.keys())
    p_explain.add_argument("--explainer", required=True, choices=explainers.list_explainers())
    p_explain.add_argument("--text", required=True, help="Testo da spiegare")

    # ---- Evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Valuta modello+explainer")
    p_eval.add_argument("--metric", required=True, 
                       choices=["robustness", "contrastivity", "consistency", "human_reasoning"])
    p_eval.add_argument("--model", required=True, choices=models.MODELS.keys())
    p_eval.add_argument("--explainer", required=True, choices=explainers.list_explainers())
    p_eval.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_SIZE,
                       help="Numero di esempi da valutare")

    # ---- Test ----
    p_test = subparsers.add_parser("test", help="Test veloce sistema")
    p_test.add_argument("--model", default="tinybert", choices=models.MODELS.keys())
    p_test.add_argument("--explainer", default="lime", choices=explainers.list_explainers())

    # ---- Human Reasoning Status ----
    p_hr_status = subparsers.add_parser("hr-status", help="Controlla status Human Reasoning")

    # ---- Human Reasoning Generate ----
    p_hr_gen = subparsers.add_parser("hr-generate", help="Genera Human Reasoning ground truth")
    p_hr_gen.add_argument("--api-key", required=True, help="OpenRouter API key")
    p_hr_gen.add_argument("--sample", type=int, default=400, help="Numero di esempi")

    # ---- List ----
    p_list = subparsers.add_parser("list", help="Lista risorse disponibili")

    args = parser.parse_args()

    # Execute command
    if args.command == "explain":
        cmd_explain(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "hr-status":
        cmd_hr_status(args)
    elif args.command == "hr-generate":
        cmd_hr_generate(args)
    elif args.command == "list":
        list_available_resources()

if __name__ == "__main__":
    main()