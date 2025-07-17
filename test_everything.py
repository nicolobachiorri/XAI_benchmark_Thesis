"""
test_everything.py - Test completi e progressivi
==============================================

Sequenza di test per verificare che tutto funzioni prima del report completo:

1. Test modelli (caricamento)
2. Test explainer (creazione + spiegazione singola)
3. Test metriche (calcolo su dati minimi)
4. Test report mini (tutto insieme)
5. Test report piccolo (con più dati)

Uso:
    python test_everything.py --level 1    # Solo test modelli
    python test_everything.py --level 2    # Fino a explainer
    python test_everything.py --level 3    # Fino a metriche
    python test_everything.py --level 4    # Test mini report
    python test_everything.py --level 5    # Test report piccolo
    python test_everything.py              # Tutti i test
"""

import argparse
import time
import traceback
from datetime import datetime

import models
import dataset
import explainers
import metrics
from utils import set_seed, Timer

set_seed(42)

def test_level_1_models():
    """LEVEL 1: Test caricamento di tutti i modelli"""
    print("=" * 60)
    print("LEVEL 1: TEST CARICAMENTO MODELLI")
    print("=" * 60)
    
    results = {}
    
    for model_key in models.MODELS.keys():
        print(f"\n[1/{len(models.MODELS)}] Testing {model_key}...")
        try:
            start_time = time.time()
            
            # Test tokenizer
            print(f"  Loading tokenizer...", end="")
            tokenizer = models.load_tokenizer(model_key)
            print(" OK")
            
            # Test model
            print(f"  Loading model...", end="")
            model = models.load_model(model_key)
            print(" OK")
            
            # Test inference veloce
            print(f"  Testing inference...", end="")
            test_text = "This is a test."
            encoded = tokenizer(test_text, return_tensors="pt", max_length=50, truncation=True)
            outputs = model(**encoded)
            logits = outputs.logits
            print(f" OK (shape: {logits.shape})")
            
            load_time = time.time() - start_time
            print(f"  SUCCESS {model_key}: ({load_time:.1f}s)")
            results[model_key] = {"status": "OK", "time": load_time, "logits_shape": logits.shape}
            
        except Exception as e:
            print(f"  FAILED {model_key}")
            print(f"     Error: {str(e)}")
            results[model_key] = {"status": "FAILED", "error": str(e)}
    
    # Riassunto
    print(f"\n{'='*60}")
    print("LEVEL 1 SUMMARY:")
    success_count = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"Successful: {success_count}/{len(results)}")
    print(f"Failed: {len(results) - success_count}/{len(results)}")
    
    if success_count == 0:
        print("CRITICAL: No models loaded successfully!")
        return False
    
    return True

def test_level_2_explainers():
    """LEVEL 2: Test creazione explainer per ogni modello"""
    print("\n" + "=" * 60)
    print("LEVEL 2: TEST EXPLAINER CREATION")
    print("=" * 60)
    
    # Usa modelli che dovrebbero funzionare
    test_models = ["distilbert", "tinybert"]  # Inizia con i più piccoli
    test_explainers = explainers.list_explainers()
    test_text = "This movie is absolutely fantastic!"
    
    results = {}
    
    for model_key in test_models:
        if model_key not in models.MODELS:
            continue
            
        print(f"\n--- Testing model: {model_key} ---")
        
        try:
            # Carica modello
            model = models.load_model(model_key)
            tokenizer = models.load_tokenizer(model_key)
            print(f"Model {model_key} loaded")
            
            results[model_key] = {}
            
            for explainer_name in test_explainers:
                print(f"  Testing {explainer_name}...", end="")
                try:
                    start_time = time.time()
                    
                    # Crea explainer
                    explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                    
                    # Test spiegazione
                    attribution = explainer(test_text)
                    
                    # Verifica output
                    if hasattr(attribution, 'tokens') and hasattr(attribution, 'scores'):
                        n_tokens = len(attribution.tokens)
                        n_scores = len(attribution.scores)
                        exec_time = time.time() - start_time
                        
                        if n_tokens == n_scores and n_tokens > 0:
                            print(f" OK ({n_tokens} tokens, {exec_time:.1f}s)")
                            results[model_key][explainer_name] = {
                                "status": "OK", 
                                "tokens": n_tokens, 
                                "time": exec_time
                            }
                        else:
                            print(f" MISMATCH (tokens:{n_tokens}, scores:{n_scores})")
                            results[model_key][explainer_name] = {
                                "status": "MISMATCH", 
                                "tokens": n_tokens, 
                                "scores": n_scores
                            }
                    else:
                        print(f" BAD OUTPUT")
                        results[model_key][explainer_name] = {"status": "BAD_OUTPUT"}
                        
                except Exception as e:
                    print(f" ERROR: {str(e)[:50]}...")
                    results[model_key][explainer_name] = {"status": "ERROR", "error": str(e)}
                    
        except Exception as e:
            print(f"Failed to load model {model_key}: {e}")
            results[model_key] = {"model_error": str(e)}
    
    # Riassunto
    print(f"\n{'='*60}")
    print("LEVEL 2 SUMMARY:")
    total_tests = 0
    success_tests = 0
    
    for model_key, model_results in results.items():
        if "model_error" in model_results:
            continue
        for explainer_name, result in model_results.items():
            total_tests += 1
            if result.get("status") == "OK":
                success_tests += 1
    
    print(f"Successful: {success_tests}/{total_tests}")
    print(f"Failed: {total_tests - success_tests}/{total_tests}")
    
    return success_tests > 0

def test_level_3_metrics():
    """LEVEL 3: Test calcolo metriche con dati minimi"""
    print("\n" + "=" * 60)
    print("LEVEL 3: TEST METRICS CALCULATION")
    print("=" * 60)
    
    # Usa combinazione che dovrebbe funzionare
    model_key = "distilbert"
    explainer_name = "lime"  # LIME è di solito il più stabile
    test_texts = [
        "This movie is great!",
        "This movie is terrible.",
        "An okay film."
    ]
    test_labels = [1, 0, 1]
    
    try:
        print(f"Loading {model_key} + {explainer_name}...")
        model = models.load_model(model_key)
        tokenizer = models.load_tokenizer(model_key)
        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
        print("Model and explainer loaded")
        
        results = {}
        
        # Test Robustness
        print("\nTesting Robustness...")
        try:
            start_time = time.time()
            score = metrics.evaluate_robustness_over_dataset(
                model, tokenizer, explainer, test_texts[:2]  # Solo 2 esempi
            )
            exec_time = time.time() - start_time
            print(f"Robustness: {score:.4f} ({exec_time:.1f}s)")
            results["robustness"] = {"status": "OK", "score": score, "time": exec_time}
        except Exception as e:
            print(f"Robustness failed: {e}")
            results["robustness"] = {"status": "FAILED", "error": str(e)}
        
        # Test Contrastivity
        print("\nTesting Contrastivity...")
        try:
            start_time = time.time()
            pos_attrs = [explainer(test_texts[0])]  # Positivo
            neg_attrs = [explainer(test_texts[1])]  # Negativo
            score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
            exec_time = time.time() - start_time
            print(f"Contrastivity: {score:.4f} ({exec_time:.1f}s)")
            results["contrastivity"] = {"status": "OK", "score": score, "time": exec_time}
        except Exception as e:
            print(f"Contrastivity failed: {e}")
            results["contrastivity"] = {"status": "FAILED", "error": str(e)}
        
        # Test Consistency (2 modelli)
        print("\nTesting Consistency...")
        try:
            start_time = time.time()
            model2 = models.load_model("tinybert")  # Modello diverso
            tokenizer2 = models.load_tokenizer("tinybert")
            explainer2 = explainers.get_explainer(explainer_name, model2, tokenizer2)
            
            score = metrics.evaluate_consistency_over_dataset(
                model, model2, tokenizer, tokenizer2, 
                explainer, explainer2, test_texts[:2]
            )
            exec_time = time.time() - start_time
            print(f"Consistency: {score:.4f} ({exec_time:.1f}s)")
            results["consistency"] = {"status": "OK", "score": score, "time": exec_time}
        except Exception as e:
            print(f"Consistency failed: {e}")
            results["consistency"] = {"status": "FAILED", "error": str(e)}
        
        # Riassunto
        print(f"\n{'='*60}")
        print("LEVEL 3 SUMMARY:")
        success_count = sum(1 for r in results.values() if r["status"] == "OK")
        print(f"Successful metrics: {success_count}/{len(results)}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"CRITICAL ERROR in Level 3: {e}")
        traceback.print_exc()
        return False

def test_level_4_mini_report():
    """LEVEL 4: Test mini report (1 metrica, 2 modelli, 2 explainer)"""
    print("\n" + "=" * 60)
    print("LEVEL 4: TEST MINI REPORT")
    print("=" * 60)
    
    # Configurazione minima
    test_models = ["distilbert", "tinybert"]
    test_explainers = ["lime", "grad_input"]
    test_metrics = ["robustness"]
    sample_size = 2
    
    print(f"Configuration:")
    print(f"  Models: {test_models}")
    print(f"  Explainers: {test_explainers}")
    print(f"  Metrics: {test_metrics}")
    print(f"  Sample size: {sample_size}")
    
    try:
        from collections import defaultdict
        import pandas as pd
        
        # Simula il report
        results = defaultdict(dict)
        total_tests = len(test_models) * len(test_explainers) * len(test_metrics)
        current_test = 0
        
        for model_key in test_models:
            for explainer_name in test_explainers:
                for metric_name in test_metrics:
                    current_test += 1
                    print(f"\n[{current_test}/{total_tests}] {model_key} + {explainer_name} + {metric_name}")
                    
                    try:
                        # Carica modello
                        model = models.load_model(model_key)
                        tokenizer = models.load_tokenizer(model_key)
                        explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                        
                        # Test con dati minimi
                        test_texts = ["This is great!", "This is bad!"]
                        
                        start_time = time.time()
                        if metric_name == "robustness":
                            score = metrics.evaluate_robustness_over_dataset(
                                model, tokenizer, explainer, test_texts
                            )
                        elif metric_name == "contrastivity":
                            pos_attrs = [explainer(test_texts[0])]
                            neg_attrs = [explainer(test_texts[1])]
                            score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
                        else:
                            score = 0.5  # Placeholder
                        
                        exec_time = time.time() - start_time
                        results[explainer_name][model_key] = score
                        print(f"  Score: {score:.4f} ({exec_time:.1f}s)")
                        
                    except Exception as e:
                        print(f"  ERROR: {str(e)[:50]}...")
                        results[explainer_name][model_key] = float('nan')
        
        # Crea DataFrame
        df = pd.DataFrame(results).T
        print(f"\n{'='*60}")
        print("MINI REPORT RESULTS:")
        print(df.to_string(float_format="%.4f"))
        
        # Verifica risultati
        valid_results = df.notna().sum().sum()
        total_cells = df.size
        print(f"\nValid results: {valid_results}/{total_cells}")
        
        return valid_results > 0
        
    except Exception as e:
        print(f"CRITICAL ERROR in Level 4: {e}")
        traceback.print_exc()
        return False

def test_level_5_small_report():
    """LEVEL 5: Test report piccolo ma completo"""
    print("\n" + "=" * 60)
    print("LEVEL 5: TEST SMALL REPORT")
    print("=" * 60)
    
    print("Running: python report.py --sample 3 --csv")
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "report.py", "--sample", "3", "--csv"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minuti timeout
        )
        
        if result.returncode == 0:
            print("Small report completed successfully!")
            print("Output:")
            print(result.stdout[-500:])  # Ultimi 500 caratteri
            return True
        else:
            print("Small report failed!")
            print("Error output:")
            print(result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print("Small report timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"Error running small report: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test progressivi sistema XAI")
    parser.add_argument("--level", type=int, choices=[1,2,3,4,5], default=5, 
                       help="Livello di test da eseguire (1-5)")
    args = parser.parse_args()
    
    print(f"TESTING XAI BENCHMARK SYSTEM")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test level: {args.level}")
    
    start_time = time.time()
    
    # Esegui test progressivi
    tests = [
        (1, "Models Loading", test_level_1_models),
        (2, "Explainers Creation", test_level_2_explainers),
        (3, "Metrics Calculation", test_level_3_metrics),
        (4, "Mini Report", test_level_4_mini_report),
        (5, "Small Report", test_level_5_small_report),
    ]
    
    passed_tests = 0
    
    for level, name, test_func in tests:
        if level > args.level:
            break
            
        print(f"\nStarting Level {level}: {name}")
        try:
            success = test_func()
            if success:
                passed_tests += 1
                print(f"Level {level} PASSED")
            else:
                print(f"Level {level} FAILED")
                if level <= 3:  # Livelli critici
                    print("STOPPING: Critical test failed")
                    break
        except Exception as e:
            print(f"Level {level} CRASHED: {e}")
            traceback.print_exc()
            break
    
    total_time = time.time() - start_time
    
    # Riassunto finale
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY ({total_time:.1f}s total)")
    print(f"Passed: {passed_tests}/{min(args.level, len(tests))}")
    print(f"Failed: {min(args.level, len(tests)) - passed_tests}")
    
    if passed_tests == min(args.level, len(tests)):
        print("\nALL TESTS PASSED! System is ready for full report")
        print("You can now run: python report.py --sample 500 --csv")
    else:
        print(f"\nSome tests failed. Fix issues before running full report.")
    
    return passed_tests == min(args.level, len(tests))

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)