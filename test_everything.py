"""
test_everything.py - Test completi e progressivi AGGIORNATO
=========================================================

Sequenza di test per verificare che tutto funzioni prima del report completo:

1. Test modelli (caricamento)
2. Test explainer (creazione + spiegazione singola) - AGGIORNATO
3. Test metriche (calcolo su dati minimi) - AGGIORNATO per consistency fix
4. Test report mini (tutto insieme)
5. Test report piccolo (con più dati)

AGGIORNAMENTI:
- Consistency ora usa inference seed invece di modelli diversi
- Test explainer aggiornati per nuovi fix (shap, attention_flow)
- Timing logs inclusi per debugging performance
- Controlli dipendenze explainer

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
        print(f"\n[{len(results)+1}/{len(models.MODELS)}] Testing {model_key}...")
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
    """LEVEL 2: Test creazione explainer per ogni modello (AGGIORNATO)"""
    print("\n" + "=" * 60)
    print("LEVEL 2: TEST EXPLAINER CREATION (UPDATED)")
    print("=" * 60)
    
    # Prima controlla dipendenze
    print("\n--- Controllo Dipendenze ---")
    deps = explainers.check_dependencies()
    
    # Lista explainer disponibili (dinamica basata su dipendenze)
    available_explainers = explainers.list_explainers()
    print(f"\nExplainer disponibili: {available_explainers}")
    
    if not available_explainers:
        print("ERROR: Nessun explainer disponibile!")
        return False
    
    # Usa modelli che dovrebbero funzionare
    test_models = ["distilbert", "tinybert"]  # Inizia con i più piccoli
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
            
            for explainer_name in available_explainers:
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
    
    # Performance summary
    if success_tests > 0:
        print("\nPerformance ranking:")
        all_times = []
        for model_key, model_results in results.items():
            if "model_error" in model_results:
                continue
            for explainer_name, result in model_results.items():
                if result.get("status") == "OK":
                    all_times.append((explainer_name, result["time"]))
        
        if all_times:
            # Media per explainer
            explainer_times = {}
            for explainer, time_val in all_times:
                if explainer not in explainer_times:
                    explainer_times[explainer] = []
                explainer_times[explainer].append(time_val)
            
            avg_times = {exp: sum(times)/len(times) for exp, times in explainer_times.items()}
            sorted_times = sorted(avg_times.items(), key=lambda x: x[1])
            
            for i, (explainer, avg_time) in enumerate(sorted_times[:5]):
                print(f"  {i+1}. {explainer}: {avg_time:.2f}s avg")
    
    return success_tests > 0

def test_level_3_metrics():
    """LEVEL 3: Test calcolo metriche con dati minimi (AGGIORNATO per consistency)"""
    print("\n" + "=" * 60)
    print("LEVEL 3: TEST METRICS CALCULATION (UPDATED)")
    print("=" * 60)
    
    # Usa combinazione che dovrebbe funzionare
    model_key = "distilbert"
    
    # Prova prima con lime (più stabile), poi con altri se disponibili
    available_explainers = explainers.list_explainers()
    explainer_name = "lime" if "lime" in available_explainers else available_explainers[0]
    
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
                model, tokenizer, explainer, test_texts[:2], show_progress=False
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
        
        # Test Consistency (AGGIORNATO: ora usa inference seed)
        print("\nTesting Consistency (con inference seed)...")
        try:
            start_time = time.time()
            seeds = [42, 123]  # Solo 2 seed per test veloce
            
            score = metrics.evaluate_consistency_over_dataset(
                model=model,
                tokenizer=tokenizer,
                explainer=explainer,
                texts=test_texts[:2],  # Solo 2 testi
                seeds=seeds,
                show_progress=False
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
        
        for metric_name, result in results.items():
            if result["status"] == "OK":
                print(f"{metric_name}: {result['score']:.4f} ({result['time']:.1f}s)")
            else:
                print(f" {metric_name}: {result.get('error', 'Unknown error')}")
        
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
    
    # Configurazione minima basata su explainer disponibili
    test_models = ["distilbert", "tinybert"]
    available_explainers = explainers.list_explainers()
    test_explainers = available_explainers[:2] if len(available_explainers) >= 2 else available_explainers
    test_metrics = ["robustness"]  # Metric più veloce per test
    sample_size = 2
    
    print(f"Configuration:")
    print(f"  Models: {test_models}")
    print(f"  Explainers: {test_explainers}")
    print(f"  Metrics: {test_metrics}")
    print(f"  Sample size: {sample_size}")
    
    if not test_explainers:
        print("ERROR: Nessun explainer disponibile per mini report!")
        return False
    
    try:
        from collections import defaultdict
        import pandas as pd
        
        # Simula il report
        results = defaultdict(dict)
        total_tests = len(test_models) * len(test_explainers) * len(test_metrics)
        current_test = 0
        
        for model_key in test_models:
            if model_key not in models.MODELS:
                continue
                
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
                                model, tokenizer, explainer, test_texts, show_progress=False
                            )
                        elif metric_name == "contrastivity":
                            pos_attrs = [explainer(test_texts[0])]
                            neg_attrs = [explainer(test_texts[1])]
                            score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
                        elif metric_name == "consistency":
                            score = metrics.evaluate_consistency_over_dataset(
                                model=model,
                                tokenizer=tokenizer,
                                explainer=explainer,
                                texts=test_texts,
                                seeds=[42, 123],
                                show_progress=False
                            )
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
            timeout=600  # 10 minuti timeout (aumentato per nuovi explainer)
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
        print("Small report timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"Error running small report: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test progressivi sistema XAI")
    parser.add_argument("--level", type=int, choices=[1,2,3,4,5], default=5, 
                       help="Livello di test da eseguire (1-5)")
    parser.add_argument("--debug-timing", action="store_true", 
                       help="Abilita debug timing negli explainer")
    args = parser.parse_args()
    
    # Configura debug timing se richiesto
    if args.debug_timing:
        explainers.DEBUG_TIMING = True
        print("Debug timing abilitato per explainer")
    
    print(f"TESTING XAI BENCHMARK SYSTEM (UPDATED)")
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
        print("\n ALL TESTS PASSED! System is ready for full report")
        print("You can now run: python report.py --sample 500 --csv")
        print("\nSUGGESTED NEXT STEPS:")
        print("1. python main.py explain --model distilbert --explainer shap --text 'Great movie!'")
        print("2. python main.py evaluate --metric robustness --model distilbert --explainer lime --sample 10")
        print("3. python report.py --sample 50 --csv")
    else:
        print(f"\n Some tests failed. Fix issues before running full report.")
        print("\nDEBUG SUGGESTIONS:")
        print("1. Check dependencies: python -c 'import explainers; explainers.check_dependencies()'")
        print("2. Test individual explainer: python explainers.py")
        print("3. Run with timing debug: python test_everything.py --debug-timing")
    
    return passed_tests == min(args.level, len(tests))

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)