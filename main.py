"""
XAI Benchmark - Main CLI
Centro di controllo per la valutazione dei metodi XAI su modelli transformer encoder-only
"""

import argparse
import logging
import sys
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

# Import dei moduli custom
from models import ModelManager, ModelConfig, print_models_info, test_model_loading
from dataset import IMDBDatasetManager, test_dataset_loading
from explainers import ExplainerManager, test_explainers
from metrics import MetricsManager, test_metrics
from evaluate import XAIEvaluationPipeline, test_evaluation_pipeline

# Setup logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup configurazione logging"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Crea parser degli argomenti CLI"""
    
    parser = argparse.ArgumentParser(
        description="XAI Benchmark: Valutazione metodi XAI per transformer encoder-only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

  # Valutazione completa (tutti i modelli e explainer)
  python main.py --models all --explainers all --output results.json

  # Valutazione specifica
  python main.py --models bert-base,roberta-large --explainers lime,shap

  # Test rapido
  python main.py --models tinybert --explainers lime --eval-size 20

  # Solo preparazione modelli
  python main.py --prepare-only --models bert-base,bert-large

  # Test di sistema
  python main.py --test-all

  # Informazioni sui modelli disponibili
  python main.py --list-models
        """
    )
    
    # Modalità operative
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--test-all', 
        action='store_true',
        help='Esegue test di tutti i moduli'
    )
    mode_group.add_argument(
        '--list-models', 
        action='store_true',
        help='Elenca modelli disponibili'
    )
    mode_group.add_argument(
        '--list-explainers', 
        action='store_true',
        help='Elenca explainer disponibili'
    )
    mode_group.add_argument(
        '--prepare-only', 
        action='store_true',
        help='Solo preparazione modelli (fine-tuning)'
    )
    
    # Configurazione valutazione
    parser.add_argument(
        '--models', 
        type=str,
        default='all',
        help='Modelli da valutare (comma-separated o "all"). Default: all'
    )
    
    parser.add_argument(
        '--explainers', 
        type=str,
        default='all',
        help='Explainer da valutare (comma-separated o "all"). Default: all'
    )
    
    parser.add_argument(
        '--eval-size', 
        type=int,
        default=100,
        help='Numero di esempi per valutazione. Default: 100'
    )
    
    parser.add_argument(
        '--dataset-size', 
        type=int,
        default=None,
        help='Dimensione dataset (per test rapidi). Default: dataset completo'
    )
    
    # Output e cache
    parser.add_argument(
        '--output', 
        type=str,
        default='./results',
        help='Directory output. Default: ./results'
    )
    
    parser.add_argument(
        '--cache-dir', 
        type=str,
        default='./cache',
        help='Directory cache. Default: ./cache'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str,
        default=None,
        help='File di log (opzionale)'
    )
    
    # Opzioni avanzate
    parser.add_argument(
        '--force-retrain', 
        action='store_true',
        help='Forza riaddestramento modelli anche se già fine-tuned'
    )
    
    parser.add_argument(
        '--no-consistency', 
        action='store_true',
        help='Salta valutazione consistency (più veloce)'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int,
        default=42,
        help='Seed per riproducibilità. Default: 42'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Livello di logging. Default: INFO'
    )
    
    return parser


def parse_model_list(models_str: str) -> List[str]:
    """Parsa lista modelli dall'input CLI"""
    if models_str.lower() == 'all':
        return list(ModelConfig.MODELS.keys())
    
    models = [m.strip() for m in models_str.split(',')]
    
    # Valida modelli
    available_models = set(ModelConfig.MODELS.keys())
    invalid_models = [m for m in models if m not in available_models]
    
    if invalid_models:
        logger.error(f"Modelli non validi: {invalid_models}")
        logger.error(f"Modelli disponibili: {list(available_models)}")
        sys.exit(1)
    
    return models


def parse_explainer_list(explainers_str: str) -> List[str]:
    """Parsa lista explainer dall'input CLI"""
    available_explainers = [
        'lime', 'shap', 'inputxgradient', 
        'lrp', 'attention_rollout', 'attention_flow'
    ]
    
    if explainers_str.lower() == 'all':
        return available_explainers
    
    explainers = [e.strip() for e in explainers_str.split(',')]
    
    # Valida explainer
    invalid_explainers = [e for e in explainers if e not in available_explainers]
    
    if invalid_explainers:
        logger.error(f"Explainer non validi: {invalid_explainers}")
        logger.error(f"Explainer disponibili: {available_explainers}")
        sys.exit(1)
    
    return explainers


def run_tests() -> bool:
    """Esegue test di tutti i moduli"""
    logger.info("AVVIO: Test completo del sistema")
    
    tests = [
        ("Models", test_model_loading),
        ("Dataset", test_dataset_loading),
        ("Explainers", test_explainers),
        ("Metrics", test_metrics),
        ("Evaluation Pipeline", test_evaluation_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"TEST: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"RESULT: {test_name} - {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"RESULT: {test_name} - ERROR: {str(e)}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"TEST SUMMARY: {passed}/{total} test passati")
    
    if passed == total:
        logger.info("SUCCESS: Tutti i test passati - sistema pronto")
        return True
    else:
        logger.warning("WARNING: Alcuni test falliti - verificare configurazione")
        return False


def list_available_models():
    """Elenca modelli disponibili"""
    print("\nMODELLI DISPONIBILI:")
    print("=" * 80)
    
    for key, info in ModelConfig.MODELS.items():
        status = "Needs fine-tuning" if info["needs_finetuning"] else "Pre-trained"
        print(f"{key:<20} | {info['architecture']:<15} | {info['parameters']:<8} | {status}")
    
    print("=" * 80)
    print(f"Totale: {len(ModelConfig.MODELS)} modelli")


def list_available_explainers():
    """Elenca explainer disponibili"""
    explainers = [
        ("lime", "Model Simplification", "LIME - Local Interpretable Model-agnostic Explanations"),
        ("shap", "Perturbation-based", "SHAP - SHapley Additive exPlanations"),
        ("inputxgradient", "Gradient-based", "Input × Gradient method"),
        ("lrp", "Layer-wise Relevance", "Layer-wise Relevance Propagation"),
        ("attention_rollout", "Attention Mechanism", "Attention Rollout through layers"),
        ("attention_flow", "Attention Mechanism", "Attention Flow (weighted average)")
    ]
    
    print("\nEXPLAINER DISPONIBILI:")
    print("=" * 90)
    
    for name, category, description in explainers:
        print(f"{name:<18} | {category:<20} | {description}")
    
    print("=" * 90)
    print(f"Totale: {len(explainers)} explainer")


def prepare_models_only(models: List[str], 
                       cache_dir: str, 
                       force_retrain: bool = False) -> bool:
    """Prepara solo i modelli (fine-tuning)"""
    logger.info(f"PREPARE: Fine-tuning {len(models)} modelli")
    
    # Inizializza pipeline per preparazione
    pipeline = XAIEvaluationPipeline(cache_dir=cache_dir)
    
    # Setup dataset
    dataset_info = pipeline.setup_dataset()
    
    # Prepara modelli
    model_status = pipeline.prepare_models(models, force_retrain=force_retrain)
    
    # Report
    successful = sum(model_status.values())
    total = len(model_status)
    
    logger.info(f"COMPLETE: {successful}/{total} modelli preparati con successo")
    
    for model, status in model_status.items():
        status_str = "SUCCESS" if status else "FAILED"
        logger.info(f"  {model}: {status_str}")
    
    return successful == total


def run_evaluation(models: List[str], 
                  explainers: List[str],
                  eval_size: int,
                  dataset_size: Optional[int],
                  output_dir: str,
                  cache_dir: str,
                  include_consistency: bool,
                  force_retrain: bool,
                  random_state: int) -> bool:
    """Esegue valutazione completa"""
    
    logger.info(f"EVAL: Avvio valutazione XAI")
    logger.info(f"  Modelli: {models}")
    logger.info(f"  Explainer: {explainers}")
    logger.info(f"  Eval size: {eval_size}")
    logger.info(f"  Dataset size: {dataset_size or 'completo'}")
    
    start_time = time.time()
    
    try:
        # Inizializza pipeline
        pipeline = XAIEvaluationPipeline(
            output_dir=output_dir,
            cache_dir=cache_dir,
            eval_subset_size=eval_size,
            random_state=random_state
        )
        
        # Setup dataset con dimensione specifica
        if dataset_size:
            dataset_info = pipeline.setup_dataset(subset_size=dataset_size)
        else:
            dataset_info = pipeline.setup_dataset()
        
        # Esegui valutazione
        results = pipeline.run_full_evaluation(
            models_to_use=models,
            explainers_to_use=explainers,
            include_consistency=include_consistency,
            save_results=True
        )
        
        # Report finale
        total_time = time.time() - start_time
        
        logger.info("COMPLETE: Valutazione completata")
        logger.info(f"  Tempo totale: {total_time:.2f}s")
        logger.info(f"  Valutazioni: {results['summary']['successful_evaluations']}")
        logger.info(f"  Fallimenti: {results['summary']['failed_evaluations']}")
        
        # Best performers
        summary = results['summary']
        if 'best_robustness' in summary:
            best = summary['best_robustness']
            logger.info(f"  Best Robustness: {best['model']} + {best['explainer']} ({best['score']:.4f})")
        
        if 'best_contrastivity' in summary:
            best = summary['best_contrastivity']
            logger.info(f"  Best Contrastivity: {best['model']} + {best['explainer']} ({best['score']:.4f})")
        
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Valutazione fallita: {str(e)}")
        return False


def main():
    """Funzione main"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logger.info("XAI Benchmark - Valutazione metodi XAI per transformer encoder-only")
    logger.info(f"Versione Python: {sys.version}")
    logger.info(f"Directory corrente: {os.getcwd()}")
    
    # Modalità lista
    if args.list_models:
        list_available_models()
        return
    
    if args.list_explainers:
        list_available_explainers()
        return
    
    # Modalità test
    if args.test_all:
        success = run_tests()
        sys.exit(0 if success else 1)
    
    # Parsa configurazione
    models = parse_model_list(args.models)
    explainers = parse_explainer_list(args.explainers)
    
    logger.info(f"Configurazione:")
    logger.info(f"  Modelli: {models}")
    logger.info(f"  Explainer: {explainers}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Cache: {args.cache_dir}")
    
    # Modalità preparazione
    if args.prepare_only:
        success = prepare_models_only(models, args.cache_dir, args.force_retrain)
        sys.exit(0 if success else 1)
    
    # Modalità valutazione completa
    success = run_evaluation(
        models=models,
        explainers=explainers,
        eval_size=args.eval_size,
        dataset_size=args.dataset_size,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        include_consistency=not args.no_consistency,
        force_retrain=args.force_retrain,
        random_state=args.random_state
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()