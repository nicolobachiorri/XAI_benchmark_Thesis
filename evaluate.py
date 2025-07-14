"""
XAI Benchmark - Evaluation Module
Orchestrare il pipeline completo di valutazione dei metodi XAI
"""

import os
import logging
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import torch
from tqdm import tqdm

# Import dei moduli custom
from models import ModelManager, ModelConfig
from dataset import IMDBDatasetManager
from explainers import ExplainerManager
from metrics import MetricsManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XAIEvaluationPipeline:
    """Pipeline completo per la valutazione dei metodi XAI"""
    
    def __init__(self, 
                 output_dir: str = "./results",
                 cache_dir: str = "./cache",
                 eval_subset_size: int = 100,
                 random_state: int = 42):
        
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.eval_subset_size = eval_subset_size
        self.random_state = random_state
        
        # Crea directory
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Inizializza manager
        self.model_manager = ModelManager(
            cache_dir=str(self.cache_dir / "models"),
            models_dir=str(self.cache_dir / "finetuned_models")
        )
        
        self.dataset_manager = IMDBDatasetManager(
            cache_dir=str(self.cache_dir / "datasets")
        )
        
        self.metrics_manager = MetricsManager()
        
        # Storage per risultati
        self.results = {}
        self.detailed_results = {}
        
        # Configurazioni
        self.models_to_evaluate = list(ModelConfig.MODELS.keys())
        self.explainers_to_evaluate = [
            'lime', 'shap', 'inputxgradient', 
            'lrp', 'attention_rollout', 'attention_flow'
        ]
        
    def setup_dataset(self, subset_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Setup del dataset IMDB
        
        Args:
            subset_size: Dimensione subset per test rapidi
            
        Returns:
            Dataset info
        """
        logger.info("SETUP: Caricamento dataset IMDB")
        
        # Carica dataset
        raw_dataset = self.dataset_manager.load_raw_dataset(
            subset_size=subset_size,
            random_state=self.random_state
        )
        
        # Info dataset
        dataset_info = self.dataset_manager.get_dataset_info()
        logger.info(f"Dataset caricato: {dataset_info['statistics']['train']['size']} train, "
                   f"{dataset_info['statistics']['test']['size']} test")
        
        return dataset_info
    
    def prepare_models(self, 
                      models_to_use: Optional[List[str]] = None,
                      force_retrain: bool = False) -> Dict[str, bool]:
        """
        Prepara tutti i modelli (caricamento + fine-tuning se necessario)
        
        Args:
            models_to_use: Lista modelli da preparare (se None, usa tutti)
            force_retrain: Se True, riaddestra anche se già fine-tuned
            
        Returns:
            Dict {model_name: success_status}
        """
        if models_to_use is None:
            models_to_use = self.models_to_evaluate
        
        logger.info(f"SETUP: Preparazione {len(models_to_use)} modelli")
        
        preparation_status = {}
        
        for model_key in models_to_use:
            try:
                logger.info(f"Preparazione {model_key}...")
                
                model_info = ModelConfig.get_model_info(model_key)
                
                # Carica modello base
                model, tokenizer = self.model_manager.load_model_and_tokenizer(model_key)
                
                # Se serve fine-tuning
                if model_info["needs_finetuning"] or force_retrain:
                    # Tokenizza dataset per questo modello
                    raw_dataset = self.dataset_manager.load_raw_dataset()
                    tokenized_dataset = self.dataset_manager.tokenize_dataset(tokenizer)
                    
                    # Fine-tuning
                    finetuned_path = self.model_manager.fine_tune_model(
                        model_key,
                        tokenized_dataset["train"],
                        tokenized_dataset["test"],
                        overwrite_output_dir=force_retrain
                    )
                    
                    logger.info(f"SUCCESS: {model_key} fine-tuned in {finetuned_path}")
                else:
                    logger.info(f"SUCCESS: {model_key} già pronto")
                
                preparation_status[model_key] = True
                
            except Exception as e:
                logger.error(f"FAILED: Errore nella preparazione di {model_key}: {str(e)}")
                preparation_status[model_key] = False
        
        successful_models = sum(preparation_status.values())
        logger.info(f"COMPLETE: {successful_models}/{len(models_to_use)} modelli preparati con successo")
        
        return preparation_status
    
    def evaluate_model_explainer_pair(self, 
                                    model_key: str, 
                                    explainer_name: str,
                                    eval_dataset: Any) -> Dict[str, Any]:
        """
        Valuta una coppia modello-explainer
        
        Args:
            model_key: Nome del modello
            explainer_name: Nome dell'explainer
            eval_dataset: Dataset di valutazione
            
        Returns:
            Risultati della valutazione
        """
        logger.info(f"EVAL: {model_key} + {explainer_name}")
        
        start_time = time.time()
        
        try:
            # Carica modello fine-tuned
            model, tokenizer = self.model_manager.load_finetuned_model(model_key)
            
            # Inizializza explainer manager
            explainer_manager = ExplainerManager(model, tokenizer)
            explainer = explainer_manager.get_explainer(explainer_name)
            
            # Genera spiegazioni per il subset di valutazione
            explanations = []
            texts = []
            true_labels = []
            
            # Converti dataset in pandas per manipolazione più facile
            eval_df = eval_dataset.to_pandas()
            
            # Limita la valutazione per evitare tempi eccessivi
            max_eval_samples = min(self.eval_subset_size, len(eval_df))
            eval_subset = eval_df.sample(n=max_eval_samples, random_state=self.random_state)
            
            logger.info(f"Valutazione su {len(eval_subset)} esempi")
            
            for idx, row in tqdm(eval_subset.iterrows(), total=len(eval_subset), desc=f"Explaining {model_key}+{explainer_name}"):
                # Decodifica il testo dai token IDs
                text = tokenizer.decode(row['input_ids'], skip_special_tokens=True)
                
                try:
                    # Genera spiegazione
                    explanation = explainer.explain(text)
                    explanations.append(explanation)
                    texts.append(text)
                    true_labels.append(int(row['labels']))
                    
                except Exception as e:
                    logger.warning(f"Errore nella spiegazione per esempio {idx}: {str(e)}")
                    continue
            
            if not explanations:
                raise ValueError("Nessuna spiegazione generata con successo")
            
            # Valuta metriche
            metrics_results = self._evaluate_metrics(explanations, texts, true_labels, explainer)
            
            evaluation_time = time.time() - start_time
            
            # Risultato finale
            result = {
                'model': model_key,
                'explainer': explainer_name,
                'num_explanations': len(explanations),
                'evaluation_time': evaluation_time,
                'metrics': metrics_results,
                'model_info': ModelConfig.get_model_info(model_key),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"SUCCESS: {model_key}+{explainer_name} completato in {evaluation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"FAILED: {model_key}+{explainer_name} - {str(e)}")
            return {
                'model': model_key,
                'explainer': explainer_name,
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _evaluate_metrics(self, 
                         explanations: List[Dict[str, Any]], 
                         texts: List[str],
                         true_labels: List[int],
                         explainer) -> Dict[str, Any]:
        """
        Valuta le metriche per un set di spiegazioni
        
        Args:
            explanations: Lista di spiegazioni
            texts: Testi originali
            true_labels: Labels vere
            explainer: Oggetto explainer per robustness
            
        Returns:
            Risultati delle metriche
        """
        metrics_results = {}
        
        # Robustness (richiede funzione explainer)
        try:
            explainer_function = lambda text: explainer.explain(text)
            robustness_result = self.metrics_manager.metrics['robustness'].evaluate(
                explanations[:10],  # Limita per velocità
                texts[:10],
                explainer_function
            )
            metrics_results['robustness'] = robustness_result
        except Exception as e:
            logger.warning(f"Robustness evaluation failed: {str(e)}")
            metrics_results['robustness'] = None
        
        # Contrastivity (richiede labels)
        try:
            contrastivity_result = self.metrics_manager.metrics['contrastivity'].evaluate(
                explanations, true_labels
            )
            metrics_results['contrastivity'] = contrastivity_result
        except Exception as e:
            logger.warning(f"Contrastivity evaluation failed: {str(e)}")
            metrics_results['contrastivity'] = None
        
        return metrics_results
    
    def evaluate_consistency_between_models(self, 
                                          model1_key: str, 
                                          model2_key: str,
                                          explainer_name: str,
                                          eval_dataset: Any) -> Dict[str, Any]:
        """
        Valuta consistency tra due modelli per uno specifico explainer
        
        Args:
            model1_key: Primo modello
            model2_key: Secondo modello  
            explainer_name: Nome explainer
            eval_dataset: Dataset di valutazione
            
        Returns:
            Risultati consistency
        """
        logger.info(f"CONSISTENCY: {model1_key} vs {model2_key} con {explainer_name}")
        
        try:
            # Carica modelli
            model1, tokenizer1 = self.model_manager.load_finetuned_model(model1_key)
            model2, tokenizer2 = self.model_manager.load_finetuned_model(model2_key)
            
            # Inizializza explainer per entrambi
            explainer1 = ExplainerManager(model1, tokenizer1).get_explainer(explainer_name)
            explainer2 = ExplainerManager(model2, tokenizer2).get_explainer(explainer_name)
            
            # Genera spiegazioni per subset
            eval_df = eval_dataset.to_pandas()
            eval_subset = eval_df.sample(n=min(50, len(eval_df)), random_state=self.random_state)
            
            explanations1 = []
            explanations2 = []
            
            for idx, row in eval_subset.iterrows():
                # Decodifica testo (usa tokenizer1 come riferimento)
                text = tokenizer1.decode(row['input_ids'], skip_special_tokens=True)
                
                try:
                    exp1 = explainer1.explain(text)
                    exp2 = explainer2.explain(text)
                    
                    explanations1.append(exp1)
                    explanations2.append(exp2)
                    
                except Exception as e:
                    logger.warning(f"Errore consistency esempio {idx}: {str(e)}")
                    continue
            
            if not explanations1 or not explanations2:
                raise ValueError("Nessuna spiegazione generata per consistency")
            
            # Valuta consistency
            consistency_result = self.metrics_manager.metrics['consistency'].evaluate(
                explanations1, explanations2
            )
            
            return {
                'model1': model1_key,
                'model2': model2_key,
                'explainer': explainer_name,
                'consistency': consistency_result,
                'num_examples': len(explanations1)
            }
            
        except Exception as e:
            logger.error(f"FAILED consistency {model1_key} vs {model2_key}: {str(e)}")
            return {
                'model1': model1_key,
                'model2': model2_key,
                'explainer': explainer_name,
                'error': str(e)
            }
    
    def run_full_evaluation(self, 
                          models_to_use: Optional[List[str]] = None,
                          explainers_to_use: Optional[List[str]] = None,
                          include_consistency: bool = True,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        Esegue valutazione completa
        
        Args:
            models_to_use: Modelli da valutare
            explainers_to_use: Explainer da valutare
            include_consistency: Se includere valutazione consistency
            save_results: Se salvare risultati
            
        Returns:
            Risultati completi
        """
        start_time = time.time()
        
        logger.info("START: Valutazione completa XAI benchmark")
        
        # Setup default
        if models_to_use is None:
            models_to_use = self.models_to_evaluate
        if explainers_to_use is None:
            explainers_to_use = self.explainers_to_evaluate
        
        # 1. Setup dataset
        dataset_info = self.setup_dataset()
        
        # 2. Prepara modelli
        model_status = self.prepare_models(models_to_use)
        successful_models = [k for k, v in model_status.items() if v]
        
        if not successful_models:
            raise RuntimeError("Nessun modello preparato con successo")
        
        # 3. Prepara dataset di valutazione
        # Usa un tokenizer di riferimento per creare il dataset di eval
        ref_model, ref_tokenizer = self.model_manager.load_finetuned_model(successful_models[0])
        raw_dataset = self.dataset_manager.load_raw_dataset()
        tokenized_dataset = self.dataset_manager.tokenize_dataset(ref_tokenizer)
        eval_dataset = self.dataset_manager.create_subset_for_evaluation(
            tokenized_dataset, eval_size=self.eval_subset_size
        )
        
        # 4. Valutazione modello-explainer pairs
        evaluation_results = []
        
        total_combinations = len(successful_models) * len(explainers_to_use)
        logger.info(f"Valutazione {total_combinations} combinazioni")
        
        for model_key in successful_models:
            for explainer_name in explainers_to_use:
                result = self.evaluate_model_explainer_pair(
                    model_key, explainer_name, eval_dataset["test"]
                )
                evaluation_results.append(result)
        
        # 5. Valutazione consistency (opzionale)
        consistency_results = []
        if include_consistency and len(successful_models) >= 2:
            logger.info("EVAL: Consistency tra modelli")
            
            # Confronta tutti i pair di modelli
            for i, model1 in enumerate(successful_models):
                for model2 in successful_models[i+1:]:
                    for explainer_name in explainers_to_use:
                        consistency_result = self.evaluate_consistency_between_models(
                            model1, model2, explainer_name, eval_dataset["test"]
                        )
                        consistency_results.append(consistency_result)
        
        # 6. Compila risultati finali
        total_time = time.time() - start_time
        
        final_results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'total_time': total_time,
                'models_evaluated': successful_models,
                'explainers_evaluated': explainers_to_use,
                'eval_subset_size': self.eval_subset_size,
                'random_state': self.random_state
            },
            'dataset_info': dataset_info,
            'model_preparation': model_status,
            'evaluation_results': evaluation_results,
            'consistency_results': consistency_results,
            'summary': self._create_summary(evaluation_results, consistency_results)
        }
        
        # 7. Salva risultati
        if save_results:
            self._save_results(final_results)
        
        logger.info(f"COMPLETE: Valutazione completata in {total_time:.2f}s")
        return final_results
    
    def _create_summary(self, 
                       evaluation_results: List[Dict[str, Any]], 
                       consistency_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Crea summary dei risultati"""
        
        # Summary delle metriche per modello-explainer
        summary_data = []
        
        for result in evaluation_results:
            if 'error' not in result and result.get('metrics'):
                metrics = result['metrics']
                
                row = {
                    'model': result['model'],
                    'explainer': result['explainer'],
                    'num_explanations': result['num_explanations'],
                    'evaluation_time': result['evaluation_time']
                }
                
                # Estrai metriche principali
                if metrics.get('robustness'):
                    row['robustness_mad'] = metrics['robustness']['mad']
                
                if metrics.get('contrastivity'):
                    row['contrastivity_kl'] = metrics['contrastivity']['kl_divergence']
                
                summary_data.append(row)
        
        # Crea DataFrame per analisi
        summary_df = pd.DataFrame(summary_data)
        
        summary = {
            'total_evaluations': len(evaluation_results),
            'successful_evaluations': len(summary_data),
            'failed_evaluations': len(evaluation_results) - len(summary_data)
        }
        
        if not summary_df.empty:
            # Best performers per metrica
            if 'robustness_mad' in summary_df.columns:
                best_robustness = summary_df.loc[summary_df['robustness_mad'].idxmin()]
                summary['best_robustness'] = {
                    'model': best_robustness['model'],
                    'explainer': best_robustness['explainer'],
                    'score': best_robustness['robustness_mad']
                }
            
            if 'contrastivity_kl' in summary_df.columns:
                best_contrastivity = summary_df.loc[summary_df['contrastivity_kl'].idxmax()]
                summary['best_contrastivity'] = {
                    'model': best_contrastivity['model'],
                    'explainer': best_contrastivity['explainer'],
                    'score': best_contrastivity['contrastivity_kl']
                }
        
        # Summary consistency
        if consistency_results:
            consistency_scores = []
            for result in consistency_results:
                if 'error' not in result and result.get('consistency'):
                    consistency_scores.append(result['consistency']['explanation_correlation'])
            
            if consistency_scores:
                summary['consistency_stats'] = {
                    'mean': np.mean(consistency_scores),
                    'std': np.std(consistency_scores),
                    'min': np.min(consistency_scores),
                    'max': np.max(consistency_scores)
                }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Salva risultati su file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva risultati completi
        results_file = self.output_dir / f"xai_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Salva summary come CSV
        if results['evaluation_results']:
            summary_data = []
            for result in results['evaluation_results']:
                if 'error' not in result and result.get('metrics'):
                    row = {
                        'model': result['model'],
                        'explainer': result['explainer'],
                        'num_explanations': result['num_explanations'],
                        'evaluation_time': result['evaluation_time']
                    }
                    
                    metrics = result['metrics']
                    if metrics.get('robustness'):
                        row['robustness_mad'] = metrics['robustness']['mad']
                    if metrics.get('contrastivity'):
                        row['contrastivity_kl'] = metrics['contrastivity']['kl_divergence']
                    
                    summary_data.append(row)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_file = self.output_dir / f"xai_summary_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
        
        logger.info(f"Risultati salvati in {results_file}")


def test_evaluation_pipeline():
    """Test rapido del pipeline di valutazione"""
    try:
        print("\nTEST: Pipeline di valutazione XAI")
        
        # Inizializza pipeline
        pipeline = XAIEvaluationPipeline(
            output_dir="./test_results",
            eval_subset_size=5  # Molto piccolo per test rapido
        )
        
        print("SUCCESS: Pipeline inizializzato")
        
        # Test setup dataset
        dataset_info = pipeline.setup_dataset(subset_size=20)
        print(f"SUCCESS: Dataset setup - {dataset_info['statistics']['train']['size']} esempi")
        
        # Test con un solo modello piccolo
        test_models = ['tinybert']
        test_explainers = ['lime']  # Solo LIME per test rapido
        
        model_status = pipeline.prepare_models(test_models)
        print(f"SUCCESS: Modelli preparati - {model_status}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Test pipeline fallito: {str(e)}")
        return False


if __name__ == "__main__":
    # Demo
    test_evaluation_pipeline()