"""
XAI Benchmark - Models Module
Gestisce il caricamento, fine-tuning e salvataggio dei modelli transformer encoder-only
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configurazione dei modelli disponibili"""
    
    MODELS = {
        # Architettura BERT
        "tinybert": {
            "model_name": "huawei-noah/TinyBERT_General_4L_312D",
            "architecture": "BERT",
            "parameters": "14M",
            "needs_finetuning": True
        },
        "distilbert": {
            "model_name": "distilbert/distilbert-base-uncased",
            "architecture": "DistilBERT", 
            "parameters": "67M",
            "needs_finetuning": True
        },
        "bert-base": {
            "model_name": "google-bert/bert-base-uncased",
            "architecture": "BERT",
            "parameters": "110M", 
            "needs_finetuning": True
        },
        "bert-large": {
            "model_name": "google-bert/bert-large-uncased",
            "architecture": "BERT",
            "parameters": "335M",
            "needs_finetuning": True
        },
        
        # Architettura RoBERTa (già fine-tuned)
        "roberta-large": {
            "model_name": "siebert/sentiment-roberta-large-english",
            "architecture": "RoBERTa",
            "parameters": "355M",
            "needs_finetuning": False  # Già fine-tuned per sentiment
        },
        
        # Architettura XLM-RoBERTa
        "xlm-roberta": {
            "model_name": "FacebookAI/xlm-roberta-large",
            "architecture": "XLM-RoBERTa",
            "parameters": "561M",
            "needs_finetuning": True
        },
        
        # Architettura DeBERTa
        "deberta-v2-xl": {
            "model_name": "microsoft/deberta-v2-xlarge", 
            "architecture": "DeBERTa-v2",
            "parameters": "900M",
            "needs_finetuning": True
        },
        "mdeberta-v3": {
            "model_name": "microsoft/mdeberta-v3-base",
            "architecture": "mDeBERTa-v3", 
            "parameters": "184M",
            "needs_finetuning": True
        }
    }
    
    @classmethod
    def get_model_names(cls) -> list:
        """Restituisce lista dei nomi dei modelli"""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_key: str) -> Dict[str, Any]:
        """Restituisce informazioni sul modello"""
        if model_key not in cls.MODELS:
            raise ValueError(f"Modello {model_key} non trovato. Disponibili: {cls.get_model_names()}")
        return cls.MODELS[model_key]


class ModelManager:
    """Gestisce caricamento, fine-tuning e salvataggio dei modelli"""
    
    def __init__(self, cache_dir: str = "./model_cache", models_dir: str = "./finetuned_models"):
        self.cache_dir = Path(cache_dir)
        self.models_dir = Path(models_dir)
        
        # Crea directory se non esistono
        self.cache_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando device: {self.device}")
        
        # Cache per modelli caricati
        self._loaded_models = {}
        self._loaded_tokenizers = {}
    
    def load_model_and_tokenizer(self, 
                                model_key: str, 
                                force_reload: bool = False) -> Tuple[Any, Any]:
        """
        Carica modello e tokenizer
        
        Args:
            model_key: Chiave del modello da caricare
            force_reload: Se True, ricarica anche se già in cache
            
        Returns:
            Tuple (model, tokenizer)
        """
        # Check cache
        if not force_reload and model_key in self._loaded_models:
            logger.info(f"Caricamento {model_key} dalla cache")
            return self._loaded_models[model_key], self._loaded_tokenizers[model_key]
        
        model_info = ModelConfig.get_model_info(model_key)
        model_name = model_info["model_name"]
        
        logger.info(f"Caricamento {model_key} ({model_info['architecture']}, {model_info['parameters']})")
        
        try:
            # Carica tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                use_fast=True
            )
            
            # Aggiungi pad_token se mancante (necessario per alcuni modelli)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Carica modello
            if model_info["needs_finetuning"]:
                # Modello base da fine-tunare
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,  # Binary sentiment classification
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            else:
                # Modello già fine-tuned
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            
            # Sposta su device appropriato
            model.to(self.device)
            
            # Cache
            self._loaded_models[model_key] = model
            self._loaded_tokenizers[model_key] = tokenizer
            
            logger.info(f"{model_key} caricato con successo")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Errore nel caricamento di {model_key}: {str(e)}")
            raise
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics per il training"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }
    
    def fine_tune_model(self, 
                       model_key: str,
                       train_dataset,
                       eval_dataset,
                       output_dir: Optional[str] = None,
                       **training_kwargs) -> str:
        """
        Fine-tuna un modello su dataset IMDB
        
        Args:
            model_key: Chiave del modello
            train_dataset: Dataset di training
            eval_dataset: Dataset di valutazione  
            output_dir: Directory di output (opzionale)
            **training_kwargs: Parametri aggiuntivi per TrainingArguments
            
        Returns:
            Path del modello salvato
        """
        model_info = ModelConfig.get_model_info(model_key)
        
        # Check se serve fine-tuning
        if not model_info["needs_finetuning"]:
            logger.info(f"SKIP: {model_key} già fine-tuned, skip fine-tuning")
            return self._get_finetuned_path(model_key)
        
        # Check se già fine-tuned
        finetuned_path = self._get_finetuned_path(model_key)
        if finetuned_path.exists() and not training_kwargs.get('overwrite_output_dir', False):
            logger.info(f"FOUND: {model_key} già fine-tuned in {finetuned_path}")
            return str(finetuned_path)
        
        logger.info(f"START: Avvio fine-tuning per {model_key}")
        
        # Carica modello e tokenizer
        model, tokenizer = self.load_model_and_tokenizer(model_key)
        
        # Setup output directory
        if output_dir is None:
            output_dir = str(finetuned_path)
        
        # Default training arguments
        default_args = {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,  # Conservativo per memoria
            "per_device_eval_batch_size": 16,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "logging_steps": 100,
            "eval_steps": 500,
            "save_steps": 500,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "save_total_limit": 2,
            "seed": 42,
            "data_seed": 42,
            "fp16": self.device.type == "cuda",  # Mixed precision se GPU disponibile
            "dataloader_pin_memory": False,
            "remove_unused_columns": False
        }
        
        # Merge con parametri custom
        default_args.update(training_kwargs)
        training_args = TrainingArguments(**default_args)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Setup trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        try:
            # Train
            logger.info(f"Training {model_key}...")
            trainer.train()
            
            # Salva modello finale
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Salva info del training
            training_info = {
                "model_key": model_key,
                "model_name": model_info["model_name"],
                "architecture": model_info["architecture"],
                "parameters": model_info["parameters"],
                "training_args": training_args.to_dict(),
                "final_metrics": trainer.state.log_history[-1] if trainer.state.log_history else {}
            }
            
            with open(Path(output_dir) / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"COMPLETE: Fine-tuning {model_key} completato: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"ERROR: Errore durante fine-tuning di {model_key}: {str(e)}")
            raise
    
    def load_finetuned_model(self, model_key: str) -> Tuple[Any, Any]:
        """
        Carica un modello già fine-tuned
        
        Args:
            model_key: Chiave del modello
            
        Returns:
            Tuple (model, tokenizer)
        """
        model_info = ModelConfig.get_model_info(model_key)
        
        # Se è già fine-tuned, usa il modello originale
        if not model_info["needs_finetuning"]:
            return self.load_model_and_tokenizer(model_key)
        
        # Altrimenti cerca la versione fine-tuned
        finetuned_path = self._get_finetuned_path(model_key)
        
        if not finetuned_path.exists():
            raise FileNotFoundError(f"Modello fine-tuned non trovato per {model_key}. "
                                  f"Esegui prima il fine-tuning.")
        
        logger.info(f"Caricamento modello fine-tuned {model_key} da {finetuned_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
            model = AutoModelForSequenceClassification.from_pretrained(finetuned_path)
            model.to(self.device)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"ERROR: Errore nel caricamento del modello fine-tuned {model_key}: {str(e)}")
            raise
    
    def _get_finetuned_path(self, model_key: str) -> Path:
        """Restituisce il path del modello fine-tuned"""
        return self.models_dir / f"{model_key}_finetuned"
    
    def get_model_info_summary(self) -> Dict[str, Any]:
        """Restituisce summary delle info sui modelli"""
        summary = {}
        for key, info in ModelConfig.MODELS.items():
            finetuned_path = self._get_finetuned_path(key)
            summary[key] = {
                **info,
                "finetuned_available": finetuned_path.exists(),
                "finetuned_path": str(finetuned_path)
            }
        return summary
    
    def cleanup_cache(self):
        """Pulisce la cache dei modelli caricati"""
        self._loaded_models.clear()
        self._loaded_tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cache pulita")


# Utility functions
def print_models_info():
    """Stampa informazioni sui modelli disponibili"""
    print("\nMODELLI DISPONIBILI:")
    print("=" * 80)
    
    for key, info in ModelConfig.MODELS.items():
        status = "Needs fine-tuning" if info["needs_finetuning"] else "Pre-trained"
        print(f"{key:<15} | {info['architecture']:<12} | {info['parameters']:<6} | {status}")
    
    print("=" * 80)


def test_model_loading():
    """Test rapido del caricamento modelli"""
    manager = ModelManager()
    
    # Test con modello piccolo
    test_model = "tinybert"
    
    try:
        print(f"\nTEST: Caricamento {test_model}")
        model, tokenizer = manager.load_model_and_tokenizer(test_model)
        
        print(f"SUCCESS: Modello caricato: {type(model).__name__}")
        print(f"SUCCESS: Tokenizer caricato: {type(tokenizer).__name__}")
        print(f"SUCCESS: Device: {model.device}")
        print(f"SUCCESS: Vocab size: {tokenizer.vocab_size}")
        
        # Test tokenization
        test_text = "This movie is great!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"SUCCESS: Test tokenization: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Test fallito: {str(e)}")
        return False
    finally:
        manager.cleanup_cache()


if __name__ == "__main__":
    # Demo
    print_models_info()
    test_model_loading()