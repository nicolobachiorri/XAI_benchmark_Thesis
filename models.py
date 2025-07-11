# models.py
from __future__ import annotations
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import torch
import tempfile
import os
import gc
import logging
from typing import Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelli testati per compatibilità XAI
MODEL_IDS = [
    "huawei-noah/TinyBERT_General_4L_312D",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english", 
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "siebert/sentiment-roberta-large-english",
    "google/electra-base-discriminator",
    # Rimossi temporaneamente per evitare conflitti:
    # "sentence-transformers/all-mpnet-base-v2",  # Sentence transformer, diversa architettura
    # "facebook/xlm-roberta-large",  # Molto grande, problemi memoria
    # "microsoft/deberta-v2-xlarge",  # Architettura diversa
    # "microsoft/mdeberta-v3-base",  # Architettura diversa
]

# Configurazioni note problematiche
PROBLEMATIC_CONFIGS = {
    "sentence-transformers/all-mpnet-base-v2": "sentence-transformer architecture",
    "microsoft/deberta-v2-xlarge": "deberta architecture differences", 
    "microsoft/mdeberta-v3-base": "mdeberta architecture differences"
}

def cleanup_memory():
    """Pulisce la memoria GPU/CPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _reset_classifier_head(model, num_labels: int):
    """Re-inizializza la Linear finale quando cambiamo num_labels."""
    head_names = ["classifier", "score", "classification_head"]
    
    for name in head_names:
        if hasattr(model, name):
            head = getattr(model, name)
            if hasattr(head, "reset_parameters"):
                head.reset_parameters()
            elif hasattr(head, "weight"):
                # Reinizializza con dimensioni corrette
                if head.weight.shape[0] != num_labels:
                    # Ricrea il layer con dimensioni corrette
                    in_features = head.weight.shape[1]
                    new_head = torch.nn.Linear(in_features, num_labels)
                    setattr(model, name, new_head)
                    logger.info(f"Recreated {name} with {num_labels} labels")
                else:
                    torch.nn.init.xavier_uniform_(head.weight)
                    if head.bias is not None:
                        torch.nn.init.zeros_(head.bias)
            return True
    
    logger.warning("Classification head not found or not reset")
    return False

def validate_model_compatibility(model_id: str) -> bool:
    """Verifica se il modello è compatibile con XAI methods"""
    if model_id in PROBLEMATIC_CONFIGS:
        logger.warning(f"Model {model_id} might have issues: {PROBLEMATIC_CONFIGS[model_id]}")
        return False
    return True

def load_model(
    model_id: str,
    num_labels: int = 2,
    device: Union[str, torch.device] = "cpu",
    minimal: bool = False,
    force_cpu: bool = False,
    trust_remote_code: bool = False,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Carica modello e tokenizer con gestione robusta degli errori.
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Cleanup preventivo
        cleanup_memory()
        
        # Validazione compatibilità
        if not validate_model_compatibility(model_id):
            logger.warning(f"Loading potentially incompatible model: {model_id}")
        
        # Carica config
        logger.info(f"Loading config for {model_id}")
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        
        # Verifica encoder-only
        if getattr(cfg, "is_decoder", False) or getattr(cfg, "is_encoder_decoder", False):
            raise ValueError(f"{model_id} non è encoder-only")
        
        # Determina device effettivo
        if force_cpu:
            device = "cpu"
        elif isinstance(device, str) and device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
        
        # Configura dtype
        torch_dtype = None
        if minimal and str(device).startswith("cuda"):
            torch_dtype = torch.float16
        
        # Salva num_labels originale
        original_num_labels = cfg.num_labels
        needs_reset = original_num_labels != num_labels
        cfg.num_labels = num_labels
        
        # Carica modello
        logger.info(f"Loading model {model_id} with {num_labels} labels")
        
        model_kwargs = {
            "config": cfg,
            "ignore_mismatched_sizes": True,
            "trust_remote_code": trust_remote_code,
        }
        
        # Gestione device mapping più robusta
        if torch_dtype and str(device).startswith("cuda"):
            model_kwargs["torch_dtype"] = torch_dtype
            # Non usare device_map="auto" se vogliamo controllare il device
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, **model_kwargs
        )
        
        # Reset classifier head se necessario
        if needs_reset:
            success = _reset_classifier_head(model, num_labels)
            if not success:
                logger.warning(f"Could not reset classifier head for {model_id}")
        
        # Sposta su device
        model = model.to(device)
        model.eval()
        
        # Carica tokenizer
        logger.info(f"Loading tokenizer for {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        
        # Aggiungi pad_token se mancante
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        
        logger.info(f"Successfully loaded {model_id}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading {model_id}: {str(e)}")
        cleanup_memory()
        raise

def fine_tune_model(
    model,
    train_ds,
    eval_ds=None,
    *,
    lr: float = 2e-5,
    epochs: int = 2,
    batch: int = 8,
    device: Union[str, torch.device] = "cpu",
    cleanup_after: bool = True,
):
    """
    Mini fine-tuning supervisionato con gestione memoria migliorata.
    """
    try:
        # Import solo quando necessario
        from transformers import TrainingArguments, Trainer
        
        tmp_out = tempfile.mkdtemp(prefix="ft_")
        
        args = TrainingArguments(
            output_dir=tmp_out,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            fp16=str(device).startswith("cuda"),
            evaluation_strategy="epoch" if eval_ds is not None else "no",
            save_total_limit=1,
            report_to=[],
            logging_steps=50,
            dataloader_pin_memory=False,  # Riduce uso memoria
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        model.eval()
        
        # Cleanup
        if cleanup_after:
            del trainer
            cleanup_memory()
            
        # Rimuovi directory temporanea
        import shutil
        try:
            shutil.rmtree(tmp_out)
        except:
            pass
            
        logger.info("Fine-tuning completed")
        return model
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        cleanup_memory()
        raise

# Utility per batch processing
def process_models_safely(model_ids: list[str], process_fn, **kwargs):
    """
    Processa una lista di modelli in modo sicuro, gestendo errori e memoria.
    """
    results = {}
    
    for model_id in model_ids:
        try:
            logger.info(f"Processing {model_id}")
            result = process_fn(model_id, **kwargs)
            results[model_id] = result
            logger.info(f"Successfully processed {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_id}: {str(e)}")
            results[model_id] = None
            
        finally:
            # Cleanup sempre dopo ogni modello
            cleanup_memory()
    
    return results