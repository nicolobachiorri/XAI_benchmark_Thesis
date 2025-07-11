# dataset.py - versione robusta
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from utils import set_seed
import logging
import torch
from typing import Tuple, Optional, Union
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_imdb(
    tokenizer_name: str, 
    max_len: int = 256,
    cache_dir: Optional[str] = None,
    subset_size: Optional[int] = None,
    force_download: bool = False
) -> Tuple[Dataset, Dataset]:
    """
    Carica e tokenizza il dataset IMDB con gestione robusta degli errori.
    
    Args:
        tokenizer_name: Nome del tokenizer da usare
        max_len: Lunghezza massima delle sequenze
        cache_dir: Directory per cache (opzionale)
        subset_size: Numero di esempi per split (per testing rapido)
        force_download: Forza download del dataset
    
    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, test_dataset)
    """
    try:
        set_seed()
        
        # Carica tokenizer con gestione errori
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        try:
            tok = AutoTokenizer.from_pretrained(
                tokenizer_name, 
                use_fast=True,
                trust_remote_code=False,  # Sicurezza
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load fast tokenizer, trying slow: {e}")
            tok = AutoTokenizer.from_pretrained(
                tokenizer_name, 
                use_fast=False,
                trust_remote_code=False,
                cache_dir=cache_dir
            )
        
        # Aggiungi pad_token se mancante
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif tok.unk_token is not None:
                tok.pad_token = tok.unk_token
            else:
                # Fallback per tokenizer problematici
                tok.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Set pad_token to: {tok.pad_token}")
        
        # Carica dataset
        logger.info("Loading IMDB dataset...")
        dataset_kwargs = {
            "cache_dir": cache_dir,
        }
        
        if force_download:
            dataset_kwargs["download_mode"] = "force_redownload"
        
        try:
            raw = load_dataset("imdb", **dataset_kwargs)
        except Exception as e:
            logger.error(f"Error loading IMDB dataset: {e}")
            # Fallback con cache disabilitata
            logger.info("Trying without cache...")
            raw = load_dataset("imdb")
        
        # Prendi subset se richiesto (per testing rapido)
        if subset_size is not None:
            logger.info(f"Taking subset of {subset_size} examples per split")
            raw["train"] = raw["train"].select(range(min(subset_size, len(raw["train"]))))
            raw["test"] = raw["test"].select(range(min(subset_size, len(raw["test"]))))
        
        # Funzione di tokenizzazione robusta
        def tokenize_function(batch):
            try:
                # Gestisce casi dove 'text' potrebbe essere None o vuoto
                texts = []
                for text in batch["text"]:
                    if text is None or text == "":
                        texts.append("[EMPTY]")  # Placeholder per testi vuoti
                    else:
                        texts.append(str(text))  # Assicura che sia stringa
                
                result = tok(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors=None,  # Lascia come liste per ora
                    add_special_tokens=True,
                )
                
                # Verifica che tutto sia andato bene
                if not all(len(ids) == max_len for ids in result["input_ids"]):
                    logger.warning("Some sequences have wrong length after tokenization")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in tokenization: {e}")
                # Fallback con tokenizzazione base
                return {
                    "input_ids": [[tok.unk_token_id] * max_len] * len(batch["text"]),
                    "attention_mask": [[1] * max_len] * len(batch["text"]),
                }
        
        # Applica tokenizzazione
        logger.info("Tokenizing dataset...")
        try:
            tokenized = raw.map(
                tokenize_function,
                batched=True,
                batch_size=1000,  # Batch size ragionevole
                num_proc=1,       # Evita problemi multiprocessing
                remove_columns=[],  # Non rimuovere colonne qui
                desc="Tokenizing"
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            # Fallback senza batching
            logger.info("Trying without batching...")
            tokenized = raw.map(
                tokenize_function,
                batched=True,
                batch_size=100,
                num_proc=1,
                desc="Tokenizing (fallback)"
            )
        
        # Rinomina colonna label
        tokenized = tokenized.rename_column("label", "labels")
        
        # Aggiungi indici
        for split in tokenized.keys():
            tokenized[split] = tokenized[split].add_column(
                "idx", 
                list(range(len(tokenized[split])))
            )
        
        # Configura formato PyTorch con gestione errori
        logger.info("Setting PyTorch format...")
        for split in tokenized.keys():
            try:
                tokenized[split].set_format(
                    type="torch",
                    columns=["input_ids", "attention_mask", "labels"],
                    output_all_columns=True,
                )
            except Exception as e:
                logger.error(f"Error setting format for {split}: {e}")
                # Fallback: converti manualmente
                tokenized[split] = convert_to_torch_manually(tokenized[split])
        
        train_dataset = tokenized["train"]
        test_dataset = tokenized["test"]
        
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  Train size: {len(train_dataset)}")
        logger.info(f"  Test size: {len(test_dataset)}")
        logger.info(f"  Max length: {max_len}")
        
        return train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Critical error in get_imdb: {e}")
        raise

def convert_to_torch_manually(dataset):
    """Fallback per conversione manuale a formato PyTorch"""
    def convert_batch(batch):
        # Converte manualmente le colonne necessarie
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch:
                batch[key] = torch.tensor(batch[key])
        return batch
    
    return dataset.map(convert_batch, batched=True)

def get_imdb_small(tokenizer_name: str, max_len: int = 256) -> Tuple[Dataset, Dataset]:
    """Versione ridotta per testing rapido"""
    return get_imdb(
        tokenizer_name=tokenizer_name,
        max_len=max_len,
        subset_size=1000  # Solo 1000 esempi per split
    )

def validate_dataset(dataset: Dataset) -> bool:
    """Valida che il dataset sia stato caricato correttamente"""
    try:
        # Controlla che abbia le colonne necessarie
        required_columns = ["input_ids", "attention_mask", "labels", "text", "idx"]
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        
        # Controlla un campione
        sample = dataset[0]
        
        # Verifica dimensioni
        if len(sample["input_ids"]) != len(sample["attention_mask"]):
            logger.error("Mismatch between input_ids and attention_mask length")
            return False
        
        # Verifica tipi
        if not isinstance(sample["labels"], (int, torch.Tensor)):
            logger.error(f"Labels should be int or tensor, got {type(sample['labels'])}")
            return False
        
        logger.info("Dataset validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

# Utility per debugging
def print_dataset_info(dataset: Dataset, name: str = "Dataset"):
    """Stampa informazioni dettagliate sul dataset"""
    logger.info(f"\n{name} Info:")
    logger.info(f"  Size: {len(dataset)}")
    logger.info(f"  Columns: {dataset.column_names}")
    logger.info(f"  Features: {dataset.features}")
    
    # Mostra un esempio
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"  Sample keys: {list(sample.keys())}")
        logger.info(f"  Input IDs shape: {len(sample['input_ids']) if 'input_ids' in sample else 'N/A'}")
        logger.info(f"  Text preview: {sample.get('text', 'N/A')[:100]}...")

# Test rapido
if __name__ == "__main__":
    # Test con un tokenizer semplice
    try:
        train, test = get_imdb_small("distilbert-base-uncased", max_len=128)
        print_dataset_info(train, "Train")
        print_dataset_info(test, "Test")
        
        if validate_dataset(train) and validate_dataset(test):
            logger.info("Dataset test passed!")
        else:
            logger.error("Dataset test failed!")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")