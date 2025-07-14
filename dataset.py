"""
XAI Benchmark - Dataset Module
Gestisce il caricamento, preprocessing e tokenization del dataset IMDB per sentiment analysis
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IMDBDatasetManager:
    """Gestisce il dataset IMDB per sentiment analysis binaria"""
    
    def __init__(self, cache_dir: str = "./dataset_cache", max_length: int = 512):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_length = max_length
        self.num_labels = 2  # Binary sentiment: 0=negative, 1=positive
        
        # Dataset raw
        self._raw_dataset = None
        self._tokenized_datasets = {}  # Cache per tokenized datasets
        
        # Statistiche
        self.dataset_stats = {}
        
    def load_raw_dataset(self, 
                        train_file: str = "Train.csv",
                        test_file: str = "Test.csv",
                        data_dir: str = ".",
                        subset_size: Optional[int] = None,
                        random_state: int = 42) -> DatasetDict:
        """
        Carica il dataset IMDB dai file CSV locali
        
        Args:
            train_file: Nome del file CSV di training
            test_file: Nome del file CSV di test
            data_dir: Directory contenente i file CSV
            subset_size: Se specificato, usa solo un subset per testing
            random_state: Seed per riproducibilità
            
        Returns:
            DatasetDict con train/test splits
        """
        if self._raw_dataset is not None and subset_size is None:
            logger.info("Dataset IMDB già caricato dalla cache")
            return self._raw_dataset
            
        logger.info("Caricamento dataset IMDB da file CSV locali")
        
        try:
            # Percorsi dei file
            data_path = Path(data_dir)
            train_path = data_path / train_file
            test_path = data_path / test_file
            
            # Verifica esistenza file
            if not train_path.exists():
                raise FileNotFoundError(f"File di training non trovato: {train_path}")
            if not test_path.exists():
                raise FileNotFoundError(f"File di test non trovato: {test_path}")
            
            logger.info(f"Caricamento file di training: {train_path}")
            logger.info(f"Caricamento file di test: {test_path}")
            
            # Carica i CSV
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Training set: {len(train_df)} righe, colonne: {list(train_df.columns)}")
            logger.info(f"Test set: {len(test_df)} righe, colonne: {list(test_df.columns)}")
            
            # Standardizza i nomi delle colonne
            train_df = self._standardize_columns(train_df)
            test_df = self._standardize_columns(test_df)
            
            # Crea il DatasetDict
            train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
            test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
            
            dataset = DatasetDict({
                'train': train_dataset,
                'test': test_dataset
            })
            
            # Se richiesto subset per testing rapido
            if subset_size is not None:
                logger.info(f"Creazione subset di {subset_size} esempi per testing")
                dataset = self._create_subset(dataset, subset_size, random_state)
            
            # Verifica e pulisci i dati
            dataset = self._clean_dataset(dataset)
            
            # Calcola statistiche
            self._compute_dataset_stats(dataset)
            
            self._raw_dataset = dataset
            logger.info("Dataset IMDB caricato con successo dai file CSV locali")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del dataset IMDB: {str(e)}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizza i nomi delle colonne del dataset
        
        Args:
            df: DataFrame con colonne originali
            
        Returns:
            DataFrame con colonne standardizzate
        """
        # Mappa nomi colonne comuni a standard
        column_mapping = {
            'review': 'text',
            'Review': 'text',
            'sentiment': 'label',
            'Sentiment': 'label',
            'polarity': 'label',
            'Polarity': 'label'
        }
        
        # Rinomina colonne se necessario
        df = df.rename(columns=column_mapping)
        
        # Verifica che abbiamo le colonne necessarie
        if 'text' not in df.columns:
            # Prova a trovare la colonna del testo
            text_candidates = [col for col in df.columns 
                             if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'content'])]
            if text_candidates:
                df = df.rename(columns={text_candidates[0]: 'text'})
            else:
                raise ValueError(f"Impossibile trovare colonna del testo. Colonne disponibili: {list(df.columns)}")
        
        if 'label' not in df.columns:
            # Prova a trovare la colonna delle label
            label_candidates = [col for col in df.columns 
                              if any(keyword in col.lower() for keyword in ['label', 'sentiment', 'target', 'class'])]
            if label_candidates:
                df = df.rename(columns={label_candidates[0]: 'label'})
            else:
                raise ValueError(f"Impossibile trovare colonna delle label. Colonne disponibili: {list(df.columns)}")
        
        # Converti label testuali in numeriche se necessario
        if df['label'].dtype == 'object':
            logger.info("Conversione label testuali in numeriche")
            # Mapping più robusto per diversi formati
            df['label'] = df['label'].astype(str).str.lower().str.strip()
            
            label_mapping = {
                'positive': 1, 'negative': 0,
                'pos': 1, 'neg': 0,
                '1': 1, '0': 0,
                'true': 1, 'false': 0,
                'good': 1, 'bad': 0
            }
            
            df['label'] = df['label'].map(label_mapping)
            
            # Rimuovi righe con label non riconosciute
            before_cleaning = len(df)
            df = df.dropna(subset=['label'])
            after_cleaning = len(df)
            
            if before_cleaning != after_cleaning:
                logger.warning(f"Rimosse {before_cleaning - after_cleaning} righe con label non riconosciute")
            
            df['label'] = df['label'].astype(int)
        
        # Converti label numeriche se necessario
        elif df['label'].dtype in ['float64', 'int64']:
            df['label'] = df['label'].astype(int)
            # Assicurati che le label siano 0 e 1
            unique_labels = df['label'].unique()
            if not all(label in [0, 1] for label in unique_labels):
                logger.warning(f"Label non standard trovate: {unique_labels}")
                # Se hai label come -1, 1 o 1, 2, etc., rimappa
                if set(unique_labels) == {-1, 1}:
                    df['label'] = df['label'].map({-1: 0, 1: 1})
                elif set(unique_labels) == {1, 2}:
                    df['label'] = df['label'].map({1: 0, 2: 1})
        
        logger.info(f"Colonne standardizzate: {list(df.columns)}")
        logger.info(f"Distribuzione label: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def _create_subset(self, dataset: DatasetDict, subset_size: int, random_state: int) -> DatasetDict:
        """
        Crea un subset bilanciato del dataset
        
        Args:
            dataset: Dataset completo
            subset_size: Numero totale di esempi nel subset
            random_state: Seed per riproducibilità
            
        Returns:
            DatasetDict con subset
        """
        train_size = int(subset_size * 0.8)  # 80% per train
        test_size = subset_size - train_size  # 20% per test
        
        train_subset = dataset["train"].shuffle(seed=random_state).select(range(min(train_size, len(dataset["train"]))))
        test_subset = dataset["test"].shuffle(seed=random_state).select(range(min(test_size, len(dataset["test"]))))
        
        return DatasetDict({
            "train": train_subset,
            "test": test_subset
        })
    
    def _clean_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Pulisce e valida il dataset
        
        Args:
            dataset: Dataset raw da pulire
            
        Returns:
            Dataset pulito
        """
        logger.info("Pulizia e validazione dataset")
        
        def clean_split(split_data):
            # Converti in pandas per manipolazione più facile
            df = split_data.to_pandas()
            
            # Rimuovi righe vuote o malformate
            initial_size = len(df)
            df = df.dropna(subset=['text', 'label'])
            df = df[df['text'].str.len() > 0]  # Rimuovi testi vuoti
            
            # Verifica range delle label (0=negative, 1=positive)
            df = df[df['label'].isin([0, 1])]
            
            # Rimuovi duplicati
            df = df.drop_duplicates(subset=['text'])
            
            final_size = len(df)
            if initial_size != final_size:
                logger.info(f"Rimossi {initial_size - final_size} esempi malformati/duplicati")
            
            # Converte di nuovo in Dataset
            return Dataset.from_pandas(df, preserve_index=False)
        
        # Applica pulizia a tutti gli split
        cleaned_dataset = DatasetDict()
        for split_name, split_data in dataset.items():
            cleaned_dataset[split_name] = clean_split(split_data)
            
        return cleaned_dataset
    
    def _compute_dataset_stats(self, dataset: DatasetDict):
        """Calcola statistiche del dataset"""
        logger.info("Calcolo statistiche dataset")
        
        stats = {}
        
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            
            # Statistiche di base
            stats[split_name] = {
                "size": len(df),
                "label_distribution": df['label'].value_counts().to_dict(),
                "text_length_stats": {
                    "mean": df['text'].str.len().mean(),
                    "median": df['text'].str.len().median(),
                    "min": df['text'].str.len().min(),
                    "max": df['text'].str.len().max(),
                    "std": df['text'].str.len().std()
                }
            }
            
            # Calcola percentuale di balance
            label_counts = df['label'].value_counts()
            total = len(df)
            stats[split_name]["label_balance"] = {
                "negative_pct": (label_counts.get(0, 0) / total) * 100,
                "positive_pct": (label_counts.get(1, 0) / total) * 100
            }
        
        self.dataset_stats = stats
        
        # Log delle statistiche principali
        for split_name, split_stats in stats.items():
            logger.info(f"{split_name.upper()}: {split_stats['size']} esempi, "
                       f"Balance: {split_stats['label_balance']['negative_pct']:.1f}% neg / "
                       f"{split_stats['label_balance']['positive_pct']:.1f}% pos")
    
    def tokenize_dataset(self, 
                        tokenizer: AutoTokenizer,
                        dataset: Optional[DatasetDict] = None) -> DatasetDict:
        """
        Tokenizza il dataset con un tokenizer specifico
        
        Args:
            tokenizer: Tokenizer da utilizzare
            dataset: Dataset da tokenizzare (se None, usa quello caricato)
            
        Returns:
            Dataset tokenizzato
        """
        if dataset is None:
            if self._raw_dataset is None:
                raise ValueError("Nessun dataset caricato. Chiama prima load_raw_dataset()")
            dataset = self._raw_dataset
        
        # Check cache
        tokenizer_name = getattr(tokenizer, 'name_or_path', 'unknown')
        cache_key = f"{tokenizer_name}_{self.max_length}"
        
        if cache_key in self._tokenized_datasets:
            logger.info(f"Dataset tokenizzato trovato in cache per {tokenizer_name}")
            return self._tokenized_datasets[cache_key]
        
        logger.info(f"Tokenizzazione dataset con {tokenizer_name}")
        
        def tokenize_function(examples):
            """Funzione di tokenizzazione"""
            # Tokenizza i testi
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # Padding dinamico durante training
                max_length=self.max_length,
                return_attention_mask=True,
                return_token_type_ids=False  # Non necessario per sentiment analysis
            )
            
            # Assicurati che le label siano nel formato corretto
            tokenized['labels'] = examples['label']
            
            return tokenized
        
        try:
            # Applica tokenizzazione a tutti gli split
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names,  # Rimuovi colonne originali
                desc="Tokenizing dataset"
            )
            
            # Imposta formato per PyTorch
            tokenized_dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"]
            )
            
            # Cache del risultato
            self._tokenized_datasets[cache_key] = tokenized_dataset
            
            logger.info("Tokenizzazione completata")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Errore durante tokenizzazione: {str(e)}")
            raise
    
    def create_subset_for_evaluation(self, 
                                   tokenized_dataset: DatasetDict,
                                   eval_size: int = 100,
                                   random_state: int = 42) -> DatasetDict:
        """
        Crea un subset per valutazione rapida dei metodi XAI
        
        Args:
            tokenized_dataset: Dataset tokenizzato
            eval_size: Numero di esempi per il subset
            random_state: Seed per riproducibilità
            
        Returns:
            Subset del dataset per valutazione XAI
        """
        logger.info(f"Creazione subset di valutazione ({eval_size} esempi)")
        
        subset_dataset = DatasetDict()
        
        for split_name, split_data in tokenized_dataset.items():
            if len(split_data) >= eval_size:
                # Seleziona subset bilanciato
                df = split_data.to_pandas()
                
                # Stratified sampling per mantenere balance delle classi
                subset_df = df.groupby('labels', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), eval_size // 2), random_state=random_state)
                ).reset_index(drop=True)
                
                # Mescola il subset finale
                subset_df = subset_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
                
                # Converte di nuovo in Dataset
                subset_dataset[split_name] = Dataset.from_pandas(subset_df, preserve_index=False)
                subset_dataset[split_name].set_format(
                    type="torch",
                    columns=["input_ids", "attention_mask", "labels"]
                )
            else:
                # Se il split è più piccolo del subset richiesto, usa tutto
                subset_dataset[split_name] = split_data
        
        return subset_dataset
    
    def get_human_annotations_sample(self, 
                                   tokenized_dataset: DatasetDict,
                                   sample_size: int = 50,
                                   random_state: int = 42) -> List[Dict[str, Any]]:
        """
        Crea un campione per annotazioni umane (per metrica Human Agreement)
        
        Args:
            tokenized_dataset: Dataset tokenizzato
            sample_size: Numero di esempi da annotare
            random_state: Seed per riproducibilità
            
        Returns:
            Lista di esempi per annotazione umana
        """
        logger.info(f"Creazione campione per annotazioni umane ({sample_size} esempi)")
        
        # Usa il test set per le annotazioni
        test_data = tokenized_dataset["test"].to_pandas()
        
        # Sample stratificato
        sample_df = test_data.groupby('labels', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 2), random_state=random_state)
        ).reset_index(drop=True)
        
        # Prepara dati per annotazione
        annotation_samples = []
        for idx, row in sample_df.iterrows():
            # Decodifica il testo dal tokenizer (richiede tokenizer)
            sample = {
                "id": idx,
                "input_ids": row["input_ids"].tolist(),
                "attention_mask": row["attention_mask"].tolist(),
                "true_label": int(row["labels"]),
                "true_label_text": "positive" if row["labels"] == 1 else "negative"
            }
            annotation_samples.append(sample)
        
        return annotation_samples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Restituisce informazioni complete sul dataset"""
        info = {
            "dataset_name": "IMDB Movie Reviews (Local CSV)",
            "task": "Binary Sentiment Classification",
            "num_labels": self.num_labels,
            "label_mapping": {0: "negative", 1: "positive"},
            "max_length": self.max_length,
            "cache_dir": str(self.cache_dir),
            "statistics": self.dataset_stats
        }
        
        return info
    
    def print_dataset_info(self):
        """Stampa informazioni sul dataset in formato leggibile"""
        info = self.get_dataset_info()
        
        print("\nDATASET INFORMATION:")
        print("=" * 60)
        print(f"Dataset: {info['dataset_name']}")
        print(f"Task: {info['task']}")
        print(f"Labels: {info['num_labels']} ({info['label_mapping']})")
        print(f"Max Length: {info['max_length']} tokens")
        print()
        
        if info['statistics']:
            print("STATISTICS:")
            for split_name, stats in info['statistics'].items():
                print(f"\n{split_name.upper()}:")
                print(f"  Size: {stats['size']:,} examples")
                print(f"  Balance: {stats['label_balance']['negative_pct']:.1f}% negative, "
                      f"{stats['label_balance']['positive_pct']:.1f}% positive")
                print(f"  Text Length: {stats['text_length_stats']['mean']:.0f} ± "
                      f"{stats['text_length_stats']['std']:.0f} chars "
                      f"(min={stats['text_length_stats']['min']}, "
                      f"max={stats['text_length_stats']['max']})")
        
        print("=" * 60)


def test_dataset_loading():
    """Test rapido del caricamento dataset"""
    try:
        print("\nTEST: Caricamento dataset IMDB da CSV locali")
        
        # Inizializza manager
        dataset_manager = IMDBDatasetManager()
        
        # Carica dataset (subset per test rapido)
        raw_dataset = dataset_manager.load_raw_dataset(
            train_file="Train.csv",
            test_file="Test.csv",
            data_dir=".",
            subset_size=200  # Subset piccolo per test
        )
        
        print("SUCCESS: Dataset raw caricato")
        dataset_manager.print_dataset_info()
        
        # Test tokenizzazione (richiede un tokenizer)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        tokenized_dataset = dataset_manager.tokenize_dataset(tokenizer)
        print("SUCCESS: Dataset tokenizzato")
        
        # Test subset per XAI evaluation
        eval_subset = dataset_manager.create_subset_for_evaluation(tokenized_dataset, eval_size=20)
        print(f"SUCCESS: Subset XAI creato - {len(eval_subset['test'])} esempi")
        
        # Test sample per annotazioni umane
        human_samples = dataset_manager.get_human_annotations_sample(tokenized_dataset, sample_size=10)
        print(f"SUCCESS: Campione annotazioni umane - {len(human_samples)} esempi")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Test fallito: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Demo
    test_dataset_loading()