# models.py
from __future__ import annotations
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch, inspect, tempfile, os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 10 encoder-only popolari
MODEL_IDS = [
    "huawei-noah/TinyBERT_General_4L_312D",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "siebert/sentiment-roberta-large-english",
    "google/electra-base-discriminator",
    "sentence-transformers/all-mpnet-base-v2",
    "facebook/xlm-roberta-large",
    "microsoft/deberta-v2-xlarge",
    "microsoft/mdeberta-v3-base",
]

# ───────────────────────────────────────── helper
def _reset_classifier_head(model):
    """Re-inizializza la Linear finale quando cambiamo num_labels."""
    for name in ["classifier", "score", "classification_head"]:
        if hasattr(model, name):
            head = getattr(model, name)
            if hasattr(head, "reset_parameters"):
                head.reset_parameters()
            elif hasattr(head, "weight"):
                torch.nn.init.xavier_uniform_(head.weight)
                if head.bias is not None:
                    torch.nn.init.zeros_(head.bias)
            return
    logger.warning("Classification head not found: not reinitialized")

# ───────────────────────────────────────── load
def load_model(
    model_id: str,
    num_labels: int = 2,
    device: str | torch.device = "cpu",
    minimal: bool = False,
):
    """
    • Verifica encoder-only
    • Forza num_labels=2 (rigenera la testa se serve)
    • Restituisce il modello in eval mode sul device indicato
    """
    try:
        logger.info(f"Loading config for {model_id}")
        cfg = AutoConfig.from_pretrained(model_id)

        if getattr(cfg, "is_decoder", False) or getattr(cfg, "is_encoder_decoder", False):
            raise ValueError(f"{model_id} non è encoder-only")

        reinit = cfg.num_labels != num_labels
        cfg.num_labels = num_labels

        torch_dtype = torch.float16 if (minimal and str(device).startswith("cuda")) else None

        logger.info(f"Loading model {model_id} with {num_labels} labels")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            config=cfg,
            ignore_mismatched_sizes=True,
            torch_dtype=torch_dtype,
        )

        if reinit:
            _reset_classifier_head(model)

        model.to(device).eval()
        logger.info(f"Successfully loaded {model_id}")
        return model

    except Exception as e:
        logger.error(f"Error loading {model_id}: {e}")
        raise

# ───────────────────────────────────────── fine-tune rapido
def fine_tune_model(
    model,
    train_ds,
    eval_ds=None,
    *,
    lr: float = 2e-5,
    epochs: int = 2,
    batch: int = 8,
    device: str | torch.device = "cpu",
):
    """
    Mini fine-tuning supervisionato (train_ds già tokenizzato).
    Ritorna il modello fine-tuned in eval mode.
    """
    try:
        # importati solo qui per evitare dipendenza se non usato
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
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        trainer.train()
        model.eval()
        
        # Cleanup
        import shutil
        try:
            shutil.rmtree(tmp_out)
        except:
            pass
            
        return model
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise