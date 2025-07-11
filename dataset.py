# dataset.py  – versione corretta
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import set_seed

def get_imdb(tokenizer_name: str, max_len: int = 256):
    set_seed()

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    raw = load_dataset("imdb")

    def tok_fn(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    data = raw.map(tok_fn, batched=True)
    data = data.rename_column("label", "labels")

    # aggiunge indice univoco + format torch conservando 'text'
    for split in data.keys():
        data[split] = data[split].add_column("idx", list(range(len(data[split]))))
        data[split].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True,   # <-- mantiene 'text'
        )

    return data["train"], data["test"]
