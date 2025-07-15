"""
main.py – CLI unico per il benchmark XAI
=======================================

Offre tre sub‑comandi semplici:

* **train**        – fine‑tuning di un modello su IMDB (usa models.py)
* **evaluate**     – chiama evaluate.py con gli stessi argomenti
* **explain**      – genera e stampa una spiegazione token‑level per una frase

Esempi rapidi
-------------
```bash
# 1. Allenare distilbert con backbone congelato (solo linear head)
python main.py train --model distilbert --freeze  

# 2. Valutare robustness del modello allenato con IG su 500 esempi
python main.py evaluate --metric robustness --model distilbert --explainer integrated_gradients --sample 500

# 3. Spiegare una frase
python main.py explain --model distilbert --explainer integrated_gradients \
                       --text "I loved this movie, it was fantastic!"
```

Dipendenze: solo quelle già presenti negli altri moduli.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import torch

import dataset
import models
import explainers
import evaluate  # il nostro evaluate.py semplificato
from utils import set_seed

# ==== 1. Seed fisso per riproducibilità globale ====
set_seed(42)

# ==== 2. Helper: carica train/val datasets ====

def _get_train_val_datasets():
    return (
        dataset.IMDBDataset(dataset.train_df),
        dataset.IMDBDataset(dataset.val_df),
    )

# ==== 3. Sub‑comandi ====

def _cmd_train(args):
    train_ds, val_ds = _get_train_val_datasets()
    print(
        f"[TRAIN] modello={args.model}, epochs={args.epochs}, lr={args.lr}, "
        f"freeze_backbone={args.freeze}"
    )
    models.fine_tune(
        model_key=args.model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=args.epochs,
        lr=args.lr,
        freeze_backbone=args.freeze,
    )


def _cmd_evaluate(args, argv_rest: List[str]):
    """Pass‑through a evaluate.py mantenendo la stessa CLI."""
    # Ricostruiamo sys.argv come se fossimo in evaluate.py
    new_argv = ["evaluate.py"] + argv_rest  # prima arg è script name
    sys.argv = new_argv
    evaluate.main()


def _cmd_explain(args):
    model = models.load_model(args.model)
    tokenizer = models.load_tokenizer(args.model)
    explainer = explainers.get_explainer(args.explainer, model, tokenizer)
    attr = explainer(args.text)

    print("\nTokens e importanza:")
    for tok, score in zip(attr.tokens, attr.scores):
        print(f"{tok:<12} {score:+.4f}")

# ==== 4. Parser CLI principale ====

def main():
    parser = argparse.ArgumentParser(description="CLI semplice per XAI Benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = subparsers.add_parser("train", help="Fine‑tune di un modello su IMDB")
    p_train.add_argument("--model", required=True, choices=models.MODELS.keys())
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--freeze", action="store_true", help="Congela il backbone (allenando solo la head)")
    p_train.set_defaults(func=_cmd_train)

    # ---- evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Valuta modello+explainer su una metrica")
    # Non ridefiniamo tutte le opzioni qui; pass‑through a evaluate.py
    #p_eval.add_argument("--", nargs=argparse.REMAINDER, help="Argomenti per evaluate.py (vedi relativo help)")
    p_eval.set_defaults(func=_cmd_evaluate) 

    # ---- explain ----
    p_expl = subparsers.add_parser("explain", help="Genera spiegazione token‑level di una frase")
    p_expl.add_argument("--model", required=True, choices=models.MODELS.keys())
    p_expl.add_argument("--explainer", required=True, choices=explainers.list_explainers())
    p_expl.add_argument("--text", required=True, help="Frase da spiegare")
    p_expl.set_defaults(func=_cmd_explain)


    # --- PARSING --- #
    # restituisce (args riconosciuti, lista degli altri)
    args, rest = parser.parse_known_args()

    # Esegui il comando appropriato
    if args.command == "evaluate":
        _cmd_evaluate(args, rest)      # passiamo la lista extra a evaluate.py
    else:
        args.func(args)

if __name__ == "__main__":
    main()
