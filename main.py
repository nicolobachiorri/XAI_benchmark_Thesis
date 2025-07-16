"""
main.py – CLI semplice per il benchmark XAI
==========================================

Due comandi principali:
* **evaluate** – valuta modello + explainer su una metrica
* **explain**  – spiega una singola frase

Esempi:
```bash
# Valutare robustness
python main.py evaluate --metric robustness --model distilbert --explainer grad_input

# Spiegare una frase
python main.py explain --model bert-base --explainer lime --text "Great movie!"
```
"""

import argparse
import sys

import models
import dataset
import explainers
import evaluate
from utils import set_seed, Timer

# Seed fisso
set_seed(42)

def cmd_explain(args):
    """Comando explain: spiega una frase."""
    print(f"Spiegazione con {args.model} + {args.explainer}")
    print(f"Testo: '{args.text}'")
    
    # Carica modello
    model = models.load_model(args.model)
    tokenizer = models.load_tokenizer(args.model)
    explainer = explainers.get_explainer(args.explainer, model, tokenizer)
    
    # Genera spiegazione
    with Timer("Spiegazione"):
        attr = explainer(args.text)
    
    # Mostra risultati
    print("\nImportanza token:")
    for token, score in zip(attr.tokens, attr.scores):
        print(f"{token:>15} {score:+.3f}")

def main():
    parser = argparse.ArgumentParser(description="XAI Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Valuta modello+explainer")
    p_eval.add_argument("--metric", required=True, 
                       choices=["robustness", "contrastivity", "consistency"])
    p_eval.add_argument("--explainer", required=True, 
                       choices=explainers.list_explainers())
    p_eval.add_argument("--model", choices=models.MODELS.keys())
    p_eval.add_argument("--model-a", choices=models.MODELS.keys())
    p_eval.add_argument("--model-b", choices=models.MODELS.keys())
    p_eval.add_argument("--sample", type=int, default=500)

    # ---- Explain ----
    p_expl = subparsers.add_parser("explain", help="Spiega una frase")
    p_expl.add_argument("--model", required=True, choices=models.MODELS.keys())
    p_expl.add_argument("--explainer", required=True, choices=explainers.list_explainers())
    p_expl.add_argument("--text", required=True)

    args = parser.parse_args()

    # Esegui comando
    if args.command == "explain":
        cmd_explain(args)
    elif args.command == "evaluate":
        # Passa tutto a evaluate.py
        eval_args = []
        if args.metric:
            eval_args.extend(["--metric", args.metric])
        if args.explainer:
            eval_args.extend(["--explainer", args.explainer])
        if args.model:
            eval_args.extend(["--model", args.model])
        if args.model_a:
            eval_args.extend(["--model-a", args.model_a])
        if args.model_b:
            eval_args.extend(["--model-b", args.model_b])
        if args.sample:
            eval_args.extend(["--sample", str(args.sample)])
        
        # Chiama evaluate.py
        sys.argv = ["evaluate.py"] + eval_args
        evaluate.main()

if __name__ == "__main__":
    main()