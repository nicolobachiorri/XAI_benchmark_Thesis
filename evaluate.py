"""
evaluate.py – Pipeline di valutazione XAI semplificata
=====================================================

Scopo: permettere di valutare rapidamente un modello + un
explainer su una metrica scelta, usando le funzioni già definite in:
    • models.py
    • dataset.py
    • explainers.py
    • metrics.py (robustness, consistency, contrastivity)

Uso CLI (esempi):
    python evaluate.py --model bert-base --explainer integrated_gradients --metric robustness
    python evaluate.py --model bert-base --explainer integrated_gradients --metric contrastivity

Per «consistency» servono due modelli: --model-a, --model-b
    python evaluate.py --model-a tinybert --model-b distilbert \
                      --explainer integrated_gradients --metric consistency

Parametri facili da cambiare in testa al file.
"""

# ==== 1. Librerie ====
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import torch

import models
import dataset  # tuo dataset.py
import explainers
import metrics

# ==== 2. Parametri globali ====
SAMPLE_SIZE = 500  # numero di esempi su cui valutare (None = tutto test)
RANDOM_STATE = 42

# ==== 3. Helper caricamento dati ====

def _load_test_texts_labels(sample_size: int | None = SAMPLE_SIZE) -> Tuple[List[str], List[int]]:
    df = dataset.test_df.sample(n=sample_size, random_state=RANDOM_STATE) if sample_size else dataset.test_df
    return df["text"].tolist(), df["label"].tolist()

# ==== 4. Funzioni di valutazione ====

def _eval_robustness(model_key: str, explainer_name: str, sample_size: int | None):
    model = models.load_model(model_key)
    tokenizer = models.load_tokenizer(model_key)
    explainer = explainers.get_explainer(explainer_name, model, tokenizer)
    texts, _ = _load_test_texts_labels(sample_size)
    score = metrics.evaluate_robustness_over_dataset(model, tokenizer, explainer, texts)
    print(f"Robustness ({model_key}, {explainer_name}): {score:.4f} (↓ meglio)")


def _eval_contrastivity(model_key: str, explainer_name: str, sample_size: int | None):
    model = models.load_model(model_key)
    tokenizer = models.load_tokenizer(model_key)
    explainer = explainers.get_explainer(explainer_name, model, tokenizer)
    texts, labels = _load_test_texts_labels(sample_size)

    pos_attrs, neg_attrs = [], []
    for t, l in zip(texts, labels):
        attr = explainer(t)
        (pos_attrs if l == 1 else neg_attrs).append(attr)

    score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
    print(f"Contrastivity ({model_key}, {explainer_name}): {score:.4f} (↑ meglio)")


def _eval_consistency(model_a: str, model_b: str, explainer_name: str, sample_size: int | None):
    model1 = models.load_model(model_a)
    model2 = models.load_model(model_b)
    tok1 = models.load_tokenizer(model_a)
    tok2 = models.load_tokenizer(model_b)
    expl1 = explainers.get_explainer(explainer_name, model1, tok1)
    expl2 = explainers.get_explainer(explainer_name, model2, tok2)

    texts, _ = _load_test_texts_labels(sample_size)
    score = metrics.evaluate_consistency_over_dataset(model1, model2, tok1, tok2, expl1, expl2, texts)
    print(f"Consistency ({model_a} vs {model_b}, {explainer_name}): {score:.4f} (↑ meglio)")

# ==== 5. CLI principale ====

def main():
    parser = argparse.ArgumentParser(description="Valutazione XAI semplice")

    parser.add_argument("--metric", required=True, choices=["robustness", "contrastivity", "consistency"], help="Nome metrica")
    parser.add_argument("--explainer", required=True, help="Nome explainer (vedi explainers.list_explainers())")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Modello unico (per robustness/contrastivity)")
    group.add_argument("--model-a", help="Modello A (per consistency)")

    parser.add_argument("--model-b", help="Modello B (solo per consistency)")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE, help="Numero di esempi da valutare (None=tutti)")

    args = parser.parse_args()

    if args.metric == "consistency":
        if not (args.model_a and args.model_b):
            parser.error("--metric consistency richiede --model-a e --model-b")
        _eval_consistency(args.model_a, args.model_b, args.explainer, args.sample)
    else:
        if not args.model:
            parser.error("--metric robustness/contrastivity richiede --model")
        fn = _eval_robustness if args.metric == "robustness" else _eval_contrastivity
        fn(args.model, args.explainer, args.sample)


if __name__ == "__main__":
    main()
