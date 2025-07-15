"""
report.py – Genera tabelle aggregate (Robustness / Consistency / Contrastivity)
==============================================================================

Questo script produce, in formato *Markdown* (stampato in console) o *CSV*, le
stesse tabelle quantitative del paper, calcolando le metriche sui modelli e
explainer definiti nel progetto.

Esempio d’uso (tutte le metriche, sample=500):
    python report.py --sample 500 --out markdown

Solo Robustness in CSV:
    python report.py --metric robustness --out csv --sample 300 > robustness.csv

Requisiti: aver già fine‑tunato (o scaricato) i modelli indicati in MODELS.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import models
import dataset
import explainers
import metrics
from utils import set_seed

# ---- Configurazioni facili da cambiare -----------------------------------
METRICS = ["robustness", "contrastivity", "consistency"]
EXPLAINERS = [
    "lime",
    "shap",
    "lrp",  # alias per grad_input (se preferisci cambiare nome)
    "integrated_gradients",  # InputXGradient nel paper → grad_input; qui IG per esempio
    "grad_input",            # gradient × input – usa tu il mapping che vuoi
    "attention_rollout",     # AMV nel paper (approx.)
]

# mappa alias per stampare nome riga paper‑style
ROW_NAMES = {
    "lrp": "LRP",
    "grad_input": "InputXGradient",
    "attention_rollout": "AMV",
    "integrated_gradients": "IntegratedGradients",
}

# -------------------------------------------------------------------------

set_seed(42)

# Pre‑carica testo e label per il test‑set (una sola volta)
TEST_TEXTS, TEST_LABELS = dataset.test_df["text"].tolist(), dataset.test_df["label"].tolist()

# -------------------------------------------------------------------------

def _get_subset(texts: List[str], labels: List[int], sample: int | None):
    if sample and sample < len(texts):
        return texts[:sample], labels[:sample]
    return texts, labels

# ------------------- funzioni metriche wrapper ---------------------------

def _score_robustness(model_key: str, explainer_name: str, sample: int | None):
    model = models.load_model(model_key)
    tok = models.load_tokenizer(model_key)
    expl = explainers.get_explainer(explainer_name, model, tok)
    texts, _ = _get_subset(TEST_TEXTS, TEST_LABELS, sample)
    return metrics.evaluate_robustness_over_dataset(model, tok, expl, texts)


def _score_contrastivity(model_key: str, explainer_name: str, sample: int | None):
    model = models.load_model(model_key)
    tok = models.load_tokenizer(model_key)
    expl = explainers.get_explainer(explainer_name, model, tok)
    texts, labels = _get_subset(TEST_TEXTS, TEST_LABELS, sample)
    pos, neg = [], []
    for t, l in zip(texts, labels):
        (pos if l == 1 else neg).append(expl(t))
    return metrics.compute_contrastivity(pos, neg)


def _score_consistency(model_key: str, explainer_name: str, sample: int | None):
    """Richiede due checkpoint dello *stesso* modello con seed diversi.
    Per semplicità carichiamo lo stesso modello due volte (seed diversi nel
    fine‑tuning) se avete salvato come "<model_key>_seed1" / "<model_key>_seed2".
    Qui si assume che esistano tali checkpoint; altrimenti la funzione ritorna
    NaN.
    """
    key_a = f"{model_key}_seed1"
    key_b = f"{model_key}_seed2"
    if key_a not in models.MODELS or key_b not in models.MODELS:
        return float("nan")

    m1 = models.load_model(key_a)
    m2 = models.load_model(key_b)
    t1 = models.load_tokenizer(key_a)
    t2 = models.load_tokenizer(key_b)
    e1 = explainers.get_explainer(explainer_name, m1, t1)
    e2 = explainers.get_explainer(explainer_name, m2, t2)
    texts, _ = _get_subset(TEST_TEXTS, TEST_LABELS, sample)
    return metrics.evaluate_consistency_over_dataset(m1, m2, t1, t2, e1, e2, texts)


_METRIC_FUNC = {
    "robustness": _score_robustness,
    "contrastivity": _score_contrastivity,
    "consistency": _score_consistency,
}

# -------------------------------------------------------------------------

def build_table(metric: str, sample: int | None) -> pd.DataFrame:
    results: Dict[str, Dict[str, float]] = defaultdict(dict)
    for expl in EXPLAINERS:
        for model_key in models.MODELS.keys():
            score = _METRIC_FUNC[metric](model_key, expl, sample)
            results[expl][model_key] = score
    df = pd.DataFrame(results).T  # righe = explainer
    df.index = [ROW_NAMES.get(idx, idx).upper() for idx in df.index]
    df.columns = [c.replace("-", "").upper() for c in df.columns]
    return df


def main():
    parser = argparse.ArgumentParser("Genera tabelle XAI")
    parser.add_argument("--metric", choices=METRICS + ["all"], default="all")
    parser.add_argument("--sample", type=int, default=500, help="Numero esempi test, None=tutti")
    parser.add_argument("--out", choices=["markdown", "csv"], default="markdown")
    args = parser.parse_args()

    metrics_to_run = METRICS if args.metric == "all" else [args.metric]

    for met in metrics_to_run:
        df = build_table(met, args.sample)
        print(f"\n### {met.capitalize()} (sample={args.sample})\n")
        if args.out == "markdown":
            print(df.to_markdown(floatfmt=".4f"))
            print()
        else:
            csv_name = f"{met}_table.csv"
            df.to_csv(csv_name, float_format="%.4f")
            print(f"Salvato {csv_name}")


if __name__ == "__main__":
    main()
