# main.py
# ────────────────────────────────────────────────────────────
import os, argparse, pandas as pd, tqdm, torch
from transformers import AutoTokenizer
from models import MODEL_IDS, load_model, fine_tune_model
from dataset import get_imdb
from explainers import EXPLAINERS
from evaluate import compute_metrics_for_example
from utils import set_seed

# -------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        default="huawei-noah/TinyBERT_General_4L_312D",
        help="'all' oppure lista separata da virgole",
    )
    p.add_argument(
        "--explainers",
        default="all",
        help="'all' oppure lista separata da virgole "
        "(ixg,lrp,lime,shap,aflow,aroll)",
    )
    p.add_argument("--n_examples", type=int, default=10)
    p.add_argument("--device", default="cpu", help="'cpu', 'cuda', oppure 'mps'")
    p.add_argument(
        "--minimal",
        action="store_true",
        help="carica i modelli in float16 (solo se device cuda)",
    )
    p.add_argument(
        "--finetune",
        action="store_true",
        help="riaddestra la nuova testa a 2 classi se rigenerata",
    )
    p.add_argument("--output", default="results.csv")
    return p.parse_args()


# -------------------------- MAIN ----------------------------
def main():
    args = parse_args()
    set_seed()

    device = args.device
    model_ids = MODEL_IDS if args.models == "all" else [
        m.strip() for m in args.models.split(",")
    ]
    expl_names = list(EXPLAINERS.keys()) if args.explainers == "all" else [
        e.strip() for e in args.explainers.split(",")
    ]

    rows = []
    for mid in model_ids:
        print(f"\n=== MODEL {mid} ===")
        model = load_model(mid, device=device, minimal=args.minimal)

        if args.finetune:
            train_ds, _ = get_imdb(tokenizer_name=mid)
            print("  › fine-tuning testa a 2 classi …")
            model = fine_tune_model(model, train_ds, device=device)

        tokenizer = AutoTokenizer.from_pretrained(mid, use_fast=True)
        _, test_ds = get_imdb(tokenizer_name=mid)
        subset = test_ds.shuffle(seed=42).select(range(args.n_examples))

        for e_name in expl_names:
            explain = EXPLAINERS[e_name]

            for ex in tqdm.tqdm(subset, desc=e_name):
                text = ex["text"]

                # attribution originale
                orig_attr = dict(
                    explain(model, tokenizer, text, device=device)
                )

                # attribution su testo perturbato (shuffle parole)
                words = text.split()
                if len(words) > 5:
                    g = torch.Generator().manual_seed(0)
                    perm = torch.randperm(len(words), generator=g)
                    pert_text = " ".join(words[i] for i in perm)
                else:
                    pert_text = text

                pert_attr = dict(
                    explain(model, tokenizer, pert_text, device=device)
                )

                metrics = compute_metrics_for_example(
                    expl_attrs=orig_attr,
                    pert_attrs=pert_attr,
                )

                rows.append(
                    dict(
                        model=mid,
                        explainer=e_name,
                        example_id=ex.get("idx", len(rows)),
                        **metrics,
                    )
                )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSalvato → {args.output}")


# ------------------------- RUN ------------------------------
if __name__ == "__main__":
    main()
