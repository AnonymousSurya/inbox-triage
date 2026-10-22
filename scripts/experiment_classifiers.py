#!/usr/bin/env python3
"""
Experiment script: compare 4 local classifier options against the baseline.

Usage:
    python scripts/experiment_classifiers.py --method all
    python scripts/experiment_classifiers.py --method tfidf_tuned
    python scripts/experiment_classifiers.py --method sentence_transformer
    python scripts/experiment_classifiers.py --method setfit

Results are printed as a comparison table and saved to results/experiment_results.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN_PATH = str(DATA / "train.jsonl.gz")
DEV_PATH   = str(DATA / "dev.jsonl")

TAIL_TURNS = 8

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def context_text(context: list[dict], tail: int = TAIL_TURNS) -> str:
    parts = []
    for t in context[-tail:]:
        speaker = t.get("speaker", "")
        text = t.get("text", "")
        parts.append(f"[{speaker}] {text}")
    return " ".join(parts)


def build_xy(rows: list[dict]) -> tuple[list[str], list[str], list[str]]:
    texts   = [context_text(r.get("context") or []) for r in rows]
    intents = [r["intent"] for r in rows]
    actions = [r["action"] for r in rows]
    return texts, intents, actions


def evaluate(pred_intents, pred_actions, gold_intents, gold_actions, label: str, elapsed_train: float, elapsed_infer: float):
    ia  = accuracy_score(gold_intents, pred_intents)
    aa  = accuracy_score(gold_actions, pred_actions)
    f1  = f1_score(gold_intents, pred_intents, average="macro", zero_division=0)
    n   = len(gold_intents)
    ms_per = elapsed_infer / n * 1000

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Intent accuracy : {ia:.4f}")
    print(f"  Action accuracy : {aa:.4f}")
    print(f"  Intent macro-F1 : {f1:.4f}")
    print(f"  Train time      : {elapsed_train:.1f}s")
    print(f"  Infer ms/sample : {ms_per:.2f}ms")
    print(f"{'='*60}")

    return {"label": label, "intent_acc": round(ia,4), "action_acc": round(aa,4),
            "macro_f1": round(f1,4), "train_s": round(elapsed_train,1),
            "infer_ms_per": round(ms_per,2)}


# ---------------------------------------------------------------------------
# Option 1 — Baseline (current production model, for reference)
# ---------------------------------------------------------------------------

def run_baseline(train_rows, dev_rows):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    print("\n[1/4] Running: Baseline (current production — TF-IDF word+char, LinearSVC, class_weight=balanced)")

    texts_tr, intents_tr, actions_tr = build_xy(train_rows)
    texts_dev, intents_dev, actions_dev = build_xy(dev_rows)

    ie = LabelEncoder(); ae = LabelEncoder()
    y_i = ie.fit_transform(intents_tr)
    y_a = ae.fit_transform(actions_tr)

    def make_pipe():
        return Pipeline([
            ("feat", FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                    max_features=60_000, sublinear_tf=True, strip_accents="unicode", min_df=2)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                    max_features=80_000, sublinear_tf=True, strip_accents="unicode", min_df=3)),
            ])),
            ("clf", CalibratedClassifierCV(
                LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"), cv=3, method="sigmoid"
            )),
        ])

    t0 = time.time()
    ip = make_pipe(); ip.fit(texts_tr, y_i)
    ap = make_pipe(); ap.fit(texts_tr, y_a)
    train_t = time.time() - t0

    t0 = time.time()
    pi = ie.inverse_transform(ip.predict(texts_dev))
    pa = ae.inverse_transform(ap.predict(texts_dev))
    infer_t = time.time() - t0

    return evaluate(pi, pa, intents_dev, actions_dev,
                    "Baseline (TF-IDF word+char, LinearSVC balanced)", train_t, infer_t)


# ---------------------------------------------------------------------------
# Option 2 — Tuned TF-IDF (trigrams, more features, C sweep)
# ---------------------------------------------------------------------------

def run_tfidf_tuned(train_rows, dev_rows):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    print("\n[2/4] Running: Tuned TF-IDF (trigrams + more features + C sweep)")

    texts_tr, intents_tr, actions_tr = build_xy(train_rows)
    texts_dev, intents_dev, actions_dev = build_xy(dev_rows)

    ie = LabelEncoder(); ae = LabelEncoder()
    y_i = ie.fit_transform(intents_tr)
    y_a = ae.fit_transform(actions_tr)

    # Try C values and pick best via 3-fold CV
    best_ia = 0.0
    best_cfg = None
    best_pipes = None

    for C in [0.5, 1.0, 2.0, 5.0]:
        def make_pipe(c=C):
            return Pipeline([
                ("feat", FeatureUnion([
                    ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,3),
                        max_features=100_000, sublinear_tf=True, strip_accents="unicode", min_df=2)),
                    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
                        max_features=120_000, sublinear_tf=True, strip_accents="unicode", min_df=3)),
                ])),
                ("clf", CalibratedClassifierCV(
                    LinearSVC(max_iter=5000, C=c, class_weight="balanced"), cv=3, method="sigmoid"
                )),
            ])

        t0 = time.time()
        ip = make_pipe(); ip.fit(texts_tr, y_i)
        ap = make_pipe(); ap.fit(texts_tr, y_a)
        elapsed = time.time() - t0

        pi = ie.inverse_transform(ip.predict(texts_dev))
        ia = accuracy_score(intents_dev, pi)
        print(f"   C={C}: intent_acc={ia:.4f}  (train {elapsed:.0f}s)")

        if ia > best_ia:
            best_ia = ia
            best_cfg = (C, elapsed)
            best_pipes = (ip, ap, ie, ae)

    C_best, train_t = best_cfg
    ip, ap, ie, ae = best_pipes
    print(f"   Best C={C_best}")

    t0 = time.time()
    pi = ie.inverse_transform(ip.predict(texts_dev))
    pa = ae.inverse_transform(ap.predict(texts_dev))
    infer_t = time.time() - t0

    return evaluate(pi, pa, intents_dev, actions_dev,
                    f"Tuned TF-IDF (trigrams, C={C_best})", train_t, infer_t)


# ---------------------------------------------------------------------------
# Option 3 — Sentence Transformers + SVM
# ---------------------------------------------------------------------------

def run_sentence_transformer(train_rows, dev_rows):
    print("\n[3/4] Running: Sentence Transformers (all-MiniLM-L6-v2) + LinearSVC")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ✗ sentence-transformers not installed.")
        print("    Install with: pip install sentence-transformers")
        return {"label": "Sentence Transformer", "error": "not installed"}

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    texts_tr, intents_tr, actions_tr = build_xy(train_rows)
    texts_dev, intents_dev, actions_dev = build_xy(dev_rows)

    ie = LabelEncoder(); ae = LabelEncoder()
    y_i = ie.fit_transform(intents_tr)
    y_a = ae.fit_transform(actions_tr)

    print("  Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    t0 = time.time()
    print("  Encoding training set ...")
    X_tr = model.encode(texts_tr, batch_size=256, show_progress_bar=True)
    print("  Training classifiers ...")
    ic = CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"), cv=3)
    ac = CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"), cv=3)
    ic.fit(X_tr, y_i)
    ac.fit(X_tr, y_a)
    train_t = time.time() - t0

    t0 = time.time()
    print("  Encoding dev set ...")
    X_dev = model.encode(texts_dev, batch_size=256, show_progress_bar=False)
    pi = ie.inverse_transform(ic.predict(X_dev))
    pa = ae.inverse_transform(ac.predict(X_dev))
    infer_t = time.time() - t0

    return evaluate(pi, pa, intents_dev, actions_dev,
                    "Sentence Transformer (all-MiniLM-L6-v2) + SVC", train_t, infer_t)


# ---------------------------------------------------------------------------
# Option 4 — SetFit
# ---------------------------------------------------------------------------

def run_setfit(train_rows, dev_rows):
    print("\n[5/5] Running: SetFit (sentence-transformers + contrastive fine-tuning)")

    try:
        from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitArgs
        from datasets import Dataset
    except ImportError:
        print("  ✗ setfit not installed.")
        print("    Install with: pip install setfit datasets")
        return {"label": "SetFit", "error": "not installed"}

    texts_tr, intents_tr, _ = build_xy(train_rows)
    texts_dev, intents_dev, _ = build_xy(dev_rows)

    ie = LabelEncoder()
    y_tr = ie.fit_transform(intents_tr).tolist()

    # SetFit is few-shot: subsample to 16 examples per class to avoid OOM
    import random; random.seed(42)
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for txt, label in zip(texts_tr, intents_tr):
        buckets[label].append(txt)
    fs_texts, fs_labels = [], []
    for label, txts in buckets.items():
        chosen = random.sample(txts, min(16, len(txts)))
        fs_texts.extend(chosen)
        fs_labels.extend([label] * len(chosen))
    print(f"  Few-shot subset: {len(fs_texts)} examples ({len(buckets)} classes, <=16 each)")

    t0 = time.time()
    print("  Loading SetFit model (paraphrase-MiniLM-L6-v2) ...")
    label_names = list(ie.classes_)
    train_ds = Dataset.from_dict({"text": fs_texts, "label": fs_labels})

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        labels=label_names,
    )
    args = SetFitArgs(
        num_epochs=1,
        batch_size=32,
        num_iterations=5,
        output_dir=str(RESULTS / "setfit_ckpt"),
        report_to="none",
    )
    trainer = SetFitTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )
    print("  Contrastive fine-tuning ...")
    trainer.train()
    train_t = time.time() - t0

    t0 = time.time()
    preds = model.predict(texts_dev)
    pi = list(preds)
    infer_t = time.time() - t0

    # Action: TF-IDF SVC (SetFit only does intent here)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC
    ae = LabelEncoder()
    y_a = ae.fit_transform([r["action"] for r in train_rows])
    ap = Pipeline([
        ("feat", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                max_features=60_000, sublinear_tf=True, strip_accents="unicode", min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                max_features=80_000, sublinear_tf=True, strip_accents="unicode", min_df=3)),
        ])),
        ("clf", CalibratedClassifierCV(
            LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"), cv=3)),
    ])
    ap.fit(texts_tr, y_a)
    pa = ae.inverse_transform(ap.predict(texts_dev))

    return evaluate(pi, pa, intents_dev, [r["action"] for r in dev_rows],
                    "SetFit (paraphrase-MiniLM-L6-v2)", train_t, infer_t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHOD_MAP = {
    "baseline":            run_baseline,
    "tfidf_tuned":         run_tfidf_tuned,
    "sentence_transformer": run_sentence_transformer,
    "setfit":              run_setfit,
}

def print_summary(results: list[dict]):
    print("\n" + "="*75)
    print(f"  {'Method':<42} {'Intent':>7} {'Action':>7} {'F1':>7} {'ms/req':>7}")
    print("="*75)
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<42} {'SKIPPED ('+r['error']+')':>25}")
        else:
            print(f"  {r['label']:<42} {r['intent_acc']:>7.4f} {r['action_acc']:>7.4f} "
                  f"{r['macro_f1']:>7.4f} {r['infer_ms_per']:>7.1f}")
    print("="*75)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="all",
                        choices=list(METHOD_MAP.keys()) + ["all"],
                        help="Which method to run (default: all)")
    parser.add_argument("--train", default=TRAIN_PATH)
    parser.add_argument("--dev",   default=DEV_PATH)
    args = parser.parse_args()

    print(f"Loading training data from {args.train} ...")
    train_rows = load_jsonl(args.train)
    print(f"Loading dev data from {args.dev} ...")
    dev_rows = load_jsonl(args.dev)
    print(f"Train: {len(train_rows)} rows  Dev: {len(dev_rows)} rows")

    methods = list(METHOD_MAP.items()) if args.method == "all" \
              else [(args.method, METHOD_MAP[args.method])]

    results = []
    for name, fn in methods:
        try:
            r = fn(train_rows, dev_rows)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"\n  ERROR in {name}: {exc}")
            results.append({"label": name, "error": str(exc)})

    if len(results) > 1:
        print_summary(results)

    out = RESULTS / "experiment_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
