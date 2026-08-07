#!/usr/bin/env python3
"""
Build the human (qualitative) assessment sheet for the GAC v1.1 test split.

Mirrors the column layout of the previous round
(``data/human_assessment_to_analyze.xlsx - Sheet1.csv``) so the two are directly
comparable and can be analysed with the same code. The four rating columns plus
Overall Quality and Notes are left blank for the assessors.

The previous round's 50 sentences cannot be reused: 49 of them became the
VALIDATION split in v1.1 and are no longer in the test set, so a fresh
stratified sample is drawn here.

Usage:
    python -m scripts.build_human_assessment
    python -m scripts.build_human_assessment --include-llms --sample-size 50
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from scripts.data_loader import MTDataLoader, normalize_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLUMNS = [
    "Index", "Direction", "Model", "Source", "Reference", "Translation",
    "Diacritic Accuracy", "Idiomatic Appropriateness",
    "Cultural Concept Handling", "Grammar", "Overall Quality", "Notes",
]

# Diacritics only apply when Gĩkũyũ is the target
NA = "NOT APPLICABLE"

NLLB_SYSTEMS = [
    "NLLB-200 (600M Distilled)", "NLLB-200 (600M) fine-tuned",
    "NLLB-200 (1.3B Distilled)", "NLLB-200 (1.3B Distilled) fine-tuned",
    "NLLB-200 (1.3B)", "NLLB-200 (1.3B) fine-tuned",
    "NLLB-200 (3.3B)", "NLLB-200 (3.3B) fine-tuned",
]
OTHER_SYSTEMS = [
    "M2M-100 (418M)",
    "Llama 3.1 8B Instruct (zero-shot)", "Llama 3.1 8B Instruct (3-shot)",
    "Gemma 3 4B Instruct (zero-shot)", "Gemma 3 4B Instruct (3-shot)",
]


def load_translations(run_dir: Path) -> dict:
    """(direction, model) -> {normalized source: translation}."""
    out = {}
    for path in sorted((run_dir / "translations").glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("sentences", [])
        if not rows or not any(r["translation"].strip() for r in rows):
            continue
        out[(payload["direction"], payload["model"])] = {
            normalize_text(r["source"]): r["translation"] for r in rows
        }
    return out


def stratified_indices(texts, size, seed=42):
    """Sample indices spread across short/medium/long source sentences."""
    import random

    rng = random.Random(seed)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i].split()))
    third = len(order) // 3
    strata = [order[:third], order[third:2 * third], order[2 * third:]]

    per = size // 3
    picked = []
    for n, stratum in enumerate(strata):
        take = per if n < 2 else size - 2 * per
        picked += rng.sample(stratum, min(take, len(stratum)))
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/benchmark_20260805_084023")
    ap.add_argument("--test-file", default="data/JulyDataUpdate/TEST_DATA.xlsx")
    ap.add_argument("--out-dir", default="data/human_assessment")
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-llms", action="store_true",
                    help="Also rate M2M-100 and the prompted LLMs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    eng, kik = MTDataLoader().load_parallel_excel(args.test_file)
    trans = load_translations(run_dir)

    systems = NLLB_SYSTEMS + (OTHER_SYSTEMS if args.include_llms else [])
    idx = stratified_indices(eng, args.sample_size, args.seed)
    logger.info(f"sampled {len(idx)} of {len(eng)} test sentences (seed {args.seed})")

    rows, skipped = [], set()
    for direction in ("eng->kik", "kik->eng"):
        src_all, ref_all = (eng, kik) if direction == "eng->kik" else (kik, eng)
        for model in systems:
            table = trans.get((direction, model))
            if table is None:
                skipped.add((direction, model))
                continue
            for i in idx:
                source = src_all[i]
                hyp = table.get(normalize_text(source))
                if hyp is None:
                    continue
                rows.append({
                    "Index": i,
                    "Direction": direction,
                    "Model": model,
                    "Source": source,
                    "Reference": ref_all[i],
                    "Translation": hyp,
                    "Diacritic Accuracy": "" if direction == "eng->kik" else NA,
                    "Idiomatic Appropriateness": "",
                    "Cultural Concept Handling": "",
                    "Grammar": "",
                    "Overall Quality": "",
                    "Notes": "",
                })

    df = pd.DataFrame(rows, columns=COLUMNS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "human_assessment_v1.1_test450_to_analyze.csv"
    df.to_csv(out, index=False)

    logger.info(f"wrote {out}")
    print(f"\nrows            : {len(df)}")
    print(f"sentences       : {df['Index'].nunique()}")
    print(f"systems         : {df['Model'].nunique()}")
    print(f"per direction   : {df.groupby('Direction').size().to_dict()}")
    if skipped:
        print("\nno translations found for (excluded):")
        for d, m in sorted(skipped):
            print(f"  {d:9s} {m}")


if __name__ == "__main__":
    main()
