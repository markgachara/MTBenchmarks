"""Corpus-level linguistic statistics on the 50-sentence human-evaluation subset.

Paper 3 Tables 4a/5a carry TTR, hapax-legomena rate, mean sentence length and
perplexity columns. Those cells were either blank for the NLLB rows or carried
values that are not reproducible from any stored benchmark run, and the one
populated NLLB row (1.3B Distilled) was taken from a *different* test set than
the rest of the table.

This script recomputes all four statistics from a single source, restricted to
exactly the 50 sentences that native speakers scored, so the quantitative and
qualitative tables describe the same text. Statistics are computed with the same
``Evaluator._linguistic_metrics`` used by the benchmark, so definitions match
(HL rate is hapax count / unique tokens, not / total tokens).

Perplexity is scored on CPU: the Goldfish LM auto-selects CUDA, and on this host
both GPUs are usually held by other users.
"""
from __future__ import annotations

import os

# Must precede any torch import so the Goldfish LM stays on CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from scripts.data_loader import normalize_text
from scripts.evaluator import MTEvaluator as Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN = REPO / "results" / "benchmark_20260508_063320"
DEFAULT_SHEET = REPO / "data" / "human_assessment_to_analyze.xlsx - Sheet1 (1).csv"
DISTILLED_JSON = REPO / "data" / "human_assessment" / "distilled_1_3b_human_eval_translations.json"

STAT_KEYS = ["ttr", "hapax_rate", "avg_sent_len", "std_sent_len", "total_tokens", "unique_tokens"]


def slugify(name: str) -> str:
    return name.replace(" ", "_").lower()


def subset_indices(sheet: Path) -> dict[str, dict[str, str]]:
    """Normalised source text -> reference, for each direction's 50 sentences."""
    df = pd.read_csv(sheet)
    out = {}
    for direction, group in df.groupby("Direction"):
        uniq = group[["Source", "Reference"]].drop_duplicates()
        out[direction] = {normalize_text(str(s)): str(r) for s, r in zip(uniq["Source"], uniq["Reference"])}
        logger.info("%s: %d human-scored sentences", direction, len(out[direction]))
    return out


def collect(run: Path, wanted: dict[str, dict[str, str]]) -> dict[str, dict[str, dict]]:
    """Pull each system's translations for the human-scored sentences."""
    full = json.loads((run / "metrics" / "full_metrics.json").read_text())
    collected: dict[str, dict[str, dict]] = {}

    for direction, models in full.items():
        keys = wanted[direction]
        per_model = {}
        for name in models:
            fn = run / "translations" / f"{slugify(name)}_{direction.replace('->', '2')}.json"
            if not fn.exists():
                logger.warning("missing translations for %s [%s]", name, direction)
                continue
            sents = json.loads(fn.read_text())["sentences"]
            hit = {normalize_text(s["source"]): s for s in sents if normalize_text(s["source"]) in keys}
            if len(hit) != len(keys):
                logger.warning("%s [%s]: matched %d/%d sentences", name, direction, len(hit), len(keys))
            ordered = [hit[k] for k in keys if k in hit]
            per_model[name] = {
                "predictions": [s["translation"] for s in ordered],
                "references": [s["reference"] for s in ordered],
            }
        collected[direction] = per_model

    if DISTILLED_JSON.exists():
        payload = json.loads(DISTILLED_JSON.read_text())
        for direction, rows in payload["directions"].items():
            collected.setdefault(direction, {})[payload["model"]] = {
                "predictions": [r["translation"] for r in rows],
                "references": [r["reference"] for r in rows],
            }
        logger.info("added %s from %s", payload["model"], DISTILLED_JSON.name)
    else:
        logger.warning("%s not found - run scripts/add_distilled_to_human_eval.py first", DISTILLED_JSON)

    return collected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--no-perplexity", action="store_true")
    args = ap.parse_args()

    wanted = subset_indices(args.sheet)
    collected = collect(args.run, wanted)
    ev = Evaluator()

    results: dict[str, dict] = {}
    for direction, per_model in collected.items():
        # Perplexity is only interpretable when the output language is Gikuyu.
        score_ppl = (not args.no_perplexity) and direction.endswith("kik")
        rows = {}
        for name, data in sorted(per_model.items()):
            stats = ev._linguistic_metrics(data["predictions"], prefix="pred")
            if score_ppl:
                stats["pred_perplexity"] = ev._compute_perplexity(data["predictions"])
            stats["n_sentences"] = len(data["predictions"])
            rows[name] = stats

        any_model = next(iter(per_model.values()))
        ref_stats = ev._linguistic_metrics(any_model["references"], prefix="pred")
        if score_ppl:
            ref_stats["pred_perplexity"] = ev._compute_perplexity(any_model["references"])
        ref_stats["n_sentences"] = len(any_model["references"])
        rows["Human Reference"] = ref_stats

        results[direction] = rows
        logger.info("%s: computed stats for %d systems (+ reference)", direction, len(per_model))

    out = args.run / "metrics" / "human_eval_subset_corpus_stats.json"
    out.write_text(json.dumps({
        "subset": "50-sentence native-speaker assessment subset",
        "source_sheet": args.sheet.name,
        "run": args.run.name,
        "directions": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)

    for direction, rows in results.items():
        print(f"\n=== {direction} (n={next(iter(rows.values()))['n_sentences']}) ===")
        table = pd.DataFrame([{
            "System": k,
            "TTR": v.get("pred_ttr"),
            "HL Rate": v.get("pred_hapax_rate"),
            "Avg Len": v.get("pred_avg_sent_len"),
            "SD": v.get("pred_std_sent_len"),
            "PPL": v.get("pred_perplexity"),
        } for k, v in rows.items()])
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
