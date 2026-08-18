"""Translate the 50 human-evaluation sentences with pretrained NLLB-200 1.3B Distilled.

The Paper 3 benchmark run (benchmark_20260508_063320) covers eight systems; the
1.3B distilled checkpoint was added to the model family afterwards and so has no
rows in the human-assessment sheet. This script produces the missing 100 rows
(50 sentences x 2 directions) using the same decoding settings as the benchmark,
and writes them in the exact column layout of the existing sheet.

Source/reference text is taken from the human-assessment sheet itself rather than
re-derived from the corpus, which guarantees the new rows are aligned to exactly
the sentences the raters already scored.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from scripts.models.transformer_mt import TransformerMT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SHEET = REPO / "data" / "human_assessment_to_analyze.xlsx - Sheet1 (1).csv"
OUT_DIR = REPO / "data" / "human_assessment"

MODEL_KEY = "nllb_200_distilled_1_3b"
COLUMNS = [
    "Index", "Direction", "Model", "Source", "Reference", "Translation",
    "Diacritic Accuracy", "Idiomatic Appropriateness", "Cultural Concept Handling",
    "Grammar", "Overall Quality", "Notes",
]
RATING_COLS = COLUMNS[6:11]


def load_sentences(sheet: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return the sheet plus the unique (Index, Source, Reference) set per direction."""
    # Read as text: the existing rating columns hold integers, and appending rows
    # with blank ratings would otherwise promote them to float and rewrite every
    # existing "1" as "1.0" in the output.
    df = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet is missing expected columns: {missing}")

    per_direction = {}
    for direction, group in df.groupby("Direction"):
        uniq = group[["Index", "Source", "Reference"]].drop_duplicates("Index")
        # Every model must have been shown identical source/reference text for a
        # given Index, otherwise the new rows would not line up with the old ones.
        collisions = group.groupby("Index")[["Source", "Reference"]].nunique()
        bad = collisions[(collisions > 1).any(axis=1)]
        if len(bad):
            raise ValueError(f"{direction}: inconsistent Source/Reference for Index {list(bad.index)}")
        per_direction[direction] = (
            uniq.iloc[uniq["Index"].astype(int).argsort(kind="stable")].reset_index(drop=True)
        )
        logger.info("%s: %d unique sentences", direction, len(uniq))
    return df, per_direction


def build_model(device: str) -> TransformerMT:
    cfg = yaml.safe_load((REPO / "config" / "models.yaml").read_text())["models"][MODEL_KEY]
    if device == "cpu":
        # Both GPUs are typically occupied by other users on this host; fp32 on
        # CPU is the safe fallback and fp16 is not well supported there anyway.
        cfg = {**cfg, "dtype": "torch.float32", "batch_size": 8}
    logger.info("Loading %s on %s", cfg["name"], device)
    model = TransformerMT(cfg, device_map=device)
    model.load()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda:0", "cuda:1"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    sheet, per_direction = load_sentences(args.sheet)
    model = build_model(args.device)
    model_name = model.name

    if model_name in set(sheet["Model"]):
        raise SystemExit(f"'{model_name}' already present in the sheet; nothing to do.")

    new_rows, payload = [], {}
    for direction, frame in per_direction.items():
        src_lang, tgt_lang = direction.split("->")
        result = model.translate(
            frame["Source"].astype(str).tolist(),
            source_lang=src_lang,
            target_lang=tgt_lang,
        )
        logger.info("%s: translated %d sentences in %.1fs",
                    direction, len(result.translations), result.inference_time)

        payload[direction] = [
            {"index": int(i), "source": s, "translation": t, "reference": r}
            for i, s, t, r in zip(frame["Index"], frame["Source"], result.translations, frame["Reference"])
        ]
        for idx, src, ref, hyp in zip(frame["Index"], frame["Source"], frame["Reference"], result.translations):
            new_rows.append({
                "Index": idx,
                "Direction": direction,
                "Model": model_name,
                "Source": src,
                "Reference": ref,
                "Translation": hyp,
                # Diacritics are only rateable when the output is Gikuyu; the
                # sheet's existing convention marks the reverse direction.
                "Diacritic Accuracy": "" if tgt_lang == "kik" else "NOT APPLICABLE",
                "Idiomatic Appropriateness": "",
                "Cultural Concept Handling": "",
                "Grammar": "",
                "Overall Quality": "",
                "Notes": "",
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    added = (
        pd.DataFrame(new_rows, columns=COLUMNS)
        .assign(_sort_key=lambda d: d["Index"].astype(int))
        .sort_values(["_sort_key", "Direction"], kind="stable")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    rows_only = args.out_dir / "distilled_1_3b_rows_to_paste.csv"
    combined = args.out_dir / "human_assessment_with_distilled_1_3b.csv"
    added.to_csv(rows_only, index=False)
    pd.concat([sheet, added], ignore_index=True)[COLUMNS].to_csv(combined, index=False)
    (args.out_dir / "distilled_1_3b_human_eval_translations.json").write_text(
        json.dumps({"model": model_name, "model_id": model.model_id,
                    "num_beams": model.config.get("num_beams"),
                    "directions": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Wrote %d new rows -> %s", len(added), rows_only)
    logger.info("Wrote %d combined rows -> %s", len(sheet) + len(added), combined)


if __name__ == "__main__":
    main()
