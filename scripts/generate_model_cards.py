#!/usr/bin/env python3
"""
Generate HuggingFace model cards for the fine-tuned NLLB-200 Gĩkũyũ models.

Cards are data-driven: training details come from each run's
``finetune_meta.json`` and ``config/finetune.yaml``, and the evaluation table
is filled from a benchmark ``full_metrics.json`` when available. A README.md is
written into each model's ``final/`` directory so it uploads alongside the
weights.

Usage:
    python -m scripts.generate_model_cards
    python -m scripts.generate_model_cards --eval-metrics results/<run>/metrics/full_metrics.json
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from scripts.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_ORG = "UEFMarkIrura"

# run_key -> descriptive fields for the card
RUN_INFO = {
    "nllb_600m": {
        "repo_base": "nllb-200-distilled-600M",
        "base_model": "facebook/nllb-200-distilled-600M",
        "size_label": "600M (distilled)",
        "method": "Full fine-tuning",
        "baseline_name": "NLLB-200 (600M Distilled)",
        "finetuned_name": "NLLB-200 (600M) fine-tuned",
        "is_lora": False,
    },
    "nllb_1_3b": {
        "repo_base": "nllb-200-1.3B",
        "base_model": "facebook/nllb-200-1.3B",
        "size_label": "1.3B (dense)",
        "method": "LoRA (r=16, alpha=32)",
        "baseline_name": "NLLB-200 (1.3B)",
        "finetuned_name": "NLLB-200 (1.3B) fine-tuned",
        "is_lora": True,
    },
    "nllb_distilled_1_3b": {
        "repo_base": "nllb-200-distilled-1.3B",
        "base_model": "facebook/nllb-200-distilled-1.3B",
        "size_label": "1.3B (distilled)",
        "method": "LoRA (r=16, alpha=32)",
        "baseline_name": "NLLB-200 (1.3B Distilled)",
        "finetuned_name": "NLLB-200 (1.3B Distilled) fine-tuned",
        "is_lora": True,
    },
    "nllb_3_3b": {
        "repo_base": "nllb-200-3.3B",
        "base_model": "facebook/nllb-200-3.3B",
        "size_label": "3.3B (dense)",
        "method": "QLoRA (4-bit NF4, r=16, alpha=32)",
        "baseline_name": "NLLB-200 (3.3B)",
        "finetuned_name": "NLLB-200 (3.3B) fine-tuned",
        "is_lora": True,
    },
}

DIRECTION_INFO = {
    "eng2kik": {
        "src_code": "eng_Latn",
        "tgt_code": "kik_Latn",
        "src_name": "English",
        "tgt_name": "Gĩkũyũ",
        "arrow": "English → Gĩkũyũ",
        "metric_key": "eng->kik",
        "example_src": "Plant the maize seeds at the onset of the rains.",
    },
    "kik2eng": {
        "src_code": "kik_Latn",
        "tgt_code": "eng_Latn",
        "src_name": "Gĩkũyũ",
        "tgt_name": "English",
        "arrow": "Gĩkũyũ → English",
        "metric_key": "kik->eng",
        "example_src": "Handa mbegũ cia mbembe kĩambĩrĩria-inĩ kĩa mbura.",
    },
}

DATASET_ID = "UEFMarkIrura/Gikuyu_Agricultural_Corpus_2025"


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _metric_row(metrics: Dict) -> str:
    if not metrics:
        return "| _pending_ | _pending_ | _pending_ | _pending_ |"
    return (
        f"| {_fmt(metrics.get('chrf_pp'))} "
        f"| {_fmt(metrics.get('bleu'))} "
        f"| {_fmt(metrics.get('bertscore_f1'), 3)} "
        f"| {_fmt(metrics.get('africomet_mtl'), 3)} |"
    )


def build_card(
    run_key: str,
    direction: str,
    meta: Dict,
    ft_cfg: Dict,
    eval_metrics: Optional[Dict],
) -> str:
    info = RUN_INFO[run_key]
    d = DIRECTION_INFO[direction]
    defaults = ft_cfg["defaults"]
    run_cfg = {**defaults, **ft_cfg["runs"][run_key]}

    repo_id = f"{HF_ORG}/{info['repo_base']}-gikuyu-{direction}"
    effective_lr = run_cfg["learning_rate"]

    is_distilled = "distilled" in info["base_model"]
    if "1.3B" in info["size_label"]:
        lineage_note = (
            f"> Base checkpoint: `{info['base_model']}`. NLLB-200 ships **two** 1.3B "
            f"models: this is the "
            f"{'**distilled** checkpoint (distilled from the 54B MoE flagship, like the 600M)' if is_distilled else '**dense (non-distilled)** checkpoint'}, "
            f"i.e. `{info['base_model']}` — not "
            f"`{'facebook/nllb-200-1.3B' if is_distilled else 'facebook/nllb-200-distilled-1.3B'}`.\n"
        )
    elif is_distilled:
        lineage_note = (
            f"> Base checkpoint: `{info['base_model']}` — a **distilled** NLLB-200 "
            f"model (distilled from the 54B MoE flagship).\n"
        )
    else:
        lineage_note = (
            f"> Base checkpoint: `{info['base_model']}` — a **dense (non-distilled)** "
            f"NLLB-200 model.\n"
        )

    # Evaluation table rows (test set), fine-tuned vs its baseline
    ft_metrics = base_metrics = None
    if eval_metrics:
        dir_block = eval_metrics.get(d["metric_key"], {})
        ft_metrics = dir_block.get(info["finetuned_name"])
        base_metrics = dir_block.get(info["baseline_name"])

    frontmatter = f"""---
license: cc-by-nc-4.0
language:
- ki
- en
base_model: {info['base_model']}
pipeline_tag: translation
library_name: transformers
tags:
- translation
- gikuyu
- kikuyu
- nllb
- low-resource
- african-languages
- agriculture
datasets:
- {DATASET_ID}
metrics:
- chrf
- bleu
- bertscore
---"""

    lora_note = ""
    if info["is_lora"]:
        lora_note = (
            f"\nThe adapter was trained with {info['method']} and **merged into the "
            f"base weights** for release, so it loads as a standalone model with no "
            f"PEFT dependency.\n"
        )

    usage = f"""## Usage

This is a standard NLLB-200 checkpoint. Set the source language on the tokenizer
and force the target-language token at generation time.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="{d['src_code']}")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = "{d['example_src']}"
inputs = tokenizer(text, return_tensors="pt")
generated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("{d['tgt_code']}"),
    max_new_tokens=256,
    num_beams=5,
)
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

### Batched translation

```python
texts = [
    "{d['example_src']}",
    # ... more {d['src_name']} sentences
]
tokenizer.src_lang = "{d['src_code']}"
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=192)
generated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("{d['tgt_code']}"),
    max_new_tokens=256,
    num_beams=5,
)
translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
```

### `pipeline` shortcut

```python
from transformers import pipeline

translate = pipeline(
    "translation",
    model="{repo_id}",
    src_lang="{d['src_code']}",
    tgt_lang="{d['tgt_code']}",
    max_length=256,
)
print(translate("{d['example_src']}")[0]["translation_text"])
```

> **Input normalization.** Gĩkũyũ text should use NFC-normalized diacritics
> (ĩ = U+0129, ũ = U+0169). Mixed NFC/NFD input degrades quality; normalize with
> `unicodedata.normalize("NFC", text)` before translating.
"""

    eval_section = f"""## Evaluation

Evaluated on the held-out **GAC v1.1 test set** (450 pairs, agricultural domain),
{d['arrow']}. Metrics: chrF++ (`char_order=6, word_order=2`), BLEU (SacreBLEU),
BERTScore F1 (`bert-base-multilingual-cased`), and AfriCOMET-MTL
(`masakhane/africomet-mtl`, the primary ranking metric for African MT).

| System | chrF++ | BLEU | BERTScore F1 | AfriCOMET-MTL |
| --- | --- | --- | --- | --- |
| **This model (fine-tuned)** {_metric_row(ft_metrics)}
| {info['base_model']} (baseline) {_metric_row(base_metrics)}

All scores are computed on NFC-normalized text with the released evaluation
pipeline (`github.com/markgachara/MTBenchmarks`), so they are directly
comparable to the benchmark in the accompanying paper.
"""

    val_chrf = meta.get("final_metrics", {}).get("eval_chrf")
    val_bleu = meta.get("final_metrics", {}).get("eval_bleu")
    best_epoch = meta.get("final_metrics", {}).get("epoch")

    card = f"""{frontmatter}

# NLLB-200 {info['size_label']} — Gĩkũyũ MT ({d['arrow']})

A {info['size_label']} [NLLB-200]({("https://huggingface.co/" + info['base_model'])})
model fine-tuned for **{d['arrow']}** translation on the agricultural-domain
Gĩkũyũ Agricultural Corpus (GAC v1.1). Gĩkũyũ (also written Kikuyu) is a Bantu
language spoken by over eight million people in central Kenya.

{lineage_note}{lora_note}
- **Base model:** `{info['base_model']}`
- **Direction:** {d['arrow']} (`{d['src_code']}` → `{d['tgt_code']}`)
- **Fine-tuning method:** {info['method']}
- **Training data:** [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID}) — GAC v1.1 train split ({meta.get('train_size', '—')} pairs)
- **License:** CC-BY-NC-4.0 (inherited from NLLB-200)

## Intended uses

- **In-domain machine translation** between {d['src_name']} and {d['tgt_name']},
  especially agricultural, extension, and practical-register text.
- A **baseline / starting point** for further Gĩkũyũ MT research and fine-tuning.
- Assisting human translators and language-technology prototypes for Gĩkũyũ.

**Out of scope:** commercial use (the NLLB-NC license prohibits it), high-stakes
settings without human review, and domains far from agriculture (legal, medical,
casual conversation), where quality is not characterized.

{usage}
{eval_section}
## Training

- **Method:** {info['method']}
- **Data:** GAC v1.1 train split ({meta.get('train_size', '—')} pairs after
  deduplication), validation split ({meta.get('val_size', '—')} pairs). Text is
  NFC-normalized and passed through Moses punctuation normalization; the test
  split is fully disjoint from training.
- **Epochs:** up to {run_cfg['num_train_epochs']} with early stopping
  (patience {run_cfg['early_stopping_patience']}) on validation chrF++;
  best checkpoint at epoch {int(best_epoch) if best_epoch else '—'}.
- **Learning rate:** {effective_lr} ({run_cfg['lr_scheduler_type']} schedule,
  warmup ratio {run_cfg['warmup_ratio']})
- **Batch:** {run_cfg['per_device_train_batch_size']} × grad-accum
  {run_cfg['gradient_accumulation_steps']}; label smoothing
  {run_cfg['label_smoothing_factor']}; max sequence length
  {run_cfg['max_source_length']}.
- **Precision / hardware:** fp16 on a single consumer GPU (RTX 2080 Ti 11 GB).
- **Best validation chrF++ / BLEU:** {_fmt(val_chrf)} / {_fmt(val_bleu)}
  (49-pair validation set; test-set numbers above are the reference figures).

## Limitations and bias

- **Domain.** Trained only on agricultural-domain text; output quality drops on
  other registers.
- **Dialect.** GAC is predominantly central-Kenya dialects (Kĩ-Mũrang'a,
  Kĩ-Mathĩra, Gĩ-Kabete); other dialects are under-represented.
- **Diacritics.** Gĩkũyũ meaning depends on the vowels ĩ and ũ. Provide
  NFC-normalized input and verify diacritics in output for critical use.
- **Not for commercial use** (CC-BY-NC-4.0).

## Citation

If you use this model, please cite the benchmark paper and the corpus:

```bibtex
@misc{{gikuyu_mt_benchmark_2026,
  title  = {{Benchmarking Machine Translation Models for G\u0129k\u0169y\u0169: A Case Study in Low-Resource Language Evaluation}},
  author = {{Irura, Mark and collaborators}},
  year   = {{2026}},
  note   = {{github.com/markgachara/MTBenchmarks}}
}}

@misc{{gac_2026,
  title     = {{G\u0129k\u0169y\u0169 Agricultural Corpus (GAC) v1.1}},
  author    = {{Irura, Mark}},
  year      = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{https://huggingface.co/datasets/{DATASET_ID}}}
}}
```

## Acknowledgements

Built on Meta AI's [NLLB-200]({("https://huggingface.co/" + info['base_model'])}).
Fine-tuned and evaluated with the
[MTBenchmarks](https://github.com/markgachara/MTBenchmarks) pipeline.
"""
    return repo_id, card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="models/finetuned")
    ap.add_argument("--finetune-config", default="config/finetune.yaml")
    ap.add_argument(
        "--eval-metrics",
        default=None,
        help="Path to a benchmark full_metrics.json to populate the eval table",
    )
    args = ap.parse_args()

    ft_cfg = load_config(args.finetune_config)
    eval_metrics = None
    if args.eval_metrics and Path(args.eval_metrics).exists():
        eval_metrics = json.load(open(args.eval_metrics))
        logger.info(f"Loaded eval metrics from {args.eval_metrics}")

    root = Path(args.models_root)
    written = []
    for meta_path in sorted(root.glob("*/final/finetune_meta.json")):
        meta = json.load(open(meta_path))
        run_key, direction = meta["run_key"], meta["direction"]
        if run_key not in RUN_INFO or direction not in DIRECTION_INFO:
            logger.warning(f"Skipping unknown {run_key}/{direction}")
            continue
        repo_id, card = build_card(run_key, direction, meta, ft_cfg, eval_metrics)
        out = meta_path.parent / "README.md"
        out.write_text(card, encoding="utf-8")
        written.append((repo_id, out))
        logger.info(f"Wrote card -> {out}  ({repo_id})")

    print(f"\nGenerated {len(written)} model card(s):")
    for repo_id, out in written:
        print(f"  {repo_id:60s} {out}")


if __name__ == "__main__":
    main()
