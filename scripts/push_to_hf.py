#!/usr/bin/env python3
"""
Stage and push the fine-tuned NLLB-200 Gĩkũyũ models to the Hugging Face Hub.

Full fine-tunes upload as-is. LoRA / QLoRA runs are merged into their base
weights on CPU first, so every published repo is a standalone model with uniform
usage code and no PEFT dependency.

SAFETY: this is a dry run by default. It prints exactly what would be pushed and
does nothing to the Hub. Pass --push to actually create repos and upload.

Usage:
    python -m scripts.push_to_hf                      # dry run (default)
    python -m scripts.push_to_hf --only nllb_600m     # stage/inspect one family
    python -m scripts.push_to_hf --push               # ACTUAL upload (private)
    python -m scripts.push_to_hf --push --public      # ACTUAL upload (public)
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

from scripts.generate_model_cards import (
    HF_ORG,
    RUN_INFO,
    DIRECTION_INFO,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def repo_id_for(run_key: str, direction: str) -> str:
    return f"{HF_ORG}/{RUN_INFO[run_key]['repo_base']}-gikuyu-{direction}"


def ensure_standalone(final_dir: Path, meta: dict) -> Path:
    """
    Return a directory holding a full, standalone model ready for upload.

    Full fine-tunes already contain ``model.safetensors``. LoRA/QLoRA runs
    contain only an adapter; this merges the adapter into a fresh fp16 base on
    CPU (the GPUs are busy / shared) and writes the merged model next to it.
    """
    if (final_dir / "model.safetensors").exists() or (final_dir / "pytorch_model.bin").exists():
        return final_dir  # already standalone

    adapter_cfg = final_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        raise FileNotFoundError(f"No model or adapter weights in {final_dir}")

    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    base_id = meta["base_model_id"]
    merged_dir = final_dir.parent / "final_merged"
    if (merged_dir / "model.safetensors").exists():
        logger.info(f"Merged model already present at {merged_dir}")
        return merged_dir

    logger.info(f"Merging adapter into {base_id} on CPU (one-time)...")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(base, str(final_dir)).merge_and_unload()

    # Bake the target-language token into the generation config so the model
    # translates in the right direction even without an explicit argument.
    merged.generation_config.forced_bos_token_id = meta["forced_bos_token_id"]
    merged.generation_config.max_length = 256

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(final_dir)).save_pretrained(str(merged_dir))

    # Carry the model card over to the merged dir.
    card = final_dir / "README.md"
    if card.exists():
        shutil.copy(card, merged_dir / "README.md")

    logger.info(f"Merged model saved to {merged_dir}")
    return merged_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="models/finetuned")
    ap.add_argument("--only", default=None, help="Restrict to one run_key (e.g. nllb_600m)")
    ap.add_argument("--push", action="store_true", help="Actually create repos and upload")
    ap.add_argument("--public", action="store_true", help="Push as public (default private)")
    ap.add_argument("--merge-only", action="store_true", help="Prepare merged models but never push")
    args = ap.parse_args()

    root = Path(args.models_root)
    metas = sorted(root.glob("*/final/finetune_meta.json"))
    if not metas:
        raise SystemExit(f"No trained models under {root}")

    plan = []
    for meta_path in metas:
        meta = json.load(open(meta_path))
        run_key, direction = meta["run_key"], meta["direction"]
        if args.only and run_key != args.only:
            continue
        final_dir = meta_path.parent
        repo_id = repo_id_for(run_key, direction)
        has_card = (final_dir / "README.md").exists()
        plan.append((run_key, direction, final_dir, meta, repo_id, has_card))

    print("\n=== PUSH PLAN ===")
    print(f"Visibility: {'PUBLIC' if args.public else 'private'} | Mode: "
          f"{'PUSH' if args.push else 'DRY RUN (nothing uploaded)'}\n")
    for run_key, direction, final_dir, meta, repo_id, has_card in plan:
        method = RUN_INFO[run_key]["method"]
        card = "card ✓" if has_card else "card ✗ MISSING"
        print(f"  {repo_id:56s}  [{method}]  {card}")
    print()

    if not args.push and not args.merge_only:
        print("Dry run only. Re-run with --push to upload, or --merge-only to "
              "pre-merge LoRA models without uploading.")
        return

    # Prepare standalone dirs (merges LoRA); this is safe and reversible.
    prepared = []
    for run_key, direction, final_dir, meta, repo_id, has_card in plan:
        upload_dir = ensure_standalone(final_dir, meta)
        prepared.append((repo_id, upload_dir, has_card))

    if args.merge_only:
        print("\nMerged/standalone models prepared. No upload performed (--merge-only).")
        return

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    for repo_id, upload_dir, has_card in prepared:
        if not has_card:
            logger.warning(f"{repo_id}: README.md missing; run generate_model_cards first")
        logger.info(f"Creating repo {repo_id} (private={not args.public})")
        create_repo(repo_id, private=not args.public, exist_ok=True, repo_type="model")
        logger.info(f"Uploading {upload_dir} -> {repo_id}")
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add fine-tuned NLLB-200 Gĩkũyũ MT model (GAC v1.1)",
        )
        logger.info(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
