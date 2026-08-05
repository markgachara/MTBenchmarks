#!/usr/bin/env python3
"""
Fine-tune NLLB-200 on the corrected GAC v1.1 corpus (data/JulyDataUpdate).

Trains one model per (checkpoint, direction) pair so that eng->kik and
kik->eng are optimised independently rather than sharing capacity.

600M is fully fine-tuned; 1.3B and 3.3B use LoRA adapters because full
fine-tuning does not fit in the available 11 GB of VRAM.

Usage:
    python -m scripts.finetune --run nllb_600m --direction eng2kik
    python -m scripts.finetune --run nllb_600m --direction both
    python -m scripts.finetune --all
    python -m scripts.finetune --run nllb_600m --direction eng2kik --dry-run
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from scripts.data_loader import MTDataLoader
from scripts.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NllbDataCollator(DataCollatorForSeq2Seq):
    """
    Collator that also emits ``decoder_input_ids``.

    transformers 4.57 dropped ``prepare_decoder_input_ids_from_labels`` from the
    M2M100/NLLB model class, so the stock collator returns only ``labels``.
    With ``label_smoothing_factor`` set, Trainer pops ``labels`` before the
    forward pass, leaving the decoder with neither ``decoder_input_ids`` nor
    ``decoder_inputs_embeds`` -- which surfaces as the misleading error
    "You cannot specify both decoder_input_ids and decoder_inputs_embeds"
    (the underlying check is an XOR that also fires when both are None).

    Seq2SeqTrainer drops ``decoder_input_ids`` again before calling generate()
    when its shape matches ``labels``, so evaluation does not see the reference.
    """

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        if "labels" in batch and "decoder_input_ids" not in batch:
            from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

            pad_id = self.tokenizer.pad_token_id
            labels = batch["labels"].masked_fill(batch["labels"] == -100, pad_id)
            batch["decoder_input_ids"] = shift_tokens_right(
                labels, pad_id, self.model.config.decoder_start_token_id
            )
        return batch


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune NLLB-200 for Gĩkũyũ MT")
    p.add_argument("--config", default="config/finetune.yaml")
    p.add_argument("--run", help="Run key from config (e.g. nllb_600m)")
    p.add_argument(
        "--direction",
        default="both",
        choices=["eng2kik", "kik2eng", "both"],
        help="Translation direction to train",
    )
    p.add_argument("--all", action="store_true", help="Train every run x direction")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Tiny subset + 1 epoch, to validate the pipeline end to end",
    )
    p.add_argument("--output-root", default=None, help="Override output root")
    return p.parse_args()


def build_dataset(
    texts_src: List[str],
    texts_tgt: List[str],
    tokenizer,
    max_source_length: int,
    max_target_length: int,
):
    """Tokenize a parallel corpus into a HF Dataset ready for Seq2SeqTrainer."""
    from datasets import Dataset

    ds = Dataset.from_dict({"src": texts_src, "tgt": texts_tgt})

    def _tokenize(batch):
        model_inputs = tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            max_length=max_source_length,
            truncation=True,
        )
        # Truncate targets separately when they use a different budget
        if max_target_length != max_source_length:
            labels = tokenizer(
                text_target=batch["tgt"],
                max_length=max_target_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(_tokenize, batched=True, remove_columns=["src", "tgt"])


def make_compute_metrics(tokenizer):
    """chrF++ and BLEU on the validation split, matching the benchmark config."""
    import sacrebleu

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        # sacrebleu takes a list of reference *streams* (one stream per
        # alternative reference), each holding one reference per sentence.
        # This is the opposite of the `evaluate` wrapper's convention of one
        # list per sentence; using that form silently deflates the scores.
        references = [[l.strip() for l in decoded_labels]]

        chrf = sacrebleu.corpus_chrf(
            decoded_preds, references, char_order=6, word_order=2
        )
        bleu = sacrebleu.corpus_bleu(decoded_preds, references)
        return {"chrf": round(chrf.score, 4), "bleu": round(bleu.score, 4)}

    return compute_metrics


def load_model(run_cfg: Dict, lora_targets: List[str]):
    """Load the base model, applying LoRA / 4-bit quantization when configured."""
    model_id = run_cfg["model_id"]
    mode = run_cfg.get("mode", "full")
    load_in_4bit = run_cfg.get("load_in_4bit", False)

    kwargs = {"low_cpu_mem_usage": True}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # Quantized weights must stay on one device
        kwargs["device_map"] = {"": 0}
    elif mode == "lora":
        # Frozen base can live in fp16; only adapter weights are trained
        kwargs["torch_dtype"] = torch.float16
    else:
        # Full fine-tuning keeps fp32 master weights; AMP handles the fp16 math
        kwargs["torch_dtype"] = torch.float32

    logger.info(f"Loading {model_id} (mode={mode}, 4bit={load_in_4bit})...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)

    if mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=run_cfg.get("gradient_checkpointing", True)
            )

        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=run_cfg.get("lora_r", 16),
            lora_alpha=run_cfg.get("lora_alpha", 32),
            lora_dropout=run_cfg.get("lora_dropout", 0.05),
            target_modules=lora_targets,
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)

        # With a frozen base and gradient checkpointing, the checkpointed blocks
        # receive inputs that do not require grad, so autograd never builds a
        # graph and backward fails with "element 0 of tensors does not require
        # grad". Forcing the input embeddings to require grad reconnects it.
        if run_cfg.get("gradient_checkpointing", True):
            model.enable_input_require_grads()

        # The fp16 GradScaler cannot unscale half-precision gradients, so the
        # trainable adapter weights are kept in fp32 while the base stays fp16.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        model.print_trainable_parameters()

    return model


def train_one(
    cfg: Dict,
    run_key: str,
    direction: str,
    dry_run: bool = False,
    output_root: str = None,
) -> Dict:
    """Fine-tune a single (checkpoint, direction) combination."""
    defaults = cfg["defaults"]
    run_cfg = {**defaults, **cfg["runs"][run_key]}
    dir_cfg = cfg["directions"][direction]
    data_cfg = cfg["data"]

    set_seed(run_cfg.get("seed", 42))

    out_root = Path(output_root or run_cfg["output_root"])
    output_dir = out_root / f"{run_key}_{direction}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"TRAINING {run_key} | {direction} -> {output_dir}")
    logger.info("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    loader = MTDataLoader()
    tr_eng, tr_kik = loader.load_parallel_excel(
        data_cfg["train_file"], deduplicate=data_cfg.get("deduplicate_train", True)
    )
    va_eng, va_kik = loader.load_parallel_excel(data_cfg["val_file"])

    if dir_cfg["source_lang"] == "eng":
        train_src, train_tgt, val_src, val_tgt = tr_eng, tr_kik, va_eng, va_kik
    else:
        train_src, train_tgt, val_src, val_tgt = tr_kik, tr_eng, va_kik, va_eng

    if dry_run:
        train_src, train_tgt = train_src[:32], train_tgt[:32]
        val_src, val_tgt = val_src[:8], val_tgt[:8]
        run_cfg["num_train_epochs"] = 1

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        run_cfg["model_id"],
        src_lang=dir_cfg["src_code"],
        tgt_lang=dir_cfg["tgt_code"],
    )
    forced_bos = tokenizer.convert_tokens_to_ids(dir_cfg["tgt_code"])
    if forced_bos == tokenizer.unk_token_id:
        raise ValueError(
            f"Target language token {dir_cfg['tgt_code']} is not in the tokenizer vocabulary"
        )
    logger.info(f"forced_bos_token_id: {dir_cfg['tgt_code']} -> {forced_bos}")

    train_ds = build_dataset(
        train_src, train_tgt, tokenizer,
        run_cfg["max_source_length"], run_cfg["max_target_length"],
    )
    val_ds = build_dataset(
        val_src, val_tgt, tokenizer,
        run_cfg["max_source_length"], run_cfg["max_target_length"],
    )
    logger.info(f"train={len(train_ds)} val={len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = load_model(run_cfg, cfg["lora_target_modules"])
    model.config.forced_bos_token_id = forced_bos
    if hasattr(model, "generation_config"):
        model.generation_config.forced_bos_token_id = forced_bos
        # Trainer sets config.use_cache=False for gradient checkpointing, which
        # also disables the KV cache during predict_with_generate. Without it
        # beam search recomputes the whole decoder sequence at every step and
        # materialises [batch*beams, seq_len, 256204] logits, which OOMs an
        # 11 GB card. Generation runs under no_grad, so the cache is safe here.
        model.generation_config.use_cache = True

    collator = NllbDataCollator(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    # ── Training arguments ────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=run_cfg["num_train_epochs"],
        learning_rate=float(run_cfg["learning_rate"]),
        per_device_train_batch_size=run_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=run_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=run_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=run_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=run_cfg["warmup_ratio"],
        weight_decay=run_cfg["weight_decay"],
        label_smoothing_factor=run_cfg["label_smoothing_factor"],
        lr_scheduler_type=run_cfg["lr_scheduler_type"],
        optim=run_cfg["optim"],
        fp16=run_cfg["fp16"],
        max_grad_norm=run_cfg.get("max_grad_norm", 1.0),
        predict_with_generate=True,
        generation_max_length=run_cfg["max_target_length"],
        generation_num_beams=run_cfg["eval_num_beams"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        eval_accumulation_steps=run_cfg.get("eval_accumulation_steps", 2),
        load_best_model_at_end=True,
        metric_for_best_model=run_cfg["metric_for_best_model"],
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
        seed=run_cfg["seed"],
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=run_cfg["early_stopping_patience"]
            )
        ],
    )

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = trainer.evaluate()
    logger.info(f"Final validation metrics: {metrics}")

    meta = {
        "run_key": run_key,
        "direction": direction,
        "name": run_cfg.get("name", run_key),
        "base_model_id": run_cfg["model_id"],
        "mode": run_cfg.get("mode", "full"),
        "load_in_4bit": run_cfg.get("load_in_4bit", False),
        "src_code": dir_cfg["src_code"],
        "tgt_code": dir_cfg["tgt_code"],
        "forced_bos_token_id": forced_bos,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_file": data_cfg["train_file"],
        "val_file": data_cfg["val_file"],
        "final_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    with open(final_dir / "finetune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {final_dir}")

    del trainer, model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    return meta


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not args.all and not args.run:
        raise SystemExit("Specify --run <key> or --all")

    run_keys = list(cfg["runs"].keys()) if args.all else [args.run]
    if args.all:
        directions = ["eng2kik", "kik2eng"]
    else:
        directions = (
            ["eng2kik", "kik2eng"] if args.direction == "both" else [args.direction]
        )

    results = []
    for run_key in run_keys:
        if run_key not in cfg["runs"]:
            raise SystemExit(f"Unknown run '{run_key}'. Available: {list(cfg['runs'])}")
        for direction in directions:
            results.append(
                train_one(cfg, run_key, direction, args.dry_run, args.output_root)
            )

    logger.info("=" * 70)
    logger.info("ALL RUNS COMPLETE")
    for r in results:
        chrf = r["final_metrics"].get("eval_chrf")
        logger.info(f"  {r['run_key']:12s} {r['direction']:8s} chrF++={chrf}")


if __name__ == "__main__":
    main()
