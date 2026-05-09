#!/usr/bin/env python3
"""
Gĩkũyũ MT Benchmarking Pipeline

Evaluates 8 MT model configurations on the GAC 500-pair test set:
NLLB-200 (600M, 1.3B, 3.3B), M2M-100, Llama 3.1 8B Instruct
(zero-shot + 3-shot), Gemma 3 4B Instruct (zero-shot + 3-shot).

Produces Tables 4 and 5 from the paper plus a qualitative-assessment
template. Companion analysis and the deep-dive plots live in
``analysis.ipynb``.

Usage:
    python main.py                                      # Full benchmark
    python main.py --dry-run                            # 10-sentence test
    python main.py --models nllb_200 m2m_100            # Specific models
    python main.py --skip-bertscore --skip-africomet    # Cheap metrics only
"""
import logging
import argparse
from typing import Dict, List, Optional

from scripts.utils import (
    load_config,
    get_hardware_info,
    print_hardware_info,
    check_model_memory_feasibility,
    setup_directories,
)
from scripts.data_loader import MTDataLoader, prepare_translation_pairs
from scripts.translator import TranslationOrchestrator
from scripts.evaluator import MTEvaluator
from scripts.reporter import MTReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Gĩkũyũ MT Benchmarking")
    parser.add_argument(
        "--models-config",
        default="config/models.yaml",
        help="Path to models config",
    )
    parser.add_argument(
        "--data-file",
        default="data/selectpairs500.xlsx",
        help="Path to GAC test set Excel",
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Specific model IDs to evaluate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test with 10 sentences only",
    )
    parser.add_argument(
        "--skip-africomet",
        action="store_true",
        help="Skip AfriCOMET-MTL computation",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore computation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_directories()

    logger.info("=" * 60)
    logger.info("Gĩkũyũ MT Benchmarking Pipeline")
    logger.info("=" * 60)

    # ── 1. Load configs ──────────────────────────────────────────
    models_config = load_config(args.models_config)

    # ── 2. Hardware detection ────────────────────────────────────
    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)

    # ── 3. Load GAC test set ─────────────────────────────────────
    loader = MTDataLoader()
    english_texts, kikuyu_texts = loader.load_gac_test_set(args.data_file)

    if args.dry_run:
        logger.info("DRY RUN: using first 10 sentences only")
        english_texts = english_texts[:10]
        kikuyu_texts = kikuyu_texts[:10]

    eng2kik, kik2eng = prepare_translation_pairs(english_texts, kikuyu_texts)

    # ── 4. Create stratified sample for qualitative assessment ───
    qual_indices, qual_eng, qual_kik = loader.create_stratified_sample(
        english_texts, kikuyu_texts, sample_size=50
    )

    # ── 5. Select models ────────────────────────────────────────
    available_models = models_config["models"]
    selected = {}
    for mid, mcfg in available_models.items():
        if args.models and mid not in args.models:
            continue
        feasible, msg = check_model_memory_feasibility(mcfg, hardware_info)
        if feasible or not mcfg.get("skip_on_low_memory", False):
            selected[mid] = mcfg
            logger.info(f"  ✓ {mcfg['name']}: {msg}")
        else:
            logger.warning(f"  ✗ Skipping {mcfg['name']}: {msg}")

    if not selected:
        logger.error("No models selected for evaluation. Exiting.")
        return

    # Few-shot examples from config
    few_shot_cfg = models_config.get("few_shot_examples", {})

    # ── 6. Initialize evaluator & reporter ───────────────────────
    evaluator = MTEvaluator()
    reporter = MTReporter(output_dir=args.output_dir)

    # ── 7. Compute human reference baseline ──────────────────────
    logger.info("Computing human reference baselines...")
    ref_baseline_eng2kik = evaluator.compute_reference_baseline(
        kikuyu_texts, "eng->kik"
    )
    ref_baseline_kik2eng = evaluator.compute_reference_baseline(
        english_texts, "kik->eng"
    )

    # ── 8. Run translations & evaluation ─────────────────────────
    directions = {
        "eng->kik": {
            "pair": eng2kik,
            "few_shot_key": "eng_to_kik",
            "ref_baseline": ref_baseline_eng2kik,
        },
        "kik->eng": {
            "pair": kik2eng,
            "few_shot_key": "kik_to_eng",
            "ref_baseline": ref_baseline_kik2eng,
        },
    }

    # Collect all results: {direction: {display_name: metrics}}
    all_results = {}
    # Collect translations for qualitative template: {display_name: {direction: [translations]}}
    all_translations = {}

    with TranslationOrchestrator(models_config) as orchestrator:
        for direction, dinfo in directions.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"DIRECTION: {direction}")
            logger.info(f"{'='*60}")

            pair = dinfo["pair"]
            source_texts = pair["source"]
            target_texts = pair["target"]
            dir_results = {}

            for model_id, model_cfg in selected.items():
                model_type = model_cfg.get("model_type")
                prompting_modes = model_cfg.get("prompting_modes", [None])

                for pmode in prompting_modes:
                    # Build display name
                    if pmode and pmode != "zero_shot":
                        display_name = f"{model_cfg['name']} (3-shot)"
                    elif pmode == "zero_shot":
                        display_name = f"{model_cfg['name']} (zero-shot)"
                    else:
                        display_name = model_cfg["name"]

                    # Get few-shot examples if needed
                    fs_examples = None
                    if pmode == "few_shot":
                        fs_examples = few_shot_cfg.get(dinfo["few_shot_key"], [])

                    # Translate
                    trans = orchestrator.translate_batch(
                        model_id,
                        model_cfg,
                        source_texts,
                        source_lang=direction.split("->")[0],
                        target_lang=direction.split("->")[1],
                        prompting_mode=pmode or "zero_shot",
                        few_shot_examples=fs_examples,
                    )

                    if not trans.get("valid"):
                        logger.warning(
                            f"Invalid translations from {display_name}: {trans.get('error')}"
                        )

                    # Evaluate (honour --skip-bertscore / --skip-africomet)
                    metrics = evaluator.compute_all_metrics(
                        predictions=trans["translations"],
                        references=target_texts,
                        sources=source_texts,
                        direction=direction,
                        skip_bertscore=args.skip_bertscore,
                        skip_africomet=args.skip_africomet,
                    )
                    metrics["speed"] = trans.get("speed", 0)
                    metrics["inference_time"] = trans.get("inference_time", 0)
                    # Release evaluator GPU memory so next model can load
                    evaluator.release_gpu()

                    dir_results[display_name] = metrics

                    # Store translations
                    if display_name not in all_translations:
                        all_translations[display_name] = {}
                    all_translations[display_name][direction] = trans["translations"]

                    # Save individual translation file
                    loader.save_translations_json(
                        translations=trans["translations"],
                        sources=source_texts,
                        references=target_texts,
                        model_name=display_name,
                        direction=direction,
                        output_dir=str(reporter.run_dir / "translations"),
                        metadata={
                            "prompting_mode": pmode,
                            "speed": trans.get("speed"),
                            "inference_time": trans.get("inference_time"),
                        },
                    )

            all_results[direction] = dir_results

    # ── 9. Generate result tables ────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 60)

    eng2kik_df = reporter.create_results_table(
        all_results.get("eng->kik", {}),
        "eng->kik",
        human_ref_metrics=ref_baseline_eng2kik,
    )
    kik2eng_df = reporter.create_results_table(
        all_results.get("kik->eng", {}),
        "kik->eng",
        human_ref_metrics=ref_baseline_kik2eng,
    )

    # Save CSVs and markdown tables
    for direction, df in [("eng->kik", eng2kik_df), ("kik->eng", kik2eng_df)]:
        reporter.save_results(df, direction, fmt="csv")
        reporter.save_results(df, direction, fmt="markdown")

    # Full metrics JSON
    reporter.save_full_metrics_json(all_results)

    # Markdown report
    reporter.generate_markdown_report(eng2kik_df, kik2eng_df, hardware_info)

    # ── 10. Generate qualitative assessment template ─────────────
    reporter.generate_qualitative_template(
        indices=qual_indices,
        english_texts=english_texts,
        kikuyu_texts=kikuyu_texts,
        model_translations=all_translations,
    )

    # ── 11. Print summary ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    print("\n--- Table 4: English → Gĩkũyũ ---")
    print(reporter._format_for_markdown(eng2kik_df).to_string(index=False))

    print("\n--- Table 5: Gĩkũyũ → English ---")
    print(reporter._format_for_markdown(kik2eng_df).to_string(index=False))

    logger.info(f"\nAll outputs saved to: {reporter.run_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
