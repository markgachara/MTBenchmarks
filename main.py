#!/usr/bin/env python3
"""
Main entry point for MT benchmarking pipeline
Orchestrates full workflow: load data, translate, evaluate, report
"""
import logging
import argparse
from typing import Dict, List, Optional
from pathlib import Path

from scripts.utils import (
    load_config,
    get_hardware_info,
    print_hardware_info,
    check_model_memory_feasibility,
    setup_directories,
    recommend_dtype_and_batch_size
)
from scripts.data_loader import MTDataLoader, prepare_translation_pairs
from scripts.translator import TranslationOrchestrator
from scripts.evaluator import MTEvaluator
from scripts.reporter import MTReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main benchmarking pipeline"""
    
    # Setup
    setup_directories()
    logger.info("MT Benchmarking Pipeline v1.0")
    
    # Load configurations
    logger.info("Loading configurations...")
    models_config = load_config(args.models_config)
    datasets_config = load_config(args.datasets_config)
    
    # Detect hardware
    logger.info("Detecting hardware...")
    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)
    
    # Get recommendations
    recommended_dtype, recommended_batch_size = recommend_dtype_and_batch_size(hardware_info)
    logger.info(f"Recommended settings: dtype={recommended_dtype}, batch_size={recommended_batch_size}")
    
    # Initialize components
    data_loader = MTDataLoader(cache_dir=args.data_dir)
    evaluator = MTEvaluator()
    reporter = MTReporter(output_dir=args.output_dir)
    
    # Load datasets
    logger.info("Loading datasets...")
    try:
        # Load primary dataset (FLORES-200)
        flores_source, flores_target = data_loader.load_flores_200(
            source_lang="kik_Latn",
            target_lang="eng_Latn",
            split=args.eval_split
        )
        
        # Prepare bidirectional pairs
        eng2kik, kik2eng = prepare_translation_pairs(flores_target, flores_source)
        
        translation_pairs = {
            'flores_eng2kik': eng2kik,
            'flores_kik2eng': kik2eng
        }
        
        # Optionally load validation dataset
        if args.include_validation:
            try:
                val_source, val_target = data_loader.load_kikuyu_kb()
                val_eng2kik, val_kik2eng = prepare_translation_pairs(val_target, val_source)
                translation_pairs.update({
                    'kikuyu_kb_eng2kik': val_eng2kik,
                    'kikuyu_kb_kik2eng': val_kik2eng
                })
                logger.info("Loaded validation dataset (KevinKibe)")
            except Exception as e:
                logger.warning(f"Could not load validation dataset: {e}")
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return
    
    # Select models to benchmark
    logger.info("Preparing models...")
    selected_models = {}
    models_to_eval = models_config['models']
    
    for model_id, model_cfg in models_to_eval.items():
        if model_id in args.models or not args.models:
            # Check memory feasibility
            feasible, msg = check_model_memory_feasibility(model_cfg, hardware_info)
            if feasible or not model_cfg.get('skip_on_low_memory', False):
                selected_models[model_id] = model_cfg
                logger.info(f"  {model_cfg['name']}: {msg}")
            else:
                logger.warning(f"  Skipping {model_cfg['name']}: {msg}")
    
    # Run translation & evaluation
    logger.info("\n" + "="*60)
    logger.info("STARTING TRANSLATIONS")
    logger.info("="*60)
    
    all_results = {}
    
    with TranslationOrchestrator(models_config) as orchestrator:
        for dataset_name, pair_data in translation_pairs.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            source_texts = pair_data['source']
            target_texts = pair_data['target']
            direction = pair_data['direction']
            
            dataset_results = {}
            
            for model_id, model_cfg in selected_models.items():
                # Translate
                trans_result = orchestrator.translate_batch(
                    model_id,
                    model_cfg,
                    source_texts,
                    source_lang=direction.split('->')[0].strip(),
                    target_lang=direction.split('->')[1].strip()
                )
                
                # Evaluate
                if trans_result.get('valid'):
                    metrics = evaluator.compute_all_metrics(
                        trans_result['translations'],
                        target_texts,
                        language_pair=direction.replace('->','-')
                    )
                    metrics['speed'] = trans_result.get('speed', 0)
                    metrics['num_sentences'] = len(source_texts)
                    metrics['inference_time'] = trans_result.get('inference_time', 0)
                else:
                    metrics = {
                        'bleu': 0.0,
                        'chrf': 0.0,
                        'bertscore': None,
                        'valid': False,
                        'notes': trans_result.get('error', 'Unknown error')
                    }
                
                dataset_results[model_id] = metrics
            
            # Store results for this dataset
            all_results[dataset_name] = dataset_results
    
    # Aggregate and report
    logger.info("\n" + "="*60)
    logger.info("AGGREGATING RESULTS")
    logger.info("="*60)
    
    # Restructure results for reporter
    reporter_results = {}
    for dataset_name, models_metrics in all_results.items():
        for model_id, metrics in models_metrics.items():
            model_name = selected_models[model_id]['name']
            
            if model_name not in reporter_results:
                reporter_results[model_name] = {}
            
            # Extract direction from dataset_name
            if 'eng2kik' in dataset_name:
                direction = 'eng->kik'
            elif 'kik2eng' in dataset_name:
                direction = 'kik->eng'
            else:
                direction = 'unknown'
            
            if direction not in reporter_results[model_name]:
                reporter_results[model_name][direction] = {}
            
            reporter_results[model_name][direction][dataset_name] = metrics
    
    # Create results dataframe
    df = reporter.create_results_dataframe(reporter_results, hardware_info)
    
    # Save results
    csv_path = reporter.save_csv(df)
    json_path = reporter.save_json(reporter_results)
    
    # Generate report
    markdown_report = reporter.generate_markdown_report(df, hardware_info)
    md_path = reporter.save_markdown_report(markdown_report)
    
    # Print summary
    reporter.print_summary(df)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  Markdown: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Machine Translation Benchmarking Pipeline for Gĩkũyũ"
    )
    
    parser.add_argument(
        "--models-config",
        default="config/models.yaml",
        help="Path to models configuration file"
    )
    parser.add_argument(
        "--datasets-config",
        default="config/datasets.yaml",
        help="Path to datasets configuration file"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Specific models to benchmark (if empty, benchmarks all)"
    )
    parser.add_argument(
        "--data-dir",
        default="./data/cache",
        help="Directory for cached data"
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory for output results"
    )
    parser.add_argument(
        "--eval-split",
        default="devtest",
        choices=["dev", "devtest"],
        help="Dataset split to use for evaluation"
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        default=True,
        help="Include validation dataset (KevinKibe)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation dataset"
    )
    
    args = parser.parse_args()
    if args.skip_validation:
        args.include_validation = False
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        exit(1)
