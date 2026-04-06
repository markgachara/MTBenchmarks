"""
Data loading and preprocessing for MT benchmarking
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


class MTDataLoader:
    """Load and preprocess MT datasets"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_flores_200(
        self,
        source_lang: str = "kik_Latn",
        target_lang: str = "eng_Latn",
        split: str = "devtest"
    ) -> Tuple[List[str], List[str]]:
        """
        Load FLORES-200 dataset
        
        Args:
            source_lang: Source language code (e.g., 'kik_Latn')
            target_lang: Target language code
            split: Dataset split ('dev' or 'devtest')
        
        Returns:
            Tuple of (source_texts, target_texts)
        """
        logger.info(f"Loading FLORES-200 ({source_lang}-{target_lang}, split={split})...")
        
        try:
            # Load from HF datasets
            dataset = load_dataset(
                "facebook/flores",
                f"{source_lang}-{target_lang}",
                split=split,
                cache_dir=str(self.cache_dir)
            )
            
            # Extract texts
            source_texts = dataset['sentence_' + source_lang]
            target_texts = dataset['sentence_' + target_lang]
            
            logger.info(f"Loaded {len(source_texts)} parallel sentences from FLORES-200")
            return source_texts, target_texts
            
        except Exception as e:
            logger.error(f"Failed to load FLORES-200: {e}")
            raise
    
    def load_kikuyu_kb(self) -> Tuple[List[str], List[str]]:
        """
        Load KevinKibe Kikuyu-English sentence pairs
        
        Returns:
            Tuple of (kikuyu_texts, english_texts)
        """
        logger.info("Loading KevinKibe Kikuyu sentence pairs...")
        
        try:
            dataset = load_dataset(
                "KevinKibe/kikuyu-sentence-pairs",
                split="train",
                cache_dir=str(self.cache_dir)
            )
            
            # Extract texts - need to inspect actual structure
            # Assuming typical format with 'kik' and 'eng' columns
            if 'kik' in dataset.column_names and 'eng' in dataset.column_names:
                kikuyu_texts = dataset['kik']
                english_texts = dataset['eng']
            elif 'text' in dataset.column_names:
                # Fallback: might be single-column format
                logger.warning("Unexpected dataset format; attempting to parse")
                kikuyu_texts = []
                english_texts = []
                for item in dataset:
                    # Try to split or extract differently
                    if isinstance(item['text'], str):
                        # Assume tab or pipe-separated
                        parts = item['text'].split('\t') or item['text'].split('|')
                        if len(parts) >= 2:
                            kikuyu_texts.append(parts[0])
                            english_texts.append(parts[1])
            else:
                raise ValueError(f"Unknown dataset schema: {dataset.column_names}")
            
            logger.info(f"Loaded {len(kikuyu_texts)} parallel sentences from KevinKibe")
            return kikuyu_texts, english_texts
            
        except Exception as e:
            logger.error(f"Failed to load KevinKibe dataset: {e}")
            raise
    
    def create_parallel_jsonl(
        self,
        source_texts: List[str],
        target_texts: List[str],
        output_path: str
    ):
        """
        Save parallel texts as JSONL for easy streaming
        
        Args:
            source_texts: List of source texts
            target_texts: List of target texts
            output_path: Path to save JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for src, tgt in zip(source_texts, target_texts):
                record = {
                    'source': src,
                    'target': tgt
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(source_texts)} pairs to {output_path}")
    
    def load_parallel_jsonl(self, path: str) -> Tuple[List[str], List[str]]:
        """Load parallel texts from JSONL"""
        path = Path(path)
        source_texts = []
        target_texts = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                source_texts.append(record['source'])
                target_texts.append(record['target'])
        
        logger.info(f"Loaded {len(source_texts)} pairs from {path}")
        return source_texts, target_texts
    
    def create_validation_sample(
        self,
        source_texts: List[str],
        target_texts: List[str],
        sample_size: int = 50
    ) -> Tuple[List[str], List[str]]:
        """
        Create small validation sample
        
        Args:
            source_texts: Full source texts
            target_texts: Full target texts
            sample_size: Number of samples
        
        Returns:
            Tuple of sampled (source, target)
        """
        import random
        indices = random.sample(range(len(source_texts)), min(sample_size, len(source_texts)))
        
        sampled_source = [source_texts[i] for i in indices]
        sampled_target = [target_texts[i] for i in indices]
        
        logger.info(f"Created validation sample with {len(sampled_source)} sentences")
        return sampled_source, sampled_target


def prepare_translation_pairs(
    eng_texts: List[str],
    kik_texts: List[str]
) -> Tuple[Dict, Dict]:
    """
    Prepare translation pairs for both directions
    
    Returns:
        Dicts with 'eng2kik' and 'kik2eng' keys
    """
    eng2kik = {
        'source': eng_texts,
        'target': kik_texts,
        'direction': 'eng->kik'
    }
    
    kik2eng = {
        'source': kik_texts,
        'target': eng_texts,
        'direction': 'kik->eng'
    }
    
    return eng2kik, kik2eng
