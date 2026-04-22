"""
Data loading and preprocessing for MT benchmarking.
Primary dataset: GAC 500-pair test set (selectpairs500.xlsx)
"""
import logging
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MTDataLoader:
    """Load and preprocess MT datasets"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_gac_test_set(
        self,
        path: str = "data/selectpairs500.xlsx",
        english_col: str = "translatedText",
        kikuyu_col: str = "data_to_quality_check",
    ) -> Tuple[List[str], List[str]]:
        """
        Load the GAC 500-pair test set from Excel.

        Returns:
            Tuple of (english_texts, kikuyu_texts)
        """
        logger.info(f"Loading GAC test set from {path}...")
        df = pd.read_excel(path)

        # Drop rows with missing translations
        before = len(df)
        df = df.dropna(subset=[english_col, kikuyu_col])
        after = len(df)
        if before != after:
            logger.warning(f"Dropped {before - after} rows with missing translations")

        english_texts = df[english_col].astype(str).tolist()
        kikuyu_texts = df[kikuyu_col].astype(str).tolist()

        logger.info(f"Loaded {len(english_texts)} parallel pairs from GAC test set")
        return english_texts, kikuyu_texts

    def create_stratified_sample(
        self,
        english_texts: List[str],
        kikuyu_texts: List[str],
        sample_size: int = 50,
        seed: int = 42,
    ) -> Tuple[List[int], List[str], List[str]]:
        """
        Create a stratified sample for qualitative assessment.
        Stratifies by sentence length to cover range of complexity.

        Returns:
            Tuple of (indices, sampled_english, sampled_kikuyu)
        """
        rng = random.Random(seed)

        # Stratify by English sentence length (short/medium/long)
        lengths = [len(s.split()) for s in english_texts]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        n = len(sorted_indices)
        third = n // 3
        short = sorted_indices[:third]
        medium = sorted_indices[third : 2 * third]
        long = sorted_indices[2 * third :]

        per_stratum = sample_size // 3
        remainder = sample_size - 3 * per_stratum

        sampled = []
        sampled += rng.sample(short, min(per_stratum, len(short)))
        sampled += rng.sample(medium, min(per_stratum, len(medium)))
        sampled += rng.sample(long, min(per_stratum + remainder, len(long)))

        sampled.sort()

        sampled_eng = [english_texts[i] for i in sampled]
        sampled_kik = [kikuyu_texts[i] for i in sampled]

        logger.info(
            f"Created stratified sample of {len(sampled)} pairs "
            f"(short={per_stratum}, medium={per_stratum}, long={per_stratum + remainder})"
        )
        return sampled, sampled_eng, sampled_kik

    def save_translations_json(
        self,
        translations: List[str],
        sources: List[str],
        references: List[str],
        model_name: str,
        direction: str,
        output_dir: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save model translations alongside sources and references."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{model_name.replace(' ', '_').lower()}_{direction.replace('->', '2')}.json"
        filepath = output_dir / filename

        data = {
            "model": model_name,
            "direction": direction,
            "num_sentences": len(translations),
            "metadata": metadata or {},
            "sentences": [
                {"source": s, "translation": t, "reference": r}
                for s, t, r in zip(sources, translations, references)
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved translations to {filepath}")
        return filepath


def prepare_translation_pairs(
    eng_texts: List[str], kik_texts: List[str]
) -> Tuple[Dict, Dict]:
    """
    Prepare translation pairs for both directions.

    Returns:
        Tuple of (eng2kik_dict, kik2eng_dict)
    """
    eng2kik = {
        "source": eng_texts,
        "target": kik_texts,
        "direction": "eng->kik",
    }
    kik2eng = {
        "source": kik_texts,
        "target": eng_texts,
        "direction": "kik->eng",
    }
    return eng2kik, kik2eng
