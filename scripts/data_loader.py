"""
Data loading and preprocessing for MT benchmarking.
Primary dataset: GAC 500-pair test set (selectpairs500.xlsx)
"""
import logging
import json
import hashlib
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column schemas across GAC releases. The July 2026 corrected release
# (data/JulyDataUpdate/) renamed the parallel columns, so loading is
# schema-agnostic rather than hardcoded to one layout.
COLUMN_SCHEMAS = [
    ("translatedText", "data_to_quality_check"),  # selectpairs500.xlsx (v1.0)
    ("English", "Gikuyu"),                        # JulyDataUpdate (v1.1)
]


def _build_non_printing_char_replacer(replace_by: str = " "):
    """Map Unicode category-C (control/format/unassigned) code points to a filler."""
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace(text: str) -> str:
        return text.translate(non_printable_map)

    return replace


_MPN = None
_REPLACE_NON_PRINTING = None

# Combining marks that appear in the GAC corpus but are not the Latin combining
# tilde, so NFC cannot compose them into ĩ/ũ and they tokenize to <unk>.
# U+0342 is the Greek perispomeni, entered by mistake in place of U+0303.
_COMBINING_MARK_REPAIRS = {
    "\u0342": "\u0303",  # COMBINING GREEK PERISPOMENI -> COMBINING TILDE
}


def _get_moses_normalizer():
    """Lazily build the Moses punctuation normalizer used by the NLLB pipeline."""
    global _MPN, _REPLACE_NON_PRINTING
    if _MPN is None:
        from sacremoses import MosesPunctNormalizer

        mpn = MosesPunctNormalizer(lang="en")
        mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
        _MPN = mpn
        _REPLACE_NON_PRINTING = _build_non_printing_char_replacer()
    return _MPN, _REPLACE_NON_PRINTING


# Excel stores unrepresentable code points as literal _xXXXX_ escapes, so the
# same corrupted character reaches us as a real control byte from the published
# CSV but as an escape sequence from the .xlsx. Decoding the escapes first makes
# both sources normalize identically.
_XLSX_ESCAPE = re.compile(r"_x([0-9A-Fa-f]{4})_")


def _decode_xlsx_escapes(text: str) -> str:
    return _XLSX_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _repair_mojibake(text: str) -> str:
    """
    Undo UTF-8 bytes that were decoded as Latin-1 (e.g. 'â€™' for '’').

    Only applied when the tell-tale 'Ã'/'â' marker is present, so correctly
    encoded text is left untouched.
    """
    if "Ã" not in text and "â" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_text(text: str, moses: bool = True) -> str:
    """
    Normalize text for NLLB-family models.

    Four independent problems are addressed:

    1. Unicode form. The GAC corpus mixes NFC and NFD encodings of the Gĩkũyũ
       diacritics (ĩ, ũ). SentencePiece always emits NFC, so an NFD reference
       is scored as a character mismatch by chrF++ even when the strings are
       visually identical. Normalizing both sides to NFC removes that
       artificial penalty.
    2. Punctuation. NLLB's training data was passed through Moses punctuation
       normalization plus non-printing character removal (the `stopes`
       pipeline). Skipping it leaves typographic apostrophes and en-dashes
       that tokenize to <unk>.
    3. Mis-entered combining marks (see ``_COMBINING_MARK_REPAIRS``).
    4. Excel escapes and Latin-1/UTF-8 mojibake, which otherwise make the
       .xlsx and published .csv releases disagree.
    """
    text = _decode_xlsx_escapes(text)
    text = _repair_mojibake(text)
    for wrong, right in _COMBINING_MARK_REPAIRS.items():
        text = text.replace(wrong, right)
    if moses:
        mpn, replace_non_printing = _get_moses_normalizer()
        text = mpn.normalize(text)
        text = replace_non_printing(text)
    # NFC last: Moses substitutions can reintroduce decomposed sequences
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def pair_hash(english: str, kikuyu: str) -> str:
    """
    Content-derived identifier for a parallel pair.

    Row position is not a safe key: the published CSV and the working .xlsx
    disagree on row content, and future re-exports may reorder. Hashing the
    normalized pair yields an identifier that joins across both formats.
    """
    payload = f"{normalize_text(english)}\u241f{normalize_text(kikuyu)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class MTDataLoader:
    """Load and preprocess MT datasets"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_columns(
        df: pd.DataFrame,
        english_col: Optional[str] = None,
        kikuyu_col: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Pick the English/Gĩkũyũ column pair, auto-detecting when not given."""
        if english_col and kikuyu_col:
            missing = [c for c in (english_col, kikuyu_col) if c not in df.columns]
            if missing:
                raise KeyError(f"Columns {missing} not found. Available: {list(df.columns)}")
            return english_col, kikuyu_col

        for eng, kik in COLUMN_SCHEMAS:
            if eng in df.columns and kik in df.columns:
                return eng, kik

        raise KeyError(
            f"No known GAC column schema in {list(df.columns)}. "
            f"Expected one of {COLUMN_SCHEMAS}."
        )

    def load_parallel_excel(
        self,
        path: str,
        english_col: Optional[str] = None,
        kikuyu_col: Optional[str] = None,
        deduplicate: bool = False,
        normalize: bool = True,
        return_ids: bool = False,
    ):
        """
        Load a parallel English/Gĩkũyũ Excel file (any GAC release schema).

        Drops rows where either side is missing or blank after stripping,
        which removes placeholder artefacts such as a lone backtick.

        Args:
            deduplicate: drop exact duplicate pairs (used for training splits).
            normalize: apply NFC + Moses normalization (see ``normalize_text``).
            return_ids: also return a stable identifier per pair.

        Returns:
            (english_texts, kikuyu_texts) or (english_texts, kikuyu_texts, ids)
        """
        logger.info(f"Loading parallel data from {path}...")
        df = pd.read_excel(path)
        eng_col, kik_col = self._resolve_columns(df, english_col, kikuyu_col)
        logger.info(f"Detected schema: english='{eng_col}', kikuyu='{kik_col}'")

        before = len(df)
        df = df.dropna(subset=[eng_col, kik_col])

        # Strip and drop blanks (e.g. whitespace-only or punctuation-only cells)
        eng = df[eng_col].astype(str).str.strip()
        kik = df[kik_col].astype(str).str.strip()
        keep = (eng.str.len() > 0) & (kik.str.len() > 0)
        eng, kik = eng[keep], kik[keep]

        dropped = before - len(eng)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with missing/blank translations")

        # Identifiers are content-derived (hash of the normalized pair) so a
        # translation joins back to its source pair regardless of file format
        # or row order. The published CSV and working .xlsx disagree on two
        # rows and may be re-exported, so row position is not a safe key.
        if normalize:
            eng = eng.map(normalize_text)
            kik = kik.map(normalize_text)

        ids = pd.Series(
            [pair_hash(e, k) for e, k in zip(eng, kik)], index=eng.index
        )

        if deduplicate:
            pairs = pd.DataFrame({"eng": eng, "kik": kik, "id": ids})
            n_before = len(pairs)
            pairs = pairs.drop_duplicates(subset=["eng", "kik"])
            if len(pairs) != n_before:
                logger.warning(f"Dropped {n_before - len(pairs)} duplicate pairs")
            eng, kik, ids = pairs["eng"], pairs["kik"], pairs["id"]

        english_texts = eng.tolist()
        kikuyu_texts = kik.tolist()
        logger.info(f"Loaded {len(english_texts)} parallel pairs from {Path(path).name}")

        if return_ids:
            return english_texts, kikuyu_texts, ids.tolist()
        return english_texts, kikuyu_texts

    def load_gac_test_set(
        self,
        path: str = "data/selectpairs500.xlsx",
        english_col: Optional[str] = None,
        kikuyu_col: Optional[str] = None,
        normalize: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        Load a GAC test set from Excel (schema auto-detected).

        Returns:
            Tuple of (english_texts, kikuyu_texts)
        """
        return self.load_parallel_excel(
            path, english_col, kikuyu_col, normalize=normalize
        )

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
        ids: Optional[List[str]] = None,
    ) -> Path:
        """Save model translations alongside sources and references."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{model_name.replace(' ', '_').lower()}_{direction.replace('->', '2')}.json"
        filepath = output_dir / filename

        if ids is None:
            ids = [f"idx:{i:04d}" for i in range(len(translations))]

        data = {
            "model": model_name,
            "direction": direction,
            "num_sentences": len(translations),
            "metadata": metadata or {},
            "sentences": [
                {"id": i, "source": s, "translation": t, "reference": r}
                for i, s, t, r in zip(ids, sources, translations, references)
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
