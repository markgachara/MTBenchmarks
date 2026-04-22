"""
Transformer-based MT Model implementations (NLLB-200, M2M-100)
Handles model-specific language code formats and tokenizer setup.
"""
import logging
import time
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from .base import BaseModel, TranslationResult

logger = logging.getLogger(__name__)

# Language code mappings per model family
LANG_CODES = {
    "flores": {  # NLLB-200
        "eng": "eng_Latn",
        "kik": "kik_Latn",
    },
    "m2m": {  # M2M-100 — Kikuyu is NOT in its 100 supported languages.
        # Swahili (sw) is the closest related Bantu language available.
        "eng": "en",
        "kik": "sw",
    },
}


class TransformerMT(BaseModel):
    """
    Transformer encoder-decoder MT model.
    Works with: NLLB-200, M2M-100.
    """

    def load(self):
        """Load transformer model and tokenizer."""
        logger.info(f"Loading {self.name} ({self.model_id})...")
        start_time = time.time()

        try:
            torch_dtype = eval(self.dtype) if isinstance(self.dtype, str) else self.dtype

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )

            self._is_loaded = True
            self._load_time = time.time() - start_time
            logger.info(f"Loaded {self.name} in {self._load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            raise

    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> TranslationResult:
        """Translate texts using the transformer model."""
        if not self.is_loaded:
            self.load()

        if batch_size is None:
            batch_size = self.config.get("batch_size", 8)

        lang_fmt = self.config.get("lang_code_format", "flores")
        src_code = self._resolve_lang(source_lang, lang_fmt)
        tgt_code = self._resolve_lang(target_lang, lang_fmt)

        logger.info(f"Language codes: {source_lang} → {src_code}, {target_lang} → {tgt_code}")

        # Warn if using a fallback language code
        if lang_fmt == "m2m" and ("kik" in source_lang or "kik" in target_lang):
            logger.warning(
                "M2M-100 does not natively support Gĩkũyũ (kik). "
                "Using Swahili (sw) as closest Bantu language proxy. "
                "Scores will reflect cross-lingual transfer, not direct support."
            )

        # Set source language on tokenizer (required by NLLB and M2M)
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_code

        # Resolve target token id for forced_bos_token_id
        tgt_token_id = self._resolve_target_token_id(tgt_code)

        translations = []
        start_time = time.time()

        for i in tqdm(
            range(0, len(texts), batch_size), desc=f"Translating with {self.name}"
        ):
            batch = texts[i : i + batch_size]
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                gen_kwargs = {
                    "max_new_tokens": self.config.get("max_new_tokens", 256),
                    "num_beams": self.config.get("num_beams", 5),
                }
                if tgt_token_id is not None:
                    gen_kwargs["forced_bos_token_id"] = tgt_token_id

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translations.extend(decoded)

            except Exception as e:
                logger.warning(f"Error translating batch {i // batch_size}: {e}")
                translations.extend([""] * len(batch))

        inference_time = time.time() - start_time
        return TranslationResult(
            translations=translations,
            model_name=self.name,
            source_lang=source_lang,
            target_lang=target_lang,
            inference_time=inference_time,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_lang(code: str, fmt: str) -> str:
        """Map generic code (eng/kik) to model-specific code."""
        mapping = LANG_CODES.get(fmt, LANG_CODES["flores"])
        # Try direct lookup, then strip _Latn and retry
        if code in mapping:
            return mapping[code]
        stripped = code.replace("_Latn", "")
        return mapping.get(stripped, code)

    def _resolve_target_token_id(self, tgt_code: str) -> Optional[int]:
        """Resolve the forced_bos_token_id for the target language."""
        try:
            # Method 1: get_lang_id (official M2M-100 API per HF docs)
            if hasattr(self.tokenizer, "get_lang_id"):
                tid = self.tokenizer.get_lang_id(tgt_code)
                logger.info(f"Target token via get_lang_id: {tgt_code} → {tid}")
                return tid

            # Method 2: lang_code_to_id dict (works for NLLB/M2M)
            if hasattr(self.tokenizer, "lang_code_to_id"):
                tid = self.tokenizer.lang_code_to_id.get(tgt_code)
                if tid is not None:
                    logger.info(f"Target token via lang_code_to_id: {tgt_code} → {tid}")
                    return tid

            # Method 3: convert_tokens_to_ids (works for NLLB)
            tid = self.tokenizer.convert_tokens_to_ids(tgt_code)
            if tid != self.tokenizer.unk_token_id:
                logger.info(f"Target token via convert_tokens_to_ids: {tgt_code} → {tid}")
                return tid

            logger.warning(f"Could not resolve target token for {tgt_code}")
            return None
        except Exception as e:
            logger.warning(f"Error resolving target token for {tgt_code}: {e}")
            return None
