"""
Main translation orchestrator.
Sequential model loading with GPU cache clearing between models.
Supports both standard inference and prompting modes (zero-shot + few-shot).
"""
import logging
import time
from typing import Dict, List, Optional

from scripts.models.base import BaseModel, TranslationResult
from scripts.models.transformer_mt import TransformerMT
from scripts.models.llm_prompting import LLMViaPrompting
from scripts.models.lora_finetuned import LoRAFineTuned
from scripts.utils import clear_gpu_cache

logger = logging.getLogger(__name__)


class TranslationOrchestrator:
    """Manages model loading and translation workflow."""

    def __init__(self, config: Dict):
        self.config = config
        self.current_model: Optional[BaseModel] = None
        self.current_model_id: Optional[str] = None
        self.translation_cache: Dict = {}

    def create_model(self, model_id: str, model_config: Dict) -> BaseModel:
        """Factory method to create model instance."""
        model_type = model_config.get("model_type")
        device_map = model_config.get("device_map", "auto")

        if model_type == "transformer_mt":
            return TransformerMT(model_config, device_map)
        elif model_type == "llm_prompting":
            return LLMViaPrompting(model_config, device_map)
        elif model_type == "lora_finetuned":
            return LoRAFineTuned(model_config, device_map)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _ensure_model_loaded(self, model_id: str, model_config: Dict):
        """Load model, unloading any previously loaded model first."""
        if self.current_model_id == model_id and self.current_model and self.current_model.is_loaded:
            return

        # Always unload previous model and clear GPU
        if self.current_model:
            logger.info(f"Unloading {self.current_model.name} to free memory")
            self.current_model.unload()
        self.current_model = None
        self.current_model_id = None

        # Aggressive GPU cleanup
        import gc
        gc.collect()
        clear_gpu_cache()

        # Load new model
        model = self.create_model(model_id, model_config)
        model.load()
        self.current_model = model
        self.current_model_id = model_id

    def translate_batch(
        self,
        model_id: str,
        model_config: Dict,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        prompting_mode: str = "zero_shot",
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Translate texts with specified model and prompting mode.

        Returns dict with translations, timing, and metadata.
        """
        cache_key = f"{model_id}_{source_lang}_{target_lang}_{prompting_mode}"
        if cache_key in self.translation_cache:
            logger.info(f"Using cached translations for {cache_key}")
            return self.translation_cache[cache_key]

        mode_label = "3-shot" if prompting_mode == "few_shot" else "zero-shot"
        model_name = model_config.get("name", model_id)
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name} | Mode: {mode_label}")
        logger.info(f"Direction: {source_lang} → {target_lang} | Sentences: {len(texts)}")
        logger.info(f"{'='*60}")

        try:
            self._ensure_model_loaded(model_id, model_config)

            # Build kwargs for translate
            translate_kwargs = {}
            if model_config.get("model_type") == "llm_prompting":
                translate_kwargs["prompting_mode"] = prompting_mode
                translate_kwargs["few_shot_examples"] = few_shot_examples

            result = self.current_model.translate(
                texts, source_lang, target_lang, **translate_kwargs
            )

            result_dict = {
                "model_id": model_id,
                "model_name": model_name,
                "prompting_mode": prompting_mode,
                "translations": result.translations,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "speed": result.speed,
                "num_tokens": result.num_tokens,
                "inference_time": result.inference_time,
                "valid": len(result.translations) == len(texts),
            }

            self.translation_cache[cache_key] = result_dict

            logger.info(f"Completed: {result.speed:.1f} tokens/sec, {result.inference_time:.2f}s")
            return result_dict

        except Exception as e:
            logger.error(f"Translation failed for {model_id}: {e}")
            # Clean up after failure to prevent cascading OOM
            if self.current_model:
                self.current_model.unload()
            self.current_model = None
            self.current_model_id = None
            import gc
            gc.collect()
            clear_gpu_cache()
            return {
                "model_id": model_id,
                "model_name": model_name,
                "prompting_mode": prompting_mode,
                "translations": [""] * len(texts),
                "error": str(e),
                "valid": False,
            }

    def cleanup(self):
        """Unload current model."""
        if self.current_model and self.current_model.is_loaded:
            self.current_model.unload()
        self.current_model = None
        self.current_model_id = None
        clear_gpu_cache()
        logger.info("Cleaned up all models")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
