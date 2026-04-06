"""
Main translation orchestrator
"""
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

from scripts.models.base import BaseModel
from scripts.models.transformer_mt import TransformerMT
from scripts.models.llm_prompting import LLMViaPrompting
from scripts.models.lora_finetuned import LoRAFineTuned
from scripts.utils import clear_gpu_cache

logger = logging.getLogger(__name__)


class TranslationOrchestrator:
    """Manages model loading and translation workflow"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.translation_cache = {}
    
    def create_model(self, model_id: str, model_config: Dict) -> BaseModel:
        """
        Factory method to create appropriate model instance
        
        Args:
            model_id: Model identifier
            model_config: Model configuration dict
        
        Returns:
            Model instance
        """
        model_type = model_config.get('model_type')
        device_map = model_config.get('device_map', 'auto')
        
        if model_type == 'transformer_mt':
            return TransformerMT(model_config, device_map)
        elif model_type == 'llm_prompting':
            return LLMViaPrompting(model_config, device_map)
        elif model_type == 'lora_finetuned':
            return LoRAFineTuned(model_config, device_map)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def translate_batch(
        self,
        model_id: str,
        model_config: Dict,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        skip_on_low_memory: bool = False
    ) -> Dict:
        """
        Translate texts with specified model
        
        Args:
            model_id: Model identifier
            model_config: Model configuration
            texts: List of source texts
            source_lang: Source language code
            target_lang: Target language code
            skip_on_low_memory: If True, skip model if insufficient memory
        
        Returns:
            Dict with translations and metadata
        """
        cache_key = f"{model_id}_{source_lang}_{target_lang}"
        
        # Check cache first
        if cache_key in self.translation_cache:
            logger.info(f"Using cached translations for {model_id}")
            return self.translation_cache[cache_key]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Translating with {model_config.get('name', model_id)}")
        logger.info(f"Direction: {source_lang} → {target_lang}")
        logger.info(f"Sentences: {len(texts)}")
        logger.info(f"{'='*60}")
        
        try:
            # Create or reuse model
            if model_id not in self.models:
                logger.info(f"Loading model: {model_id}")
                model = self.create_model(model_id, model_config)
                model.load()
                self.models[model_id] = model
            else:
                model = self.models[model_id]
            
            # Translate
            result = model.translate(
                texts,
                source_lang,
                target_lang
            )
            
            # Cache result
            result_dict = {
                'model_id': model_id,
                'model_name': model_config.get('name', model_id),
                'translations': result.translations,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'speed': result.speed,
                'num_tokens': result.num_tokens,
                'inference_time': result.inference_time,
                'valid': len(result.translations) == len(texts)
            }
            
            self.translation_cache[cache_key] = result_dict
            
            # Log summary
            logger.info(f"Completed translation")
            logger.info(f"  Speed: {result.speed:.1f} tokens/sec")
            logger.info(f"  Time: {result.inference_time:.2f}s")
            
            # Unload to free memory for next model
            if model_id != list(self.models.keys())[-1]:  # Don't unload last model yet
                model.unload()
                del self.models[model_id]
                clear_gpu_cache()
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Translation failed for {model_id}: {e}")
            return {
                'model_id': model_id,
                'model_name': model_config.get('name', model_id),
                'translations': [""] * len(texts),
                'error': str(e),
                'valid': False
            }
    
    def cleanup(self):
        """Unload all models"""
        for model_id, model in self.models.items():
            if model.is_loaded:
                model.unload()
        self.models.clear()
        clear_gpu_cache()
        logger.info("Cleaned up all models")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
