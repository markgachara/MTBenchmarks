"""
Transformer-based MT Model implementations (NLLB, M2M-100, SeamlessM4T)
"""
import logging
import time
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from tqdm import tqdm

from .base import BaseModel, TranslationResult

logger = logging.getLogger(__name__)


class TransformerMT(BaseModel):
    """
    Generic Transformer encoder-decoder MT model
    Works with: NLLB, M2M-100, SeamlessM4T
    """
    
    def load(self):
        """Load transformer model and tokenizer"""
        logger.info(f"Loading {self.name} ({self.model_id})...")
        start_time = time.time()
        
        try:
            # Determine dtype
            torch_dtype = eval(self.dtype) if isinstance(self.dtype, str) else self.dtype
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
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
        **kwargs
    ) -> List[str]:
        """
        Translate texts using transformer model
        
        Args:
            texts: List of source texts
            source_lang: Source language code (e.g., 'eng_Latn', 'eng')
            target_lang: Target language code (e.g., 'kik_Latn', 'kik')
            batch_size: Batch size for inference
        """
        if not self.is_loaded:
            self.load()
        
        if batch_size is None:
            batch_size = self.config.get('batch_size', 8)
        
        translations = []
        start_time = time.time()
        
        # Normalize language codes based on model type
        src_lang = self._normalize_lang_code(source_lang, 'source')
        tgt_lang = self._normalize_lang_code(target_lang, 'target')
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating with {self.name}"):
            batch = texts[i:i+batch_size]
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to same device as model
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Set language tokens for generation
                if hasattr(self.model.config, 'forced_bos_token_id'):
                    # Model supports forced tokens (NLLB, M2M, etc.)
                    try:
                        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
                        if tgt_token_id == self.tokenizer.unk_token_id:
                            # Try alternative format
                            tgt_token_id = self.tokenizer.lang_code_to_id.get(tgt_lang)
                        if tgt_token_id is not None:
                            inputs["forced_bos_token_id"] = tgt_token_id
                    except:
                        pass  # Continue without forced token
                
                # Generate translations
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.get('max_new_tokens', 256),
                        num_beams=self.config.get('num_beams', 5),
                    )
                
                # Decode
                batch_translations = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                
                translations.extend(batch_translations)
                
            except Exception as e:
                logger.warning(f"Error translating batch {i//batch_size}: {e}")
                # Add empty translations as placeholders
                translations.extend([""] * len(batch))
        
        inference_time = time.time() - start_time
        return TranslationResult(
            translations=translations,
            model_name=self.name,
            source_lang=source_lang,
            target_lang=target_lang,
            inference_time=inference_time
        )
    
    def _normalize_lang_code(self, lang_code: str, position: str) -> str:
        """
        Normalize language code to format expected by specific model
        NLLB/M2M use 'eng_Latn', 'kik_Latn' format
        """
        # Basic normalization
        lang_map = {
            'eng': 'eng_Latn',
            'eng_Latn': 'eng_Latn',
            'kik': 'kik_Latn',
            'kik_Latn': 'kik_Latn',
        }
        return lang_map.get(lang_code, lang_code)
