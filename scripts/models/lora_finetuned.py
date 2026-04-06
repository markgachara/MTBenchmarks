"""
LoRA Fine-tuned Model (Kikuyu-Translator on Gemma-3)
"""
import logging
import time
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from .base import BaseModel, TranslationResult

logger = logging.getLogger(__name__)


class LoRAFineTuned(BaseModel):
    """
    LoRA fine-tuned model for bidirectional translation
    Works with: InterstellarCG/kikuyu-translator-final
    """
    
    def load(self):
        """Load base model and LoRA weights"""
        logger.info(f"Loading {self.name} ({self.model_id})...")
        start_time = time.time()
        
        try:
            # Determine dtype
            torch_dtype = eval(self.dtype) if isinstance(self.dtype, str) else self.dtype
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model with LoRA weights
            # The model_id typically includes the LoRA weights already
            self.model = AutoModelForCausalLM.from_pretrained(
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
        Translate texts using LoRA-fine-tuned model
        
        Args:
            texts: List of source texts
            source_lang: Source language ('eng' or 'kik')
            target_lang: Target language ('eng' or 'kik')
            batch_size: Batch size
        """
        if not self.is_loaded:
            self.load()
        
        if batch_size is None:
            batch_size = self.config.get('batch_size', 4)
        
        translations = []
        start_time = time.time()
        
        for i, text in enumerate(tqdm(texts, desc=f"Translating with {self.name}")):
            try:
                # Build prompt for the fine-tuned model
                prompt = self._build_translation_prompt(text, source_lang, target_lang)
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.get('max_new_tokens', 256),
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=False
                    )
                
                # Decode
                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                translation = self._extract_translation_from_output(full_output, prompt)
                translations.append(translation)
                
            except Exception as e:
                logger.warning(f"Error translating text {i}: {e}")
                translations.append("")
        
        inference_time = time.time() - start_time
        return TranslationResult(
            translations=translations,
            model_name=self.name,
            source_lang=source_lang,
            target_lang=target_lang,
            inference_time=inference_time
        )
    
    def _build_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Build instruction-based prompt for Kikuyu translator"""
        # This model is specifically trained for English ↔ Kikuyu
        if 'eng' in source_lang.lower():
            # English to Kikuyu
            prompt = f"Translate to Kikuyu: {text}\n\nTranslation:"
        elif 'kik' in source_lang.lower():
            # Kikuyu to English
            prompt = f"Translate to English: {text}\n\nTranslation:"
        else:
            # Fallback
            prompt = f"Translate: {text}\n\nTranslation:"
        
        return prompt
    
    def _extract_translation_from_output(self, full_output: str, prompt: str) -> str:
        """Extract translation from model output"""
        # Remove prompt from output
        if prompt in full_output:
            translation = full_output[len(prompt):].strip()
        else:
            translation = full_output.strip()
        
        # Clean up (take first line, remove artifacts)
        translation = translation.split('\n')[0].strip()
        
        # Remove common artifacts
        artifacts = ['<|end|>', '<eos>', '[END]']
        for artifact in artifacts:
            translation = translation.replace(artifact, '').strip()
        
from tqdm import tqdm
