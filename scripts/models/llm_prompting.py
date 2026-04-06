"""
LLM-based MT via Prompting (Mistral models)
"""
import logging
import time
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .base import BaseModel, TranslationResult

logger = logging.getLogger(__name__)


class LLMViaPrompting(BaseModel):
    """
    General-purpose LLM for MT via instruction prompting
    Works with: Mistral-7B, Mistral-Small, etc.
    """
    
    def load(self):
        """Load LLM model and tokenizer"""
        logger.info(f"Loading {self.name} ({self.model_id})...")
        start_time = time.time()
        
        try:
            # Determine dtype
            torch_dtype = eval(self.dtype) if isinstance(self.dtype, str) else self.dtype
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
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
        Translate texts using LLM prompting
        
        Args:
            texts: List of source texts
            source_lang: Source language (e.g., 'kik', 'eng')
            target_lang: Target language
            batch_size: Batch size
        """
        if not self.is_loaded:
            self.load()
        
        if batch_size is None:
            batch_size = self.config.get('batch_size', 4)
        
        translations = []
        start_time = time.time()
        
        # Get prompting mode from config
        prompting_mode = self.config.get('prompting_mode', 'zero_shot')
        
        for i, text in enumerate(tqdm(texts, desc=f"Translating with {self.name}")):
            try:
                # Build prompt
                if prompting_mode == 'zero_shot':
                    prompt = self._build_zero_shot_prompt(text, source_lang, target_lang)
                else:
                    prompt = self._build_few_shot_prompt(text, source_lang, target_lang)
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.get('max_new_tokens', 256),
                        temperature=self.config.get('temperature', 0.3),
                        top_p=self.config.get('top_p', 0.9),
                        do_sample=False
                    )
                
                # Decode and extract translation
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
    
    def _build_zero_shot_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Build zero-shot translation prompt"""
        lang_names = {
            'eng': 'English',
            'kik': 'Kikuyu',
            'kik_Latn': 'Kikuyu'
        }
        src_name = lang_names.get(source_lang, source_lang)
        tgt_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"""Translate the following text from {src_name} to {tgt_name}.
Only output the translation, nothing else.

{src_name}: {text}
{tgt_name}:"""
        return prompt
    
    def _build_few_shot_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Build few-shot translation prompt with examples"""
        lang_names = {
            'eng': 'English',
            'kik': 'Kikuyu',
            'kik_Latn': 'Kikuyu'
        }
        src_name = lang_names.get(source_lang, source_lang)
        tgt_name = lang_names.get(target_lang, target_lang)
        
        # Simple few-shot examples
        examples = self._get_few_shot_examples(source_lang, target_lang)
        
        prompt = f"""Translate from {src_name} to {tgt_name}:

"""
        for src_ex, tgt_ex in examples:
            prompt += f"{src_name}: {src_ex}\n{tgt_name}: {tgt_ex}\n\n"
        
        prompt += f"{src_name}: {text}\n{tgt_name}:"
        return prompt
    
    def _get_few_shot_examples(self, source_lang: str, target_lang: str) -> List[tuple]:
        """Get few-shot examples for language pair"""
        # Hardcoded examples (can be expanded)
        if ('eng' in source_lang or 'eng' in target_lang) and ('kik' in source_lang or 'kik' in target_lang):
            if 'eng' in source_lang:  # eng -> kik
                return [
                    ("Hello", "Wĩ"),
                    ("Thank you", "Wĩ ũndũ mwega"),
                    ("Good morning", "Kirima kĩa wĩ")
                ]
            else:  # kik -> eng
                return [
                    ("Wĩ", "Hello"),
                    ("Wĩ ũndũ mwega", "Thank you"),
                    ("Kirima kĩa wĩ", "Good morning")
                ]
        return []
    
    def _extract_translation_from_output(self, full_output: str, prompt: str) -> str:
        """Extract translation from model output"""
        # Remove prompt from output
        if prompt in full_output:
            translation = full_output[len(prompt):].strip()
        else:
            translation = full_output.strip()
        
        # Clean up output (remove extra tokens, etc.)
        translation = translation.split('\n')[0].strip()
        
        return translation
