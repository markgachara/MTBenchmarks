"""
LoRA Fine-tuned Model: InterstellarCG/kikuyu-translator-final (Gemma 3 4B + LoRA).
Uses unsloth for efficient loading per the model card.
"""
import logging
import time
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from .base import BaseModel, TranslationResult

logger = logging.getLogger(__name__)

# Safe dtype lookup — avoids eval() on config strings
_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
}

class LoRAFineTuned(BaseModel):
    """
    LoRA fine-tuned model for bidirectional English ↔ Gĩkũyũ translation.
    Uses unsloth + Gemma-3 chat template per the HuggingFace model card.
    """

    def load(self):
        """Load model using unsloth (preferred) or standard transformers + PEFT."""
        logger.info(f"Loading {self.name} ({self.model_id})...")
        start_time = time.time()

        use_unsloth = self.config.get("use_unsloth", True)
        load_in_4bit = self.config.get("load_in_4bit", True)

        try:
            if use_unsloth:
                try:
                    self._load_with_unsloth(load_in_4bit)
                except ImportError:
                    logger.warning("unsloth not installed; falling back to transformers + PEFT")
                    self._load_with_transformers()
            else:
                self._load_with_transformers()

            self._is_loaded = True
            self._load_time = time.time() - start_time
            logger.info(f"Loaded {self.name} in {self._load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            raise

    def _load_with_unsloth(self, load_in_4bit: bool):
        """Load using unsloth (efficient inference)."""
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        self.model, self.tokenizer = FastModel.from_pretrained(
            self.model_id,
            max_seq_length=2048,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma-3")
        self._use_chat_template = True

    def _load_with_transformers(self):
        """Fallback: load with standard transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = _DTYPE_MAP.get(self.dtype, torch.float16) if isinstance(self.dtype, str) else self.dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        load_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Use 4-bit quantization if configured (saves VRAM)
        if self.config.get("load_in_4bit", False):
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
                logger.info("Using 4-bit quantization via bitsandbytes")
            except ImportError:
                logger.warning("bitsandbytes not available; loading in fp16")
                load_kwargs["torch_dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **load_kwargs
        )
        self._use_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> TranslationResult:
        """Translate texts using the fine-tuned kikuyu-translator model."""
        if not self.is_loaded:
            self.load()

        translations = []
        start_time = time.time()

        for i, text in enumerate(tqdm(texts, desc=f"Translating with {self.name}")):
            try:
                prompt_text = self._build_prompt(text, source_lang, target_lang)

                if getattr(self, "_use_chat_template", False):
                    translation = self._generate_with_chat_template(prompt_text)
                else:
                    translation = self._generate_plain(prompt_text)

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
            inference_time=inference_time,
        )

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Build prompt per the model card format."""
        if "eng" in source_lang.lower() or "en" == source_lang.lower():
            return f"Translate the following English text to Kikuyu.\n\n{text}"
        else:
            return f"Translate the following Kikuyu text to English.\n\n{text}"

    def _generate_with_chat_template(self, prompt_text: str) -> str:
        """Generate using unsloth chat template (per model card)."""
        messages = [{"role": "user", "content": prompt_text}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)

        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                temperature=0.3,
                do_sample=False,
            )

        generated = outputs[0][prompt_len:]
        translation = self.tokenizer.decode(generated, skip_special_tokens=True)
        return translation.strip().split("\n")[0].strip()

    def _generate_plain(self, prompt_text: str) -> str:
        """Generate using plain tokenization (fallback)."""
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                temperature=0.3,
                do_sample=False,
            )

        generated = outputs[0][prompt_len:]
        translation = self.tokenizer.decode(generated, skip_special_tokens=True)
        return translation.strip().split("\n")[0].strip()
