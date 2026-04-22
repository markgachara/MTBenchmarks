"""
LLM-based MT via Prompting.
Supports: Aya-101 (encoder-decoder / seq2seq) and Llama 3 8B Instruct (decoder-only).
Both zero-shot and 3-shot prompting modes.
"""
import logging
import time
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from .base import BaseModel, TranslationResult

logger = logging.getLogger(__name__)


class LLMViaPrompting(BaseModel):
    """
    LLM for MT via instruction prompting.
    Handles both encoder-decoder (Aya-101) and decoder-only (Llama 3) architectures.
    """

    def load(self):
        """Load LLM model and tokenizer."""
        logger.info(f"Loading {self.name} ({self.model_id})...")
        start_time = time.time()

        try:
            torch_dtype = eval(self.dtype) if isinstance(self.dtype, str) else self.dtype
            is_enc_dec = self.config.get("is_encoder_decoder", False)

            # Check if quantization is needed
            quantization = self.config.get("quantization", None)
            load_kwargs = {
                "device_map": self.device_map,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }

            if quantization == "int8":
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    # Remove torch_dtype when using quantization
                    load_kwargs.pop("torch_dtype", None)
                    logger.info(f"Using INT8 quantization for {self.name}")
                except ImportError:
                    logger.warning("bitsandbytes not available; loading without quantization")

            if is_enc_dec:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id, **load_kwargs
                )
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **load_kwargs
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
        prompting_mode: str = "zero_shot",
        few_shot_examples: Optional[List[Dict]] = None,
        **kwargs,
    ) -> TranslationResult:
        """Translate texts using LLM prompting."""
        if not self.is_loaded:
            self.load()

        if batch_size is None:
            batch_size = self.config.get("batch_size", 4)

        is_enc_dec = self.config.get("is_encoder_decoder", False)
        translations = []
        start_time = time.time()

        mode_label = "3-shot" if prompting_mode == "few_shot" else "zero-shot"
        desc = f"Translating with {self.name} ({mode_label})"

        for i, text in enumerate(tqdm(texts, desc=desc)):
            try:
                prompt = self._build_prompt(
                    text, source_lang, target_lang, prompting_mode, few_shot_examples
                )

                if is_enc_dec:
                    translation = self._generate_seq2seq(prompt)
                else:
                    translation = self._generate_causal(prompt)

                translations.append(translation)

            except Exception as e:
                logger.warning(f"Error translating text {i}: {e}")
                translations.append("")

        inference_time = time.time() - start_time
        return TranslationResult(
            translations=translations,
            model_name=f"{self.name} ({mode_label})",
            source_lang=source_lang,
            target_lang=target_lang,
            inference_time=inference_time,
        )

    def _generate_seq2seq(self, prompt: str) -> str:
        """Generate translation using encoder-decoder model (Aya-101)."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available() and hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                num_beams=self.config.get("num_beams", 5),
            )

        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation.strip()

    def _generate_causal(self, prompt: str) -> str:
        """Generate translation using decoder-only model (Llama 3)."""
        # Use chat template if available (Llama 3 requires this)
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if torch.cuda.is_available() and hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)

        prompt_len = input_ids.shape[1]

        # Build EOS token list (Llama 3 uses <|eot_id|> as additional terminator)
        eos_token_ids = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id != self.tokenizer.unk_token_id:
            eos_token_ids.append(eot_id)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                eos_token_id=eos_token_ids,
                do_sample=True,
                temperature=self.config.get("temperature", 0.6),
                top_p=self.config.get("top_p", 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (exclude prompt)
        generated = outputs[0][prompt_len:]
        translation = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Clean: take first line, strip artifacts
        translation = translation.split("\n")[0].strip()
        return translation

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        prompting_mode: str,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """Build translation prompt."""
        src_name = self._lang_name(source_lang)
        tgt_name = self._lang_name(target_lang)

        if prompting_mode == "few_shot" and few_shot_examples:
            return self._build_few_shot_prompt(
                text, src_name, tgt_name, few_shot_examples
            )
        else:
            return self._build_zero_shot_prompt(text, src_name, tgt_name)

    def _build_zero_shot_prompt(
        self, text: str, src_name: str, tgt_name: str
    ) -> str:
        """Paper-specified zero-shot prompt format."""
        return (
            f"Translate the following sentence from {src_name} to {tgt_name}, "
            f"preserving all diacritical marks accurately: {text}"
        )

    def _build_few_shot_prompt(
        self,
        text: str,
        src_name: str,
        tgt_name: str,
        examples: List[Dict],
    ) -> str:
        """Paper-specified 3-shot prompt format."""
        prompt = (
            f"Translate sentences from {src_name} to {tgt_name}, "
            f"preserving all diacritical marks accurately.\n\n"
        )
        for idx, ex in enumerate(examples, 1):
            prompt += f"Example {idx}:\n"
            prompt += f"{src_name}: {ex['source']}\n"
            prompt += f"{tgt_name}: {ex['target']}\n\n"

        prompt += f"Now translate:\n{src_name}: {text}\n{tgt_name}:"
        return prompt

    @staticmethod
    def _lang_name(code: str) -> str:
        names = {
            "eng": "English",
            "eng_Latn": "English",
            "en": "English",
            "kik": "Gĩkũyũ",
            "kik_Latn": "Gĩkũyũ",
            "ki": "Gĩkũyũ",
        }
        return names.get(code, code)
