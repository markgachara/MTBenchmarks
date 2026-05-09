"""
LLM-based MT via Prompting.
Supports: Aya-101 (encoder-decoder / seq2seq) and Llama 3 8B Instruct (decoder-only).
Both zero-shot and 3-shot prompting modes.
"""
import logging
import re
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

_PREAMBLE_RE = re.compile(
    r"^\s*(?:okay|sure|here(?:'s| is)|the translation|translation)[^\n:]*[:.]?\s*\n",
    re.IGNORECASE,
)


def _clean_llm_translation(text: str) -> str:
    """Heuristic post-processing for chatty LLM responses.

    Drops common preambles like "Okay, here's the translation: ...", strips
    markdown bold/italic/code markers, and returns the first non-empty
    content line. The aim is to recover the actual translated sentence
    even when the model adds explanatory framing.
    """
    if not text:
        return ""

    # Strip leading whitespace + an optional preamble line that ends in ':'.
    cleaned = _PREAMBLE_RE.sub("", text, count=1)

    # Look for the first non-empty line; if it's bracketed in markdown bold,
    # keep the inner text. Skip lines that are clearly meta-commentary.
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip markdown bold/italic/code markers
        line = re.sub(r"^\*+\s*|\s*\*+$", "", line)  # **...**
        line = re.sub(r"^_+\s*|\s*_+$", "", line)
        line = re.sub(r"^`+\s*|\s*`+$", "", line)
        # Strip common label prefixes that LLMs add
        line = re.sub(r"^(?:Translation|Gĩkũyũ|English|Kikuyu|Kik|Output|Answer)\s*[:.\-]\s*", "", line, flags=re.IGNORECASE)
        # Drop surrounding quotes
        line = line.strip().strip("\"'")
        if line:
            return line
    return ""


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
            torch_dtype = _DTYPE_MAP.get(self.dtype, torch.float16) if isinstance(self.dtype, str) else self.dtype
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
                        load_in_8bit=True,
                        # Allow accelerate to spill modules to CPU when the model
                        # doesn't fit on available GPUs (e.g. Aya-101 13B on
                        # dual-11 GB GPUs); CPU layers run unquantized fp32.
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                    # Remove torch_dtype when using quantization
                    load_kwargs.pop("torch_dtype", None)
                    logger.info(f"Using INT8 quantization for {self.name} (CPU offload enabled)")
                except ImportError:
                    logger.warning("bitsandbytes not available; loading without quantization")
            elif quantization == "int4":
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_quant_type="nf4",
                    )
                    load_kwargs.pop("torch_dtype", None)
                    logger.info(f"Using INT4 (NF4) quantization for {self.name}")
                except ImportError:
                    logger.warning("bitsandbytes not available; loading without quantization")

            model_class = self.config.get("model_class")

            if is_enc_dec:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id, **load_kwargs
                )
            else:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Gemma-3 ships as a multimodal Gemma3ForConditionalGeneration; use
                # the text-only Gemma3ForCausalLM head to avoid loading the vision
                # tower we never use.
                if model_class == "gemma3":
                    from transformers import Gemma3ForCausalLM

                    self.model = Gemma3ForCausalLM.from_pretrained(
                        self.model_id, **load_kwargs
                    )
                else:
                    from transformers import AutoModelForCausalLM

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
        translations: List[str] = []
        start_time = time.time()

        mode_label = "3-shot" if prompting_mode == "few_shot" else "zero-shot"
        desc = f"Translating with {self.name} ({mode_label})"

        # Build all prompts up front so we can batch them.
        prompts = [
            self._build_prompt(t, source_lang, target_lang, prompting_mode, few_shot_examples)
            for t in texts
        ]

        for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
            batch = prompts[i : i + batch_size]
            try:
                if is_enc_dec:
                    out_batch = self._generate_seq2seq_batch(batch)
                else:
                    out_batch = self._generate_causal_batch(batch)
            except Exception as e:
                logger.warning(f"Error translating batch starting at {i}: {e}")
                out_batch = [""] * len(batch)
            translations.extend(out_batch)

        inference_time = time.time() - start_time
        return TranslationResult(
            translations=translations,
            model_name=f"{self.name} ({mode_label})",
            source_lang=source_lang,
            target_lang=target_lang,
            inference_time=inference_time,
        )

    def _generate_seq2seq_batch(self, prompts: List[str]) -> List[str]:
        """Batched generation for encoder-decoder models (e.g. Aya-101)."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        first_dev = next(iter(self.model.parameters())).device
        enc = {k: v.to(first_dev) for k, v in enc.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **enc,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                num_beams=self.config.get("num_beams", 5),
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def _generate_causal_batch(self, prompts: List[str]) -> List[str]:
        """Batched generation for decoder-only models (Llama, Gemma)."""
        # Build chat-template inputs as plain strings, then batch-tokenize
        # with left padding so the prompts align at the right edge.
        if hasattr(self.tokenizer, "apply_chat_template"):
            tmpl = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
        else:
            tmpl = prompts

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(tmpl, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        finally:
            self.tokenizer.padding_side = prev_side

        first_dev = next(iter(self.model.parameters())).device
        enc = {k: v.to(first_dev) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        eos_token_ids = [self.tokenizer.eos_token_id]
        # convert_tokens_to_ids returns unk_token_id (not None) for unknown tokens;
        # only add <|eot_id|> when it's genuinely in the vocabulary (Llama-style).
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id != self.tokenizer.unk_token_id:
            eos_token_ids.append(eot_id)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=self.config.get("max_new_tokens", 256),
                eos_token_id=eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.config.get("do_sample", False):
                gen_kwargs.update(
                    do_sample=True,
                    temperature=self.config.get("temperature", 0.6),
                    top_p=self.config.get("top_p", 0.9),
                )
            else:
                gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            outputs = self.model.generate(**enc, **gen_kwargs)

        # Slice off the prompt tokens from each row, decode, then post-process
        # to strip preambles, markdown, etc. (LLMs love to be chatty).
        gen = outputs[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [_clean_llm_translation(d) for d in decoded]

    # Single-prompt helpers kept for backward compatibility (tests / callers).

    def _generate_seq2seq(self, prompt: str) -> str:
        return self._generate_seq2seq_batch([prompt])[0]

    def _generate_causal(self, prompt: str) -> str:
        return self._generate_causal_batch([prompt])[0]

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
            f"preserving all diacritical marks accurately. "
            f"Output ONLY the {tgt_name} translation as a single line, with no preamble, "
            f"no explanation, no quotation marks, and no markdown formatting.\n\n"
            f"{src_name}: {text}\n{tgt_name}:"
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
            f"preserving all diacritical marks accurately. "
            f"Output ONLY the {tgt_name} translation as a single line, with no preamble.\n\n"
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
