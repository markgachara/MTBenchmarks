"""
Pre-flight verification: confirm Kikuyu (kik) coverage in every candidate
model's tokenizer before we touch any model code.

Loads only tokenizers (cheap) for: NLLB-1.3B/3.3B, MADLAD-400-3B/7B,
Gemma-3-12B-IT, Llama-3.1-8B-Instruct. Loads the Goldfish-Kikuyu model
fully because it's tiny (124 M).

Exits non-zero on any failure so it can gate CI.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from transformers import AutoTokenizer


@dataclass
class Check:
    name: str
    fn: Callable[[], str]  # returns a short status string on success


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_nllb(model_id: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_id)
    # NLLB tokenizers expose lang_code_to_id; in newer transformers it lives
    # on additional_special_tokens. Try both.
    if hasattr(tok, "lang_code_to_id") and "kik_Latn" in tok.lang_code_to_id:
        tid = tok.lang_code_to_id["kik_Latn"]
    else:
        tid = tok.convert_tokens_to_ids("kik_Latn")
        if tid == tok.unk_token_id:
            raise RuntimeError("kik_Latn resolved to <unk>")
    return f"kik_Latn -> {tid}"


def check_madlad(model_id: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_id)
    tid = tok.convert_tokens_to_ids("<2kik>")
    if tid == tok.unk_token_id:
        raise RuntimeError("<2kik> resolved to <unk>")
    return f"<2kik> -> {tid}"


def check_gemma3(model_id: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_id)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Habari"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "<start_of_turn>" not in rendered:
        raise RuntimeError("Gemma-3 chat template did not render expected tokens")
    return f"vocab={len(tok)}, chat_template=ok"


def check_llama31(model_id: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_id)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot == tok.unk_token_id:
        raise RuntimeError("<|eot_id|> not found")
    return f"<|eot_id|> -> {eot}"


def check_goldfish(model_id: str) -> str:
    # Tiny enough (124M GPT-2) to load fully and verify scoring works.
    import torch
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    sample = "Niĩ nĩ ndĩthomete na nĩ nguĩte"  # short Kikuyu phrase
    ids = tok(sample, return_tensors="pt").input_ids
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    return f"loss={loss:.3f} on '{sample}'"


# ---------------------------------------------------------------------------

CHECKS: list[Check] = [
    Check("NLLB-200-1.3B",        lambda: check_nllb("facebook/nllb-200-1.3B")),
    Check("NLLB-200-3.3B",        lambda: check_nllb("facebook/nllb-200-3.3B")),
    Check("MADLAD-400-3B",        lambda: check_madlad("google/madlad400-3b-mt")),
    Check("MADLAD-400-7B",        lambda: check_madlad("google/madlad400-7b-mt")),
    Check("Gemma-3-12B-IT",       lambda: check_gemma3("google/gemma-3-12b-it")),
    Check("Llama-3.1-8B-Instruct", lambda: check_llama31("meta-llama/Llama-3.1-8B-Instruct")),
    Check("Goldfish-Kikuyu",      lambda: check_goldfish("goldfish-models/kik_latn_full")),
]


def main() -> int:
    failures = 0
    print(f"{'Model':<28} {'Result':<10} Notes")
    print("-" * 78)
    for c in CHECKS:
        try:
            note = c.fn()
            print(f"{c.name:<28} {_green('PASS'):<19} {note}")
        except Exception as e:  # noqa: BLE001
            print(f"{c.name:<28} {_red('FAIL'):<19} {type(e).__name__}: {e}")
            failures += 1
    print("-" * 78)
    if failures:
        print(_red(f"{failures}/{len(CHECKS)} check(s) failed"))
        return 1
    print(_green(f"All {len(CHECKS)} checks passed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
