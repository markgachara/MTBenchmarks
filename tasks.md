# Tasks: MTBenchmarks Model Expansion

Detailed, dependency-ordered task list for the work described in [plans.md](plans.md).
Each `[ ]` is a single mergeable unit. Indented sub-tasks must be done in
order; siblings at the same level can be parallelised.

**Branch**: `feature/evaluation-pipeline-alignment`
**Target end-state**: 9 evaluated MT models + Goldfish perplexity scorer,
producing fully populated paper Tables 4 & 5 on the GAC 500-pair test set.

---

## Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` dropped / out-of-scope
- ⚠ blocking risk
- 🔒 requires HF gated-model acceptance
- 💾 large download (> 10 GB)

---

## Phase 0 — Preparation (do first, no code)

These unblock every later phase.

- [ ] **0.1** Confirm `.venv` has `transformers >= 4.50`, `accelerate`,
      `bitsandbytes`, `sentencepiece`, `protobuf` installed
  - [ ] Run `python -c "import transformers; print(transformers.__version__)"`
  - [ ] If `< 4.50`, update [pyproject.toml](pyproject.toml) → `transformers>=4.50,<5`
  - [ ] `uv sync` (or `pip install -U -r requirements.txt`)
- [ ] **0.2** 🔒 Accept HF licence terms for new gated models (in browser,
      logged in as the same user as `huggingface-cli login`)
  - [ ] [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it)
  - [ ] [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)
  - [ ] [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [ ] **0.3** Verify `huggingface-cli whoami` returns the licensed account
- [ ] **0.4** Confirm disk space for model cache: `df -h ~/.cache/huggingface`
      should show ≥ 100 GB free (combined new downloads ≈ 80 GB)
  - [ ] If short, set `HF_HOME=/path/with/space` in `.envrc` or shell

---

## Phase 1 — Verification script (depends on Phase 0)

⚠ **Blocking gate.** Do NOT proceed to Phase 2+ until every check passes.
Catches the most expensive failure mode (token id missing for `kik`) before
any model code is written.

- [ ] **1.1** Create [`scripts/verify_lang_support.py`](scripts/verify_lang_support.py)
  - [ ] Loads only the tokenizer (cheap, no weights) for each candidate
  - [ ] Asserts MADLAD-3B: `tok.convert_tokens_to_ids("<2kik>") != tok.unk_token_id`
  - [ ] Asserts NLLB-1.3B / 3.3B: `tok.lang_code_to_id["kik_Latn"]` exists
  - [ ] Asserts Gemma-3-12B-IT: tokenizer loads, `apply_chat_template`
        renders a sample message
  - [ ] Asserts Llama-3.1-8B-IT: tokenizer loads, `<|eot_id|>` token id != unk
  - [ ] Asserts Goldfish: tokenizer + model load, can score one sentence
  - [ ] Prints a green/red summary table and exits with non-zero on any failure
- [ ] **1.2** Run the script, paste output into PR description
  - ⚠ If MADLAD `<2kik>` resolves to unk → escalate; consider whether
    paper claims (Kikuyu in 419-lang dataset) survive without it
- [ ] **1.3** Commit + push as PR-1 (no model code yet)

---

## Phase 2 — NLLB-200 scaling family (depends on 1.3)

Zero new code. Pure YAML. Lowest-risk addition; gives a within-NLLB
scaling curve that strengthens the paper.

- [ ] **2.1** Edit [`config/models.yaml`](config/models.yaml)
  - [ ] Add `nllb_200_1_3b` block (copy of existing `nllb_200`, change
        `model_id`, `parameters: 1_300_000_000`, `vram_fp16_gb: 2.6`,
        `batch_size: 4`)
  - [ ] Add `nllb_200_3_3b` block (`facebook/nllb-200-3.3B`,
        `parameters: 3_300_000_000`, `vram_fp16_gb: 6.6`, `batch_size: 2`)
- [ ] **2.2** 💾 Pre-download to verify license + cache
  - [ ] `huggingface-cli download facebook/nllb-200-1.3B`
  - [ ] `huggingface-cli download facebook/nllb-200-3.3B`
- [ ] **2.3** Smoke test
  - [ ] `python main.py --models nllb_200_1_3b --dry-run` → BLEU > 0
  - [ ] `python main.py --models nllb_200_3_3b --dry-run` → BLEU > 0
  - [ ] Inspect one Gĩkũyũ sample in
        [`results/benchmark_<ts>/translations/`](results) — must look like
        Kikuyu, not English/Swahili
- [ ] **2.4** Commit as PR-2

---

## Phase 3 — Llama 3.1 8B swap (depends on 1.3, parallel with 2)

Pure config swap. The existing [`llm_prompting.py`](scripts/models/llm_prompting.py)
already handles Llama-3 chat template + `<|eot_id|>`; Llama-3.1 uses the
same special tokens.

- [ ] **3.1** In [`config/models.yaml`](config/models.yaml), edit the
      existing `llama3_8b` block
  - [ ] Change `model_id: meta-llama/Meta-Llama-3-8B-Instruct` →
        `meta-llama/Llama-3.1-8B-Instruct`
  - [ ] Update `name: "Llama 3.1 8B Instruct"`
  - [ ] (Optional) rename top-level key `llama3_8b` → `llama3_1_8b` for
        clarity in result file names
- [ ] **3.2** 💾 Pre-download
  - [ ] `huggingface-cli download meta-llama/Llama-3.1-8B-Instruct`
- [ ] **3.3** Smoke test
  - [ ] `python main.py --models llama3_1_8b --dry-run` → both zero-shot
        and 3-shot complete; output non-empty
- [ ] **3.4** Commit as PR-3

---

## Phase 4 — MADLAD-400 (DROPPED after Phase 1 verification)

**Verified 7 May 2026**: `<2kik>` resolves to `<unk>` in MADLAD-400's tokenizer.
The model was not trained to produce Kikuyu. All Phase 4 sub-tasks are
out-of-scope; do not start. See [plans.md §0](plans.md).

<details><summary>Original plan (kept for record)</summary>

- [-] **4.1** Edit [`scripts/models/transformer_mt.py`](scripts/models/transformer_mt.py)
  - [ ] Add to `LANG_CODES` dict:
        ```python
        "madlad": {"eng": "en", "kik": "kik"},
        ```
  - [ ] In `translate()`, branch on `lang_fmt == "madlad"`:
    - [ ] Skip `tokenizer.src_lang` assignment (no such attribute on T5 tok)
    - [ ] Set `tgt_token_id = None` (no forced BOS)
    - [ ] Prepend `f"<2{tgt_code}> "` to every text in the batch before
          tokenisation
    - [ ] Add a comment explaining the prefix convention
  - [ ] Add a unit-style smoke check: log the first prefixed text in INFO
        so dry-run output makes the change visible
- [-] **4.2** Edit [`config/models.yaml`](config/models.yaml)
  - [ ] Add `madlad400_3b` block:
    - `model_id: google/madlad400-3b-mt`, `model_type: transformer_mt`,
      `lang_code_format: madlad`, `parameters: 3_000_000_000`,
      `vram_fp16_gb: 6.0`, `batch_size: 4`, `num_beams: 5`
  - [ ] Add `madlad400_7b` block:
    - `parameters: 7_200_000_000`, `vram_fp16_gb: 14.4`, `batch_size: 2`,
      `quantization: "int8"` (so it fits on one 11 GB GPU)
- [-] **4.3** 💾 Pre-download
  - [ ] `huggingface-cli download google/madlad400-3b-mt` (~12 GB)
  - [ ] `huggingface-cli download google/madlad400-7b-mt` (~28 GB)
- [-] **4.4** Smoke test
  - [ ] `python main.py --models madlad400_3b --dry-run`
  - [ ] Visually inspect at least 1 `eng->kik` and 1 `kik->eng` sample
  - [ ] Confirm chrF > 0 (sanity, not score quality)
  - [ ] `python main.py --models madlad400_7b --dry-run` (slower; OK)
- [-] **4.5** Commit as PR-4

</details>

---

## Phase 5 — Gemma-3 IT models (depends on 1.3 + Phase 0.1)

Touches [`scripts/models/llm_prompting.py`](scripts/models/llm_prompting.py)
to use the text-only `Gemma3ForCausalLM` class instead of the multimodal
default.

- [ ] **5.1** Edit [`scripts/models/llm_prompting.py`](scripts/models/llm_prompting.py)
  - [ ] In `load()`, add a branch:
        ```python
        model_class = self.config.get("model_class")
        if model_class == "gemma3":
            from transformers import Gemma3ForCausalLM
            self.model = Gemma3ForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        ```
  - [ ] Default path unchanged for existing Aya/Llama configs
  - [ ] Verify `apply_chat_template` is used (Gemma-3 requires it)
- [ ] **5.2** Edit [`config/models.yaml`](config/models.yaml)
  - [ ] Add `gemma3_12b_it`:
    - `model_class: gemma3`, `is_encoder_decoder: false`,
      `parameters: 12_000_000_000`, `vram_fp16_gb: 24.0`,
      `vram_int8_gb: 12.0`, `quantization: int8`, `batch_size: 2`,
      `device_map: auto`, `prompting_modes: [zero_shot, few_shot]`
  - [ ] Add `gemma3_27b_it` with `quantization: int4`,
        `skip_on_low_memory: true`, `device_map: auto` (split across 2 GPUs)
- [ ] **5.3** 💾 Pre-download
  - [ ] `huggingface-cli download google/gemma-3-12b-it` (~25 GB)
  - [ ] `huggingface-cli download google/gemma-3-27b-it` (~55 GB)  ← only
        if we keep 27B; can defer
- [ ] **5.4** Smoke test
  - [ ] `python main.py --models gemma3_12b_it --dry-run`
  - [ ] `nvidia-smi` during run — confirm INT8 keeps both GPUs ≤ ~10 GB
  - [ ] Visually inspect zero-shot vs 3-shot outputs (3-shot should look
        more Kikuyu-like)
  - [ ] (Optional) `gemma3_27b_it` dry run — only after 12B passes
- [ ] **5.5** Commit as PR-5

---

## Phase 6 — Goldfish-Kikuyu perplexity (depends on 1.3, parallel with 2-5)

Most invasive change to the evaluation pipeline. Finally fills the empty
"Perplexity" column in the result tables.

- [ ] **6.1** Edit [`scripts/evaluator.py`](scripts/evaluator.py)
  - [ ] Add private fields: `self._goldfish_model = None`,
        `self._goldfish_tok = None`
  - [ ] Add `_get_goldfish()` lazy loader (FP16, `device_map="auto"`)
  - [ ] Implement `_compute_perplexity(predictions, direction)`:
    - [ ] Early-return `{"perplexity": None}` if `direction != "eng->kik"`
    - [ ] For each prediction: tokenise, compute mean per-token NLL with
          stride/sliding-window for sequences > 512 tokens
    - [ ] Aggregate: `perplexity = math.exp(corpus_mean_nll)`
    - [ ] Catch and log OOM/empty-string failures; return `None` on error
  - [ ] Wire into `compute_all_metrics`: append result to the `results`
        dict
  - [ ] Update `release_gpu()` to also release the Goldfish model
- [ ] **6.2** Add baseline row generation
  - [ ] In [`main.py`](main.py), after computing reference baselines,
        also compute Goldfish perplexity on `kikuyu_texts` (human ref)
  - [ ] Store under `ref_baseline_eng2kik["perplexity"]`
- [ ] **6.3** Edit [`scripts/reporter.py`](scripts/reporter.py)
  - [ ] In the table-builder, change the Perplexity column to read
        `metrics.get("perplexity")` (currently always `None`)
  - [ ] Format as `"{val:.1f}"` if not None, else `"—"`
- [ ] **6.4** 💾 Pre-download Goldfish (~500 MB, trivial)
  - [ ] `huggingface-cli download goldfish-models/kik_latn_full`
- [ ] **6.5** Smoke test
  - [ ] `python main.py --models nllb_200 --dry-run` → confirm perplexity
        column in resulting [`results/.../report.md`](results) is populated
        for `eng->kik` rows and `—` for `kik->eng`
  - [ ] Confirm a Human Reference row also has a perplexity value
- [ ] **6.6** Commit as PR-6

---

## Phase 7 — Pre-existing bug-fixes (depends on Phases 1-6 merged)

These are issues already noted in the codebase audit but not directly
addressed by adding new models. Doing them in the same release window
keeps the final benchmark run clean.

- [ ] **7.1** Investigate `kikuyu-translator-final` returning empty strings
      ([results](results/benchmark_20260422_201704/translations/kikuyu-translator-final_eng2kik.json))
  - [ ] Read [`scripts/models/lora_finetuned.py`](scripts/models/lora_finetuned.py) `_generate_with_chat_template`
  - [ ] Check if `unsloth.FastModel.for_inference(model)` is being called
  - [ ] Check if `apply_chat_template` correctly returns a tensor on the
        model's device
  - [ ] Add a `--debug-print-prompts` flag to dump the first 3 raw
        prompt/response pairs
  - [ ] Fix root cause; smoke test
- [ ] **7.2** Investigate M2M-100 low chrF++ (~10) for both directions
  - [ ] Confirm whether `sw` (Swahili) is truly the closest fallback or
        whether M2M's tokeniser actually has a `ki` token even though
        the model claims 100-language coverage
  - [ ] Decide: keep M2M with explicit "Kikuyu via Swahili proxy" footnote,
        or remove from primary table
- [ ] **7.3** Confirm Aya-101 + Llama 3.1 actually run end-to-end on the full
      500-pair set (the 6th dry-run had them missing — possibly OOM)
  - [ ] If OOM: lower `batch_size` to 1 in their config, document
- [ ] **7.4** Commit as PR-7 (one PR per fix is also fine)

---

## Phase 8 — Full benchmark execution (depends on Phases 1-7)

- [ ] **8.1** Pre-flight
  - [ ] `git status` clean, on `feature/evaluation-pipeline-alignment`
  - [ ] `nvidia-smi` shows both GPUs idle, ≥ 11 GB free each
  - [ ] `df -h ~/.cache/huggingface` ≥ 50 GB free (downloads done)
  - [ ] `python scripts/verify_lang_support.py` still passes
- [ ] **8.2** `python main.py 2>&1 | tee logs/full_run_$(date +%Y%m%d_%H%M).log`
- [ ] **8.3** Post-run verification
  - [ ] All 9 evaluated models present in
        [`results/benchmark_<ts>/metrics/full_metrics.json`](results)
  - [ ] No row has all-zero metrics
  - [ ] Perplexity column populated for `eng->kik` rows
  - [ ] [`results/benchmark_<ts>/qualitative/`](results) spreadsheet has
        50 stratified samples × 9 models × 2 directions
- [ ] **8.4** Commit results directory + final
      [`report.md`](results/benchmark_20260422_201704/report.md) snapshot

---

## Phase 9 — Documentation & PR cleanup (depends on 8.4)

- [ ] **9.1** Update [`README.md`](README.md) model count (5 → 9) and
      add MADLAD/NLLB-3.3B/Gemma-3 to the feature list
- [ ] **9.2** Update [`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) Table 1 to
      match the executed model list
- [ ] **9.3** Mark all Phase 1-8 PRs as ready-for-review; final umbrella
      PR squash-merge into `feature/evaluation-pipeline-alignment`
- [ ] **9.4** Eventually rebase/merge into `main` once paper draft uses
      the results

---

## Out of scope (do not start)

- [-] Toucan-1.2B / 3.7B — verified `kik` not in MT-supported language list
      (see [plans.md §1](plans.md))
- [-] Aya 23 / Aya Expanse 8B / 32B — only 23 supported langs, none Bantu
- [-] AfriTeVa V2 — Kikuyu coverage uncertain; defer
- [-] Helsinki-NLP `opus-mt-en-bnt` — explicitly excludes `kik`
- [-] NLLB-MoE-54B — does not fit; CPU offload would take days
- [-] InkubaLM — only Swahili/Yoruba/Hausa/isiZulu/isiXhosa
- [-] MADLAD-400-10.7B — marginal value over 7B; reviewer-on-demand only

---

## Dependency graph (TL;DR)

```
0  ──▶ 1 ──┬──▶ 2 (NLLB scaling)        ──┐
           ├──▶ 3 (Llama 3.1 swap)       ─┤
           ├──▶ 4 (MADLAD)               ─┼──▶ 7 (bug-fixes) ──▶ 8 (full run) ──▶ 9 (docs)
           ├──▶ 5 (Gemma-3)              ─┤
           └──▶ 6 (Goldfish perplexity)  ─┘
```

Phases 2-6 are mutually independent and can be parallelised across separate
PRs once Phase 1 has confirmed Kikuyu coverage for all models.
