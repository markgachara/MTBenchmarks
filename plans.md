# Plan: Adding New Publishable Models to MTBenchmarks

**Branch**: `feature/evaluation-pipeline-alignment`
**Date**: 7 May 2026
**Hardware**: RTX 2080 Ti (10.7 GB) + RTX 3060 (11.7 GB), 1.5 TB RAM, 32-core Xeon

---

## 0. Executive summary

After a second research pass with deeper verification I am **revising the
earlier recommendation**. One model has been dropped, four remain.

| Recommendation | Verdict | Reason |
|---|---|---|
| ~~MADLAD-400 (3B / 7B)~~ | ❌ **DROP** (verified 7 May 2026) | `<2kik>` resolves to `<unk>` in the 256 K SentencePiece vocab. MADLAD's MT model was not trained with Kikuyu as a language token — the 419-lang figure is for the *monolingual corpus*, not the MT model's supported pairs. Confirmed by [`scripts/verify_lang_support.py`](scripts/verify_lang_support.py). |
| NLLB-200 (1.3B / 3.3B) | ✅ **Add** | Same family as existing 600M. Zero code changes. CC-BY-NC. |
| Gemma-3 12B / 27B IT | ✅ **Add** | Multilingual (140+ langs). Paper reports FloRes chrF up to 48.8. Gemma license. |
| Llama 3.1 8B Instruct | ✅ **Add** (replace existing Llama 3) | Direct upgrade, same code path, Llama-3.1 license. |
| Goldfish-Kikuyu (`kik_latn_full`) | ✅ **Add** (not as MT, as **perplexity scorer**) | Fixes the empty perplexity column in Tables 4/5. |
| ~~Toucan-1.2B / 3.7B~~ | ❌ **DROP** | Kikuyu is NOT in Toucan's 46 supported translation directions (only the Cheetah base LM was pretrained on it). Including it would be misleading. |

This brings the benchmark to **7 evaluated models** (was 5):
NLLB-600M / 1.3B / 3.3B, M2M-100, kikuyu-translator, Aya-101,
Gemma-3-12B-IT, Llama-3.1-8B-IT — plus Goldfish for perplexity.

---

## 1. Verification: Why Toucan was dropped

The Cheetah pretraining corpus (517 African languages) **does** include Kikuyu
(`kik` is in [supported-languages.txt](https://github.com/UBC-NLP/Cheetah/blob/main/supported-languages.txt)).

But Toucan is the *MT-fine-tuned* version of Cheetah, and its translation
prefix dictionary (the only languages it was trained to translate) lists only
46 codes. **`kik` is absent**. Confirmed in
[UBC-NLP/Toucan README](https://github.com/UBC-NLP/Toucan):

```python
# Toucan supported translation langs (excerpt)
"kbp", "lgg", "lug", "mlg", "nyn", "swa", "swc", "swh", "yor", ...
# kik is NOT here
```

Translating `eng -> kik` with Toucan would silently fall through to whatever
the closest token resolves to (likely Swahili `swa`/`swh`), producing
non-Kikuyu output. Misleading for a benchmark paper.

## 2. Verification of remaining models for `kik` coverage

### MADLAD-400 (verified)
- **Paper** ([arXiv:2309.04662](https://arxiv.org/abs/2309.04662)): MADLAD-400
  monolingual covers 419 languages including Kikuyu. The MT models are
  trained on 250B tokens spanning 450+ languages, with parallel data for 157.
- **Tokenizer**: T5 SentencePiece (256 K vocab) with `<2xx>` prefix tokens
  prepended to the *source* sentence (not `forced_bos_token_id` like NLLB).
- **Verification step before merge**: run a one-line script
  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
  print(tok.convert_tokens_to_ids("<2kik>"))   # must NOT equal unk_token_id
  ```
- **License**: Apache-2.0 (most permissive of any candidate).

### NLLB-200 1.3B / 3.3B (verified)
- Identical architecture/tokenizer to the existing
  [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
  already in [`config/models.yaml`](config/models.yaml). Uses `kik_Latn`.
- **License**: CC-BY-NC 4.0 — already accepted in the existing pipeline.

### Gemma-3 12B-IT / 27B-IT (verified)
- Model card claims **140+ languages**, not language-specific guarantee for
  Kikuyu but [card multilingual table](https://huggingface.co/google/gemma-3-27b-it)
  reports **FloRes chrF = 48.8 (27B)**, **46.0 (12B)**.
- Architecture: decoder-only causal LM with chat template (Gemma-3).
- Requires `transformers >= 4.50.0`.
- Use `Gemma3ForCausalLM` (text-only) — *not* `Gemma3ForConditionalGeneration`
  (multimodal, larger memory).
- **License**: Gemma — research-friendly, gated on HF (need to accept).

### Llama 3.1 8B Instruct (verified)
- Officially supports 8 languages (English, German, French, Italian,
  Portuguese, Hindi, Spanish, Thai). **Kikuyu is NOT in supported list**, but
  the card explicitly states "trained on a broader collection of languages
  than the 8 supported languages" — testing emergent zero-shot/few-shot
  Kikuyu translation is a legitimate publishable comparison.
- Same chat-template path as the existing Llama-3 entry.

### Goldfish-Kikuyu (verified)
- [`goldfish-models/kik_latn_full`](https://huggingface.co/goldfish-models/kik_latn_full):
  124 M GPT-2, trained on 8 MB of Kikuyu (NLLB CommonCrawl + eBible + Wiki +
  Glot500). Apache-2.0.
- **Use case**: perplexity scoring only, not a translator. Compute mean
  per-token loss on each model's `eng->kik` outputs.
- **Caveat for paper**: training data is heavily biblical/religious — may
  give artificially high perplexity to non-religious agricultural text.
  Note this in the paper's "Limitations" section.

---

## 3. Hardware feasibility (FP16 / INT8 on 10.7 + 11.7 GB)

Sizes are weight memory only; activations add 1–3 GB headroom.

| Model | Params | FP16 | INT8 | Strategy |
|---|---:|---:|---:|---|
| NLLB-600M (existing) | 0.6 B | 1.2 GB | — | single GPU FP16 |
| NLLB-200-1.3B | 1.3 B | 2.6 GB | 1.3 GB | single GPU FP16 |
| NLLB-200-3.3B | 3.3 B | 6.6 GB | 3.3 GB | single GPU FP16 |
| MADLAD-3B | 3.0 B | 6.0 GB | 3.0 GB | single GPU FP16 |
| MADLAD-7B | 7.2 B | 14.4 GB | 7.2 GB | **INT8 single GPU** *or* `device_map="auto"` split across both |
| Llama-3.1-8B | 8.0 B | 16.0 GB | 8.0 GB | INT8 single GPU |
| Gemma-3-12B-IT | 12.0 B | 24.0 GB | 12.0 GB | INT8 split across 2 GPUs *or* INT4 single GPU |
| Gemma-3-27B-IT | 27.0 B | 54.0 GB | 27.0 GB | **INT4 (~14 GB)** split across 2 GPUs — tight but feasible |
| Aya-101 (existing) | 13 B | 26 GB | 13 GB | already INT8 split |
| Goldfish-kik | 0.12 B | 0.5 GB | — | trivial, runs anywhere |

**Plan**: rely on existing sequential-loading orchestrator
([`scripts/translator.py`](scripts/translator.py)) — only one model in VRAM at
a time. Set `quantization: "int8"` for everything ≥ 7 B and
`quantization: "int4"` for Gemma-3-27B. Use `device_map="auto"` for models
that require splitting.

---

## 4. Implementation plan (phased)

Each phase is independently mergeable. Start with the cheapest (zero-code)
additions to grow the result table fast, then tackle the harder ones.

### Phase 1 — Verification & dependencies
**Goal**: prove kik token ids exist before writing any model code.

- [ ] Add `transformers>=4.50` to [`pyproject.toml`](pyproject.toml) (Gemma-3 requirement).
- [ ] Write a 30-line `scripts/verify_lang_support.py` that, for each candidate
      model id, loads only the tokenizer (cheap) and checks:
  - MADLAD-3B: `tok.convert_tokens_to_ids("<2kik>")` ≠ unk
  - NLLB-3.3B: `tok.lang_code_to_id["kik_Latn"]` exists
  - Gemma-3-12B-IT: tokenizer loads, chat template renders
  - Llama-3.1-8B-IT: tokenizer loads
  - Goldfish-kik: tokenizer loads
- [ ] Run on the dev box, capture output as part of the PR.

### Phase 2 — NLLB-200 scaling family (zero code)
**Effort**: ~10 min. Just YAML.

- [ ] Add `nllb_200_1_3b` and `nllb_200_3_3b` entries in
      [`config/models.yaml`](config/models.yaml), copying the existing
      `nllb_200` block with adjusted `parameters`, `vram_fp16_gb`, `batch_size`.
      `model_id` = `facebook/nllb-200-1.3B` / `facebook/nllb-200-3.3B`.
- [ ] Set `batch_size: 4` for 1.3B and `batch_size: 2` for 3.3B (FP16).
- [ ] Dry-run with `python main.py --models nllb_200_1_3b nllb_200_3_3b --dry-run`.

### Phase 3 — Llama 3.1 swap (config-only)
**Effort**: ~5 min.

- [ ] Replace `model_id` in the existing `llama3_8b` entry from
      `meta-llama/Meta-Llama-3-8B-Instruct` to
      `meta-llama/Llama-3.1-8B-Instruct`. Same chat template, same code path.
- [ ] Optionally rename key `llama3_8b` → `llama3_1_8b` (cosmetic; updates
      result file names).
- [ ] Verify the existing `<|eot_id|>` handling in
      [`scripts/models/llm_prompting.py`](scripts/models/llm_prompting.py)
      still applies (it does — Llama-3.1 uses the same special tokens).

### Phase 4 — MADLAD-400 (small code change)
**Effort**: ~1 hour, including testing.

MADLAD uses prefix tokens **on the source side**, not `forced_bos_token_id`.
Existing [`scripts/models/transformer_mt.py`](scripts/models/transformer_mt.py)
already supports per-family `lang_code_format` switching, so the change is
small.

- [ ] In [`scripts/models/transformer_mt.py`](scripts/models/transformer_mt.py):
  1. Add new entry to the `LANG_CODES` dict:
     ```python
     "madlad": {"eng": "en", "kik": "kik"}
     ```
  2. In `translate()`, if `lang_fmt == "madlad"`, **prepend** `f"<2{tgt_code}> "`
     to every input text and skip the `forced_bos_token_id` branch (set
     `tgt_token_id = None`).
- [ ] In [`config/models.yaml`](config/models.yaml), add:
  ```yaml
  madlad400_3b:
    name: "MADLAD-400 (3B MT)"
    model_id: "google/madlad400-3b-mt"
    model_type: "transformer_mt"
    lang_code_format: "madlad"
    parameters: 3_000_000_000
    vram_fp16_gb: 6.0
    vram_int8_gb: 3.0
    dtype: "torch.float16"
    is_encoder_decoder: true
    batch_size: 4
    max_new_tokens: 256
    num_beams: 5
    device_map: "auto"
    description: "Google MADLAD-400 3B MT model (450+ langs, Apache-2.0)."
  madlad400_7b:
    # similar, parameters: 7_200_000_000, vram_fp16_gb: 14.4, batch_size: 2,
    # quantization: "int8" to fit on 1 GPU
  ```
- [ ] Smoke test: `python main.py --models madlad400_3b --dry-run` and visually
      inspect at least one output that resembles Gĩkũyũ.

### Phase 5 — Gemma-3 IT models (config + minor llm_prompting tweak)
**Effort**: ~1 hour.

The existing [`scripts/models/llm_prompting.py`](scripts/models/llm_prompting.py)
calls `apply_chat_template` for any decoder-only model with the attribute,
so it should work for Gemma-3 once we ensure the right model class is loaded.

- [ ] In [`scripts/models/llm_prompting.py`](scripts/models/llm_prompting.py)
      `load()`, when `is_encoder_decoder: false` and the config has a new
      flag `model_class: "gemma3"`, use
      `Gemma3ForCausalLM.from_pretrained(...)` instead of
      `AutoModelForCausalLM` (the latter resolves to the multimodal class
      and requires extra image tokens). `transformers>=4.50`.
- [ ] In [`config/models.yaml`](config/models.yaml), add:
  ```yaml
  gemma3_12b_it:
    name: "Gemma 3 12B Instruct"
    model_id: "google/gemma-3-12b-it"
    model_type: "llm_prompting"
    model_class: "gemma3"
    is_encoder_decoder: false
    parameters: 12_000_000_000
    vram_fp16_gb: 24.0
    vram_int8_gb: 12.0
    dtype: "torch.float16"
    batch_size: 2
    max_new_tokens: 256
    quantization: "int8"
    device_map: "auto"
    prompting_modes: ["zero_shot", "few_shot"]
  gemma3_27b_it:
    # same, parameters 27B, quantization: "int4" via bitsandbytes
    skip_on_low_memory: true   # opt-in only
  ```
- [ ] Sanity check: temperature defaults of 0.6/top_p 0.9 from existing code
      are fine for Gemma-3 (matches the model card).

### Phase 6 — Goldfish-Kikuyu perplexity
**Effort**: ~1.5 hours.

This is the only addition that touches
[`scripts/evaluator.py`](scripts/evaluator.py) and finally fills the
"Perplexity" column that is currently always blank.

- [ ] Add a new lazy-loader to `MTEvaluator`:
  ```python
  def _get_goldfish(self):
      if self._goldfish_model is None:
          from transformers import AutoModelForCausalLM, AutoTokenizer
          self._goldfish_tok = AutoTokenizer.from_pretrained(
              "goldfish-models/kik_latn_full"
          )
          self._goldfish_model = AutoModelForCausalLM.from_pretrained(
              "goldfish-models/kik_latn_full",
              torch_dtype=torch.float16,
              device_map="auto",
          ).eval()
      return self._goldfish_model, self._goldfish_tok
  ```
- [ ] Implement `_compute_perplexity(predictions, direction)`:
  - Only run when `direction == "eng->kik"` (Goldfish is Kikuyu LM).
  - For each prediction, compute mean per-token NLL with sliding-window for
    sequences > 512 tokens. Return `{"perplexity": exp(mean_nll)}`.
- [ ] Wire it into `compute_all_metrics`.
- [ ] Update [`scripts/reporter.py`](scripts/reporter.py) so the Perplexity
      column reads `pred_perplexity` instead of leaving "—".
- [ ] Add a baseline row: Goldfish perplexity on the human-reference Kikuyu.

### Phase 7 — Pipeline test & full run
- [ ] Re-run `python main.py --dry-run` (10 sentences) including all new
      models. Confirm:
  - No model returns 0.00 across the board.
  - Translations look like Gĩkũyũ for kik-targeted models.
  - Perplexity column populated.
- [ ] Update [`results/`](results/) gitignore as needed.
- [ ] Run full 500-pair benchmark. Total estimated wall-clock with sequential
      loading on the dev box: ~3-4 hrs (Aya-101 dominates).

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| MADLAD `<2kik>` token resolves to `<unk>` | Low | Phase 1 verification script catches this before any code is written. |
| Gemma-3-27B INT4 OOM on 11 GB GPU | Medium | Use `device_map="auto"` to split across both 11 GB GPUs; fall back to CPU offload (1.5 TB RAM). Mark as `skip_on_low_memory`. |
| Goldfish perplexity dominated by domain mismatch (biblical training data vs agricultural test) | High | Document explicitly in paper. Report perplexity *delta* between models rather than absolute values. |
| `transformers>=4.50` breaks the `unsloth`-loaded kikuyu-translator | Medium | Pin `transformers` only at runtime; isolate the kikuyu-translator load path; the existing fallback to standard `transformers + PEFT` already exists in [`scripts/models/lora_finetuned.py`](scripts/models/lora_finetuned.py). |
| Llama-3.1 license/gating block | Low | License already accepted for Llama-3; same Meta gating model. Re-accept if needed. |
| Gemma-3 / MADLAD download size (~50 GB combined) | Medium | Pre-download with `huggingface-cli download` before benchmark run. Set `HF_HOME` to a partition with ≥ 100 GB free. |
| MADLAD MT model trained on 157 parallel pairs only — Kikuyu may have low parallel coverage | Medium | This is an empirical question the benchmark itself answers. Even low scores are publishable. |

---

## 6. Sequencing (recommended PR breakdown)

1. **PR-1**: Phase 1 (verification script) + Phase 2 (NLLB scaling). Lowest
   risk, immediately useful — gives a within-NLLB scaling curve.
2. **PR-2**: Phase 3 (Llama 3.1 swap).
3. **PR-3**: Phase 4 (MADLAD). Includes the small `transformer_mt.py` change.
4. **PR-4**: Phase 5 (Gemma-3 12B; 27B optional).
5. **PR-5**: Phase 6 (Goldfish perplexity in `evaluator.py`).
6. **PR-6**: Phase 7 (full re-run, results commit).

Each PR should include a `--dry-run` smoke test confirmation before merge.

---

## 7. Out of scope (intentionally not added)

- **NLLB-MoE-54B** — 108 GB FP16; even INT8 (54 GB) doesn't fit on the dev
  GPUs. CPU-offload with 1.5 TB RAM technically works but inference would
  take days. Defer.
- **MADLAD-10.7B** — feasible at INT8 split across both GPUs but adds
  marginal value over 7B. Skip unless reviewers ask for it.
- **Aya 23 / Aya Expanse 8B & 32B** — 23 supported languages, none of which
  are Kikuyu. The existing Aya-101 is the relevant Cohere model.
- **AfriTeVa V2** — only ~16 African languages, Kikuyu coverage uncertain;
  worth a tokenizer check later, but not a priority.
- **Helsinki-NLP/opus-mt-en-bnt** — Bantu family but excludes `kik`
  explicitly.
- **Toucan-1.2B / 3.7B** — see §1.

---

## 8. Definition of done

- All 9 evaluated models present in [`config/models.yaml`](config/models.yaml).
- `python main.py --dry-run` (10 sentences) completes with non-zero scores
  for every model in both directions.
- `python main.py` (full 500 pairs) completes and produces:
  - [`results/benchmark_<ts>/metrics/table_eng2kik.md`](results) populated
    for all 9 models, with Perplexity column populated for `eng->kik`.
  - [`results/benchmark_<ts>/metrics/table_kik2eng.md`](results) populated
    for all 9 models.
  - [`results/benchmark_<ts>/qualitative/`](results) spreadsheet with
    columns for all 9 models × 50 stratified samples × 2 directions.
- All four pre-existing failure modes (M2M-100 lang code, kikuyu-translator
  empty output, Aya/Llama not running in latest run) confirmed fixed in the
  same run.
