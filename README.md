# MT Benchmarking Pipeline for Gĩkũyũ

Reproducible benchmark of machine-translation models on **English ↔ Gĩkũyũ**, aligned with the paper *"Benchmarking Machine Translation Models for Gĩkũyũ"* (Irura, 2026). Evaluates **8 model configurations** (6 unique models, with 2 of them run in both zero-shot and 3-shot prompting modes) on the 500-pair GAC agricultural test set with a multi-metric framework, statistical significance tests, and six deep-dive analyses (diacritics, length effect, hallucination, code-switching, per-sentence metric agreement).

See [results/benchmark_20260508_063320/FINAL_REPORT.md](results/benchmark_20260508_063320/FINAL_REPORT.md) for the latest run's full report (8 plots, all tables, all p-values).

---

## What's in the benchmark

### Models evaluated (8 configurations)

| Model | Type | Quantization | Notes |
|---|---|---|---|
| NLLB-200 600M / 1.3B / 3.3B | Encoder-decoder MT | FP16 | Within-family scaling curve |
| M2M-100 418M | Encoder-decoder MT | FP16 | No `kik` token; uses Swahili proxy |
| Llama 3.1 8B Instruct | Decoder-only LLM | INT8 | Zero-shot + 3-shot prompting |
| Gemma 3 4B Instruct | Decoder-only LLM | BF16 | Zero-shot + 3-shot prompting |

### Models declared but skipped (consumer-GPU constraints)

| Model | Reason |
|---|---|
| `InterstellarCG/kikuyu-translator-final` | Built on `unsloth/gemma-3n-e4b-it-unsloth-bnb-4bit` (multimodal Gemma-3n); vision/audio towers exceed VRAM headroom on dual-11 GB GPUs even with text weights at nf4. |
| `CohereForAI/aya-101` (13 B mT5) | INT8 weights ~13 GB; CPU offload OOMs during forward pass. |

Documented in [config/models.yaml](config/models.yaml). Treated as a finding about reproducibility for the African-NLP community, not a flaw.

### Verified non-runnable (dropped from the plan)

`MADLAD-400-3B/7B` — `<2kik>` resolves to `<unk>`. Confirmed via [scripts/verify_lang_support.py](scripts/verify_lang_support.py).

### Test set

- **GAC 500-pair test set** ([data/selectpairs500.xlsx](data/selectpairs500.xlsx)) — agricultural domain, predominantly KI-MURANGA dialect, 79/499 references contain `[cs]english_borrowing[cs]` code-switch markers.
- Both directions evaluated: English → Gĩkũyũ and Gĩkũyũ → English.
- 1 pair dropped for missing translation; final n = 499.

### Metrics

Reference-based, per [config/datasets.yaml](config/datasets.yaml):

- **BLEU** via SacreBLEU (Post 2018) — secondary, surface n-gram.
- **chrF++** with `char_order=6, word_order=2` — surface, character-level. Robust to morphology.
- **BERTScore F1** with `bert-base-multilingual-cased` — semantic. **Disagrees with chrF++ for eng→kik** in this benchmark (Spearman ρ ≈ 0.6 vs 0.98 the other way).
- **AfriCOMET-MTL** (`masakhane/africomet-mtl`) — primary ranking metric for African languages.
- **Goldfish-Kikuyu perplexity** (`goldfish-models/kik_latn_full`) — fluency relative to a Kikuyu LM. Note: the LM is biblical/Wikipedia-trained, so absolute values reflect domain mismatch with the agricultural test set.

Plus corpus-level linguistic metrics: TTR, hapax legomena rate, average sentence length.

### Statistical analysis

- **Sentence-level chrF++** computed per (model, direction) sentence — see [analysis.ipynb](analysis.ipynb).
- **5 000-sample paired bootstrap** (Koehn 2004) for all pairwise model comparisons.
- **95 % bootstrap CIs** on every chrF++ point estimate.
- **Spearman correlations** between metrics across 8 model rows per direction.

### Deep-dive analyses (notebook §13)

Six paper-publishable findings exploit metadata that the v1.0 corpus-level pipeline left unused:

| § | Analysis | Headline result |
|---|---|---|
| 13.1 | Diacritic F1 on `ĩ ũ Ĩ Ũ` | Only NLLB learnt Gĩkũyũ orthography (F1 0.74). Llama zero-shot = 0.15, M2M = 0.004. |
| 13.2 | chrF++ ~ source length OLS | All 16 slopes positive — **chrF++ rewards verbosity**. |
| 13.3 | Wrong-script + 4-gram repetition | Llama zero-shot eng→kik is 69 % English-script; Gemma 3-shot has 22 % looping 4-grams. |
| 13.4 | Code-switching robustness | NLLB loses 5–7 chrF++ on `[cs]` strata (p < 0.001). |
| 13.5 | Per-sentence metric agreement | chrF++ vs AfriCOMET disagree on the per-sentence winner ~60 % of the time. |
| 14 | Synthesis + paper recommendations | Add diacritic F1 as Table 4a; report length-stratified scores; flag wrong-script rate; report chrF++ + AfriCOMET *together*. |

---

## Quick start

### Hardware

Tested on dual NVIDIA consumer GPUs (RTX 3060 12 GB + RTX 2080 Ti 11 GB) with 1.5 TB RAM and a 32-core Xeon. CPU-only mode works but is impractical for the LLMs.

### Setup

```bash
git clone https://github.com/markgachara/MTBenchmarks
cd MTBenchmarks

uv sync                              # or: pip install -r requirements.txt
huggingface-cli login                 # required for gated Gemma-3 / Llama-3.1
python scripts/verify_lang_support.py # confirms tokenizers know about kik
```

### Run

```bash
# Full benchmark (~6 hours on dual 11 GB GPUs)
python main.py

# Subset of models for fast iteration
python main.py --models nllb_200_3_3b llama3_1_8b --dry-run    # 10 sentences

# Skip expensive metrics if iterating
python main.py --skip-bertscore --skip-africomet
```

Run-time tips:
- Sequential model loading is automatic; one model in VRAM at a time.
- Cache is at `~/.cache/huggingface` (symlink to wherever you have space; we use `/work/irura/hf_cache`).
- Aya-101 / kikuyu-translator-final are auto-skipped via `skip_on_low_memory` in the YAML; force-enable at your own peril.

### Reproduce the analysis

```bash
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb
```

This regenerates eight plots and writes `FINAL_REPORT.md` next to the most recent run.

---

## Project layout

```
MTBenchmarks/
├── main.py                                # entry point, CLI
├── analysis.ipynb                         # statistical + deep-dive analysis
├── config/
│   ├── models.yaml                        # model registry (per-model VRAM, batch size, prompting)
│   └── datasets.yaml                      # GAC test set + metric configuration
├── data/
│   └── selectpairs500.xlsx                # GAC 500-pair test set
├── scripts/
│   ├── data_loader.py
│   ├── evaluator.py                       # BLEU, chrF++, BERTScore, AfriCOMET, perplexity
│   ├── reporter.py                        # CSV + Markdown + JSON outputs
│   ├── translator.py                      # sequential-loading orchestrator
│   ├── verify_lang_support.py             # Phase-1 gate; checks Kikuyu token coverage
│   └── models/
│       ├── base.py                        # BaseModel abstract class
│       ├── transformer_mt.py              # NLLB, M2M-100
│       ├── llm_prompting.py               # Llama-3.1, Gemma-3, Aya-101
│       └── lora_finetuned.py              # kikuyu-translator-final (PEFT)
├── plans.md / tasks.md                    # implementation plan + dependency-ordered tasks
├── EVALUATION_PLAN.md                     # original paper-alignment plan
└── results/
    └── benchmark_<timestamp>/
        ├── translations/                  # per-model JSONL of (src, hyp, ref)
        ├── metrics/                       # CSV + Markdown tables, full_metrics.json
        ├── qualitative/                   # human assessment template (50 stratified pairs)
        ├── *.png                          # 8 analysis plots
        ├── report.md                      # auto-generated summary
        └── FINAL_REPORT.md                # paper-ready writeup
```

---

## Headline results (run `benchmark_20260508_063320`, 499 pairs)

### Table 4 — English → Gĩkũyũ (sorted by chrF++)

| Model | BLEU | chrF++ | 95 % CI | AfriCOMET | Diacritic F1 | Wrong-script |
|---|---:|---:|---|---:|---:|---:|
| NLLB-200 (3.3B) | 8.42 | **34.20** | [33.00, 35.45] | 0.36 | 0.73 | 0.4 % |
| NLLB-200 (1.3B) | 6.87 | 32.91 | [31.71, 34.14] | 0.35 | 0.74 | 0.4 % |
| NLLB-200 (600M) | 5.47 | 30.57 | [29.47, 31.71] | 0.33 | 0.74 | 0.2 % |
| M2M-100 | 1.16 | 13.11 | [12.70, 13.54] | 0.27 | **0.004** | **100 %** |
| Llama 3.1 (zero-shot) | 0.93 | 12.99 | [12.60, 13.40] | 0.14 | 0.15 | 68.9 % |
| Gemma 3 (3-shot) | 0.55 | 12.61 | [12.09, 13.16] | −0.01 | 0.48 | 0.0 % |
| Gemma 3 (zero-shot) | 0.51 | 11.75 | [11.30, 12.21] | −0.03 | 0.51 | 0.4 % |
| Llama 3.1 (3-shot) | 0.63 | 11.58 | [11.05, 12.12] | −0.02 | 0.47 | 0.8 % |

### Table 5 — Gĩkũyũ → English (sorted by chrF++)

| Model | BLEU | chrF++ | 95 % CI | AfriCOMET |
|---|---:|---:|---|---:|
| NLLB-200 (3.3B) | 7.28 | **31.55** | [30.07, 33.03] | 0.42 |
| NLLB-200 (1.3B) | 3.96 | 30.36 | [28.91, 31.81] | 0.39 |
| NLLB-200 (600M) | 5.50 | 27.44 | [26.19, 28.68] | 0.31 |
| Llama 3.1 (3-shot) | 3.20 | 18.82 | [18.07, 19.60] | 0.22 |
| Gemma 3 (3-shot) | 2.45 | 18.40 | [17.72, 19.10] | 0.21 |
| Gemma 3 (zero-shot) | 2.15 | 17.45 | [16.83, 18.12] | 0.16 |
| Llama 3.1 (zero-shot) | 2.43 | 17.18 | [16.51, 17.90] | 0.19 |
| M2M-100 | 0.43 | 10.33 | [9.96, 10.73] | **−0.26** |

### Six headline findings

1. **NLLB-200 dominates.** Even 600M beats every LLM by 9–18 chrF++.
2. **NLLB scaling is real.** All five 600M→1.3B→3.3B increments significant at p ≤ 0.020 worst-case (5 000-sample paired bootstrap).
3. **Few-shot prompting is direction-dependent and family-dependent.** Llama-3.1 eng→kik 3-shot regresses by 1.4 chrF++ (p < 0.001) — but its diacritic F1 *triples* (0.15 → 0.47).
4. **The LLM chrF++ floor on eng→kik is hallucination, not partial competence.** Llama zero-shot is 69 % English-script; Gemma 3-shot has 22 % repeated 4-grams.
5. **Code-switching markers cost NLLB 5–7 chrF++** (p < 0.001). LLMs show no CS sensitivity because their baseline is already at the floor.
6. **chrF++ and AfriCOMET disagree on the per-sentence winner ~60 % of the time** (33.7 % eng→kik, 41.9 % kik→eng) despite Spearman ρ ≥ 0.95 on aggregate scores. Argues against picking a single primary metric.

Full prose findings: [analysis.ipynb §12 + §14](analysis.ipynb), tables: [results/benchmark_20260508_063320/FINAL_REPORT.md](results/benchmark_20260508_063320/FINAL_REPORT.md).

---

## Configuration cheat sheet

[config/models.yaml](config/models.yaml) entries support:

```yaml
my_model:
  name: "Display name for tables"
  model_id: "huggingface/repo"
  model_type: "transformer_mt | llm_prompting | lora_finetuned"
  parameters: 600_000_000
  vram_fp16_gb: 1.2
  dtype: "torch.bfloat16"
  is_encoder_decoder: true
  batch_size: 8
  max_new_tokens: 256
  num_beams: 5
  device_map: "auto"
  lang_code_format: "flores"   # for transformer_mt — flores | m2m
  model_class: "gemma3"        # for llm_prompting — opt-in to Gemma3ForCausalLM
  quantization: "int4 | int8"  # opt-in bitsandbytes
  prompting_modes: ["zero_shot", "few_shot"]   # for llm_prompting
  skip_on_low_memory: false
```

[config/datasets.yaml](config/datasets.yaml) declares the test set, metric configuration, and qualitative-assessment parameters.

---

## Development log

Implementation followed a phased plan with paired bootstrap validation at every step:

- **Phase 0–1**: Environment setup, [verify_lang_support.py](scripts/verify_lang_support.py) gate. **Caught MADLAD-400 unsupported before any wasted code.**
- **Phase 2**: NLLB-200 1.3B + 3.3B (config-only).
- **Phase 3**: Llama 3 → Llama 3.1 swap with INT8.
- **Phase 5**: Gemma 3 4B with `model_class: gemma3` and forced-greedy decoding (Gemma's shipped `do_sample=true` collides with bnb-INT8 on Turing).
- **Phase 6**: Goldfish-Kikuyu perplexity scoring in [evaluator.py](scripts/evaluator.py).
- **Phase 7**: Documented kikuyu-translator-final and Aya-101 as hardware-skipped findings.
- **Phase 8**: Full 500-pair run with sequential loading + batched LLM generation.
- **Phase 9**: [analysis.ipynb](analysis.ipynb) end-to-end execution, eight plots, FINAL_REPORT.
- **Phase 9b**: Six novel deep-dive analyses (diacritics, length effect, hallucination, CS robustness, metric agreement, synthesis).

Branch history: `git log --oneline feature/evaluation-pipeline-alignment`. The default branch (`main`) is the v1.0 starting point.

---

## Citation

```bibtex
@misc{mtbenchmarks2026,
  title  = {MT Benchmarking Pipeline for Gĩkũyũ},
  author = {Irura, Mark},
  year   = {2026},
  url    = {https://github.com/markgachara/MTBenchmarks}
}
```

## References

- NLLB-200 — Costa-jussà et al. 2022, [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)
- M2M-100 — Fan et al. 2020, [arXiv:2010.11125](https://arxiv.org/abs/2010.11125)
- Aya-101 — Üstün et al. 2024, [arXiv:2402.07827](https://arxiv.org/abs/2402.07827)
- Llama 3.1 — Meta 2024, [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Gemma 3 — Google 2025, [Gemma 3 technical report](https://goo.gle/Gemma3Report)
- Goldfish — Chang et al. 2024, [arXiv:2408.10441](https://arxiv.org/abs/2408.10441)
- AfriCOMET — Wang et al. 2024, [masakhane/africomet-mtl](https://huggingface.co/masakhane/africomet-mtl)
- chrF++ / SacreBLEU — Popović 2017, Post 2018
- Paired bootstrap test — Koehn 2004, *Statistical Significance Tests for Machine Translation Evaluation*

---

**Last updated**: 9 May 2026
**Latest run**: [`benchmark_20260508_063320`](results/benchmark_20260508_063320/)
**Current branch**: `feature/evaluation-pipeline-alignment`
