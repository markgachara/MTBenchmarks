# Gĩkũyũ MT Benchmarking — Evaluation Plan

## Overview

This plan aligns the codebase with the paper "Benchmarking Machine Translation Models for Gĩkũyũ" and Mark's email (11 Apr 2026). The goal is a reproducible evaluation of 5 MT models across 2 translation directions on the 500-pair GAC test set, using a multi-metric framework combining quantitative MT metrics, corpus-level linguistic metrics, and qualitative human assessment.

---

## 1. Models to Evaluate

Per Table 1 of the paper and Mark's email:

| # | Model | HuggingFace ID | Category | Params | Eval Mode | VRAM Est. (FP16) |
|---|-------|---------------|----------|--------|-----------|-----------------|
| 1 | **NLLB-200 (600M)** | `facebook/nllb-200-distilled-600M` | Encoder-Decoder | 600M | Language token | ~1.2 GB |
| 2 | **M2M-100 (418M)** | `facebook/m2m100_418M` | Encoder-Decoder | 418M | Language token | ~0.8 GB |
| 3 | **SeamlessM4T** | `facebook/seamless-m4t-v2-large` | Reference Only | 2.3B | **NOT EVALUATED** | N/A |
| 4 | **kikuyu-translator-final** | `InterstellarCG/kikuyu-translator-final` | LoRA Fine-tuned (Gemma 3 4B) | 4B | Instruction prompt | ~6 GB |
| 5 | **Aya-101** | `CohereForAI/aya-101` | Encoder-Decoder Instruction-tuned (mT5-xxl) | 13B | Zero-shot + 3-shot prompting | ~26 GB (FP16), ~13 GB (INT8) |
| 6 | **Llama 3 8B Instruct** | `meta-llama/Meta-Llama-3-8B-Instruct` | Decoder-only LLM | 8B | Zero-shot + 3-shot prompting | ~16 GB (FP16), ~8 GB (INT8) |

### Hardware Available
- **GPU 0**: NVIDIA RTX 3060 (12 GB VRAM)
- **GPU 1**: NVIDIA RTX 2080 Ti (11 GB VRAM)
- **RAM**: 1.5 TB
- **CPUs**: 64 cores

### Hardware Strategy
- Models 1-2 (NLLB, M2M): Fit easily on either GPU (~1 GB each)
- Model 4 (kikuyu-translator): Fits on GPU 0 at FP16 (6 GB) or either GPU at INT8 (4 GB)
- Model 6 (Llama 3 8B): Fits on GPU 0 at INT8 (8 GB) or split across both GPUs
- Model 5 (Aya-101, 13B): Largest model — use INT8 quantization + split across both GPUs (~23 GB total), or offload to CPU RAM (1.5 TB available)
- **Sequential loading** — one model at a time, clear GPU cache between models

---

## 2. Test Dataset

### Primary: GAC 500-Pair Test Set (`data/selectpairs500.xlsx`)
- **500 sentence pairs** — human-validated English ↔ Gĩkũyũ
- **Domain**: Agricultural (non-religious, non-FLORES)
- **Columns**: `translatedText` (English), `data_to_quality_check` (Gĩkũyũ)
- **Dialect**: Predominantly KI-MURANGA (477/500), some GĨ-KABETE (20), KI-MATHIRA (2)
- **Avg sentence length**: English ~11 words, Gĩkũyũ ~12 words
- **Critical**: Excludes FLORES-200 sentences to avoid data leakage for NLLB/M2M

### Translation Directions
1. **English → Gĩkũyũ** (eng → kik): Source = `translatedText`, Reference = `data_to_quality_check`
2. **Gĩkũyũ → English** (kik → eng): Source = `data_to_quality_check`, Reference = `translatedText`

---

## 3. Evaluation Metrics

### 3.1 Quantitative MT-Specific Metrics

| Metric | Library | Notes |
|--------|---------|-------|
| **BLEU** | `sacrebleu` | Standardized tokenization via SacreBLEU (Post, 2018). Secondary metric. |
| **chrF++** | `sacrebleu` | Character n-gram order 6, word n-gram order 2. Primary surface-level metric. |
| **BERTScore F1** | `bert-score` | Using `bert-base-multilingual-cased`. Semantic similarity signal. |
| **AfriCOMET-MTL** | `unbabel-comet` + `masakhane/africomet-mtl` | **PRIMARY RANKING METRIC**. Requires (source, translation, reference) triplets. |

### 3.2 Quantitative Corpus-Level Linguistic Metrics

| Metric | Computation | Purpose |
|--------|------------|---------|
| **Perplexity** | Score generated Gĩkũyũ text against a Gĩkũyũ language model | Lower = more natural/predictable output |
| **Type-Token Ratio (TTR)** | Unique tokens / Total tokens for model output vs. reference | Lexical diversity; deviation from reference signals pathological behavior |
| **Hapax Legomena Rate** | Words appearing exactly once / Total unique words | Vocabulary richness indicator |
| **Avg Sentence Length** | Mean word count per sentence + std dev | Detect systematic compression/expansion |

### 3.3 Qualitative Human Assessment (Scaffolding Only)
- Stratified sample of **50 sentence pairs** (per paper Section 3.4.3) from the 500-pair test set
- Two fluent Gĩkũyũ speakers assess on 1-5 scale:
  - Diacritic accuracy
  - Idiomatic appropriateness
  - Avoidance of biblical register
  - Cultural concept handling
  - Addition, Omission, Mistranslation, Grammar (MQM categories)
- We will generate the output spreadsheet for human annotators

---

## 4. Prompting Strategy

### Encoder-Decoder Models (NLLB-200, M2M-100)
- Standard inference via language tokens
- Beam search, beam size = 5, max_new_tokens = 256

### LoRA Fine-Tuned (kikuyu-translator-final)
- Uses Gemma-3 chat template via `unsloth`
- Eng→Kik prompt: `"Translate the following English text to Kikuyu.\n\n{source}"`
- Kik→Eng prompt: `"Translate the following Kikuyu text to English.\n\n{source}"`

### LLMs via Prompting (Aya-101, Llama 3 8B)

**Zero-shot prompt:**
```
Translate the following sentence from English to Gĩkũyũ, preserving all diacritical marks accurately: {source}
```

**3-shot prompt (same examples for both models, drawn from OUTSIDE the test set):**
```
Translate sentences from English to Gĩkũyũ, preserving all diacritical marks accurately.

Example 1:
English: The government announced new policies to support farmers.
Gĩkũyũ: Thirikari nĩ ĩrahũrire watho mũerũ wa gũteithia arĩmi.

Example 2:
English: The children went to school early in the morning.
Gĩkũyũ: Ciana nĩ ciathiire thukuru rũciinĩ tene.

Example 3:
English: We need to plant trees to protect our environment.
Gĩkũyũ: Nĩ tũhĩthagio gũhaanda mĩtĩ nĩ guo tũrĩnde mahĩndĩ maitũ.

Now translate:
English: {source}
Gĩkũyũ:
```

**Note**: The 3 few-shot examples must be validated by Mark/a Gĩkũyũ speaker before use, and must NOT come from the 500-pair test set.

---

## 5. Implementation Steps

### Phase 1: Data & Config Alignment
1. ✅ Install dependencies: `sacrebleu`, `bert-score`, `unbabel-comet`, `unsloth`, `transformers`, `torch`, `pandas`, `openpyxl`
2. Write data loader for `selectpairs500.xlsx` (English ↔ Gĩkũyũ pairs)
3. Update `config/models.yaml` to match the paper's model list (replace Mistral with Aya-101 + Llama 3)
4. Update `config/datasets.yaml` to use GAC test set as primary

### Phase 2: Model Implementations
5. Update `scripts/models/llm_prompting.py` to handle:
   - Aya-101 (`AutoModelForSeq2SeqLM` — encoder-decoder, NOT decoder-only)
   - Llama 3 8B Instruct (decoder-only with chat template)
   - Both zero-shot and 3-shot prompting modes
6. Update `scripts/models/lora_finetuned.py` to use `unsloth` for kikuyu-translator-final
7. Verify encoder-decoder models (NLLB, M2M) work with correct language codes

### Phase 3: Metrics Implementation
8. Add AfriCOMET-MTL to evaluator (`unbabel-comet` library, `masakhane/africomet-mtl` model)
9. Replace simple chrF with chrF++ (word_order=2 in sacrebleu)
10. Add corpus-level linguistic metrics: perplexity, TTR, Hapax Legomena, avg sentence length
11. Compute baseline corpus-level metrics for human reference translations

### Phase 4: Pipeline Integration
12. Update `main.py` to:
    - Load GAC test set from Excel
    - Run models sequentially (memory management)
    - Support both zero-shot and 3-shot for LLMs
    - Compute all metrics per model per direction
    - Generate results in Tables 4 & 5 format from paper
13. Generate qualitative assessment spreadsheet (50 stratified samples × 5 models × 2 directions)

### Phase 5: Execution & Reporting
14. Run dry-run with 10 sentences to validate pipeline
15. Run full benchmark (500 pairs × 5 models × 2 directions)
16. Generate reports: CSV, JSON, Markdown matching paper tables
17. Export human assessment spreadsheet for Mark/annotators

---

## 6. Dependencies to Install

```bash
uv add torch transformers accelerate bitsandbytes   # Core ML
uv add sacrebleu bert-score unbabel-comet            # MT metrics
uv add sentencepiece protobuf                        # Tokenizers
uv add unsloth                                       # For kikuyu-translator (Gemma LoRA)
uv add pandas openpyxl                               # Data handling (already installed)
uv add pyyaml                                        # Config (already installed)
uv add evaluate                                      # HuggingFace evaluate
```

---

## 7. Expected Output Structure

```
results/
├── benchmark_YYYYMMDD/
│   ├── translations/
│   │   ├── nllb_200_eng2kik.json
│   │   ├── nllb_200_kik2eng.json
│   │   ├── m2m_100_eng2kik.json
│   │   ├── m2m_100_kik2eng.json
│   │   ├── kikuyu_translator_eng2kik.json
│   │   ├── kikuyu_translator_kik2eng.json
│   │   ├── aya_101_zeroshot_eng2kik.json
│   │   ├── aya_101_3shot_eng2kik.json
│   │   ├── aya_101_zeroshot_kik2eng.json
│   │   ├── aya_101_3shot_kik2eng.json
│   │   ├── llama3_8b_zeroshot_eng2kik.json
│   │   ├── llama3_8b_3shot_eng2kik.json
│   │   ├── llama3_8b_zeroshot_kik2eng.json
│   │   └── llama3_8b_3shot_kik2eng.json
│   ├── metrics/
│   │   ├── table4_eng2kik.csv          # Paper Table 4 format
│   │   ├── table5_kik2eng.csv          # Paper Table 5 format
│   │   └── full_metrics.json           # All metrics, all models
│   ├── qualitative/
│   │   └── human_assessment_template.xlsx  # 50 stratified samples for annotators
│   └── report.md                       # Full markdown report
```

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Aya-101 (13B) doesn't fit in GPU memory | Use INT8 quantization + device_map="auto" to split across both GPUs (23 GB total). Fallback: CPU offloading (1.5 TB RAM). |
| AfriCOMET unreliable for Gĩkũyũ (uncovered language) | Paper acknowledges this; use as primary but triangulate with chrF++ and BERTScore |
| kikuyu-translator uses `unsloth` which may conflict | Install in separate venv or use standard transformers with PEFT if issues arise |
| Few-shot examples may be inaccurate | Flag for Mark's review before final run |
| 3-shot examples contaminate test set | Examples explicitly drawn from outside the 500-pair set |
| Long runtime for 5 models × 2 directions × 500 sentences | Sequential GPU loading; estimate ~2-3 hours total with available hardware |

---

## 9. Alignment Checklist (Paper ↔ Implementation)

- [ ] Models match Table 1 exactly (5 evaluated + SeamlessM4T reference-only)
- [ ] Test set is GAC 500-pair (NOT FLORES-200)
- [ ] BLEU via SacreBLEU with standardized tokenization
- [ ] chrF++ with char_order=6, word_order=2
- [ ] BERTScore F1 with bert-base-multilingual-cased
- [ ] AfriCOMET-MTL via masakhane/africomet-mtl checkpoint
- [ ] Perplexity against a Gĩkũyũ language model
- [ ] TTR computed for model outputs AND human references
- [ ] Hapax Legomena rate computed
- [ ] Avg sentence length + std dev computed
- [ ] Zero-shot prompting for Aya-101 and Llama 3
- [ ] 3-shot prompting for Aya-101 and Llama 3
- [ ] Beam search (beam=5) for encoder-decoder models
- [ ] Results format matches Tables 4 & 5 in paper
- [ ] Qualitative assessment template generated (50 stratified pairs)
- [ ] Human reference baseline row in results tables
