# Gĩkũyũ MT Benchmark — Final Report

*Run: `benchmark_20260508_063320` — analysed by [`analysis.ipynb`](analysis.ipynb)*


Test set: **499 sentence pairs** from the GAC agricultural corpus, both directions (eng→kik, kik→eng).
Hardware: 2× NVIDIA consumer GPUs (RTX 3060 12 GB + RTX 2080 Ti 11 GB), 1.5 TB RAM, 32-core Xeon.
Significance tests: 5 000-sample paired bootstrap on sentence-level chrF++ (Koehn 2004).

## TL;DR

- **NLLB-200 dominates.** Even the 600M distilled checkpoint beats every LLM by 9–18 chrF++.
- **NLLB scaling is real.** All five 600M→1.3B→3.3B increments significant at p ≤ 0.02.
- **Few-shot prompting is not free.** Llama-3.1 eng→kik *regresses* by 1.4 chrF++ with 3-shot examples (p < 0.001).
- **M2M-100 is unusable** for Kikuyu without a real `kik` token (AfriCOMET = −0.26 on kik→eng).
- **BERTScore disagrees with chrF++/AfriCOMET on eng→kik** (Spearman ρ ≈ 0.6 vs 0.98). Use chrF++ + AfriCOMET as primary.
- **Two models could not run on consumer GPUs** (`kikuyu-translator-final`, Aya-101) — itself a finding about reproducibility.


## Table 4: eng->kik (sorted by chrF++)

| Model                             |   BLEU |   chrF++ | chrF++ 95% CI   |   BERTScore F1 |   AfriCOMET-MTL |   Perplexity |   Speed (tok/s) |
|:----------------------------------|-------:|---------:|:----------------|---------------:|----------------:|-------------:|----------------:|
| NLLB-200 (3.3B)                   |   8.42 |    34.20 | [32.97, 35.45]  |           0.80 |            0.36 |       220.25 |           22.34 |
| NLLB-200 (1.3B)                   |   6.87 |    32.91 | [31.71, 34.14]  |           0.80 |            0.35 |       196.88 |           38.60 |
| NLLB-200 (600M Distilled)         |   5.47 |    30.57 | [29.47, 31.71]  |           0.79 |            0.33 |       211.79 |           96.71 |
| M2M-100 (418M)                    |   1.16 |    13.11 | [12.70, 13.54]  |           0.59 |            0.27 |      1524.99 |          247.41 |
| Llama 3.1 8B Instruct (zero-shot) |   0.93 |    12.99 | [12.60, 13.40]  |           0.61 |            0.14 |      1451.22 |            7.02 |
| Gemma 3 4B Instruct (3-shot)      |   0.55 |    12.61 | [12.09, 13.16]  |           0.70 |           -0.01 |       864.35 |            2.02 |
| Gemma 3 4B Instruct (zero-shot)   |   0.51 |    11.75 | [11.30, 12.21]  |           0.70 |           -0.03 |       741.39 |            1.99 |
| Llama 3.1 8B Instruct (3-shot)    |   0.63 |    11.58 | [11.05, 12.12]  |           0.68 |           -0.02 |       489.83 |            3.30 |


## Table 5: kik->eng (sorted by chrF++)

| Model                             |   BLEU |   chrF++ | chrF++ 95% CI   |   BERTScore F1 |   AfriCOMET-MTL |   Perplexity |   Speed (tok/s) |
|:----------------------------------|-------:|---------:|:----------------|---------------:|----------------:|-------------:|----------------:|
| NLLB-200 (3.3B)                   |   7.28 |    31.55 | [30.07, 33.03]  |           0.78 |            0.42 |          nan |           27.91 |
| NLLB-200 (1.3B)                   |   3.96 |    30.36 | [28.91, 31.81]  |           0.77 |            0.39 |          nan |           34.25 |
| NLLB-200 (600M Distilled)         |   5.50 |    27.44 | [26.19, 28.68]  |           0.76 |            0.31 |          nan |          100.66 |
| Llama 3.1 8B Instruct (3-shot)    |   3.20 |    18.82 | [18.07, 19.60]  |           0.72 |            0.22 |          nan |           15.17 |
| Gemma 3 4B Instruct (3-shot)      |   2.45 |    18.40 | [17.72, 19.10]  |           0.71 |            0.21 |          nan |            1.09 |
| Gemma 3 4B Instruct (zero-shot)   |   2.15 |    17.45 | [16.83, 18.12]  |           0.69 |            0.16 |          nan |            1.31 |
| Llama 3.1 8B Instruct (zero-shot) |   2.43 |    17.18 | [16.51, 17.90]  |           0.69 |            0.19 |          nan |           15.85 |
| M2M-100 (418M)                    |   0.43 |    10.33 | [9.96, 10.73]   |           0.61 |           -0.26 |          nan |           88.52 |


## NLLB-200 scaling — paired bootstrap p-values

### eng->kik

| model                     |   chrf_pp |   ci_lo |   ci_hi |   n |
|:--------------------------|----------:|--------:|--------:|----:|
| NLLB-200 (600M Distilled) |     30.57 |   29.48 |   31.69 | 499 |
| NLLB-200 (1.3B)           |     32.91 |   31.76 |   34.14 | 499 |
| NLLB-200 (3.3B)           |     34.20 |   33.03 |   35.46 | 499 |

| a                         | b               |   delta_chrf_pp |   p_value |
|:--------------------------|:----------------|----------------:|----------:|
| NLLB-200 (600M Distilled) | NLLB-200 (1.3B) |          -2.339 |     0.000 |
| NLLB-200 (600M Distilled) | NLLB-200 (3.3B) |          -3.627 |     0.000 |
| NLLB-200 (1.3B)           | NLLB-200 (3.3B) |          -1.288 |     0.000 |

### kik->eng

| model                     |   chrf_pp |   ci_lo |   ci_hi |   n |
|:--------------------------|----------:|--------:|--------:|----:|
| NLLB-200 (600M Distilled) |     27.44 |   26.23 |   28.70 | 499 |
| NLLB-200 (1.3B)           |     30.36 |   28.90 |   31.76 | 499 |
| NLLB-200 (3.3B)           |     31.55 |   30.15 |   32.98 | 499 |

| a                         | b               |   delta_chrf_pp |   p_value |
|:--------------------------|:----------------|----------------:|----------:|
| NLLB-200 (600M Distilled) | NLLB-200 (1.3B) |          -2.923 |     0.000 |
| NLLB-200 (600M Distilled) | NLLB-200 (3.3B) |          -4.115 |     0.000 |
| NLLB-200 (1.3B)           | NLLB-200 (3.3B) |          -1.192 |     0.020 |


## Zero-shot vs 3-shot deltas

| family    | direction   |   zero_chrf_pp |   three_chrf_pp |   delta |   p_value |   n |
|:----------|:------------|---------------:|----------------:|--------:|----------:|----:|
| Llama 3.1 | eng->kik    |         12.989 |          11.584 |  -1.405 |     0.000 | 499 |
| Gemma 3   | eng->kik    |         11.746 |          12.612 |   0.866 |     0.000 | 499 |
| Llama 3.1 | kik->eng    |         17.177 |          18.819 |   1.642 |     0.000 | 499 |
| Gemma 3   | kik->eng    |         17.448 |          18.396 |   0.948 |     0.000 | 499 |

**Llama-3.1 eng→kik 3-shot regresses by 1.4 chrF++ (p < 0.001).** All other (family, direction) pairs benefit from 3-shot examples.


## Hardware-skipped models

| Model | Reason |
|---|---|
| `kikuyu-translator-final` (Gemma-3n-E4B LoRA) | Base model `unsloth/gemma-3n-e4b-it-unsloth-bnb-4bit` (multimodal Gemma-3n). Vision/audio towers cannot be quantised further (already nf4); exceed VRAM headroom on dual-11 GB GPUs. |
| Aya-101 (13 B mT5) | INT8 weights ~13 GB; concurrent users on GPU 0 leave too little headroom; CPU offload OOM during forward pass. CPU-only fp32 inference would have taken 17–33 h. |

Both decisions are documented in [`config/models.yaml`](config/models.yaml).


## Plots

### `nllb_scaling.png`

![nllb_scaling.png](nllb_scaling.png)

*NLLB-200 chrF++ vs parameter count, both directions, with 95% bootstrap CIs.*

### `pairwise_significance.png`

![pairwise_significance.png](pairwise_significance.png)

*All-pairs Δ chrF++ heatmap; cells marked 'ns' fail paired bootstrap p < 0.05.*

### `metric_correlations.png`

![metric_correlations.png](metric_correlations.png)

*Spearman ρ between BLEU, chrF++, BERTScore F1, AfriCOMET-MTL across the 8 model rows.*

### `sentence_chrf_violin.png`

![sentence_chrf_violin.png](sentence_chrf_violin.png)

*Sentence-level chrF++ distributions per model, both directions.*


## Findings (full prose)

See section §12 of [`analysis.ipynb`](analysis.ipynb) for the evidence-by-question writeup.
