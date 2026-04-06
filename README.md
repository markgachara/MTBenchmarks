# MT Benchmarking Pipeline for Gĩkũyũ - v1.0

Lean, resource-efficient benchmarking system for machine translation models and datasets from Hugging Face, with focus on English ↔ Gĩkũyũ translation.

## Features

✅ **6 Machine Translation Models**
- NLLB-200 (600M distilled)
- M2M-100 (418M)
- SeamlessM4T (medium, 1.2B)
- Kikuyu-Translator (LoRA fine-tuned Gemma-3)
- Mistral-7B-Instruct
- Mistral-Small-3.1 (24B)

✅ **2 Evaluation Directions**
- English → Gĩkũyũ
- Gĩkũyũ → English

✅ **Multiple Datasets**
- FLORES-200 (primary benchmark)
- KevinKibe (validation)

✅ **Comprehensive Metrics**
- BLEU
- ChrF / ChrF++
- BERTScore (if GPU available)

✅ **Hardware-Aware**
- Automatic hardware detection
- Quantization recommendations (FP16, INT8)
- CPU-fallback mode (slow but works)
- Memory-efficient sequential loading

✅ **Complete Reporting**
- CSV results export
- JSON detailed results
- Markdown summary reports
- Speed metrics (tokens/sec)

## Installation

### Prerequisites
- Python 3.10+
- At least 12 GB RAM
- Optional: GPU with 6+ GB VRAM (for faster inference)

### Setup

```bash
# Clone/download the project
cd MTBenchmarks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_validation.py
```

## Quick Start

### 1. Run Full Benchmarking Pipeline

```bash
python main.py
```

This will:
- Detect your hardware
- Load all 6 models
- Download FLORES-200 + KevinKibe datasets (first run only)
- Translate with both directions
- Evaluate with all metrics
- Generate CSV, JSON, and Markdown reports

**Expected runtime:**
- GPU (12GB+ VRAM): ~30-60 minutes
- CPU-only: ~3-4 hours

### 2. Benchmark Specific Models

```bash
# Only test lightweight models
python main.py --models nllb_200 m2m_100 seamless_m4t

# Only test the Kikuyu-specific model
python main.py --models kikuyu_translator

# Only test Mistral-7B
python main.py --models mistral_7b
```

### 3. Use Different Dataset Split

```bash
# Use full dev split instead of devtest (smaller)
python main.py --eval-split dev

# Skip validation dataset
python main.py --skip-validation
```

### 4. Custom Output Location

```bash
python main.py \
  --output-dir ./my_results \
  --data-dir ./my_data_cache
```

## Project Structure

```
MTBenchmarks/
├── config/
│   ├── models.yaml           # Model specifications & registry
│   └── datasets.yaml         # Dataset configuration
├── scripts/
│   ├── __init__.py
│   ├── utils.py              # Utilities (hardware detection, config loading)
│   ├── data_loader.py        # Dataset loading & preprocessing
│   ├── translator.py         # Translation orchestrator
│   ├── evaluator.py          # Metric computation
│   ├── reporter.py           # Results aggregation & reporting
│   └── models/
│       ├── __init__.py
│       ├── base.py           # Abstract model class
│       ├── transformer_mt.py # NLLB, M2M, SeamlessM4T
│       ├── llm_prompting.py  # Mistral models
│       └── lora_finetuned.py # Kikuyu-Translator
├── data/                     # Downloaded datasets (created on first run)
├── results/                  # Output results
├── main.py                   # Main entry point
├── test_validation.py        # Validation tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

### models.yaml
Defines all 6 MT models with:
- Model IDs and Hugging Face repositories
- Parameter counts and download sizes
- VRAM requirements (FP32, FP16, INT8)
- Batch sizes and generation parameters
- Special handling (language code formats, prompting modes)

### datasets.yaml
Specifies datasets:
- FLORES-200 (professionals translated, 1000+ Gĩkũyũ sentences)
- KevinKibe (crowdsourced, ~420 sentences)
- Evaluation splits and metrics

### Override via Command-line
All paths can be overridden:
```bash
python main.py \
  --models-config ./my_models.yaml \
  --datasets-config ./my_datasets.yaml
```

## Output

Results are saved to `./results/` with timestamp:

### 1. CSV Results (`mt_benchmark_YYYYMMDD_HHMMSS.csv`)
Spreadsheet-friendly format with columns:
- `model`: Model name
- `direction`: eng->kik or kik->eng
- `dataset`: Dataset used (FLORES, KevinKibe)
- `bleu`: BLEU score
- `chrf`: ChrF score
- `bertscore`: BERTScore (if computed)
- `speed_tokens_per_sec`: Inference speed
- `valid`: Whether translation succeeded

### 2. JSON Results (`mt_benchmark_YYYYMMDD_HHMMSS.json`)
Detailed results including:
- Full metric values
- Metric precisions (BLEU n-grams)
- Model load times
- Batch statistics

### 3. Markdown Report (`mt_benchmark_YYYYMMDD_HHMMSS.md`)
Human-readable summary:
- Hardware configuration
- Best performing models per metric
- Speed comparison
- Data tables
- Notes on interpretation

## Advanced Usage

### Monitor GPU Memory

```bash
# In another terminal, monitor GPU usage
nvidia-smi -l 1
```

### Profile Model Memory

The pipeline automatically profiles each model's memory usage:
```
Loading NLLB-200 (600M Distilled)
  Speed: 245.3 tokens/sec
  Time: 34.21s
```

### Extend with Custom Models

1. Add model definition to `config/models.yaml`
2. Create subclass in `scripts/models/` if needed
3. Update model factory in `scripts/translator.py`

### Extend with Custom Datasets

1. Add dataset config to `config/datasets.yaml`
2. Implement loader in `scripts/data_loader.py`
3. Ensure parallel format (source/target pairs)

## Hardware Recommendations

| Configuration | Expected Runtime | Setup |
|---|---|---|
| **CPU-only (8 cores)** | 3-4 hours | Baseline, very slow |
| **GPU (6GB)** | 1-2 hours | Good for lightweight models only |
| **GPU (12GB)** | 30-60 min | Ideal; can test all 6 models |
| **GPU (24GB)** | 20-30 min | Excellent; all models with larger batches |

## Troubleshooting

### Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size in `config/models.yaml`
- Use INT8 quantization instead of FP16
- Skip large models (Mistral-24B)
- Use CPU-only mode

### Model Download Fails
```
ConnectionError: Failed to download model
```
**Solution:**
- Check internet connection
- Increase HF timeout: `export HF_DATASETS_TIMEOUT=60`
- Manually download from https://huggingface.co

### Dataset Not Found
```
FileNotFoundError: FLORES-200 not found
```
**Solution:**
- First run requires internet connection to download datasets
- Check `./data/cache/` for partially downloaded files
- Delete and retry

## Performance Tips

1. **On CPU**: Use only lightweight models (NLLB, M2M)
2. **On GPU**: Increase batch size for better throughput
3. **Long texts**: Reduce `max_new_tokens` in config
4. **Quick test**: Use `--eval-split dev` (smaller dataset)
5. **Memory critical**: Skip BERTScore (doesn't affect ranking)

## Benchmarking Methodology

### Metrics Interpretation
- **BLEU** (0-100): Surface-level n-gram match. Baseline metric.
- **ChrF** (0-100): Character-level F-score. Better for morphology.
- **ChrF++** (0-100): ChrF with word level. Best for low-resource.
- **BERTScore** (0-1): Semantic similarity via embeddings. Correlates with human.

### Important Notes
- Gĩkũyũ is low-resource; expect baseline scores lower than high-resource pairs
- FLORES-200 is professional translation; high quality reference
- Results depend on quantization (FP32 > FP16 > INT8 quality)
- Batch size affects reproducibility; use same settings for comparison

## Extending v1.0

Planned features for future versions:
- [ ] Human evaluation interface
- [ ] Statistical significance testing
- [ ] Domain-specific evaluations (agriculture, news, etc.)
- [ ] Real-time serving/API
- [ ] Multi-GPU distribution
- [ ] Custom fine-tuning workflows

## Contributing

To add support for new models or datasets:

1. **New Model**: Create subclass of `BaseModel` in `scripts/models/`
2. **New Dataset**: Add loader in `scripts/data_loader.py`
3. **New Metric**: Add to evaluator in `scripts/evaluator.py`
4. **Update configs**: Add entries to `config/models.yaml` or `datasets.yaml`

## License

[Specify your license]

## Citation

If you use this benchmarking system, please cite:

```bibtex
@software{mtbenchmarks2026,
  title={MT Benchmarking Pipeline for Gĩkũyũ},
  version={1.0},
  year={2026}
}
```

## Support

For issues, questions, or contributions:
- Check existing documentation in this README
- Review output logs in `./results/` 
- Validate setup with `python test_validation.py`

## References

- **FLORES-200**: https://github.com/openlanguagedata/flores
- **NLLB**: https://huggingface.co/facebook/nllb-200-distilled-600M
- **M2M-100**: https://huggingface.co/facebook/m2m100_418M
- **SeamlessM4T**: https://huggingface.co/facebook/seamless-m4t-medium
- **SacreBLEU**: https://github.com/mjpost/sacreBLEU
- **Evaluate Library**: https://huggingface.co/docs/evaluate

---

**Last updated**: April 2026
**Status**: v1.0 Production Ready
