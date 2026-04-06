# Quick Start Guide for MT Benchmarking

## 30-Second Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate setup
python test_validation.py

# 3. Run benchmarking
python main.py
```

## What Will Happen?

1. **Initialization** (~1 min)
   - Detect your hardware (CPU, GPU, RAM)
   - Print recommendations for batch size & quantization
   - Load model & dataset configurations

2. **Data Loading** (~5 min, first run only)
   - Download FLORES-200 benchmark (~100 MB)
   - Download KevinKibe validation set (~1 MB)
   - Cache locally for future runs

3. **Translation** (~30-60 min on GPU, 2-4 hours on CPU)
   - Each model translates test sentences
   - English→Gĩkũyũ (2500 sentences from FLORES + KevinKibe)
   - Gĩkũyũ→English (same sentences reversed)
   - Sequential loading to manage memory (~1 model at a time)

4. **Evaluation** (~2 min)
   - Compute BLEU scores
   - Compute ChrF scores
   - Compute BERTScore (if GPU available)
   - Compare speeds (tokens/sec)

5. **Reporting** (~30 sec)
   - Generate CSV, JSON, Markdown results
   - Print summary to console
   - Save to `./results/`

## Expected Runtime by Hardware

| Hardware | Runtime | Notes |
|----------|---------|-------|
| GPU 6GB | 90 min | Skips Mistral-24B (too large) |
| GPU 12GB | 45 min | All models tested |
| GPU 24GB | 30 min | Parallel-friendly batch sizes |
| CPU (8 cores) | 3+ hours | Slow but functional |

## Common Commands

```bash
# Full benchmark (all 6 models, both directions)
python main.py

# Fast test with 3 lightweight models only
python main.py --models nllb_200 m2m_100 seamless_m4t

# Test Kikuyu-specific model on small dev split
python main.py --models kikuyu_translator --eval-split dev

# Just test prompting capability (Mistral-7B only)
python main.py --models mistral_7b --skip-validation

# Custom output location
python main.py --output-dir ./my_bench_results

# Show all options
python main.py --help
```

## Interpreting Results

Results CSV has columns:
- `model`: Model name
- `direction`: eng→kik or kik→eng
- `dataset`: Which evaluation set (FLORES or KevinKibe)
- `bleu`: BLEU score (0-100)
- `chrf`: ChrF score (0-100)
- `bertscore`: Semantic similarity (0-1, if GPU available)
- `speed_tokens_per_sec`: Inference speed

**Expected score ranges for low-resource translation:**
- BLEU: 15-35 (Gĩkũyũ is underresourced)
- ChrF: 30-50
- Speed: 100-400 tokens/sec depending on model size

## Troubleshooting

### GPU not detected
```bash
# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

If False:
- Reinstall PyTorch for your GPU (https://pytorch.org)
- Pipeline will auto-fallback to CPU (slow)

### Out of memory
```bash
# Reduce batch size or model count
python main.py --models nllb_200 m2m_100  # Lightweight only
```

### Dataset download fails
```bash
# Use stored cache
rm -rf data/cache/
python main.py  # Re-downloads if needed
```

## Next Steps

Once completed:

1. **Review Results**: Open `./results/mt_benchmark_*.csv` in spreadsheet app
2. **Check Report**: Read `./results/mt_benchmark_*.md` for human summary
3. **Analyze**: Compare metric scores, identify best model for use case
4. **Iterate**: Run with different models (add custom ones via config)

## Quick Hardware Check

Before running full benchmark:

```python
# Check your machine
python3 << 'EOF'
from scripts.utils import get_hardware_info, print_hardware_info
hw = get_hardware_info()
print_hardware_info(hw)
EOF
```

This will show:
- CPU cores, speed
- Total/available RAM
- GPU (if present)
- Recommended batch size

## Next: Production Use

v1.0 is ready for:
- ✅ Comparing model performance
- ✅ Measuring translation speed
- ✅ Evaluating against FLORES benchmark
- ✅ Documenting results reproducibly

v2.0 planned features:
- Human evaluation
- Custom domain datasets
- Statistical significance testing
- Real-time serving
