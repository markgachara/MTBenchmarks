#!/usr/bin/env bash
#
# Unattended post-training orchestrator.
#
# Waits for ALL fine-tuning to finish (the 6 main runs + the 2 distilled-1.3B
# runs), then runs the comprehensive evaluation on the corrected GAC v1.1 test
# set, populates the model cards with real numbers, and pre-merges the LoRA
# adapters into standalone models. It deliberately STOPS before any Hub push.
#
# Progress is written to logs/finetune/POSTTRAIN_STATUS.txt and a final
# POSTTRAIN_DONE marker line signals completion.
#
# Launched in the background; safe to run unattended.
set -uo pipefail
cd /home/irura/MTBenchmarks
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=logs/finetune/POSTTRAIN_STATUS.txt
: > "$STATUS"
log() { echo "$(date -Is)  $*" | tee -a "$STATUS"; }

log "orchestrator started; waiting for training to finish"

# The distilled-1.3B runner (ft2) only prints this line after BOTH distilled
# runs finish, and ft2 itself waits for the main ft queue — so this single
# marker guarantees all 8 fine-tuning runs are complete.
while ! grep -q "DISTILLED 1.3B DONE" logs/finetune/runner.log 2>/dev/null; do
  sleep 180
done
# Belt and braces: make sure no training process lingers.
while pgrep -f "scripts.finetune" >/dev/null; do sleep 60; done
log "all training finished"

n=$(ls -d models/finetuned/*/final 2>/dev/null | wc -l)
log "fine-tuned model dirs present: $n (expected 8)"

# Gate: never evaluate an incomplete model set. If fewer than 8 models are
# present, abort loudly rather than publishing a partial leaderboard.
if [[ "$n" -lt 8 ]]; then
  log "ABORT: only $n/8 fine-tuned models present; not running eval. Fix training and re-launch the orchestrator."
  log "POSTTRAIN_ABORTED missing_models=$((8 - n))"
  exit 1
fi

# ── Comprehensive eval: baselines + fine-tuned, corrected 450-pair test set ──
# Fine-tuned config is listed first so the card-critical NLLB models and their
# baselines are evaluated (and incrementally saved) before the slow LLMs.
log "starting evaluation on data/JulyDataUpdate/TEST_DATA.xlsx"
python main.py \
  --models-config config/models_finetuned.yaml config/models.yaml \
  --data-file data/JulyDataUpdate/TEST_DATA.xlsx \
  --output-dir results > logs/finetune/posttrain_eval.log 2>&1
EVAL_RC=$?
log "eval finished with exit code $EVAL_RC"

RUN=$(ls -dt results/benchmark_* 2>/dev/null | head -1)
log "latest results dir: $RUN"

# ── Populate model cards with real eval numbers ──────────────────────────────
if [[ -f "$RUN/metrics/full_metrics.json" ]]; then
  python -m scripts.generate_model_cards \
    --eval-metrics "$RUN/metrics/full_metrics.json" >> "$STATUS" 2>&1
  log "model cards regenerated with eval metrics"
else
  log "WARNING: no full_metrics.json found; cards left with pending values"
fi

# ── Pre-merge LoRA adapters into standalone models (NO push) ─────────────────
python -m scripts.push_to_hf --merge-only >> "$STATUS" 2>&1
log "LoRA models merged; standalone weights staged (no upload performed)"

log "POSTTRAIN_DONE eval_run=$RUN eval_rc=$EVAL_RC"
