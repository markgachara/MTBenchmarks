#!/usr/bin/env bash
# Distilled-1.3B runner. Only writes the DONE marker if BOTH directions succeed,
# so a failed run cannot trip the post-training orchestrator into evaluating an
# incomplete model set.
cd /home/irura/MTBenchmarks
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ok=1
for dir in eng2kik kik2eng; do
  log="logs/finetune/nllb_distilled_1_3b_${dir}.log"
  echo "=== START nllb_distilled_1_3b ${dir} $(date -Is) ===" >> logs/finetune/runner.log
  if python -m scripts.finetune --run nllb_distilled_1_3b --direction "$dir" > "$log" 2>&1; then
    echo "=== OK nllb_distilled_1_3b ${dir} $(date -Is) ===" >> logs/finetune/runner.log
  else
    echo "=== FAILED nllb_distilled_1_3b ${dir} (see $log) ===" >> logs/finetune/runner.log
    tail -5 "$log" >> logs/finetune/runner.log
    ok=0
  fi
done
if [[ "$ok" == "1" ]]; then
  echo "=== DISTILLED 1.3B DONE $(date -Is) ===" >> logs/finetune/runner.log
else
  echo "=== DISTILLED 1.3B FAILED $(date -Is) ===" >> logs/finetune/runner.log
fi
