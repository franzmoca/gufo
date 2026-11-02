#!/usr/bin/env bash
set -euo pipefail
model=/mnt/data/models/ds41/unsloth-ud-iq2xxs/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf
for label in base2 cand2 cand3 base3; do
 bin=/tmp/gufo-prefill-baseline
 if [[ $label == cand* ]]; then bin=/home/franz/projects/gufo/build/release/gufo; fi
 "$bin" bench -m "$model" --stream-experts --expert-cache-bytes 103079215104 -p 2048 -n 128 -d 8192 -r 3 > "/tmp/ds41-prefill-$label.log" 2>&1
 echo "$label complete"
done
