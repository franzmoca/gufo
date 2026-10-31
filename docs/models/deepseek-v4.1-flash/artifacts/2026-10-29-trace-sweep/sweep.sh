#!/usr/bin/env bash
# Timing (unprofiled) then profiled run of each model x mode; MODELS/MODES select a subset.
cd "$(dirname "$0")/../../../../.."
B=${GUFO:-build/release/gufo}; O=${OUT:-/tmp/ds41-trace}; A=docs/models/deepseek-v4.1-flash/artifacts/2026-10-29-trace-sweep
R=${GUFO_MODEL_DIR:-/mnt/data/models/ds41}/ista-gsq-rco-gguf
D=${GUFO_MODEL_DIR:-/mnt/data/models/ds41}/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf
mkdir -p "$O"
declare -A MODEL=( [iq1m]="-m $R/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf"
                   [iq2xxs]="-m $R/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf --stream-experts" )
P="-f $A/code.txt --no-display-prompt -n 256 --think off -v"
declare -A MODE=( [pp]="bench -p 2048 -n 0 -r 2"
                  [ar-greedy]="prompt $P -t 0 --speculative off"
                  [ar-sampled]="prompt $P -s 1 --speculative off"
                  [ds-greedy]="prompt $P -t 0 --speculative dspark --dspark-model $D"
                  [ds-sampled]="prompt $P -s 1 --speculative dspark --dspark-model $D" )
for m in ${MODELS:-iq1m iq2xxs}; do
  for k in ${MODES:-pp ar-greedy ar-sampled ds-greedy ds-sampled}; do
    sub=${MODE[$k]%% *}; args=${MODE[$k]#* }
    echo "=== $m $k $(date +%T)"
    timeout 900 $B $sub ${MODEL[$m]} $args > $O/$m-$k.time.txt 2>&1; echo "time rc=$?"
    rm -rf $O/$m-$k.prof
    timeout 1200 nix develop -c python3 tools/prof/prof.py run -o $O/$m-$k.prof -- $B $sub ${MODEL[$m]} $args > $O/$m-$k.prof.txt 2>&1; echo "prof rc=$?"
  done
done
echo DONE
