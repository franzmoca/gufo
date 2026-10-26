#!/bin/bash
# bench.sh OUTDIR REPS LABEL=BIN... : alternating gufo bench pp512/pp2048 (+tg16)
out=$1; reps=$2; shift 2; mkdir -p $out
M=/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf
for r in $(seq 1 $reps); do
  pairs=("$@"); [ $((r % 2)) = 0 ] && pairs=($(printf '%s\n' "$@" | tac))
  for lb in "${pairs[@]}"; do l=${lb%%=*}; b=${lb#*=}
    $b bench -m $M -p 512,2048 -n 16 -d 0 -r 1 > $out/$l-$r.txt 2>$out/$l-$r.err || echo "FAIL $l $r"
    echo "$l $r $(grep -E 'pp|tg' $out/$l-$r.txt | awk -F'|' '{printf "%s=%s ", $5, $6}')"
  done
done
