#!/bin/bash
# usage: edit.sh BIN LABEL
cd /tmp/ds41-review
M=/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf
D=/mnt/data/models/ds41/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf
P="Here is a C++ file:

$(cat p1c/src.txt)

Output the complete file again unchanged, except rename the function lin to linear_q4. Output only the code."
GUFO_DEEPSEEK41_DSPARK_TRACE=1 timeout 1200 ./$1 prompt -m $M --dspark-model $D --speculative dspark --think off -n 900 --no-display-prompt -v "$P" > p1c/$2.out 2> p1c/$2.err
echo "$2 rc=$?"; grep -E "Generated|Decode|tok/s|\[Speculative\]|TokenTrace" p1c/$2.err | grep -v "^ds41:"
