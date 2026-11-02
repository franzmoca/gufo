#!/usr/bin/env bash
# run.sh TAG SPEC(ar|dspark) SAMPLED(0|1) PROF(0|1) [BIN]; set D (model dir), CODE and DOC (client inputs)
set -uo pipefail
tag=$1 spec=$2 sampled=$3 prof=$4 bin=${5:-build/release/gufo}
D=${D:-/mnt/data/models/ds41}
M=$D/unsloth-ud-iq2xxs/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf
O=${OUT:-prof}/$tag; rm -rf $O; mkdir -p $O
PORT=8099
args=(serve --host 127.0.0.1 --port $PORT --sessions 1 llm --model $M --context 65536 --stream-experts --served-model-name ds41)
[ "$spec" = dspark ] && args+=(--speculative dspark --dspark-model $D/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf)
if [ "$prof" = 1 ]; then
  nix develop -c rocprofv3 --kernel-trace --memory-copy-trace --output-format rocpd -d $O -o t -- $bin "${args[@]}" > $O/server.log 2>&1 &
else
  $bin "${args[@]}" > $O/server.log 2>&1 &
fi
SPID=$!
for i in $(seq 1 900); do curl -sf http://127.0.0.1:$PORT/ready >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:$PORT/ready >/dev/null || { echo "not ready"; kill $SPID; exit 1; }
python3 -I "$(dirname "$0")/client.py" http://127.0.0.1:$PORT/v1 $sampled $O/client.json "$CODE" "$DOC" > $O/client.txt 2>&1 || echo "client rc=$?"
GPID=$(ss -ltnpH "sport = :$PORT" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
kill -INT $GPID; wait $SPID
grep -o 'request=r[0-9]* .*decode_tps=[0-9.]*\|draft_accepted=[0-9]* draft_proposed=[0-9]* draft_rounds=[0-9]*' $O/server.log | sed 's/method.*prompt_tokens/prompt_tokens/' 
cat $O/client.txt
