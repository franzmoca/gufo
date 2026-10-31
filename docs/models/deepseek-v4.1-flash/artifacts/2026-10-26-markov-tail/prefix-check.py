import subprocess
from pathlib import Path
root=Path('/tmp/gufo-ds41-markov-tail')
line=next(x for x in Path('/tmp/gufo-ds41-tokenizer/baseline-docs.stdout').read_text().splitlines() if x.startswith('ids '))
ids=','.join(line.split()[1:129])
(root/'prefix-ids.txt').write_text(ids+'\n')
(root/'dump').mkdir(exist_ok=True)
cmd=['build/gpu-test/tests/models/deepseek_v41_flash/ds41_load_probe','/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf','--dspark','/mnt/data/models/ds41/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf','--decode',ids,'--dspark-dump',str(root/'dump')]
with (root/'prefix-check.stdout').open('w') as out,(root/'prefix-check.stderr').open('w') as err:
 r=subprocess.run(cmd,stdout=out,stderr=err)
print('prefix-check returncode',r.returncode,flush=True)
raise SystemExit(r.returncode)
