import subprocess,sys
from pathlib import Path
root=Path('/tmp/gufo-ds41-verify')
label=sys.argv[1];width=int(sys.argv[2]);cycles=width
ids=Path('/tmp/ds41-gpu/'+('score3k.ids' if label.endswith('-short') else 'score18k.ids')).read_text().strip()
probe=str(root/'probe-baseline') if label=='baseline' else 'build/gpu-test/tests/models/deepseek_v41_flash/ds41_load_probe'
cmd=[probe,'/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf','--decode',ids,'--spec-check',str(cycles),'--spec-width',str(width)]
with (root/(label+'-width'+str(width)+'.log')).open('w') as log:
 r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
print(label,width,'returncode',r.returncode,flush=True)
raise SystemExit(r.returncode)
