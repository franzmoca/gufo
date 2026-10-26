import subprocess,json,math
from pathlib import Path
root=Path('/tmp/gufo-ds41-device-plan')
ids=Path('/tmp/ds41-gpu/score3k.ids').read_text().strip();scores={}
for label,probe in [('baseline',str(root/'probe-baseline')),('candidate','build/gpu-test/tests/models/deepseek_v41_flash/ds41_load_probe')]:
 out=root/(label+'-quality');out.mkdir(exist_ok=True)
 cmd=[probe,'/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf','--decode',ids,'--batch','--score','--out',str(out)]
 with (root/(label+'-quality.stdout')).open('w') as log,(root/(label+'-quality.stderr')).open('w') as err:r=subprocess.run(cmd,stdout=log,stderr=err)
 if r.returncode:raise SystemExit(r.returncode)
 rows=[line.split()[:4] for line in (out/'score.txt').read_text().splitlines()];scores[label]=rows
 print(label,'rows',len(rows),'ppl',math.exp(sum(float(row[3]) for row in rows)/len(rows)),flush=True)
assert scores['baseline']==scores['candidate'],'NLL or argmax changed'
(root/'quality-summary.json').write_text(json.dumps({'positions':len(scores['candidate']),'argmax_and_nll_columns_identical':True,'nll_precision':'six decimal places in existing probe output','ppl':math.exp(sum(float(row[3]) for row in scores['candidate'])/len(scores['candidate']))},indent=2)+'\n')
print('all argmax/NLL rows identical',flush=True)
