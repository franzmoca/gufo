import subprocess,os,json,hashlib
from pathlib import Path
root=Path('/tmp/gufo-ds41-verify')
env=os.environ.copy();env['GUFO_DEEPSEEK41_DSPARK_TRACE']='1'
for label in ['baseline','candidate']:
 binary=root/('gufo-'+label)
 cmd=[str(binary),'bench','-m','/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf','--dspark-model','/mnt/data/models/ds41/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf','--speculative','dspark','--draft-tokens','5','--temperature','0','--top-p','1','--seed','73','-p','2048','-n','128','-d','0,32768','-r','1','-v']
 (root/(label+'-bench-command.json')).write_text(json.dumps({'cmd':cmd,'trace':True,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2)+'\n')
 with (root/(label+'-bench.stdout')).open('w') as out,(root/(label+'-bench.stderr')).open('w') as err:
  r=subprocess.run(cmd,stdout=out,stderr=err,env=env)
 print(label,'bench returncode',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
