import subprocess,json,hashlib
from pathlib import Path
root=Path('/tmp/gufo-ds41-device-plan')
for label in ['baseline','candidate']:
 binary=root/('gufo-'+label)
 cmd=[str(binary),'bench','-m','/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf','-p','16,128','-n','0','-d','0','-r','1','-v']
 (root/(label+'-followup-command.json')).write_text(json.dumps({'cmd':cmd,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2)+'\n')
 with (root/(label+'-followup.stdout')).open('w') as out,(root/(label+'-followup.stderr')).open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
 print(label,'returncode',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
