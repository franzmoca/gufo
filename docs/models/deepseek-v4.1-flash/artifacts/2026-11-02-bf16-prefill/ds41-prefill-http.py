import json, subprocess, time, urllib.request
from pathlib import Path
model='/mnt/data/models/ds41/unsloth-ud-iq2xxs/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf'
for label, binary in [('base','/tmp/gufo-prefill-baseline'),('cand','build/release/gufo'),('cand2','build/release/gufo'),('base2','/tmp/gufo-prefill-baseline')]:
 with open('/tmp/ds41-http-'+label+'.log','w') as log:
  p=subprocess.Popen([binary,'serve','--port','8097','llm','--model',model,'--served-model-name','ds41','--context','65536','--stream-experts','--expert-cache-bytes','96636764160','--sessions','1'],stdout=log,stderr=log)
  try:
   for _ in range(180):
    if p.poll() is not None: raise RuntimeError('server exited')
    try:
     urllib.request.urlopen('http://127.0.0.1:8097/health',timeout=1);break
    except Exception: time.sleep(1)
   else: raise RuntimeError('startup timeout')
   results=[]
   for name in ['server.md','expert_store.cpp']:
    doc=Path('/tmp/qcmp/served/docs/'+name).read_text()
    body={'model':'ds41','max_tokens':1,'temperature':0,'reasoning_effort':'none','messages':[{'role':'user','content':f'<document>\n{doc}\n</document>\n\nSummarize this document in three sentences.'}]}
    start=time.monotonic()
    response=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8097/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=300))
    results.append({'document':name,'wall_seconds':time.monotonic()-start,'response':response})
   Path('/tmp/ds41-http-'+label+'.json').write_text(json.dumps(results,indent=2))
   print(label,[(x['document'],x['wall_seconds']) for x in results],flush=True)
  finally:
   p.terminate()
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.wait()
