import subprocess,json,pathlib,time
root=pathlib.Path('/home/franz/projects/gufo')
out=root/'docs/models/gemma-4-31b/artifacts/2026-11-04-host-registration'
models={
'qwen27b':'/home/franz/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf',
'flash-next':'/home/franz/models/qwen3.8-flash-next/gufo/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'}
results=[]
for model,path in models.items():
 for label,binary in [('A1','/tmp/gufo-2m-baseline'),('B1','/tmp/gufo-2m-candidate'),('B2','/tmp/gufo-2m-candidate'),('A2','/tmp/gufo-2m-baseline')]:
  target=out/f'{model}-{label}'
  cmd=['python3','tests/functional/run.py','--output',str(target),'--sampling-preset','qwen38','--suite','cache-growth','--through-case','cache-growth:cache_growth_drop_reasoning_cold_control_3','--startup-timeout','300','--request-timeout','180']
  cmd+=['--record-baseline'] if label=='A1' else ['--baseline',str(out/f'{model}-A1')]
  cmd+=['--',binary,'serve','llm','--model',path,'--context','16384','--sessions','2','--speculative','off','--cache-ram-bytes','2147483648']
  print('START',model,label,flush=True)
  start=time.time()
  with (out/f'{model}-{label}-runner.log').open('w') as log:
   p=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  row={'model':model,'run':label,'command':cmd,'returncode':p.returncode,'elapsed_s':time.time()-start}
  results.append(row)
  (out/'http-invocations.json').write_text(json.dumps(results,indent=2)+'\n')
  print('END',model,label,p.returncode,round(row['elapsed_s'],1),flush=True)
  if label=='A1' and p.returncode:
   break
