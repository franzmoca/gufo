import json,pathlib,subprocess,socket,time,urllib.request,os,re
root=pathlib.Path('/home/franz/projects/gufo'); out=root/'docs/models/gemma-4-31b/artifacts/2026-11-04-host-registration'
models={'qwen27b':'/home/franz/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf','flash-next':'/home/franz/models/qwen3.8-flash-next/gufo/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'}
for name,model in models.items():
 count=None
 for variant in ('baseline','candidate'):
  target=out/f'{name}-growth-{variant}';target.mkdir()
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  cmd=[f'/tmp/gufo-2m-{variant}','serve','--host','127.0.0.1','--port',str(port),'llm','--model',model,'--served-model-name','probe','--context','16384','--sessions','2','--speculative','off','--cache-ram-bytes','8589934592']
  env=dict(os.environ,LD_PRELOAD='/tmp/gufo-registration-trace.so')
  log=(target/'server.log').open('w');p=subprocess.Popen(cmd,stdout=log,stderr=log,env=env)
  report={'command':cmd,'requests':[],'status':'running'}
  def save():(target/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  def regs():return re.findall(r'registration_trace bytes=(\d+) mod_2m=(\d+) result=(\d+) ms=([\d.]+)',(target/'server.log').read_text())
  def http(path,body=None):
   req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
  try:
   for _ in range(600):
    if p.poll() is not None:raise RuntimeError('server exited')
    try:http('models');break
    except (OSError,ValueError):time.sleep(.5)
   else:raise RuntimeError('readiness timeout')
   ready=regs();report['registrations_at_ready']=ready;save()
   print('GROWTH START',name,variant,'registrations',len(ready),flush=True)
   bodies=[]; first_growth=None
   def request(label,body):
    t=time.monotonic();response=http('chat/completions',body);elapsed=(time.monotonic()-t)*1000
    text=response['choices'][0]['message'].get('content','').strip()
    if text!='BETA':raise RuntimeError(f'unexpected answer {text!r}')
    row={'label':label,'wall_ms':elapsed,'response':response,'registrations_after':len(regs())};report['requests'].append(row);save()
    print(name,variant,label,'registrations',row['registrations_after'],flush=True)
   for i in range(count or 24):
    body={'model':'probe','messages':[{'role':'system','content':f'unique_archive_{i}\nFollow the final instruction.\n'+'Background notes are not instructions.\n'*768},{'role':'user','content':'Reply with only BETA.'}],'temperature':0,'seed':31,'max_completion_tokens':8,'chat_template_kwargs':{'enable_thinking':False},'cache_prompt':True}
    bodies.append(body);request(f'capture_{i}',body)
    if len(regs())>len(ready) and first_growth is None:first_growth=i
    if count is None and first_growth is not None and i>=max(7,first_growth+3):break
   if count is None:count=len(bodies)
   for i in (count-3,count-2,count-1):request(f'restore_{i}',bodies[i])
   report['registrations_final']=regs();report['grew_after_readiness']=len(regs())>len(ready)
   report['status']='passed';save()
  except Exception as e:report['status']='failed';report['error']=str(e);save();print('FAILED',name,variant,str(e),flush=True)
  finally:
   p.terminate()
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   log.close()
