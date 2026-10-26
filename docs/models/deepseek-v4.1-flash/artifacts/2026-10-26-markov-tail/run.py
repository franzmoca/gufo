import subprocess,sys,os,json,re,hashlib
from pathlib import Path
root=Path('/tmp/gufo-ds41-markov-tail');results=[]
for label,binary,trace in [(sys.argv[1]+'-coding','gufo-'+sys.argv[1].split('-')[0],True),(sys.argv[1]+'-prose','gufo-'+sys.argv[1].split('-')[0],True)]:
    env=os.environ.copy()
    if trace:env['GUFO_DEEPSEEK41_DSPARK_TRACE']='1'
    cmd=[str(root/binary),'prompt','-m','/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf','--dspark-model','/mnt/data/models/ds41/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf','--speculative','dspark','--think','off','--temperature','1','--top-p','1','--seed','73','-n','256','--no-display-prompt','-v',('Implement a small Python LRU cache with get and put. Explain eviction briefly.' if label.endswith('coding') else 'Write a 500-word imaginative story about a librarian finding a map in an old cookbook. Use vivid details and end with a surprising choice.')]
    with (root/(label+'.txt')).open('w') as out,(root/(label+'.stderr')).open('w') as err:
        r=subprocess.run(cmd,stdout=out,stderr=err,env=env)
    output=(root/(label+'.txt')).read_text();err=(root/(label+'.stderr')).read_text();stats=re.search(r'Generated (\d+) tokens on ROCm in ([\d.]+)s \(([\d.]+) tok/s\)',output);tokenhash=re.search(r'\[TokenTrace\]: count=(\d+) sha256=(\w+)',err)
    row={'run':label,'returncode':r.returncode,'tokens':int(stats[1]) if stats else None,'seconds':float(stats[2]) if stats else None,'tps':float(stats[3]) if stats else None,'token_sha256':tokenhash[2] if tokenhash else None,'pq_trace_rows':len(re.findall('DSpark sample row=',err)),'binary_sha256':hashlib.sha256((root/binary).read_bytes()).hexdigest()}
    line=next((s for s in err.splitlines() if s.startswith('[Speculative]:')),'');row['speculation']={k:float(v) for k,v in re.findall(r'([a-z_]+)=([\d.]+)',line)};results.append(row);(root/(sys.argv[1]+'-results.json')).write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
    if r.returncode:sys.exit(r.returncode)
