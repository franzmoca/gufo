import json, sys, urllib.request, time
base=sys.argv[1]
for name in ["server.md", "expert_store.cpp"]:
    doc=open(f"/tmp/qcmp/served/docs/{name}").read()
    body={"model":"ds41","max_tokens":1,"temperature":0,"reasoning_effort":"none",
          "messages":[{"role":"user","content":f"<document>\n{doc}\n</document>\n\nSummarize this document in three sentences."}]}
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(base+"/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=3600))
    print(name, r["usage"]["prompt_tokens"], f"{time.time()-t:.2f}s", flush=True); time.sleep(3)
