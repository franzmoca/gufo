#!/usr/bin/env python3
"""DS4.1 disk prompt cache in the production setup: native context, one
session, DSpark. Conversation -> restart (TERM, then KILL) -> replay must
restore from disk and match uncached controls."""
import json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True)
CACHE = Path(sys.argv[2]); BIN = sys.argv[3]
M = "/mnt/data/models/ds41"
CMD = [BIN, "serve", "--host", "127.0.0.1", "--port", "8091", "--sessions", "1", "llm",
       "--model", f"{M}/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf",
       "--speculative", "dspark", "--dspark-model",
       f"{M}/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf",
       "--context", "0", "--cache-disk", str(CACHE), "--cache-disk-bytes", str(64 << 30)] + sys.argv[4:]
REPO = Path("/home/franz/projects/gufo")
doc = "\n\n".join((REPO / p).read_text() for p in ("docs/KV-CACHE.md", "docs/TESTING.md", "docs/RELEASING.md"))
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 2**20
def start(tag):
    log = open(OUT / f"server-{tag}.log", "w")
    p = subprocess.Popen(CMD, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2); break
        except Exception:
            if p.poll() is not None: sys.exit(f"{tag}: server exited {p.returncode}")
            time.sleep(1)
    print(f"{tag}: ready {time.time()-t0:.1f}s MemAvailable {avail():.2f} GiB", flush=True)
    return p
def chat(msgs, cache=True):
    body = {"model": "DeepSeek V4.1 Flash", "messages": msgs, "max_tokens": 64, "temperature": 0,
            "thinking": {"type": "disabled"}, "cache_prompt": cache}
    req = urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"]["content"], r.get("usage", {})
def turn(tag, msgs, cache=True):
    text, usage = chat(msgs, cache)
    print(f"{tag}: avail {avail():.2f} GiB usage {json.dumps(usage)}", flush=True)
    return text
def published(tag, n):
    log = OUT / f"server-{tag}.log"
    t0 = time.time()
    while time.time() - t0 < 60:
        if len(re.findall(r"disk.*(publish|commit|stored)|(publish|commit|stored).*disk", log.read_text(), re.I)) >= n: return True
        time.sleep(1)
    return False
first = [{"role": "user", "content": doc + "\n\nSummarize the cache design above in three sentences."}]
results = {}
p = start("cold")
a1 = turn("cold-t1", first)
second = first + [{"role": "assistant", "content": a1},
                  {"role": "user", "content": "Which of these documents covers releases? One sentence."}]
a2 = turn("cold-t2", second)
time.sleep(5)
results["cold"] = [a1, a2]
for sig, tag in (() if os.environ.get("NO_RESTART") else ((signal.SIGTERM, "term"), (signal.SIGKILL, "kill"))):
    p.send_signal(sig); p.wait()
    p = start(tag)
    b1 = turn(f"{tag}-t1", first); b2 = turn(f"{tag}-t2", second)
    results[tag] = [b1, b2]
    time.sleep(5)
c1 = turn("uncached-t1", first, cache=False); c2 = turn("uncached-t2", second, cache=False)
results["uncached"] = [c1, c2]
p.send_signal(signal.SIGTERM); p.wait()
(OUT / "replies.json").write_text(json.dumps(results, indent=1))
ok = all(results[k] == results["uncached"] for k in results)
print("identical replies:", ok, {k: [v[i] == results["uncached"][i] for i in (0, 1)] for k, v in results.items()})
