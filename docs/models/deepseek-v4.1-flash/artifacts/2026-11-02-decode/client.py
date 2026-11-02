"""Decode workloads: chat, code, after a long document.
argv: base_url sampled(0/1) out_json code_file long_document
(the profiles used tools/ds41/replay_sim.py at c9faa470 and the HTTP server documentation, 23,974 prompt tokens)."""
import json, sys, time, urllib.request
base, sampled, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
code = open(sys.argv[4]).read()
doc = open(sys.argv[5]).read()
reqs = [
    ("chat", "Write a long, detailed short story (at least 700 words) about a lighthouse keeper on a remote island "
             "who finds a stranded robot after a storm. Use dialogue and vivid description."),
    ("code", f"```python\n{code}\n```\n\nRewrite the `readahead` function of this file so that it also charges the "
             "wasted reads (predicted experts that are not routed) to the read queue, returns the wasted bytes too, "
             "and has a full docstring. Then write a unit test for it with pytest. Output complete code."),
    ("long", f"<document>\n{doc}\n</document>\n\nExplain in detail, section by section, how this server schedules, "
             "batches and caches requests. Be thorough (at least 600 words)."),
]
res = []
for name, content in reqs:
    body = {"model": "ds41", "max_tokens": 512, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": content}]}
    if sampled: body.update(temperature=0.7, top_p=0.95, seed=1234)
    else: body.update(temperature=0)
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"}), timeout=3600))
    dt = time.time() - t
    u = r["usage"]
    res.append({"name": name, "prompt": u["prompt_tokens"], "gen": u["completion_tokens"], "wall": dt,
                "text": r["choices"][0]["message"]["content"]})
    print(name, u["prompt_tokens"], u["completion_tokens"], f"{dt:.2f}s", flush=True)
    time.sleep(3)
json.dump(res, open(out, "w"), indent=1)
