#!/usr/bin/env python3
"""Alternating C1 sampled DSpark runs: run.py OUT.jsonl REPS SEEDS LABEL=BINARY..."""
import hashlib, json, re, subprocess, sys, time
from pathlib import Path

MODEL = '/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf'
DRAFT = '/mnt/data/models/ds41/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf'
PROMPTS = {
    'coding': 'Implement a small Python LRU cache with get and put. Explain eviction briefly.',
    'prose': 'Write a 500-word imaginative story about a librarian finding a map in an old cookbook. Use vivid details and end with a surprising choice.',
    'explain': 'Explain how TCP congestion control works, covering slow start, congestion avoidance, fast retransmit and fast recovery.',
    'rust': 'Write a Rust function that parses a CSV line with quoted fields into a Vec<String>, with unit tests.',
}

out = Path(sys.argv[1]); reps = int(sys.argv[2]); seeds = [int(s) for s in sys.argv[3].split(',')]
bins = [a.split('=', 1) for a in sys.argv[4:]]
prompts = sys.argv[5:] and None
for rep in range(reps):
    for seed in seeds:
        for name, text in PROMPTS.items():
            order = bins if (rep + seed) % 2 == 0 else bins[::-1]
            for label, binary in order:
                cmd = [binary, 'prompt', '-m', MODEL, '--dspark-model', DRAFT, '--speculative', 'dspark', '--think', 'off',
                       '--temperature', '1', '--top-p', '1', '--seed', str(seed), '-n', '256', '--no-display-prompt', '-v', text]
                r = subprocess.run(cmd, capture_output=True, text=True)
                stats = re.search(r'Generated (\d+) tokens on ROCm in ([\d.]+)s \(([\d.]+) tok/s\)', r.stdout + r.stderr)
                th = re.search(r'\[TokenTrace\]: count=(\d+) sha256=(\w+)', r.stderr)
                line = next((s for s in r.stderr.splitlines() if s.startswith('[Speculative]:')), '')
                row = {'label': label, 'rep': rep, 'seed': seed, 'prompt': name, 'rc': r.returncode,
                       'tokens': int(stats[1]) if stats else None, 'tps': float(stats[3]) if stats else None,
                       'hash': th[2] if th else hashlib.sha256(r.stdout.encode()).hexdigest()[:16],
                       'spec': {k: float(v) for k, v in re.findall(r'([a-z_]+)=([\d.]+)', line)}, 't': time.time()}
                with out.open('a') as f: f.write(json.dumps(row) + '\n')
                print(label, rep, seed, name, row['tps'], row['spec'].get('acceptance'), row['hash'][:12], flush=True)
                if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
