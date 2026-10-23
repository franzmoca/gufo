"""Chat-format calibration rows: conversations the BF16 model completes itself.

Prompts come from sources disjoint from eval_set.py (Hipfire's benchmark
prompts and Kyojin's creative/translation requests); each conversation is
rendered with the Gemma 4 template, completed by the BF16 checkpoint and
the token stream is cut into windows. Adds `calib_chat` to a tokens file.

    python -I tools/gemma4/quant_sim/calib_chat.py --model DIR --tokens FILE
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_set import encode_prompt, generate, user  # noqa: E402

HIPFIRE = Path.home() / "projects/hipfire/benchmarks/prompts"
KYOJIN = Path.home() / "projects/kyojin/eval/prompts"


def clip(text, chars):
    return text[:chars]


def conversations():
    convs = []
    long_text = [
        ("qwen38_issue693_longcode_20676.txt", "Review this code. List the "
         "three most important bugs or design problems and propose fixes."),
        ("deepseek4_mq2r_prose_ctx8192.txt", "Summarize this text and "
         "describe its tone."),
        ("cohere2moe_long.txt", "What are the main points of this document? "
         "Answer as a bulleted list, then give a one-paragraph critique."),
        ("ttft_5900.txt", "Explain the content below to a newcomer."),
        ("embed_long.txt", "Extract the key facts from this text as a table."),
        ("glimmer_prefill_4096.txt", "Continue the following text in the "
         "same style for a few paragraphs."),
        ("ds4-gfx942-ar-2048.source.txt", "Explain what this source does and "
         "how it could be tested."),
        ("ds4_heterogeneous_code_2048.txt", "Refactor the most complex "
         "function below and explain the change."),
    ]
    for name, ask in long_text:
        convs.append({"messages": user(
            ask + "\n\n" + clip((HIPFIRE / name).read_text(), 22000))})
    for name in ["longprose_multidoc.jsonl", "longcode_pflash.jsonl"]:
        line = (HIPFIRE / name).read_text().splitlines()[0]
        record = json.loads(line)
        convs.append({"messages": user(
            clip(record["filler_text"], 22000) + "\n\n" + record["question"])})
    for name in ["creative_02", "creative_03", "translate_01", "translate_02",
                 "coding_01", "coding_02", "coding_03"]:
        request = json.loads((KYOJIN / f"{name}.json").read_text())
        convs.append({"messages": request["messages"],
                      "tools": request.get("tools")})
    tools = [{"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}}]
    convs.append({"messages": [
        {"role": "system", "content": (HIPFIRE / "agentic_pi_system.txt").read_text()},
        *user((HIPFIRE / "agentic_user_multistep.txt").read_text())],
        "tools": tools})
    convs.append({"messages": [
        {"role": "system", "content": (HIPFIRE / "agentic_hermes_system.txt").read_text()},
        *user((HIPFIRE / "agentic_user_read.txt").read_text())],
        "tools": tools})
    for name in ["humaneval_0_has_close_elements.txt", "humaneval_2_truncate.txt",
                 "merge_sort_thinking_off.txt", "mixed_code_then_prose.txt",
                 "trains-meet.txt", "bare_factual.txt", "prose_river_short.txt"]:
        convs.append({"messages": user((HIPFIRE / name).read_text())})
    return convs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--length", type=int, default=2048)
    args = p.parse_args()
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    template = (args.model / "chat_template.jinja").read_text()
    convs = conversations()
    prompts = [encode_prompt(template, c, tok) for c in convs]
    for c, ids in zip(convs, prompts):
        print(f"prompt {len(ids):6d}", flush=True)
    replies = generate(args.model, prompts, 768,
                       [f"calib{i}" for i in range(len(prompts))], seed=5000)
    stream = []
    for ids, reply in zip(prompts, replies):
        stream += ids + reply
    span = args.length
    rows = [stream[i:i + span] for i in range(0, len(stream) - span + 1, span)]
    print(f"{len(stream)} tokens, {len(rows)} windows", flush=True)
    if len(rows) < args.rows:
        raise SystemExit(f"only {len(rows)} windows")
    # Spread the chosen windows over the stream.
    step = len(rows) / args.rows
    chosen = [rows[int(i * step)] for i in range(args.rows)]
    data = torch.load(args.tokens)
    data["calib_chat"] = torch.tensor(chosen)
    torch.save(data, args.tokens)


if __name__ == "__main__":
    main()
