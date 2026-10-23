"""Builds the token sets for the Gemma 4 quantization simulation.

Evaluation and calibration come from disjoint sources: evaluation uses prose
from the Kyojin eval texts, agentic coding conversations rendered through the
Gemma 4 chat template, and Gufo C++/Python sources; calibration uses the
exllamav3 standard calibration mix (web, wiki, code, technical,
multilingual).

    python -I tools/gemma4/quant_sim/data.py --model DIR --out DIR
"""

import argparse
import json
import random
from pathlib import Path

import jinja2
import torch
from tokenizers import Tokenizer

BOS = 2
KYOJIN = Path.home() / "projects/kyojin"
GUFO = Path(__file__).resolve().parents[3]


def windows(ids, length, count, rng):
    """`count` non-overlapping windows of `length - 1` tokens after a BOS."""
    span = length - 1
    starts = list(range(0, len(ids) - span, span))
    rng.shuffle(starts)
    if len(starts) < count:
        raise ValueError(f"only {len(starts)} windows, {count} wanted")
    return [[BOS] + ids[s:s + span] for s in sorted(starts[:count])]


def render_chat(template, request, generation_prompt=False):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise jinja2.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    # The template wants tool-call arguments as mappings, as a server would
    # pass them after parsing the OpenAI request.
    for message in request["messages"]:
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    return env.from_string(template).render(
        messages=request["messages"], tools=request.get("tools"),
        add_generation_prompt=generation_prompt, bos_token="")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--calib-rows", type=int, default=64)
    args = p.parse_args()
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    rng = random.Random(1234)

    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids

    evaluation = []  # (category, name, ids)
    texts = KYOJIN / "eval/eval_texts"
    for name, count in [("pride_prejudice_mod.txt", 2),
                        ("variable_man_mod.txt", 1),
                        ("illustrious_client.txt", 1)]:
        ids = encode((texts / name).read_text())
        for i, w in enumerate(windows(ids, args.length, count, rng)):
            evaluation.append(("prose", f"{name}:{i}", w))
    template = (args.model / "chat_template.jinja").read_text()
    for name in ["agentic_code_10", "agentic_code_20", "agentic_curl_10",
                 "agentic_curl_16"]:
        request = json.loads((KYOJIN / f"eval/prompts/{name}.json").read_text())
        ids = encode(render_chat(template, request))
        # The conversation tail: tool results and replies, not the shared
        # system prompt.
        evaluation.append(("agentic", name, [BOS] + ids[-(args.length - 1):]))
    for path in ["src/models/gemma4/engine.cpp",
                 "src/models/gemma4/kernels/rocm/executor.cpp",
                 "tests/functional/pi_agent.py",
                 "tools/prof/prof.py"]:
        ids = encode((GUFO / path).read_text())
        evaluation.append(("code", path, [BOS] + ids[:args.length - 1]))
    for c, n, ids in evaluation:
        assert len(ids) == args.length, (n, len(ids))

    calib_src = KYOJIN / "exllamav3/conversion/standard_cal_data"
    share = {"c4.utf8": 0.25, "wiki.utf8": 0.25, "code.utf8": 0.25,
             "technical.utf8": 0.125, "multilingual.utf8": 0.125}
    calibration = []
    for name, frac in share.items():
        ids = encode((calib_src / name).read_text())
        calibration += windows(ids, args.length,
                               round(frac * args.calib_rows), rng)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "eval": torch.tensor([ids for _, _, ids in evaluation]),
        "eval_meta": [(c, n) for c, n, _ in evaluation],
        "calib": torch.tensor(calibration),
    }, args.out / "tokens.pt")
    print(f"eval {len(evaluation)} x {args.length}, "
          f"calib {len(calibration)} x {args.length}")


if __name__ == "__main__":
    main()
