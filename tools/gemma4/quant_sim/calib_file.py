"""Adds a pre-rendered calibration text (special tokens written out, e.g.
bartowski's calibration-v6 rendered through the Gemma 4 template) to a
tokens file as `calib_<name>`: the whole text in windows of `--length`
tokens, each after a BOS.

    python -I tools/gemma4/quant_sim/calib_file.py --model DIR --tokens FILE \
        --text FILE --name v6
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

BOS = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--length", type=int, default=2048)
    args = p.parse_args()
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    # Special tokens in the text (<|turn>, <|tool_call>, ...) map to their
    # ids, as llama-imatrix --parse-special does.
    ids = tok.encode(args.text.read_text(), add_special_tokens=False).ids
    span = args.length - 1
    rows = [[BOS] + ids[i:i + span] for i in range(0, len(ids) - span + 1, span)]
    data = torch.load(args.tokens)
    data[f"calib_{args.name}"] = torch.tensor(rows)
    torch.save(data, args.tokens)
    print(f"{len(ids)} tokens -> {len(rows)} windows of {args.length}")


if __name__ == "__main__":
    main()
