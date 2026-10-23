"""Builds the chat evaluation set: prompts the BF16 model answers itself.

Each item is a conversation rendered with the Gemma 4 chat template up to the
generation prompt, plus the reply the BF16 checkpoint samples for it
(exllamav3, Gemma's default sampling). sim.py scores only the reply tokens.

    python -I tools/gemma4/quant_sim/eval_set.py --model DIR --out FILE \
        [--calib FILE]

--calib copies the calibration rows of an earlier tokens file (data.py).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import render_chat  # noqa: E402

KYOJIN = Path.home() / "projects/kyojin"
GUFO = Path(__file__).resolve().parents[3]
# <eos>, <turn|>, and <|tool_response>: a server stops there to run the tool.
EOS = [1, 106, 50]
MAX_PROMPT = 17000


def user(text):
    return [{"role": "user", "content": text}]


def source(path, start_line=0, lines=400):
    text = (GUFO / path).read_text().splitlines()
    return "\n".join(text[start_line:start_line + lines])


def chapter(name, chars, offset):
    text = (KYOJIN / "eval/eval_texts" / name).read_text()
    return text[offset:offset + chars]


def prompts():
    items = []
    # Agentic coding: recorded opencode/curl sessions, cut before the latest
    # assistant turn whose prompt still fits (filled in by agentic()).
    # The sessions share prefixes, so each pair cuts at two prompt budgets.
    # The tool schemas alone are ~8K tokens.
    for name, budget in [("agentic_code_20", 10500), ("agentic_code_29", 17000),
                         ("agentic_curl_10", 11000), ("agentic_curl_16", 14000)]:
        items.append(("agentic", name, budget))
    tools = [{
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the repository and return "
                           "its stdout and stderr.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]}}}, {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]}}}]
    items.append(("agentic", "tool_first_step", {
        "messages": user(
            "The CI job `cpu-test` fails after my last commit with a linker "
            "error mentioning `gufo::models::gemma4::Reference`. Find the "
            "cause and propose a fix."),
        "tools": tools}))
    items.append(("agentic", "tool_after_result", {
        "messages": [
            *user("Why is `tools/prof/prof.py` slow on large traces? Use the "
                  "tools to look."),
            {"role": "assistant", "tool_calls": [{"type": "function",
             "function": {"name": "read_file",
                          "arguments": {"path": "tools/prof/prof.py"}}}]},
            {"role": "tool", "name": "read_file",
             "content": source("tools/prof/prof.py", 0, 260)}],
        "tools": tools}))
    # Coding.
    items += [
        ("code", "cpp_explain", {"messages": user(
            "Explain what this C++ code does, then point out any bugs or "
            "performance problems:\n\n```cpp\n" +
            source("src/models/gemma4/reference.cpp", 0, 230) + "\n```")}),
        ("code", "python_write", {"messages": user(
            "Write a Python function that parses a GGUF file header and "
            "returns the tensor names, types and shapes, without third-party "
            "packages. Include type hints and a short test.")}),
        ("code", "rust_fix", {"messages": user(
            "This Rust code does not compile. Fix it and explain each "
            "change:\n\n```rust\nuse std::collections::HashMap;\n\n"
            "fn count_words(text: &str) -> HashMap<&str, usize> {\n"
            "    let mut counts = HashMap::new();\n"
            "    for word in text.split_whitespace() {\n"
            "        let lower = word.to_lowercase();\n"
            "        *counts.entry(&lower).or_insert(0) += 1;\n"
            "    }\n    counts\n}\n\nfn main() {\n"
            "    let text = String::from(\"a b A c b\");\n"
            "    let counts = count_words(&text);\n"
            "    drop(text);\n    println!(\"{:?}\", counts);\n}\n```")}),
        ("code", "hip_kernel", {"messages": user(
            "Write a HIP kernel for gfx1151 (wave32) that computes RMSNorm "
            "over rows of 5376 floats with one block per row, plus the host "
            "launch code. Explain the reduction strategy.")}),
        ("code", "long_edit", {"messages": user(
            "Refactor the following Python script so the report printing is "
            "separated from the data loading, and add argparse help for "
            "every option. Return the full new file.\n\n```python\n" +
            source("tests/functional/pi_agent.py", 0, 300) + "\n```")}),
        ("code", "sql", {"messages": user(
            "Given tables orders(id, customer_id, total, created_at) and "
            "customers(id, name, country), write SQL for the top five "
            "countries by revenue in 2025 with month-over-month growth, and "
            "explain the query plan you would expect.")}),
    ]
    # Long-context prose.
    items += [
        ("prose", "pp_summary", {"messages": user(
            "Summarize the following chapter in about 300 words, then list "
            "the main characters and their motivations.\n\n" +
            chapter("pride_prejudice_mod.txt", 20000, 150000))}),
        ("prose", "vm_questions", {"messages": user(
            "Read this story excerpt and answer: what does the protagonist "
            "want, what stands in the way, and what do you expect to happen "
            "next? Quote the text where useful.\n\n" +
            chapter("variable_man_mod.txt", 18000, 40000))}),
        ("prose", "story", {"messages": user(
            "Write the opening of a short story (about 700 words) set in a "
            "lighthouse during a storm, told by the keeper's daughter.")}),
        ("prose", "letter", {"messages": user(
            "Write a polite but firm letter to a landlord about a heating "
            "system that has failed three times this winter, asking for a "
            "permanent repair and a rent reduction.")}),
        ("prose", "ic_continue", {"messages": user(
            "Continue this story for about 600 words in the same style:\n\n" +
            chapter("illustrious_client.txt", 12000, 10000))}),
        ("prose", "essay", {"messages": user(
            "Discuss the trade-offs of remote work for early-career software "
            "engineers in a structured essay of about 600 words.")}),
    ]
    # Reasoning and multilingual.
    items += [
        ("reasoning", "math", {"messages": user(
            "A tank fills through pipe A in 6 hours and pipe B in 9 hours, "
            "and drains through pipe C in 12 hours. Starting empty with all "
            "three open, after 2 hours pipe B is closed. How long until the "
            "tank is full? Show your work.")}),
        ("reasoning", "explain", {"messages": user(
            "Explain how trellis-coded quantization works and why it can "
            "beat scalar quantization at the same bit rate.")}),
        ("reasoning", "italian", {"messages": user(
            "Spiegami in italiano le differenze tra la memoria unificata "
            "di un APU e la VRAM dedicata di una GPU discreta, con esempi "
            "pratici per l'inferenza di modelli linguistici.")}),
        ("reasoning", "german", {"messages": user(
            "Erkläre auf Deutsch, wie ein Wärmepumpenheizsystem funktioniert "
            "und wann es sich in einem Altbau lohnt.")}),
        ("reasoning", "chinese", {"messages": user(
            "请用中文解释什么是注意力机制，以及为什么滑动窗口注意力能降低长上下文的计算成本。")}),
        ("reasoning", "logic", {"messages": user(
            "Five people sit in a row. Anna is not at either end. Ben sits "
            "directly right of Carla. Dario is somewhere left of Ben. Elena "
            "sits at an end next to Anna. List every valid seating.")}),
    ]
    return items


def agentic(name, budget, template, tok):
    """The session up to the last assistant turn whose prompt fits."""
    request = json.loads((KYOJIN / f"eval/prompts/{name}.json").read_text())
    messages = request["messages"]
    tools = request.get("tools")
    for cut in range(len(messages) - 1, 0, -1):
        if messages[cut]["role"] != "assistant":
            continue
        conv = {"messages": messages[:cut], "tools": tools}
        ids = encode_prompt(template, conv, tok)
        if len(ids) <= budget:
            return conv, ids
    # Too long even at the first reply: drop the tool schemas.
    for cut in range(1, len(messages)):
        if messages[cut]["role"] == "assistant":
            conv = {"messages": messages[:cut], "tools": None}
            return conv, encode_prompt(template, conv, tok)
    raise ValueError(name)


def encode_prompt(template, conv, tok):
    text = render_chat(template, {"messages": [dict(m) for m in conv["messages"]],
                                  "tools": conv.get("tools")},
                       generation_prompt=True)
    return [2] + tok.encode(text, add_special_tokens=False).ids


def generate(model_dir, prompts, max_new, names, seed=1000):
    """BF16 replies (token ids, stop token included) with Gemma's default
    sampling, batched through exllamav3."""
    sys.path.insert(0, str(KYOJIN))
    from exllamav3 import Cache, Config, Generator, Job, Model
    from exllamav3 import Tokenizer as ExlTokenizer
    from exllamav3.generator.sampler import ComboSampler
    config = Config.from_directory(str(model_dir))
    model = Model.from_config(config)
    tokens = sum(len(ids) for ids in prompts) + len(prompts) * (max_new + 256)
    cache = Cache(model, max_num_tokens=(tokens + 255) // 256 * 256,
                  max_batch_size=len(prompts))
    model.load(progressbar=False)
    generator = Generator(model, cache, ExlTokenizer.from_config(config))
    sampler = ComboSampler(temperature=1.0, top_k=64, top_p=0.95)
    for i, ids in enumerate(prompts):
        generator.enqueue(Job(input_ids=torch.tensor([ids]),
                              max_new_tokens=max_new, sampler=sampler,
                              seed=seed + i, stop_conditions=EOS,
                              identifier=i))
    replies = [[] for _ in prompts]
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            i = r["identifier"]
            if "token_ids" in r:
                replies[i] += r["token_ids"].flatten().tolist()
            if r.get("eos"):
                # The stop token is not streamed; the model sampled it, so
                # it is scored too.
                if r.get("eos_reason") == "stop_token":
                    replies[i].append(r["eos_triggering_token_id"])
                print(f"done {names[i]} ({len(replies[i])} tokens, "
                      f"{r.get('eos_reason')})", flush=True)
    return replies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calib", type=Path)
    p.add_argument("--max-new", type=int, default=1024)
    args = p.parse_args()
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    template = (args.model / "chat_template.jinja").read_text()

    items = []
    for cat, name, conv in prompts():
        if isinstance(conv, int):
            conv, ids = agentic(name, conv, template, tok)
        else:
            ids = encode_prompt(template, conv, tok)
        assert len(ids) <= MAX_PROMPT, (name, len(ids))
        items.append((cat, name, ids))
        print(f"{cat:9s} {name:18s} prompt {len(ids)}", flush=True)

    replies = generate(args.model, [ids for _, _, ids in items], args.max_new,
                       [name for _, name, _ in items])

    sequences, meta, score_from = [], [], []
    for (cat, name, ids), reply in zip(items, replies):
        sequences.append(torch.tensor(ids + reply))
        meta.append((cat, name))
        score_from.append(len(ids))
        print(f"{cat:9s} {name:18s} prompt {len(ids):5d} reply {len(reply):5d}")
    out = {"eval": sequences, "eval_meta": meta, "eval_score_from": score_from,
           "texts": [tok.decode(s[f:].tolist(), skip_special_tokens=False)
                     for s, f in zip(sequences, score_from)]}
    if args.calib:
        out["calib"] = torch.load(args.calib)["calib"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)


if __name__ == "__main__":
    main()
