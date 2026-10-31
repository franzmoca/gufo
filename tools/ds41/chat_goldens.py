#!/usr/bin/env python3
"""Write DeepSeek V4.1 chat-template goldens from the official encoder.

Runs the official ``encoding/encoding.py`` (DeepSeek-V4.1-Flash-1017) over a
fixed set of conversations and writes each case's input, in the shape the
compiled renderer takes, together with the prompt the official encoder
produces. Serving conventions follow deepseek-recipe: request tools go into a
leading system message (an empty one is inserted when the conversation has
none) and the ``developer`` role is a system message.

  python3 -I tools/ds41/chat_goldens.py --encoding PATH/encoding.py \
      --output tests/models/deepseek_v41_flash/fixtures/encoding-1017.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
}
SEARCH = {
    "name": "search",
    "description": "Search the web — returns snippets.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object"}}},
    "strict": True,
}


def image(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def text(value: str) -> dict:
    return {"type": "text", "text": value}


def call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


CASES = [
    {"name": "chat_single", "thinking": False, "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "thinking_default_effort", "thinking": True, "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"name": "thinking_low_system", "thinking": True, "effort": "low",
     "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Explain RoPE."}]},
    {"name": "thinking_max_history", "thinking": True, "effort": "max", "messages": [
        {"role": "user", "content": "Pick a number."},
        {"role": "assistant", "reasoning_content": "Any number works.", "content": "7"},
        {"role": "user", "content": "Double it."}]},
    {"name": "chat_history_system", "thinking": False, "messages": [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Bonjour !"},
        {"role": "user", "content": "How are you?"}]},
    {"name": "tools_thinking_parallel", "thinking": True, "tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Use tools when helpful."},
        {"role": "user", "content": "Weather in Rome and news about it?"},
        {"role": "assistant", "reasoning_content": "Two independent calls.", "content": "",
         "tool_calls": [call("call_a", "get_weather", {"city": "Rome", "days": 2}),
                        call("call_b", "search", {"query": "Rome news", "filters": {"lang": "it", "max": 3}})]},
        {"role": "tool", "tool_call_id": "call_b", "content": "Snippet: Colosseum restored."},
        {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp\": 21}"}]},
    {"name": "tools_chat_no_system", "thinking": False, "tools": [WEATHER],
     "messages": [{"role": "user", "content": "Weather in Oslo?"}]},
    {"name": "mid_conversation_system", "thinking": True, "messages": [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Tell me a fact."},
        {"role": "assistant", "reasoning_content": "Pick something simple.", "content": "Water boils at 100 °C."},
        {"role": "system", "content": "From now on answer in one word."},
        {"role": "user", "content": "And ice?"}]},
    {"name": "tool_result_then_user", "thinking": True, "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "reasoning_content": "Call the tool.", "content": "",
         "tool_calls": [call("c1", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 25°C"},
        {"role": "user", "content": "Thanks, and tomorrow?"}]},
    {"name": "developer_role", "thinking": False, "messages": [
        {"role": "developer", "content": "Reply with JSON only."}, {"role": "user", "content": "List two colors."}]},
    {"name": "assistant_last", "thinking": False, "messages": [
        {"role": "user", "content": "Say hi."}, {"role": "assistant", "content": "Hi!"}]},
    {"name": "unicode_and_escapes", "thinking": True, "effort": "high", "tools": [SEARCH], "messages": [
        {"role": "user", "content": "日本語で \"quotes\" and \\backslash\\ \n new line?"},
        {"role": "assistant", "reasoning_content": "", "content": "",
         "tool_calls": [call("x", "search", {"query": "引用 \"x\"", "filters": {"tags": ["a", "b"], "n": 1.5, "ok": True}})]},
        {"role": "tool", "tool_call_id": "x", "content": "no results"}]},
    {"name": "image_between_text", "thinking": False, "messages": [
        {"role": "user", "content": [text("第一张图"), image("a.png"), text("有什么内容？")]}]},
    {"name": "images_only_then_text", "thinking": True, "messages": [
        {"role": "user", "content": [image("a.png"), image("b.png")]},
        {"role": "assistant", "reasoning_content": "Two pictures.", "content": "Two pictures."},
        {"role": "user", "content": [text("Compare them."), image("c.png")]}]},
    {"name": "image_tool_result", "thinking": True, "tools": [SEARCH], "messages": [
        {"role": "user", "content": "Find a photo of Rome."},
        {"role": "assistant", "reasoning_content": "Search.", "content": "",
         "tool_calls": [call("s1", "search", {"query": "Rome photo"})]},
        {"role": "tool", "tool_call_id": "s1", "content": [text("Top result:"), image("rome.jpg")]},
        {"role": "user", "content": [image("mine.jpg"), text("Is it like mine?")]}]},
]


def load_encoder(path: Path):
    spec = importlib.util.spec_from_file_location("ds41_official_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_messages(case: dict) -> list:
    """The case as the official encoder takes it, with deepseek-recipe serving conventions."""
    messages = copy.deepcopy(case["messages"])
    for message in messages:
        if message["role"] == "developer":
            message["role"] = "system"
    if case.get("tools"):
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = [{"type": "function", "function": tool} for tool in case["tools"]]
    return messages


def renderer_input(case: dict) -> dict:
    """The case in the compiled renderer's message shape."""
    messages = []
    for message in case["messages"]:
        content = message.get("content", "")
        out = {"role": message["role"], "content": content}
        if isinstance(content, list):
            # The HTTP parsers append text parts and place each image at the
            # current end of the text.
            out["content"], out["images"] = "", []
            for block in content:
                if block["type"] == "text":
                    out["content"] += block["text"]
                else:
                    out["images"].append(len(out["content"].encode()))
        if "reasoning_content" in message:
            out["reasoning_content"] = message["reasoning_content"]
        if "tool_call_id" in message:
            out["tool_call_id"] = message["tool_call_id"]
        if message.get("tool_calls"):
            out["tool_calls"] = []
            for tool_call in message["tool_calls"]:
                arguments = json.loads(tool_call["function"]["arguments"])
                out["tool_calls"].append({
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "arguments": [
                        {"name": key, "is_string": isinstance(value, str),
                         "value": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}
                        for key, value in arguments.items()
                    ],
                })
        messages.append(out)
    tools = [{"name": tool["name"], "definition_json": json.dumps(tool, ensure_ascii=False, separators=(",", ":"))}
             for tool in case.get("tools", [])]
    return {"messages": messages, "tools": tools}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoding", type=Path, required=True, help="official encoding/encoding.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoder = load_encoder(args.encoding)
    cases = []
    for case in CASES:
        prompt = encoder.encode_messages(
            official_messages(case), thinking_mode="thinking" if case["thinking"] else "chat",
            reasoning_effort=case.get("effort"))
        cases.append({"name": case["name"], "thinking": case["thinking"], "effort": case.get("effort"),
                      **renderer_input(case), "expected": prompt})
    fixture = {
        "schema": "gufo.ds41-chat-goldens.v1",
        "source": "deepseek-ai/DeepSeek-V4.1-Flash-1017 encoding/encoding.py",
        "encoding_sha256": hashlib.sha256(args.encoding.read_bytes()).hexdigest(),
        "conventions": "deepseek-recipe: request tools in a leading system message; developer is system",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
