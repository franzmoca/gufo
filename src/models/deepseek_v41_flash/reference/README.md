# DeepSeek V4 Flash 0731 encoding reference

The compiled formatter is pinned to
`deepseek-ai/DeepSeek-V4-Flash-0731` revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

- Official `encoding/encoding_dsv4.py` SHA-256:
  `abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6`
- Official `tokenizer.json` SHA-256:
  `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`
- Recognized antirez GGUF `tokenizer.chat_template` SHA-256:
  `872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27`

The Python encoder and embedded Jinja are reference data; Gufo implements the
official history, effort, tool-loop, and generation-prefix semantics in C++.
