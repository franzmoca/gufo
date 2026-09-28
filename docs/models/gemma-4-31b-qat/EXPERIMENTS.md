# Gemma 4 31B QAT experiments

The QAT checkpoint runs on the standard 31B engine; its kernel decisions,
including the Q4_0 paths, are recorded in the
[31B experiments](../gemma-4-31b/EXPERIMENTS.md).

| Experiment | Decision / evidence |
| --- | --- |
| Drafter for the QAT target | Retained: the repository's Q4_0 QAT `gemma4-assistant` (the file `-hf` auto-discovery selects). Its drafter head is already Q4_0, so the Q8_0 → Q4_K head repack does not apply; it runs through the split-K GEMV. |
| QAT vision sidecar | Retained: the repository's own `mmproj-BF16.gguf`; its weights differ from the standard 31B sidecar (SHA-256 `d904b357…` vs `7a4601b1…`). |
