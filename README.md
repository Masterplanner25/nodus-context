# nodus-context

**LLM context window management: token budget, compaction strategies, and
tool result guards for AI systems.**

Keeps agent conversations within finite LLM context windows without manual
message pruning. No required external dependencies — token counting uses a
word-count estimate by default; install `tiktoken` for accurate counts.

> **Status:** v0.1.0 — published on [PyPI](https://pypi.org/project/nodus-context/).

---

## Install

```bash
pip install nodus-context

# With accurate tiktoken-based token counting:
pip install "nodus-context[tiktoken]"
```

---

## What it provides

| Component | Purpose |
|---|---|
| `ContextBudget` | Token budget with utilization tracking and overflow detection |
| `ContextMessage` | Normalized message (role, content, token count) |
| `ContextWindow` | Message list with budget enforcement and compaction |
| `DropToolInternalsStrategy` | Remove intermediate tool calls, keep final pairs |
| `SummarizeStrategy` | Keep head + tail, replace middle with a summary message |
| `estimate_tokens` / `count_tokens` | Word-estimate or tiktoken-accurate counting |

---

## Quick start

```python
from nodus_context import ContextWindow, ContextBudget, ROLE_USER, ROLE_ASSISTANT

budget = ContextBudget(max_tokens=8192)
window = ContextWindow(budget=budget)

window.add(ROLE_USER, "What is the capital of France?")
window.add(ROLE_ASSISTANT, "The capital of France is Paris.")

messages = window.messages()   # list of ContextMessage
print(f"Using {budget.used}/{budget.max_tokens} tokens")
```

---

## ContextBudget

```python
from nodus_context import ContextBudget

budget = ContextBudget(max_tokens=4096, reserve_tokens=512)
budget.add(150)                # record token usage
budget.utilization             # float 0.0–1.0
budget.available               # tokens remaining (respects reserve)
budget.is_over_budget          # True if used > max_tokens
budget.reset()
```

---

## ContextWindow

```python
from nodus_context import ContextWindow, ContextBudget, DropToolInternalsStrategy

window = ContextWindow(
    budget=ContextBudget(max_tokens=8192),
    compaction_strategy=DropToolInternalsStrategy(),
    compaction_threshold=0.85,   # compact when 85% full
)

window.add(role, content)         # adds message, tracks tokens
window.add_raw(context_message)   # add pre-built ContextMessage
window.messages()                 # current message list
window.compact()                  # trigger compaction manually
window.guard_tool_results(n=2)    # keep only the last n tool result pairs
window.token_count                # current total
window.message_count
```

Compaction runs automatically when `utilization >= compaction_threshold`.

---

## Compaction strategies

### DropToolInternalsStrategy

Removes intermediate `tool_use` / `tool_result` pairs, keeping only the
final `n` pairs. Reduces context without losing the outcome.

```python
from nodus_context import DropToolInternalsStrategy
strategy = DropToolInternalsStrategy(keep_last_n_pairs=1)
```

### SummarizeStrategy

Keeps the first `head_count` and last `tail_count` messages; replaces
everything in between with a single summary message.

```python
from nodus_context import SummarizeStrategy
strategy = SummarizeStrategy(
    head_count=2,
    tail_count=4,
    summary_fn=lambda msgs: "Summary of intermediate steps.",
)
```

`summary_fn` can call an LLM — any callable that takes a list of
`ContextMessage` and returns a string.

---

## Token counting

```python
from nodus_context import estimate_tokens, count_tokens

estimate_tokens("Hello, world!")   # fast word-count estimate (no dep)
count_tokens("Hello, world!")      # tiktoken if installed, else estimate
```

Install `nodus-context[tiktoken]` for accurate counts.

---

## Role constants

```python
from nodus_context import (
    ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM,
    ROLE_TOOL_USE, ROLE_TOOL_RESULT, ROLE_THINKING,
)
```

---

## Design

- **No required dependencies.** Token estimation uses `~4 chars per token`
  heuristic. Accurate counting requires `tiktoken` (optional extra).
- **Protocol-based strategies.** Any callable satisfying `CompactionStrategy`
  works — no inheritance needed.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

---

## License

MIT — see [LICENSE](LICENSE).
