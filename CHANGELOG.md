# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2026-05-30

Initial release — prepared, not yet published.

### Added

- **`ContextMessage`** — normalized message record. Fields: `role`, `content`,
  `token_count`. Role constants: `ROLE_USER`, `ROLE_ASSISTANT`, `ROLE_SYSTEM`,
  `ROLE_TOOL_USE`, `ROLE_TOOL_RESULT`, `ROLE_THINKING`.

- **`ContextBudget`** — token budget with utilization and overflow tracking.
  `max_tokens`, `reserve_tokens`, `used`, `available`, `utilization`,
  `is_over_budget`. `add(n)`, `reset()`.

- **`ContextWindow`** — message list with automatic compaction.
  `add(role, content)`, `add_raw(msg)`, `messages()`, `compact()`,
  `guard_tool_results(n)`. Compaction fires when
  `utilization >= compaction_threshold` (default 0.85).

- **`estimate_tokens(text)`** — word-count token estimate (~4 chars/token).
  No dependencies.

- **`count_tokens(text)`** — uses `tiktoken` if installed, falls back to
  `estimate_tokens`. Requires `nodus-context[tiktoken]` for accuracy.

- **`CompactionStrategy`** — protocol: `compact(messages) -> list[ContextMessage]`.

- **`DropToolInternalsStrategy`** — removes intermediate `tool_use` /
  `tool_result` pairs, keeping the last `n` pairs (default 1).

- **`SummarizeStrategy`** — keeps `head_count` + `tail_count` messages;
  replaces the middle with a single summary message produced by `summary_fn`.

- **29 tests** across three test files (budget, strategies, window).

- **No required dependencies** — pure stdlib. Optional `[tiktoken]` extra for
  accurate token counting.

[0.1.0]: https://github.com/Masterplanner25/nodus-context/releases/tag/v0.1.0
