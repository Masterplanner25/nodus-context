# Contributing to nodus-context

## Setup

```bash
git clone https://github.com/Masterplanner25/nodus-context.git
cd nodus-context
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -q
```

## Code style

- Python 3.11+
- No required external dependencies (stdlib only; tiktoken is optional)
- Type hints on all public functions
- `CompactionStrategy` is a protocol — new strategies satisfy it by
  structure, not inheritance

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Add tests for any new behaviour
3. Ensure `pytest tests/ -q` passes
4. Open a pull request with a description of what changes and why
