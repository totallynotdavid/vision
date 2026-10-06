# Contributing

## Setup

```bash
mise install            # pins Python, uv and ruff (mise.toml)
uv sync --extra dev
```

Without mise, `uv` fetches Python 3.14 on the first sync. The `train` extra
(`uv sync --extra train`) adds Ultralytics, SAHI and `lap`; it pulls torch and
is needed only for real-backend runs.

## Check

CI runs `mise run check` on every pull request
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Run it before you
push:

```console
$ mise run check
```

It runs three tasks from [`mise.toml`](mise.toml):

| Task                    | Command                        |
| ----------------------- | ------------------------------ |
| `mise run lint`         | `uv run ruff check .`          |
| `mise run format-check` | `uv run ruff format --check .` |
| `mise run test`         | `uv run pytest`                |

`mise run format` applies the Ruff formatter to the repository.

## Tests

[`tests/test_smoke.py`](tests/test_smoke.py) runs on the CPU with the dummy
backend. It covers rotated IoU, the corner round trip, the text-cell round trip,
the metric on perfect and partial predictions, the clip-grouped split, the
`git_sha` fallback, and a full sweep over synthetic data in a temporary
directory. Add a test for the behavior you change. Real-backend code needs the
`train` extra and a GPU and has no test.

## Documentation

The manual is in [`docs/`](docs/readme.md) and the module map is
[`architecture.md`](architecture.md). Markdown is wrapped at 80 columns:

```bash
bunx prettier --print-width 80 --prose-wrap always --write readme.md architecture.md contributing.md AGENTS.md 'docs/*.md'
```

Run the commands you document and paste their real output.
