# Contributing

## Setup

```bash
mise install        # pins Python, uv and oxfmt (mise.toml)
mise run install    # uv sync --locked
```

Without mise, `uv` fetches Python 3.14 on the first sync. The `train` extra
(`uv sync --extra train`) adds Ultralytics, SAHI and `lap`; it pulls torch and
is needed only for real-backend runs.

## Check

CI runs `mise run check` on every pull request and every push to `master`
([`.github/workflows/ci.yml`](workflows/ci.yml)). Run it before you push:

```console
$ mise run check
```

It runs three tasks from [`mise.toml`](../mise.toml) and writes no files:

| Task                 | Checks                                                                   |
| -------------------- | ------------------------------------------------------------------------ |
| `mise run lint`      | Ruff on Python (`lint:py`), oxfmt on Markdown, YAML and TOML (`lint:md`) |
| `mise run typecheck` | mypy in strict mode on `src` and `tests`                                 |
| `mise run test`      | pytest                                                                   |

`mise run fix` applies the safe Ruff fixes and formats Python, Markdown, YAML
and TOML.

## Tests

[`tests/test_smoke.py`](../tests/test_smoke.py) runs on the CPU with the dummy
backend. It covers rotated IoU, the corner round trip, the text-cell round trip,
the metric on perfect and partial predictions, the clip-grouped split, the
`git_sha` fallback, and a full sweep over synthetic data in a temporary
directory. Add a test for the behavior you change. Real-backend code needs the
`train` extra and a GPU and has no test, except the conversion of Ultralytics
output to IR in [`tests/test_infer.py`](../tests/test_infer.py), which runs
against a fake model.

## Documentation

The manual is in [`docs/`](../docs/readme.md) and the module map is
[`docs/architecture.md`](../docs/architecture.md). oxfmt wraps Markdown at 80
columns; `mise run fix` formats it and `mise run lint:md` checks it.

Run the commands you document and paste their real output.
