# Contributing

Treat the repository as research code. Preserve existing experiment contracts unless a schema or interface change is intentional.

## Setup

```bash
make install
```

## Before review

- Run `make test`.
- Run `make smoke` when you only need a fast local check.
- Update documentation when command names, config expectations, or output locations change.
- Keep runnable examples and templates aligned when changing the workflow surface.
- Keep generated outputs and downloaded datasets out of version control.

## Style

- Prefer small changes that preserve existing research logic.
- Keep user-facing documentation and help text in concise English.
- Treat output filenames, CSV schemas, and manifest keys as public interfaces unless there is a strong reason to change them.

## Changes that affect reproducibility

If a change affects training behavior, config defaults, or artifact schemas, document the impact in the review notes and add or update a focused test when practical.
