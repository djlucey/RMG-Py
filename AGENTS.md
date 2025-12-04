# Repository Guidelines

## Project Structure & Module Organization
- Core libraries live in `rmgpy/` (generation algorithms, kinetics, thermo, solvers) and `arkane/` (thermo and rate calculations). Shared utilities sit in `utilities.py`.
- Domain data and examples are under `kinetics_database/`, `thermo`, and `examples/`. The ammonia Cantera tutorials you just added reside in `examples/rmg/ammonia/`.
- Tests live alongside code in `rmgpy/**/tests` and in the top-level `test/` folder. Documentation sources are under `documentation/` (built HTML ends up in `documentation/build/html`).

## Build, Test, and Development Commands
- Ensure dependencies: `python utilities.py check-dependencies` (checks compilers, libraries) and `python utilities.py check-pydas`.
- Editable install: `python -m pip install -e .` (use an env from `environment.yml`).
- Run all tests: `python -m pytest` (or `make test` for the default unit test selection). Functional/database subsets: `python -m pytest -m "functional"` or `python -m pytest -m "database"`.
- Build docs locally: `make documentation` (then open `documentation/build/html/index.html`).
- Quick examples: `make eg1` (minimal ethane pyrolysis) or other `eg*` targets to exercise end-to-end flows.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indentation, and prefer explicit imports. Use type hints where practical.
- Match existing naming: modules and packages are `snake_case`, classes `CapWords`, constants `UPPER_SNAKE`.
- Keep functions cohesive and documented with short docstrings; align with existing logging and error-handling patterns in `rmgpy`.

## Testing Guidelines
- Add or update unit tests near the code they cover (e.g., `rmgpy/<module>/tests/`). Name tests `test_<behavior>` and group by feature.
- For new examples or workflows, include minimal regression checks where feasible; prefer fast-running tests over long examples.
- Run `python -m pytest` before opening a PR; for database-heavy changes, also run the marked `functional` or `database` suites if relevant.

## Commit & Pull Request Guidelines
- Use concise, present-tense commit messages (e.g., `add cantera tutorial notebook`, `fix pressure controller logic`). Squash locally only when it clarifies history.
- PRs should describe the change, rationale, and verification (tests run, examples exercised). Link issues when applicable and include screenshots or plots for user-facing changes (e.g., new notebooks or docs).
- Keep diffs focused; avoid reformatting unrelated files. Ensure CI passes and documentation builds if you touch docs.

## Security & Configuration Tips
- Do not commit credentials or external data; keep secrets out of notebooks and config. Prefer environment variables for any local paths or tokens.
- When adding dependencies, update `environment.yml` and justify their scope; avoid widening the trusted surface without review.
