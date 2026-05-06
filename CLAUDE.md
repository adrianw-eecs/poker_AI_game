# Claude Code Configuration

## Testing

**When editing code** — run the smoke suite only:
```bash
pytest -m smoke
```
Runs ~22 integration-level tests in < 30 seconds. Covers every major subsystem boundary end-to-end.

**Before pushing to GitHub** — run the full suite:
```bash
pytest
```
Runs ~128 tests in < 120 seconds. Full coverage with no redundant unit tests.

### Test Organization

The test suite is organized into two tiers:

- **Smoke tests** (`@pytest.mark.smoke`): Fast integration tests that validate core functionality across subsystem boundaries. Designed to catch regressions quickly during development.
- **Full tests**: Comprehensive unit and integration tests that verify detailed behavior, edge cases, and invariants.

Key files:
- `tests/integration/test_3player_scenarios.py`: Multi-hand scenarios validating chip conservation and game flow
- `tests/engine/test_session.py`: Session lifecycle and chip management
- `tests/evaluation/test_evaluator.py`: Hand evaluation correctness
- `tests/stats/test_aggregator.py`: Statistics computation from game logs
- `tests/ml/test_env.py`: RL environment contract and state encoding

Never run `pytest` with `--collect-only` or without a filter on the full test suite — the old 500+ test suite has been consolidated to ~128 focused tests.
