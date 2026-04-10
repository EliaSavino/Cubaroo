# Testing

The repo uses Python's built-in `unittest` framework.

## Command

```bash
python -m unittest discover -s tests
```

## Coverage Focus

- deterministic cube scrambles with explicit seeds
- inverse move correctness and solved-state invariants
- cube structural consistency and facelet-count validity
- history bookkeeping across scramble and solve phases
- scorer dispatch, state-only consistency, and monotonic sanity checks
- environment reset/step behavior
- planner integration without mutating the live environment
- top-level package exports through `src` and `src.index`

## Test Design Rules

- tests avoid printing to stdout during normal runs
- tests prefer deterministic seeds over random behavior
- helper builders stay in `tests/test_functions.py`
- exploratory scripts remain separate from the `test_*.py` suites that are
  executed during discovery
