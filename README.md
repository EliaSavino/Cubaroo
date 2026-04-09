# Cubaroo

`Cubaroo` is a Rubik's Cube experimentation repo built around an explicit cubie
state model, heuristic scorers, reinforcement-learning environments, and search
planners.

The project is now organized around a small public API:

- `src/index.py`: stable convenience entrypoint for the most common classes.
- `src/cube.py`: cube state, move application, history tracking, and rendering.
- `src/cubies.py`: cubie dataclasses and lookup tables.
- `src/scorer.py`: heuristic and pattern-database based scoring functions.
- `src/solvers/`: encoders, environment wrappers, planners, and training code.
- `src/models/`: policy/value model helpers.

## Quick Start

```python
from src.index import Cube, CubeGymCubie, IndexCubieEncoder

cube = Cube()
cube.scramble(length=10, seed=7)
print(cube.format_net(use_color=False))

env = CubeGymCubie(encoder=IndexCubieEncoder())
obs = env.reset(scramble_len=3, seed=7)
```

## Repository Layout

See [docs/architecture.md](/Users/es/GitHub/Cubaroo/docs/architecture.md) for a
module-level map of the repo.

See [docs/testing.md](/Users/es/GitHub/Cubaroo/docs/testing.md) for the current
test strategy and the command used in CI/local verification.

## Running Tests

```bash
python -m unittest discover -s tests
```

## Design Notes

- The cube state is stored as corner and edge cubie objects. Rendered facelets
  and encoded observations are derived views.
- Scrambles support deterministic seeds so tests and experiments can reproduce
  the same states.
- Rich reward shaping lives in `src.scorer`; the `Cube` object keeps only a
  simple baseline score.
