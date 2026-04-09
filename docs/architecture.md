# Architecture

## Core Modules

- `src/index.py`
  Public convenience entrypoint. Import from here when you want the common cube,
  scorer, encoder, and planner classes without reaching through package internals.

- `src/cube.py`
  The main cube state container. Handles rotations, scrambling, move history,
  solved checks, array conversion, and text/3D rendering helpers.

- `src/cubies.py`
  Dataclasses and lookup tables for corners and edges, including canonical slot
  order, facelet coordinates, and sticker color definitions.

- `src/scorer.py`
  Contains basic solved-fraction scoring, weighted placement/orientation scores,
  and phase-1 heuristics including cached pattern-database support.

## Solver Modules

- `src/solvers/encoders.py`
  Encodes a cube into one-hot, flat float, or index-based feature vectors.

- `src/solvers/cube_gym.py`
  Small environment wrapper that exposes `reset()` and `step()` for training or
  search code.

- `src/solvers/tree_search_planner.py`
  Monte Carlo tree search planner that can use either the scorer or a learned
  Q-function to evaluate leaf states.

- `src/solvers/manager.py`
  DQN training loop and configuration for longer-running experiments.

## Models

- `src/models/`
  Model wrappers and network definitions used by the trainer. This directory is
  intentionally left mostly untouched by the cleanup except for package docs and
  import normalization.

## Tests

- `tests/test_cube_core.py`
  Cube mechanics, history, deterministic scramble behavior, and text formatting.

- `tests/test_scorer_suite.py`
  Scoring behavior and scorer dispatch.

- `tests/test_environment.py`
  Environment reset/step behavior and move validation.

- `tests/test_mcts_planner.py`
  Planner integration with the real environment.

- `tests/test_package_api.py`
  Smoke coverage for the curated package entrypoints.
