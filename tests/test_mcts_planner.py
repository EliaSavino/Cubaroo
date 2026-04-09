'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
import unittest
import numpy as np

from src.cube import Cube
from src.solvers.tree_search_planner import MCTSPlanner
from src.solvers.cube_gym import CubeGymCubie, MOVES, apply_move
from src.solvers.encoders import IndexCubieEncoder
from src.scorer import Scorer, ScoringOption


class TestMCTSPlannerRealEnv(unittest.TestCase):

    def setUp(self):
        # Real env with real scorer/encoder
        self.env = CubeGymCubie(
            encoder=IndexCubieEncoder(),
            scorer=Scorer(option=ScoringOption.SOLVED_FRACTION),
            alpha=1.0,
            max_steps=100
        )
        self.obs = self.env.reset(scramble_len=3, seed=17)

        # MCTS tuned light for unittest speed
        self.planner = MCTSPlanner(
            max_depth=5,
            n_sim=32,
            cpuct=1.5
        )

    def test_choose_action_returns_valid_index(self):
        a = self.planner.choose_action(self.env)
        self.assertIsInstance(a, int)
        self.assertTrue(0 <= a < len(MOVES), "Planner returned invalid action index")

    def test_best_sequence_respects_depth_and_is_applicable(self):
        seq, val = self.planner.best_sequence(self.env)
        # 1) sequence type/length
        self.assertIsInstance(seq, list)
        self.assertTrue(all(isinstance(x, int) for x in seq))
        self.assertLessEqual(len(seq), self.planner.max_depth)

        # 2) sequence is actually applicable to a cube copy
        cube_copy = self._deepcopy_cube(self.env)
        for a in seq:
            apply_move(cube_copy, MOVES[a])
        self.assertIsInstance(cube_copy, Cube)

    def test_search_tree_structure(self):
        info = self.planner.search_tree(self.env)
        self.assertIn("children", info)
        self.assertIsInstance(info["children"], list)
        # Children entries have expected keys
        if info["children"]:
            child0 = info["children"][0]
            for k in ("action_idx", "move", "visits", "Q"):
                self.assertIn(k, child0)

    def test_planner_does_not_mutate_env(self):
        # Record current env “solve” history length (or fallback to a cheap proxy)
        hist_len_before = self._solve_phase_len(self.env)
        _ = self.planner.choose_action(self.env)
        hist_len_after = self._solve_phase_len(self.env)
        self.assertEqual(
            hist_len_before, hist_len_after,
            "Planner should not mutate live env state/history"
        )

    # --- helpers ---
    def _deepcopy_cube(self, env: CubeGymCubie):
        try:
            return env.cube.copy()
        except Exception:
            import copy
            return copy.deepcopy(env.cube)

    def _solve_phase_len(self, env: CubeGymCubie) -> int:
        try:
            hist = env.cube.get_history()
            if "phase" in hist.columns:
                return int((hist["phase"] == "solve").sum())
            return len(hist)
        except Exception:
            # If history not available, use a stable proxy from encoder output
            # This merely ensures the planner didn’t call env.step(...)
            obs = env.encoder.encode(env.cube)
            return int(np.sum(obs))  # any deterministic proxy works for equality check


if __name__ == "__main__":
    unittest.main()
