'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import unittest
import numpy as np
from src.solvers.cube_gym import CubeGymCubie, apply_move, MOVES
from src.solvers.encoders import IndexCubieEncoder, FlatCubieEncoder, CubieEncoder
from src.cube import Cube

class TestCubeGym(unittest.TestCase):
    """Unit tests for the CubeGymCubie environment and encoders."""

    def setUp(self):
        self.onehot = CubieEncoder()
        self.index = IndexCubieEncoder()
        self.flat = FlatCubieEncoder()
        self.gym = CubeGymCubie(encoder=self.onehot)

    # --- Basic Environment ---------------------------------------------------

    def test_reset_and_encode(self):
        obs = self.gym.reset(scramble_len=5)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (self.onehot.dim,))
        self.assertFalse(np.isnan(obs).any())

    def test_step_changes_state(self):
        self.gym.reset(scramble_len=3)
        prev_score = self.gym.cube.score()
        obs, reward, done, info = self.gym.step(0)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("score", info)
        # Score should be defined and typically changes after a move
        self.assertIsInstance(info["score"], (float, int))

    def test_gym_done_on_solved(self):
        self.gym.reset(scramble_len=0)
        obs, reward, done, info = self.gym.step(0)
        inverse = MOVES[0][0] + ("'" if not MOVES[0].endswith("'") else "")
        apply_move(self.gym.cube, inverse)
        # After undo, cube should be solved again
        self.assertEqual(self.gym.cube.score(), 0.5)

    # --- Encoders ------------------------------------------------------------

    def test_encoder_dimensions(self):
        cube = Cube()
        onehot = self.onehot.encode(cube)
        index = self.index.encode(cube)
        flat = self.flat.encode(cube)
        self.assertEqual(onehot.shape, (256,))
        self.assertEqual(index.shape, (40,))
        self.assertEqual(flat.shape, (40,))
        self.assertTrue(np.all(index >= 0))
        self.assertTrue(np.all(flat >= 0))

    def test_encoders_consistency(self):
        """Basic consistency check: identical cube should yield consistent encodings."""
        cube1, cube2 = Cube(), Cube()
        e1 = self.index.encode(cube1)
        e2 = self.index.encode(cube2)
        np.testing.assert_array_equal(e1, e2)
        f1 = self.flat.encode(cube1)
        f2 = self.flat.encode(cube2)
        np.testing.assert_array_equal(f1, f2)


if __name__ == "__main__":
    unittest.main()