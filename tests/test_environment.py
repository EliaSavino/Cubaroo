import unittest

from src.cube import Cube
from src.scorer import Scorer, ScoringOption
from src.solvers.cube_gym import CubeGymCubie, apply_move
from src.solvers.encoders import IndexCubieEncoder


class TestCubeGym(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CubeGymCubie(
            encoder=IndexCubieEncoder(),
            scorer=Scorer(option=ScoringOption.SOLVED_FRACTION),
            alpha=1.0,
            max_steps=10,
        )

    def test_reset_supports_deterministic_seed(self) -> None:
        first = self.env.reset(scramble_len=4, seed=9)
        first_key = self.env.cube.state_key()

        second = self.env.reset(scramble_len=4, seed=9)
        second_key = self.env.cube.state_key()

        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(first_key, second_key)

    def test_step_updates_solve_history(self) -> None:
        self.env.reset(scramble_len=2, seed=4)

        _obs, _reward, done, info = self.env.step(0)

        self.assertFalse(done)
        self.assertEqual(info["history_len"], 1)
        self.assertEqual(info["move"], "U")

    def test_step_rejects_invalid_action(self) -> None:
        self.env.reset(scramble_len=0, seed=1)

        with self.assertRaises(IndexError):
            self.env.step(99)

    def test_apply_move_validates_notation(self) -> None:
        cube = Cube()

        with self.assertRaises(ValueError):
            apply_move(cube, "bad")


if __name__ == "__main__":
    unittest.main()
