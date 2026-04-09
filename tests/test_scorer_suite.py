import unittest

from src.cube import Cube
from src.scorer import SCORE_FN, Scorer, ScoringOption


class TestScorers(unittest.TestCase):
    def setUp(self) -> None:
        self.solved = Cube()
        self.scrambled = Cube()
        self.scrambled.scramble(length=6, seed=21)

    def test_all_registered_scorers_return_floats(self) -> None:
        for option in ScoringOption:
            with self.subTest(option=option):
                scorer = Scorer(option=option)
                self.assertIsInstance(scorer(self.scrambled), float)

    def test_solved_cube_outscores_scrambled_cube(self) -> None:
        for option in ScoringOption:
            with self.subTest(option=option):
                scorer = Scorer(option=option)
                self.assertGreaterEqual(scorer(self.solved), scorer(self.scrambled))

    def test_dispatch_matches_registry(self) -> None:
        for option, score_fn in SCORE_FN.items():
            with self.subTest(option=option):
                scorer = Scorer(option=option)
                self.assertEqual(scorer(self.scrambled), float(score_fn(self.scrambled)))


if __name__ == "__main__":
    unittest.main()
