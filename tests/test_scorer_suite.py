import unittest

from src.cube import Cube
from src.scorer import SCORE_FN, Scorer, ScoringOption
from tests.test_functions import (
    make_f2l,
    make_first_layer,
    make_top_cross_in_place,
)


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

    def test_default_scorer_is_state_only(self) -> None:
        scorer = Scorer()
        a = Cube()
        b = Cube()

        for move in ["R", "U", "F", "U'", "F'", "R'"]:
            a.rotate(move[0], clockwise=not move.endswith("'"))
        for move in ["R", "U", "F", "U'", "F'", "R'"]:
            b.rotate(move[0], clockwise=not move.endswith("'"))

        b.clear_history()

        self.assertEqual(a.state_key(), b.state_key())
        self.assertEqual(scorer(a), scorer(b))

    def test_consistent_progress_respects_major_subgoals(self) -> None:
        scorer = Scorer(option=ScoringOption.CONSISTENT_PROGRESS)
        solved = Cube()
        f2l = make_f2l(seed=3)
        first_layer = make_first_layer(seed=3)
        top_cross = make_top_cross_in_place(seed=3)
        scrambled = Cube()
        scrambled.scramble(length=12, seed=3)

        self.assertGreaterEqual(scorer(solved), scorer(f2l))
        self.assertGreaterEqual(scorer(f2l), scorer(first_layer))
        self.assertGreaterEqual(scorer(first_layer), scorer(top_cross))
        self.assertGreaterEqual(scorer(top_cross), scorer(scrambled))

    def test_lookahead_progress_is_not_more_myopic_than_base(self) -> None:
        base = Scorer(option=ScoringOption.CONSISTENT_PROGRESS)
        lookahead = Scorer(option=ScoringOption.LOOKAHEAD_PROGRESS)
        cube = Cube()
        cube.scramble(length=8, seed=19)

        self.assertGreaterEqual(lookahead(cube), base(cube))


if __name__ == "__main__":
    unittest.main()
