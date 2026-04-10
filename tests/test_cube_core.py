import unittest

from src.cube import Cube, VALID_FACES
from src.solvers.cube_gym import apply_move


class TestCubeCore(unittest.TestCase):
    def test_new_cube_starts_solved(self) -> None:
        cube = Cube()

        self.assertTrue(cube.is_solved())
        self.assertEqual(cube.solved_fraction(), 1.0)
        self.assertEqual(cube.moves_since_scramble(), 0)

    def test_each_face_turn_is_invertible(self) -> None:
        for face in VALID_FACES:
            with self.subTest(face=face):
                cube = Cube()
                start = cube.state_key()

                cube.rotate(face, True)
                cube.rotate(face, False)

                self.assertEqual(cube.state_key(), start)
                self.assertTrue(cube.is_solved())

    def test_scramble_is_deterministic_with_seed(self) -> None:
        left = Cube()
        right = Cube()

        left.scramble(length=8, seed=13)
        right.scramble(length=8, seed=13)

        self.assertEqual(left.state_key(), right.state_key())
        self.assertEqual(left.get_history().to_dict("records"), right.get_history().to_dict("records"))
        self.assertEqual(left.moves_since_scramble(), 0)
        self.assertTrue((left.get_history()["phase"] == "scramble").all())
        self.assertEqual(left.consistency_issues(), [])

    def test_clear_history_and_format_net(self) -> None:
        cube = Cube()
        cube.scramble(length=3, seed=5)
        cube.rotate("R")

        self.assertGreater(len(cube.get_history()), 0)

        cube.clear_history()
        net = cube.format_net(use_color=False)

        self.assertEqual(len(cube.get_history()), 0)
        self.assertEqual(cube.moves_since_scramble(), 0)
        self.assertIsInstance(net, str)
        self.assertIn("0", net)

    def test_invalid_face_raises(self) -> None:
        cube = Cube()

        with self.assertRaises(ValueError):
            cube.rotate("X")

    def test_random_move_sequence_preserves_consistency(self) -> None:
        cube = Cube()
        moves = ["R", "U", "F", "L'", "D", "B'", "R'", "U'"]

        for move in moves:
            apply_move(cube, move)
            cube.assert_consistent()

    def test_inverse_sequence_returns_to_solved_state(self) -> None:
        cube = Cube()
        moves = ["R", "U", "F", "L'", "D", "B'"]

        for move in moves:
            apply_move(cube, move)
        for move in reversed(moves):
            inverse = move[0] if move.endswith("'") else f"{move}'"
            apply_move(cube, inverse)

        cube.assert_consistent()
        self.assertTrue(cube.is_solved())


if __name__ == "__main__":
    unittest.main()
