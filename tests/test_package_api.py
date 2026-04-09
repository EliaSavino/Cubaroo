import unittest

import src
from src import index


class TestPackageAPI(unittest.TestCase):
    def test_top_level_exports_match_index(self) -> None:
        self.assertIs(src.Cube, index.Cube)
        self.assertIs(src.Scorer, index.Scorer)
        self.assertIs(src.CubeGymCubie, index.CubeGymCubie)
        self.assertIs(src.MCTSPlanner, index.MCTSPlanner)

    def test_index_exports_constructible_types(self) -> None:
        cube = index.Cube()
        env = index.CubeGymCubie(encoder=index.IndexCubieEncoder())

        self.assertTrue(cube.is_solved())
        self.assertEqual(env.encoder.dim, 40)


if __name__ == "__main__":
    unittest.main()
