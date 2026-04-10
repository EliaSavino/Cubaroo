import unittest

import torch.nn as nn

from src.scorer import Scorer, ScoringOption
from src.solvers.cube_gym import CubeGymCubie
from src.solvers.encoders import IndexCubieEncoder
from src.solvers.manager import DQNConfig, DQNTrainer


class TestCurriculumHelpers(unittest.TestCase):
    def setUp(self) -> None:
        env = CubeGymCubie(
            encoder=IndexCubieEncoder(),
            scorer=Scorer(option=ScoringOption.CONSISTENT_PROGRESS),
            alpha=1.0,
            max_steps=10,
        )
        model = nn.Linear(40, 12)
        cfg = DQNConfig(total_steps=10, eval_every=5, use_mcts=False)
        self.trainer = DQNTrainer(env, model, cfg)

    def test_curriculum_schedule_is_gradual(self) -> None:
        schedule = self.trainer._curriculum_schedule(start_scramble=1)

        self.assertEqual(schedule[:6], [1, 2, 3, 4, 5, 6])
        self.assertEqual(schedule[-1], self.trainer.cfg.curriculum_max_scramble)

    def test_curriculum_target_relaxes_for_harder_scrambles(self) -> None:
        schedule = self.trainer._curriculum_schedule(start_scramble=1)
        easy = self.trainer._curriculum_target(1, schedule)
        mid = self.trainer._curriculum_target(6, schedule)
        hard = self.trainer._curriculum_target(schedule[-1], schedule)

        self.assertGreater(easy, mid)
        self.assertGreater(mid, hard)
        self.assertAlmostEqual(easy, self.trainer.cfg.curriculum_easy_success)
        self.assertAlmostEqual(hard, self.trainer.cfg.curriculum_success)


if __name__ == "__main__":
    unittest.main()
