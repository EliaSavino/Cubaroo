import unittest

import numpy as np
import torch

from src.models.mlpq_net import MLPQNet
from src.models.tiny_transformer import TransformerQNet
from src.solvers.manager import DQNConfig
from src.solvers.solver_utilities import linear_epsilon


class TestTrainingDefaults(unittest.TestCase):
    def test_cosine_epsilon_hold_and_decay(self) -> None:
        cfg = DQNConfig(
            eps_start=1.0,
            eps_end=0.1,
            eps_hold_steps=10,
            eps_decay_steps=90,
            eps_schedule="cosine",
        )

        self.assertEqual(linear_epsilon(0, cfg), 1.0)
        self.assertEqual(linear_epsilon(10, cfg), 1.0)
        self.assertGreater(linear_epsilon(20, cfg), 0.1)
        self.assertAlmostEqual(linear_epsilon(200, cfg), 0.1)

    def test_linear_epsilon_schedule_still_supported(self) -> None:
        cfg = DQNConfig(
            eps_start=1.0,
            eps_end=0.2,
            eps_hold_steps=0,
            eps_decay_steps=100,
            eps_schedule="linear",
        )

        self.assertAlmostEqual(linear_epsilon(50, cfg), 0.6)

    def test_mlp_qnet_output_shape(self) -> None:
        model = MLPQNet(in_dim=256, hidden=256)
        x = torch.zeros(4, 256)
        q = model(x)

        self.assertEqual(tuple(q.shape), (4, 12))

    def test_transformer_qnet_output_shape(self) -> None:
        model = TransformerQNet(d_model=64, nhead=8, num_layers=2)
        tokens = torch.from_numpy(np.zeros((3, 40), dtype=np.int64))
        q = model(tokens)

        self.assertEqual(tuple(q.shape), (3, 12))


if __name__ == "__main__":
    unittest.main()
