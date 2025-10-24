'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
import argparse
import os
import random
import numpy as np
import torch

# project imports
from src.solvers.manager import DQNTrainer, DQNConfig
from src.solvers.cube_gym import CubeGymCubie
from src.solvers.encoders import IndexCubieEncoder  # <-- token/index encoder
from src.models.graph_message_passing import CubeGNNQNet        # <-- your GNN model file

N_ACTIONS = 12  # fixed: 6 faces * 2 directions * 1 (quarter/half handled in env if needed)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser("Train GNN DQN on Cube (token/index encoder)")
    # training loop / env
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--scramble", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0, help="reward scale for Δprogress")
    p.add_argument("--max-steps", type=int, default=200, dest="max_steps")

    # optimization
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--target-sync-every", type=int, default=2_000)

    # epsilon-greedy
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=200_000)

    # eval + curriculum
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--curriculum-success", type=float, default=0.30)
    p.add_argument("--curriculum-max", type=int, default=20)

    # model hyperparams
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)

    # bookkeeping
    p.add_argument("--exp", type=str, default="gnn_dqn")
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.models_dir, exist_ok=True)

    # --- Env + Model ----------------------------------------------------------
    # Index/token encoder expected to produce integer tokens per cubie/node.
    env = CubeGymCubie(
        encoder=IndexCubieEncoder(),  # <- emits token indices for CubieTokenEmbedding
        alpha=args.alpha,
        max_steps=args.max_steps,
    )

    model = CubeGNNQNet(
        d_model=args.d_model,
        layers=args.layers,
        n_actions=N_ACTIONS,
    )

    # --- Config ---------------------------------------------------------------
    cfg = DQNConfig(
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        total_steps=args.steps,
        warmup_steps=4 * args.batch_size,
        target_sync_every=args.target_sync_every,  # ignored if Polyak in trainer
        train_every=args.train_every,
        eval_every=args.eval_every,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        curriculum_success=args.curriculum_success,
        curriculum_max_scramble=args.curriculum_max,
        save_path=os.path.join(args.models_dir, "cube_gnn.pt"),
        output_dir=args.outdir,
        experiment_name=args.exp,
    )

    # --- Train ---------------------------------------------------------------
    trainer = DQNTrainer(env, model, cfg)
    trainer.train(start_scramble=args.scramble)


if __name__ == "__main__":
    main()