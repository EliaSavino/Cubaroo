'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import torch

# project imports
from src.solvers.manager import DQNTrainer, DQNConfig
from src.solvers.cube_gym import CubeGymCubie, CubieEncoder  # one-hot encoder
from src.models.mlpq_net import MLPQNet  # or DuelingMLPQNet if you added it

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser("Train MLP DQN on Cube (one-hot)")
    p.add_argument("--steps", type=int, default=5000000)
    p.add_argument("--scramble", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--alpha", type=float, default=1.0, help="reward scale for Δprogress")
    p.add_argument("--max-steps", type=int, default=200, dest="max_steps")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-decay-steps", type=int, default=200000)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--curriculum-success", type=float, default=0.30)
    p.add_argument("--curriculum-max", type=int, default=20)
    p.add_argument("--exp", type=str, default="mlp_dqn_weekend")
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.models_dir, exist_ok=True)

    # Env + Model
    env = CubeGymCubie(encoder=CubieEncoder(), alpha=args.alpha, max_steps=args.max_steps)
    model = MLPQNet(in_dim=256, hidden=args.hidden)

    # Config wired to trainer’s experiment directory logic
    cfg = DQNConfig(
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        total_steps=args.steps,
        warmup_steps=4 * args.batch_size,
        target_sync_every=2_000,   # ignored if you enabled Polyak in trainer
        train_every=4,             # fewer updates → smoother & faster
        eval_every=args.eval_every,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=args.eps_decay_steps,
        curriculum_success=args.curriculum_success,
        curriculum_max_scramble=args.curriculum_max,
        save_path=os.path.join(args.models_dir, "cube_mlp.pt"),
        output_dir=args.outdir,
        experiment_name=args.exp,
    )

    trainer = DQNTrainer(env, model, cfg)
    trainer.train(start_scramble=args.scramble)

if __name__ == "__main__":
    main()