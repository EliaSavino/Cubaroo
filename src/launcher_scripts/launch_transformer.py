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
from src.scorer import Scorer, ScoringOption
from src.solvers.manager import DQNTrainer, DQNConfig
from src.solvers.cube_gym import CubeGymCubie  # index encoder for Transformer
from src.solvers.encoders import IndexCubieEncoder
from src.models.tiny_transformer import TransformerQNet

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser("Train Transformer DQN on Cube (index tokens)")
    p.add_argument("--steps", type=int, default=3000000)
    p.add_argument("--scramble", type=int, default=1)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0, help="reward scale for Δprogress")
    p.add_argument("--max-steps", type=int, default=200, dest="max_steps")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-hold-steps", type=int, default=5000)
    p.add_argument("--eps-decay-steps", type=int, default=250000)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--curriculum-success", type=float, default=0.55)
    p.add_argument("--curriculum-easy-success", type=float, default=0.95)
    p.add_argument("--curriculum-window", type=int, default=3)
    p.add_argument("--curriculum-max", type=int, default=20)
    p.add_argument("--scorer", choices=["consistent", "lookahead"], default="lookahead")
    p.add_argument("--use-mcts", action="store_true")
    p.add_argument("--exp", type=str, default="transformer_dqn_weekend")
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.models_dir, exist_ok=True)

    scorer_option = (
        ScoringOption.LOOKAHEAD_PROGRESS
        if args.scorer == "lookahead"
        else ScoringOption.CONSISTENT_PROGRESS
    )

    # Env + Model
    env = CubeGymCubie(
        encoder=IndexCubieEncoder(),
        alpha=args.alpha,
        max_steps=args.max_steps,
        scorer=Scorer(option=scorer_option),
    )
    model = TransformerQNet(d_model=args.d_model, nhead=args.nhead, num_layers=args.layers)

    cfg = DQNConfig(
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        total_steps=args.steps,
        warmup_steps=4 * args.batch_size,
        target_sync_every=2_000,
        train_every=4,
        eval_every=args.eval_every,
        eps_start=1.0,
        eps_end=0.05,
        eps_hold_steps=args.eps_hold_steps,
        eps_decay_steps=args.eps_decay_steps,
        eps_schedule="cosine",
        curriculum_success=args.curriculum_success,
        curriculum_easy_success=args.curriculum_easy_success,
        curriculum_eval_window=args.curriculum_window,
        curriculum_max_scramble=args.curriculum_max,
        use_mcts=args.use_mcts,
        eval_use_mcts=args.use_mcts,
        save_path=os.path.join(args.models_dir, "cube_tr.pt"),
        output_dir=args.outdir,
        experiment_name=args.exp,
    )

    trainer = DQNTrainer(env, model, cfg)
    trainer.train(start_scramble=args.scramble)

if __name__ == "__main__":
    main()
