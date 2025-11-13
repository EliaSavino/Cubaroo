'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple
import os, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scorer import ScoringOption, Scorer
from src.solvers.replay import Replay
from src.models.actor import build_actor
from src.solvers.solver_utilities import polyak_update, linear_epsilon, to_device, infer_obs_dtype, _maybe_compile
from src.solvers.tree_search_planner import MCTSPlanner



@dataclass
class DQNConfig:
    """
    Configuration for DQN training on the CubeGym.

    Key ideas:
    - We learn Q(s, a) from replayed experience using Double DQN (online selects, target evaluates).
    - Exploration follows an epsilon-greedy schedule from `eps_start` to `eps_end`.
    - We periodically evaluate the greedy policy and (optionally) increase scramble length (curriculum).
    - We snapshot the final model to `save_path`.

    Tuning tips:
    - If rewards are tiny, increase env.alpha (reward scale) rather than lr.
    - If SR wobbles but doesn’t rise, decay epsilon faster and/or increase terminal bonus.
    - If loss is noisy/large, try smaller lr or gradient clip (already on at 1.0).
    """

    gamma: float = 0.95
    # Discount factor for future returns. Higher → plans further ahead.
    # 0.99 is standard; 0.95 can stabilize if rewards are very noisy.

    batch_size: int = 1000
    # Minibatch size sampled from the replay buffer per update step.

    lr: float = 3e-4
    # Adam learning rate for the Q-network.

    total_steps: int = 50000000
    # Total environment interaction steps (collect + learn).

    warmup_steps: int = 2_000
    # Steps to fill replay before the first gradient update. Ensures batches have diversity.

    target_sync_every: int = 2_000
    # How often (in env steps) to hard-copy online network weights into the target network.

    train_every: int = 5
    # Learn every N env steps. Use >1 to do less-frequent, larger updates (not usually needed here).

    eval_every: int = 7000
    # How often to run a greedy evaluation (no exploration) and print SR/avg_len.

    eps_start: float = 1.0
    # Initial epsilon (exploration rate) for epsilon-greedy policy.

    eps_end: float = 0.05
    # Final epsilon after decay. Lower → more greedy late training.

    eps_decay_steps: int = 80000
    # Number of steps over which epsilon linearly decays from eps_start to eps_end.
    # For quicker consolidation on easy scrambles, try 80k–120k.

    curriculum_success: float = 0.4
    # When greedy solve-rate (SR) at current scramble_len ≥ this, bump difficulty.
    # Consider 0.30 for early stages to move 4→6 sooner.

    curriculum_max_scramble: int = 20
    # Upper bound for scramble length during curriculum progression.

    save_path: str = "cube_dqn.pt"
    # File path to save final model weights (best checkpoints optional).
    experiment_name: str = "cube_dqn"
    # Name for this experiment (used for output directory).
    output_dir: str = "runs"
    # Root directory to save experiment runs/logs.
    action_space = 12  # fixed for Rubik's Cube

    use_mcts: bool = True   # whether to use MCTS planning during training
    mcts_every: int = 1    # run MCTS every N steps
    mcts_n_sim: int = 5   # how many carlos to monte
    mcts_max_depth: int = 5 # lookahead depth for MCTS
    mcts_cpuct: float = 1.5 # exploration constant for MCTS
    mcts_use_model: bool = False  # whether to use Q-network for leaf eval in MCTS
    mcts_use_priors: bool = False     # whether to use policy priors in MCTS
    eval_use_mcts: bool = True  # whether to use MCTS during evaluation
    lambda_mcts_value: float = 0.5  # weight for combining MCTS value with Q-value during eval
    beta_mcts_policy: float = 1.0  # temperature for MCTS policy during eval
    eps_mcts_gate: float = 1.0  # only use MCTS when epsilon <= this during training

class DQNTrainer:
    """
    Generic DQN trainer for **CubeGymCubie**.

    Works with two observation encodings:
      1) One-hot encoder → `obs: float[256]` with an MLP-style Q-network.
      2) Index encoder   → `obs: int[40]`   with a Transformer-style Q-network.

    The provided `model` must implement:
      - `forward(x) -> Q` returning Q-values of shape `[B, n_actions]`
      - (optionally) an `.act(x, epsilon, action_mask=None)` method; however the
        trainer constructs a dedicated `actor` via `build_actor(...)` and uses that.

    Parameters
    ----------
    env : CubeEnv
        Environment exposing `reset(scramble_len)` and `step(a)` that returns
        `(next_obs, reward, done, info)`.
    model : nn.Module
        Q-network mapping observations to action-values.
    cfg : DQNConfig
        Configuration object (see `DQNConfig` Protocol for required fields).

    Attributes
    ----------
    env : CubeEnv
    model : nn.Module
    cfg : DQNConfig
    device : torch.device
        `"mps"` if available on Apple Silicon, else `"cpu"`. (Extend as needed.)
    target : nn.Module
        Frozen target network (Polyak-updated by default).
    actor : Actor
        Epsilon-greedy actor built from `model`.
    optim : torch.optim.Optimizer
        Adam optimizer over model parameters.
    replay : Replay
        Prioritized replay buffer (PER-lite) used for training batches.
    exp_dir : str
        Output directory for this run (`<cfg.output_dir>/<cfg.experiment_name>`).
    final_path, best_path, last_path : str
        Checkpoint file paths.
    log_path : str
        CSV training log path under `exp_dir`.
    obs_dtype : Optional[torch.dtype]
        Inferred observation dtype (`torch.long` or `torch.float32`) from first obs.
    last_loss : float
        Last computed training loss (scalar).
    _best_sr : float
        Best solve rate observed during periodic evaluations.
    _td_mean, _td_max : float
        TD error statistics for logging.

    Notes
    -----
    - Target network is maintained via **Polyak averaging** (τ=0.005). You can
      switch to periodic hard copies by enabling the commented block in `train`.
    - The trainer logs a CSV with compact stats and prints periodic summaries.
    - Curriculum learning: when success rate (SR) ≥ `cfg.curriculum_success` and
      `current_scramble < cfg.curriculum_max_scramble`, the scramble length is
      bumped after a few locked evaluation windows.

    Examples
    --------
    >>> trainer = DQNTrainer(env, model, cfg)
    >>> trainer.train(start_scramble=4)
    >>> sr, avg_len = trainer.evaluate(episodes=100, scramble_len=6)
    """
    def __init__(self, env, model: nn.Module, cfg: DQNConfig):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)



        try:
            self.model = _maybe_compile(self.model, self.device)
        except Exception:
            print("  ! Model compilation failed; continuing without it.")

        # Target net (frozen periodically)
        self.target = copy.deepcopy(self.model).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.actor = build_actor(
            model=self.model,
            device=self.device,
            n_actions=self.cfg.action_space,
            prefer_model_act=True,
        )
        
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.replay = Replay(cap=100_000, prioritized=True, alpha=0.6, drop_after_sample=False)

        # metrics/logging
        self._best_sr = -1.0
        self.log_path = getattr(self.cfg, "log_path", f"train_log{self.cfg.experiment_name}.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "scramble", "epsilon", "sr", "avg_len", "avg_return", "loss", "replay_fill", "td_mean",
                     "td_max"]
                )
        self._td_mean = 0.0
        self._td_max = 0.0
        self.obs_dtype: Optional[torch.dtype] = None
        self.last_loss = 0.0
        exp_name = getattr(self.cfg, "experiment_name", None)
        if not exp_name:
            exp_name = time.strftime("exp_%Y%m%d-%H%M%S")
        out_root = getattr(self.cfg, "output_dir", "runs")

        self.exp_dir = os.path.join(out_root, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # save paths
        self.final_path = os.path.join(self.exp_dir, "final.pt")
        self.best_path = os.path.join(self.exp_dir, "best.pt")
        self.last_path = os.path.join(self.exp_dir, "last.pt")

        # CSV log lives in the exp dir
        self.log_path = os.path.join(self.exp_dir, "train_log.csv")

        # init CSV if new
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "scramble", "epsilon", "sr", "avg_len", "avg_return",
                     "loss", "replay_fill", "td_mean", "td_max"]
                )

        # best SR tracker
        self._best_sr = -1.0
        self._td_mean = 0.0
        self._td_max = 0.0

        self._planner = None
        if getattr(self.cfg, "use_mcts", False):
            q_fn = self._q_fn_from_model if getattr(self.cfg, "mcts_use_model", True) else None
            policy_prior = self._policy_prior_from_model if getattr(self.cfg, "mcts_use_priors", True) else None
            self._planner = MCTSPlanner(
                q_fn=q_fn,
                policy_prior=policy_prior,
                gamma=self.cfg.gamma,
                cpuct=getattr(self.cfg, "mcts_cpuct", 1.5),
                n_sim=getattr(self.cfg, "mcts_n_sim", 400),
                max_depth=getattr(self.cfg, "mcts_max_depth", 5),
            )
            self._mcts_value: Optional[dict[int, float]] = {}  # cache for eval
            self._mcts_policy: Optional[dict[int, np.ndarray]] = {}  # cache for eval

    def _encode_cube(self):
        # Single helper to get a model-ready tensor from the current cube.
        obs = self.env.encoder.encode(self.env.cube)
        if self.obs_dtype is None:
            self.obs_dtype = infer_obs_dtype(obs)
        return to_device(obs[None, ...], self.device, self.obs_dtype)

    def _q_fn_from_model(self, cube) -> np.ndarray:
        """
        Leaf evaluator: returns Q(s,·) for the given cube using the current model.
        Used by MCTS as V_leaf = max_a Q(s,a).
        """
        # Encode *that* cube, not env.cube
        obs = self.env.encoder.encode(cube)
        if self.obs_dtype is None:
            self.obs_dtype = infer_obs_dtype(obs)
        x = to_device(obs[None, ...], self.device, self.obs_dtype)  # [1, d]
        with torch.no_grad():
            q = self.model(x).squeeze(0).detach().cpu().numpy()     # [12]
        return q

    def _policy_prior_from_model(self, cube) -> np.ndarray:
        """
        Optional priors for PUCT: softmax over Q(s,·) from the current model.
        """
        q = self._q_fn_from_model(cube)
        q = q - q.max()
        p = np.exp(q)
        s = p.sum()
        return (p / s) if s > 0 else np.ones_like(p) / len(p)

    def _save_ckpt(self, tag: str, step: Optional[int] = None):
        """
        Save a model checkpoint.

        Parameters
        ----------
        tag : str
            One of {"best", "last"} or a custom tag. For custom tags, a file
            name is created as `"{tag}.pt"` or `"{tag}_{step}.pt"` if `step` is provided.
        step : int, optional
            Optional step index to append to filename for custom tags.

        Returns
        -------
        None
        """
        if tag == "best":
            path = self.best_path
        elif tag == "last":
            path = self.last_path
        else:
            fname = f"{tag}.pt" if step is None else f"{tag}_{step}.pt"
            path = os.path.join(self.exp_dir, fname)
        torch.save(self.model.state_dict(), path)
        print(f"saved {tag} → {path}")

    # simple one-episode eval (greedy)
    @torch.no_grad()
    def evaluate(self, episodes: int = 70, scramble_len: Optional[int] = None) -> Tuple[float, float]:
        """
        Run greedy evaluations (ε=0) and report solve rate and average length.

        Parameters
        ----------
        episodes : int, default=70
            Number of evaluation episodes.
        scramble_len : int or None, optional
            Scramble length to pass to `env.reset`. If None, uses 0.

        Returns
        -------
        (solve_rate, avg_length) : tuple[float, float]
            Solve rate in [0, 1] and average number of steps (over all episodes).

        Notes
        -----
        - A run is counted as solved when the environment signals `done` and
          `info.get("score", 0.0) > 1.0` (matching your environment’s convention).
        - Each episode is capped at 200 steps.
        """
        solved = 0
        avg_len = 0.0
        use_plan = (self._planner is not None and self.cfg.use_mcts and self.cfg.eval_use_mcts)
        for _ in range(episodes):
            obs = self.env.reset(scramble_len=scramble_len if scramble_len is not None else 0)
            if self.obs_dtype is None:
                self.obs_dtype = infer_obs_dtype(obs)
            steps = 0
            last_action_idx: Optional[int] = None  # per-episode

            for _ in range(200):
                # build mask
                action_mask = None
                if last_action_idx is not None:
                    inv = inverse_action_idx(last_action_idx)
                    mask = torch.ones(1, self.cfg.action_space, dtype=torch.bool, device=self.device)
                    mask[0, inv] = False
                    action_mask = mask

                if use_plan:
                    a = self._planner.choose_action(self.env,
                                                    prev_action_idx=last_action_idx)
                else:
                    x = to_device(obs[None, ...], self.device, self.obs_dtype)
                    a = self.actor.act(x, epsilon=0.0, action_mask=action_mask).item()

                obs, r, done, info = self.env.step(a)
                last_action_idx = a
                steps += 1
                if done and info.get("score", 0.0) > 1.0:
                    solved += 1
                    break
            avg_len += steps
        return solved / episodes, (avg_len / episodes)

    # --- Training loop ---
    def train(self, start_scramble: int = 4):
        """
        Train the DQN agent with prioritized replay and Double-DQN targets.

        The loop performs:
        1) Environment interaction (ε-greedy with linear schedule).
        2) Replay storage (`Replay.push`).
        3) Mini-batch updates every `cfg.train_every` steps (after warmup).
           - Importance-weighted Huber loss using PER weights.
           - Double-DQN target: online argmax, target value.
        4) Target soft updates (Polyak).
        5) Periodic evaluation & CSV logging.
        6) Simple curriculum increases in scramble length.

        Parameters
        ----------
        start_scramble : int, default=4
            Initial scramble length used for `env.reset`.

        Returns
        -------
        None
        """
        obs = self.env.reset(scramble_len=start_scramble)
        self.obs_dtype = infer_obs_dtype(obs)

        step = 0
        ep_returns = []
        ep_ret = 0.0
        t0 = time.time()
        last_eval_at = -self.cfg.eval_every
        last_log_at = 0
        current_scramble = start_scramble
        beta0 = 0.4
        beta1 = 1.0

        def per_beta(step):
            t = min(1.0, step / self.cfg.total_steps)
            return beta0 + (beta1 - beta0) * t

        lock_eval_windows = 5  # min evals between curriculum bumps
        since_bump = 0

        print(f"Starting training on device={self.device} | steps={self.cfg.total_steps:,} | scramble={current_scramble}")
        print("-" * 90)
        last_action_idx: Optional[int] = None
        while step < self.cfg.total_steps:
            eps = linear_epsilon(step, self.cfg)
            x = to_device(obs[None, ...], self.device, self.obs_dtype)
            # Build mask that forbids the inverse of the last real action
            action_mask = None
            if last_action_idx is not None:
                inv = inverse_action_idx(last_action_idx)
                mask = torch.ones(1, self.cfg.action_space, dtype=torch.bool, device=self.device)
                mask[0, inv] = False  # disallow inverse move only
                action_mask = mask

            use_planner_now = (
                    self._planner is not None
                    and self.cfg.use_mcts
                    and (step % self.cfg.mcts_every == 0)
                    and eps <= self.cfg.eps_mcts_gate
            )

            if use_planner_now:
                # pass previous action into planning so tree also avoids immediate undo at root
                a, q_mcts, pi_mcts = self._planner.plan_root(self.env,
                                                             prev_action_idx=last_action_idx)
            else:
                a = self.actor.act(x, epsilon=eps, action_mask=action_mask).item()
                q_mcts, pi_mcts = None, None

            # Interact
            nobs, r, done, info = self.env.step(a)
            last_action_idx = a
            idx_push = self.replay.push(obs, a, r, nobs, float(done))
            if q_mcts is not None and pi_mcts is not None:
                self._mcts_value[idx_push] = float(q_mcts[a])
                self._mcts_policy[idx_push] = pi_mcts

            obs = nobs
            ep_ret += r
            step += 1

            # Episode end
            if done:
                ep_returns.append(ep_ret)
                obs = self.env.reset(scramble_len=current_scramble)
                ep_ret = 0.0
                last_action_idx = None

            # Learn
            if len(self.replay) >= self.cfg.warmup_steps and (step % self.cfg.train_every == 0):
                # --- sample with indices + IS weights ---
                s, a_b, r_b, ns, d_b, idx, w_b = self.replay.sample(self.cfg.batch_size)

                s_t = to_device(s, self.device, self.obs_dtype)
                ns_t = to_device(ns, self.device, self.obs_dtype)
                a_t = torch.from_numpy(a_b).long().to(self.device)
                r_t = torch.from_numpy(r_b).to(self.device)
                d_t = torch.from_numpy(d_b).to(self.device)



                # anneal beta
                beta = per_beta(step)
                w_t = torch.from_numpy(w_b).to(self.device) ** beta  # [B]

                # Q(s,a)
                q_sa = self.model(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

                # Double-DQN target
                with torch.no_grad():
                    next_best = self.model(ns_t).argmax(dim=1)
                    q_next = self.target(ns_t).gather(1, next_best.unsqueeze(1)).squeeze(1)
                    y = r_t + self.cfg.gamma * (1.0 - d_t) * q_next
                if self.cfg.lambda_mcts_value > 0.0:
                    y_mcts_list = [self._mcts_value.get(i, float('nan')) for i in idx]
                    y_mcts = torch.tensor(y_mcts_list, dtype=y.dtype, device=y.device)

                    m = torch.isfinite(y_mcts)
                    if m.any():
                        y = torch.where(
                            m,
                            (1.0 - self.cfg.lambda_mcts_value) * y + self.cfg.lambda_mcts_value * y_mcts,y)

                aux_loss = 0.0
                if self.cfg.beta_mcts_policy > 0.0:
                    A = self.cfg.action_space
                    pi_targets = torch.full((len(idx), A), float('nan'), device=self.device, dtype=torch.float32)
                    any_pi = False
                    for row, irep in enumerate(idx):
                        pi_mcts = self._mcts_policy.get(irep, None)
                        if pi_mcts is not None:
                            pi_targets[row, :] = torch.from_numpy(pi_mcts.astype(np.float32)).to(self.device)
                            any_pi = True

                    if any_pi:
                        q_all = self.model(s_t)  # [B, A]
                        logp = torch.log_softmax(q_all, dim=1)  # [B, A]
                        m = torch.isfinite(pi_targets).all(dim=1)
                        ce = -(pi_targets[m]*logp[m]).sum(dim=1).mean()
                        aux_loss = self.cfg.beta_mcts_policy * ce# [M]


                # TD error & PER-weighted Huber loss
                td_error = y - q_sa  # [B]
                per_sample = F.smooth_l1_loss(q_sa, y, reduction="none")
                loss = (w_t * per_sample).mean() + aux_loss

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                self.last_loss = loss.item()

                # priorities: |TD error|
                with torch.no_grad():
                    prios = td_error.detach().abs().clamp_(min=1e-6).cpu().numpy()
                self.replay.update_priorities(idx, prios)

                # ---- target update: choose ONE of these ----
                # (A) SOFT/POLYAK (recommended for smoother learning)
                polyak_update(self.target, self.model, tau=0.005)
                # (B) or keep your periodic hard copy (then comment Polyak above)
                # if step % self.cfg.target_sync_every == 0:
                #     self.target.load_state_dict(self.model.state_dict())

                # td stats for CSV logs
                self._td_mean = float(td_error.detach().abs().mean().item())
                self._td_max = float(td_error.detach().abs().max().item())

            # --- lightweight console log every 1000 steps ---
            if step - last_log_at >= 1000:
                last_log_at = step
                avg_ret = np.mean(ep_returns[-20:]) if len(ep_returns) > 0 else 0.0
                replay_fill = len(self.replay) / self.replay.cap * 100
                print(f"step {step:>7d} | eps={eps:5.3f} | "
                      f"loss={self.last_loss:8.3e} | "
                      f"td|mean={self._td_mean:7.2e} max={self._td_max:7.2e} | "
                      f"replay={replay_fill:6.2f}% | "
                      f"cur_scr={current_scramble:2d} | "
                      f"avg_ret={avg_ret:7.3e}", flush=True)

            # --- periodic evaluation + curriculum ---
            if step - last_eval_at >= self.cfg.eval_every or step == self.cfg.total_steps:
                last_eval_at = step
                sr, avg_len = self.evaluate(scramble_len=current_scramble)
                avg_ret = np.mean(ep_returns[-100:]) if ep_returns else 0.0
                dt = (time.time() - t0) / 60
                print("=" * 90)
                print(f"[{step:>8}] SCR={current_scramble:2d} | eps={eps:5.3f} | "
                      f"SR={sr * 100:5.1f}% | avg_len={avg_len:5.1f} | "
                      f"avg_return={avg_ret:7.3f} | "
                      f"loss={self.last_loss:8.5f} | t={dt:5.1f} min")
                print("=" * 90)

                # CSV log
                replay_fill = len(self.replay) / self.replay.cap
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, current_scramble, eps, sr, avg_len, avg_ret,
                                            self.last_loss, replay_fill, self._td_mean, self._td_max])

                # Checkpoint if best SR
                if sr > self._best_sr:
                    self._best_sr = sr
                    self._save_ckpt("best")

                # curriculum with lock
                since_bump += 1
                if sr >= self.cfg.curriculum_success and current_scramble < self.cfg.curriculum_max_scramble and since_bump >= lock_eval_windows:
                    current_scramble += 2
                    since_bump = 0
                    print(f"  ↪ Curriculum bump: scramble → {current_scramble}")

        # --- save model at the end ---
        os.makedirs(self.exp_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.final_path)
        total_time = (time.time() - t0) / 60
        print(f"\n Training finished in {total_time:.1f} min — final model saved to {self.final_path}")

# ----- Example usage ----------------------------------------------------------
if __name__ == "__main__":
    from src.solvers.cube_gym import CubeGymCubie, inverse_action_idx
    from src.solvers.encoders import CubieEncoder
    from src.models.mlpq_net import MLPQNet

    # One-hot + MLP
    env = CubeGymCubie(encoder=CubieEncoder(), alpha=1.0, max_steps=50, scorer=Scorer(ScoringOption.WEIGHTED_SLOT_AND_ORI))
    model = MLPQNet(in_dim=256, hidden=512)          # baseline
    cfg = DQNConfig(total_steps=1000_000, save_path="models/cube_mlp_actor.pt",
                    experiment_name="cube_dqn_mlp_testing", output_dir="runs")
    DQNTrainer(env, model, cfg).train(start_scramble=4)

    # # Index + Transformer
    # env = CubeGymCubie(encoder=IndexCubieEncoder(), alpha=1.0, max_steps=200)
    # model = TransformerQNet(d_model=128, nhead=8, num_layers=3)
    # cfg = DQNConfig(total_steps=300_000, save_path="models/cube_tr.pt", output_dir="runs", experiment_name="cube_dqn_transformer")
    # DQNTrainer(env, model, cfg).train(start_scramble=4)
#