'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import os, time, math, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Minimal replay buffer --------------------------------------------------

class Replay:
    """
    Experience replay buffer for off-policy RL agents.

    Stores transitions of the form (state, action, reward, next_state, done)
    up to a fixed capacity and allows random minibatch sampling for training.

    By default, transitions are sampled uniformly. Optional modes:
      - drop_after_sample=True: remove sampled transitions (each used once).
      - prioritized=True: sample according to TD-error priorities (PER-lite).

    Args:
        cap (int): Maximum number of transitions to store.
        prioritized (bool): If True, use TD-error-based prioritized replay.
        alpha (float): How strongly to prioritize high-error samples (0=uniform, 1=full).
        drop_after_sample (bool): If True, sampled items are deleted from buffer.
    """


    def __init__(self, cap: int = 300_000, prioritized: bool = False, alpha: float = 0.6, drop_after_sample: bool = False):
        self.cap = cap
        self.buf: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.pos = 0
        self.prioritized = prioritized
        self.alpha = alpha
        self.drop_after_sample = drop_after_sample  # keep False for PER
        self.priorities = np.zeros((cap,), dtype=np.float32)
        self.eps = 1e-6  # min priority to avoid zeros/NaNs

    def push(self, s, a: int, r: float, ns, d: float, td_error: float | None = None):
        item = (s, a, r, ns, d)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item

        if self.prioritized:
            if td_error is None:
                current_len = max(1, len(self.buf))
                max_prio = self.priorities[:current_len].max()
                if not np.isfinite(max_prio) or max_prio <= 0:
                    max_prio = 1.0
                self.priorities[self.pos] = max_prio
            else:
                p = abs(float(td_error))
                if not np.isfinite(p) or p <= 0:
                    p = 1.0
                self.priorities[self.pos] = p

        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch: int):
        n = len(self.buf)
        assert n > 0, "Cannot sample from empty buffer"

        if self.prioritized:
            raw = self.priorities[:n]
            raw = np.where(np.isfinite(raw) & (raw > 0), raw, self.eps)
            ps = np.power(raw, self.alpha)
            Z = ps.sum()
            if not np.isfinite(Z) or Z <= 0:
                probs = np.full(n, 1.0 / n, dtype=np.float32)
            else:
                probs = ps / Z
            idx = np.random.choice(n, size=batch, p=probs)
            weights = (n * probs[idx]) ** (-1)
            weights /= weights.max()
        else:
            idx = np.random.randint(0, n, size=batch)
            weights = np.ones(batch, dtype=np.float32)

        s, a, r, ns, d = zip(*[self.buf[i] for i in idx])
        out = (np.stack(s),
               np.array(a),
               np.array(r, dtype=np.float32),
               np.stack(ns),
               np.array(d, dtype=np.float32),
               idx,
               weights.astype(np.float32))

        if self.drop_after_sample:
            self._delete_indices(idx)

        return out

    def update_priorities(self, idx, td_errors):
        if not self.prioritized:
            return
        td = np.abs(np.asarray(td_errors, dtype=np.float32))
        td[~np.isfinite(td)] = 1.0
        td = np.maximum(td, self.eps)
        n = len(self.buf)
        idx = np.asarray(idx)
        valid = (idx >= 0) & (idx < n)
        self.priorities[idx[valid]] = td[valid]

    def _delete_indices(self, idx):
        idx = sorted(set(int(i) for i in idx), reverse=True)
        for i in idx:
            if i < len(self.buf):
                self.buf.pop(i)
                # keep priorities length aligned (simple but O(n))
                self.priorities = np.delete(self.priorities, i)

    def __len__(self): return len(self.buf)

# ----- Config ----------------------------------------------------------------
def polyak_update(target: nn.Module, online: nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

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

    eps_decay_steps: int = 150000
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
    output_dir: str = "runs"

# ----- Utilities --------------------------------------------------------------

def linear_epsilon(step: int, cfg: DQNConfig) -> float:
    t = min(1.0, step / cfg.eps_decay_steps)
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * t

def to_device(x: np.ndarray, device, dtype):
    t = torch.from_numpy(x)
    if dtype is torch.long: t = t.long()
    else: t = t.float()
    return t.to(device)

# Determine obs dtype from a single observation (float one-hot vs int index)
def infer_obs_dtype(obs: np.ndarray):
    return torch.long if np.issubdtype(obs.dtype, np.integer) else torch.float32

# ----- Trainer ----------------------------------------------------------------

class DQNTrainer:
    """
    Generic DQN trainer for CubeGymCubie.
    Works with:
      - One-hot encoder (obs: float[256]) + MLPQNet
      - Index encoder (obs: int[40]) + TransformerQNet
    The model must implement forward(x)->Q and .act(x, epsilon, action_mask=None).
    """
    def __init__(self, env, model: nn.Module, cfg: DQNConfig):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        # Target net (frozen periodically)
        self.target = copy.deepcopy(self.model).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        try:
            self.model.compile()
        except Exception:
            print("  ! Model compilation failed; continuing without it.")

        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.replay = Replay(cap=100_000, prioritized=True, alpha=0.6, drop_after_sample=False)

        # metrics/logging
        self._best_sr = -1.0
        self.log_path = getattr(self.cfg, "log_path", "train_log.csv")
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

    def _save_ckpt(self, tag: str, step: Optional[int] = None):
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
        solved = 0
        avg_len = 0.0
        for _ in range(episodes):
            obs = self.env.reset(scramble_len=scramble_len if scramble_len is not None else 0)
            if self.obs_dtype is None:
                self.obs_dtype = infer_obs_dtype(obs)
            steps = 0
            for _ in range(200):
                x = to_device(obs[None, ...], self.device, self.obs_dtype)
                a = self.model.act(x, epsilon=0.0).item()
                obs, r, done, info = self.env.step(a)
                steps += 1
                if done and info.get("score", 0.0) > 1.0:
                    solved += 1
                    break
            avg_len += steps
        return solved / episodes, (avg_len / episodes)

    # --- Training loop ---
    def train(self, start_scramble: int = 4):
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

        while step < self.cfg.total_steps:
            eps = linear_epsilon(step, self.cfg)
            x = to_device(obs[None, ...], self.device, self.obs_dtype)
            a = self.model.act(x, epsilon=eps).item()

            # Interact
            nobs, r, done, info = self.env.step(a)
            self.replay.push(obs, a, r, nobs, float(done))
            obs = nobs
            ep_ret += r
            step += 1

            # Episode end
            if done:
                ep_returns.append(ep_ret)
                obs = self.env.reset(scramble_len=current_scramble)
                ep_ret = 0.0

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

                # TD error & PER-weighted Huber loss
                td_error = y - q_sa  # [B]
                per_sample = F.smooth_l1_loss(q_sa, y, reduction="none")
                loss = (w_t * per_sample).mean()

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
    from src.solvers.cube_gym import CubeGymCubie, CubieEncoder, IndexCubieEncoder, FlatCubieEncoder
    from src.models.mlpq_net import MLPQNet
    from src.models.tiny_transformer import TransformerQNet

    # One-hot + MLP
    env = CubeGymCubie(encoder=CubieEncoder(), alpha=1.0, max_steps=200)
    model = MLPQNet(in_dim=256, hidden=512)          # baseline
    cfg = DQNConfig(total_steps=200_000, save_path="models/cube_mlp.pt")
    DQNTrainer(env, model, cfg).train(start_scramble=4)

    # # Index + Transformer
    # env = CubeGymCubie(encoder=IndexCubieEncoder(), alpha=1.0, max_steps=200)
    # model = TransformerQNet(d_model=128, nhead=8, num_layers=3)
    # cfg = DQNConfig(total_steps=300_000, save_path="models/cube_tr.pt")
    # DQNTrainer(env, model, cfg).train(start_scramble=4)
#