'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
import pandas as pd
from src.cube import Cube
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def simulate_scramble_scores(
    trials: int = 3000,
    min_len: int = 0,
    max_len: int = 100,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Monte Carlo sanity check: sample random scramble lengths, scramble fresh cubes,
    and record the post-scramble score (and solved fraction).

    Notes
    -----
    Right after scrambling, `moves_since_scramble()==0`, so:
        score = solved_fraction / max(1, 0) = solved_fraction
    Thus the plot effectively shows how solved_fraction decays with scramble length.

    Args:
        trials: Number of cubes to simulate.
        min_len: Minimum scramble length (inclusive).
        max_len: Maximum scramble length (inclusive).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns:
            ['trial', 'scramble_len', 'score', 'solved_fraction']
    """
    import random
    rng = random.Random(seed)
    rows: list[dict] = []

    for t in range(trials):
        L = rng.randint(min_len, max_len)
        c = Cube()
        # c.clear_history()
        # Use a per-trial seed for determinism if desired, or None for full random:
        c.scramble(length=L, seed=rng.randint(0, 10**9))
        rows.append(
            dict(
                trial=t,
                scramble_len=L,
                score=float(c.score()),
                solved_fraction=float(c.solved_fraction()),
            )
        )

    return pd.DataFrame(rows)

def plot_scramble_score(df: pd.DataFrame) -> None:
    """
    Scatter plot of score vs scramble length with median trend.

    Args:
        df: Output of `simulate_scramble_scores(...)`.
    """


    # Jitter for visibility at integer x
    x = df["scramble_len"].to_numpy()
    y = df["score"].to_numpy()
    jitter = (np.random.rand(len(x)) - 0.5) * 0

    # Aggregate medians
    med = df.groupby("scramble_len")["score"].median()

    plt.figure(figsize=(8, 5))
    plt.scatter(x + jitter, y, alpha=0.35, s=12)
    plt.plot(med.index, med.values, linewidth=2)
    plt.xlabel("Scramble length (quarter-turns)")
    plt.ylabel("Score after scramble (≈ solved fraction)")
    plt.title("Monte Carlo: Score vs Scramble Length")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

def summarize_scramble_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabular summary (count/mean/median/IQR) by scramble length.

    Args:
        df: Output of `simulate_scramble_scores(...)`.

    Returns:
        Summary DataFrame indexed by scramble_len.
    """
    q1 = df.groupby("scramble_len")["score"].quantile(0.25)
    q3 = df.groupby("scramble_len")["score"].quantile(0.75)
    out = df.groupby("scramble_len")["score"].agg(
        count="size", mean="mean", median="median", std="std"
    )
    out["iqr"] = q3 - q1
    return out
def exp_decay(L, S_inf, k):
    """Exponential decay to steady-state entropy plateau."""
    return S_inf + (1 - S_inf) * np.exp(-k * L)

def fit_decay(df: pd.DataFrame) -> tuple[float, float]:
    """
    Fit exponential decay of score vs scramble length:
        S(L) = S_inf + (1 - S_inf)*exp(-k*L)

    Returns:
        (S_inf, k) best-fit parameters.
    """
    med = df.groupby("scramble_len")["score"].median()
    x = med.index.to_numpy(dtype=float)
    y = med.to_numpy(dtype=float)

    # initial guesses: S_inf≈mean tail, k≈1/10
    p0 = [y[-1], 0.1]
    popt, _ = curve_fit(exp_decay, x, y, p0=p0, bounds=([0, 0], [1, 10]))
    S_inf, k = popt
    print(f"Fitted steady-state S_inf={S_inf:.3f}, decay constant k={k:.3f}")
    return S_inf, k

def plot_decay_fit(df: pd.DataFrame) -> None:
    """
    Plot median score with fitted exponential decay curve.
    """
    med = df.groupby("scramble_len")["score"].median()
    x = med.index.to_numpy(dtype=float)
    y = med.to_numpy(dtype=float)
    S_inf, k = fit_decay(df)
    xfit = np.linspace(x.min(), x.max(), 300)
    yfit = exp_decay(xfit, S_inf, k)

    plt.figure(figsize=(8, 5))
    plt.scatter(df["scramble_len"], df["score"], alpha=0.2, s=10, label="samples")
    plt.plot(x, y, "o-", label="median")
    plt.plot(xfit, yfit, "r-", lw=2.5, label=f"fit: S_inf={S_inf:.3f}, k={k:.3f}")
    plt.xlabel("Scramble length (quarter-turns)")
    plt.ylabel("Score (≈ solved fraction)")
    plt.title("Entropy Decay Fit: Score vs Scramble Length")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_scores = simulate_scramble_scores(trials=300, min_len=0, max_len=100, seed=42)
    print(summarize_scramble_score(df_scores))
    plot_scramble_score(df_scores)
    plot_decay_fit(df_scores)
