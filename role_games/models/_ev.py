"""
Shared expected-value helpers used by both FictitiousPlay and BayesianToM.
Scalar versions are used by FP; vectorized versions by BToM.
"""
import numpy as np


# ── Scalar (used by FictitiousPlay and BayesianToM.decide()) ─────────────────

def _poisson_binomial_pmf(probs: list[float]) -> list[float]:
    """Exact PMF of the sum of independent Bernoulli(p_i) RVs via DP."""
    dp = [0.0] * (len(probs) + 1)
    dp[0] = 1.0
    for p in probs:
        for k in range(len(probs), 0, -1):
            dp[k] = dp[k] * (1 - p) + dp[k - 1] * p
        dp[0] *= (1 - p)
    return dp


def _ev_stag(teammate_probs: list[float], n_players: int) -> float:
    """EV of hunting stag given beliefs about N-1 teammates."""
    pmf = _poisson_binomial_pmf(teammate_probs)
    ev = 0.0
    for k, prob in enumerate(pmf):
        stag_count = 1 + k
        hare_count = n_players - stag_count
        group_total = (4.0 if stag_count >= 2 else 0.0) + hare_count
        ev += prob * group_total / n_players
    return ev


def _ev_hare(teammate_probs: list[float], n_players: int) -> float:
    """EV of hunting hare given beliefs about N-1 teammates."""
    pmf = _poisson_binomial_pmf(teammate_probs)
    ev = 0.0
    for k, prob in enumerate(pmf):
        stag_count = k
        hare_count = n_players - stag_count
        group_total = (4.0 if stag_count >= 2 else 0.0) + hare_count
        ev += prob * group_total / n_players
    return ev


# ── Vectorized (used by BayesianToM for grid inference) ──────────────────────

def _pb_pmf_vec(prob_grids: list[np.ndarray]) -> np.ndarray:
    """
    Vectorized Poisson-Binomial PMF over a batch of probability arrays.

    prob_grids : list of M arrays, each of the same shape S
    Returns    : array of shape (*S, M+1)  where result[..., k] = P(sum == k)
    """
    shape = prob_grids[0].shape
    M = len(prob_grids)
    dp = np.zeros((*shape, M + 1))
    dp[..., 0] = 1.0
    for p in prob_grids:
        for k in range(M, 0, -1):
            dp[..., k] = dp[..., k] * (1 - p) + dp[..., k - 1] * p
        dp[..., 0] *= (1 - p)
    return dp


def _ev_stag_vec(prob_grids: list[np.ndarray], n_players: int) -> np.ndarray:
    """Vectorized EV of hunting stag; returns array of same shape as each prob_grid."""
    pmf = _pb_pmf_vec(prob_grids)          # (*S, M+1)
    ev = np.zeros(prob_grids[0].shape)
    for k in range(len(prob_grids) + 1):
        stag_count = 1 + k
        hare_count = n_players - stag_count
        group_total = (4.0 if stag_count >= 2 else 0.0) + hare_count
        ev += pmf[..., k] * (group_total / n_players)
    return ev


def _ev_hare_vec(prob_grids: list[np.ndarray], n_players: int) -> np.ndarray:
    """Vectorized EV of hunting hare; returns array of same shape as each prob_grid."""
    pmf = _pb_pmf_vec(prob_grids)
    ev = np.zeros(prob_grids[0].shape)
    for k in range(len(prob_grids) + 1):
        stag_count = k
        hare_count = n_players - stag_count
        group_total = (4.0 if stag_count >= 2 else 0.0) + hare_count
        ev += pmf[..., k] * (group_total / n_players)
    return ev


# ── Group decision (tag-based only) ──────────────────────────────────────────

def _payoff_per_player(stag_count: int, n_players: int) -> float:
    hare_count = n_players - stag_count
    return ((4.0 if stag_count >= 2 else 0.0) + hare_count) / n_players


def _ev_group(n_own: int, n_other: int, q, n_players: int):
    """
    EV of my group hunting stag vs hare, assuming each group acts as a block.

    n_own   : size of my tag group, including me
    n_other : size of the other tag group
    q       : P(other group hunts stag); float or ndarray (EV is linear in q)
    Returns (ev_stag, ev_hare) with the same type/shape as q.
    """
    ev_stag = (q * _payoff_per_player(n_own + n_other, n_players)
               + (1 - q) * _payoff_per_player(n_own, n_players))
    ev_hare = (q * _payoff_per_player(n_other, n_players)
               + (1 - q) * _payoff_per_player(0, n_players))
    return ev_stag, ev_hare
