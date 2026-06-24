"""
Bayesian Theory of Mind (BToM) agent model — depth-1.

Agent i models each other agent j as a utility-maximizer with their own
beliefs θ about the group.  When j acts, i does Bayesian inference over
what θ j must hold; i then forward-simulates j from that posterior to
predict j's next action.

Belief vector θ dimensionality by condition:
  Anonymous  : 1D  (θ ∈ [0,1]  — pool stag rate)
  Tag-based  : 2D  ((θ_red, θ_blue) ∈ [0,1]²)
  Identity   : (n-1)D  per tracked agent  → (n-1)² total dimensions
"""
import math
import random

import numpy as np

from ..agents import DecisionModel
from ..conditions import Condition
from ._ev import _ev_stag, _ev_hare, _ev_stag_vec, _ev_hare_vec


class BayesianToM(DecisionModel):

    def __init__(
        self,
        condition: Condition,
        own_agent_id: int,
        own_tag: str | None,
        tau: float,
        n_players: int,
        teammates_info: list[dict],   # [{"agent_id": int, "tag": str | None}]
        discount: float = 1.0,
        n_grid: int = 10,
    ) -> None:
        self.condition     = condition
        self.own_agent_id  = own_agent_id
        self.own_tag       = own_tag
        self.tau           = tau
        self.n_players     = n_players
        self.discount      = discount
        self.n_grid        = n_grid

        self._theta_vals = np.linspace(
            1 / (2 * n_grid), 1 - 1 / (2 * n_grid), n_grid
        )
        self._teammate_ids:  list[int]            = [t["agent_id"] for t in teammates_info]
        self._teammate_tags: dict[int, str | None] = {t["agent_id"]: t["tag"]
                                                       for t in teammates_info}

        # Complete picture of who is in the game and their tags
        self._all_ids: list[int] = sorted(
            [own_agent_id] + self._teammate_ids
        )
        self._all_tags: dict[int, str | None] = {own_agent_id: own_tag}
        for t in teammates_info:
            self._all_tags[t["agent_id"]] = t["tag"]

        # Posteriors and their associated P(stag | θ) grids
        self._posteriors:    dict[str, np.ndarray] = {}
        self._p_stag_grids:  dict[str, np.ndarray] = {}

        self._init_posteriors(teammates_info)
        self._precompute_p_stag_grids()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _uniform(self, d: int) -> np.ndarray:
        shape = (self.n_grid,) * d
        return np.ones(shape) / float(self.n_grid ** d)

    def _init_posteriors(self, teammates_info: list[dict]) -> None:
        if self.condition == Condition.ANONYMOUS:
            self._posteriors["pool"] = self._uniform(1)

        elif self.condition == Condition.TAG_BASED:
            # One 2D posterior per tag: dim-0 = θ_red, dim-1 = θ_blue
            all_tags = {self.own_tag} | {t["tag"] for t in teammates_info}
            for tag in sorted(t for t in all_tags if t is not None):
                self._posteriors[tag] = self._uniform(2)

        else:  # IDENTITY
            d = len(teammates_info)   # = n_players - 1
            for t in teammates_info:
                self._posteriors[str(t["agent_id"])] = self._uniform(d)

    # ── Precompute P(stag | θ) grids ─────────────────────────────────────────
    #
    # These are computed ONCE at construction and reused every round for both
    # the Bayesian update (likelihood) and the decision step (prediction).

    def _softmax_grid(self, ev_s: np.ndarray, ev_h: np.ndarray) -> np.ndarray:
        m = np.maximum(ev_s, ev_h) / self.tau
        exp_s = np.exp(ev_s / self.tau - m)
        exp_h = np.exp(ev_h / self.tau - m)
        return exp_s / (exp_s + exp_h)

    def _precompute_p_stag_grids(self) -> None:
        tv = self._theta_vals
        n  = self.n_players

        if self.condition == Condition.ANONYMOUS:
            # 1D grid; all agents have identical teammate compositions
            ev_s = np.array([_ev_stag([t] * (n - 1), n) for t in tv])
            ev_h = np.array([_ev_hare([t] * (n - 1), n) for t in tv])
            self._p_stag_grids["pool"] = self._softmax_grid(ev_s, ev_h)

        elif self.condition == Condition.TAG_BASED:
            # 2D grid shared by all agents of the same tag
            n_red  = sum(1 for v in self._all_tags.values() if v == "red")
            n_blue = sum(1 for v in self._all_tags.values() if v == "blue")
            TR, TB = np.meshgrid(tv, tv, indexing="ij")  # (n_grid, n_grid) each

            for key_tag in self._posteriors:
                # Teammate tags seen by an agent OF key_tag
                if key_tag == "red":
                    tm_tags = ["red"] * (n_red - 1) + ["blue"] * n_blue
                else:
                    tm_tags = ["red"] * n_red + ["blue"] * (n_blue - 1)

                prob_grids = [TR if t == "red" else TB for t in tm_tags]
                ev_s = _ev_stag_vec(prob_grids, n)
                ev_h = _ev_hare_vec(prob_grids, n)
                self._p_stag_grids[key_tag] = self._softmax_grid(ev_s, ev_h)

        else:  # IDENTITY
            # (n-1)D grid; EV is symmetric over teammate positions so one grid
            # serves all j's posteriors.
            d = n - 1
            grids = list(np.meshgrid(*[tv] * d, indexing="ij"))
            ev_s = _ev_stag_vec(grids, n)
            ev_h = _ev_hare_vec(grids, n)
            p_stag_id = self._softmax_grid(ev_s, ev_h)
            for key in self._posteriors:
                self._p_stag_grids[key] = p_stag_id   # shared reference

    # ── Bayesian update helpers ───────────────────────────────────────────────

    def _decay_posteriors(self) -> None:
        if self.discount == 1.0:
            return
        log_discount = math.log(self.discount)
        for key in self._posteriors:
            posterior = self._posteriors[key]
            n_pts     = posterior.size
            log_p     = np.log(posterior + 1e-300)
            log_u     = -math.log(n_pts)            # log(1 / n_pts)
            log_dec   = self.discount * log_p + (1 - self.discount) * log_u
            p         = np.exp(log_dec - log_dec.max())
            self._posteriors[key] = p / p.sum()

    def _update_posterior(self, key: str, action: str) -> None:
        p_stag     = self._p_stag_grids[key]
        likelihood = p_stag if action == "stag" else (1.0 - p_stag)
        posterior  = self._posteriors[key] * likelihood
        total      = posterior.sum()
        if total > 0:
            self._posteriors[key] = posterior / total
        else:
            # Numerical underflow — reset to uniform
            self._posteriors[key] = self._uniform(len(posterior.shape))

    # ── DecisionModel interface ───────────────────────────────────────────────

    def update(self, observation: dict) -> None:
        self._decay_posteriors()

        for tm in observation["teammates"]:
            action = tm["action"]

            if self.condition == Condition.ANONYMOUS:
                self._update_posterior("pool", action)

            elif self.condition == Condition.IDENTITY:
                key = str(tm["agent_id"])
                if key in self._posteriors:
                    self._update_posterior(key, action)

            else:  # TAG_BASED
                tag = tm["tag"]
                if tag in self._posteriors:
                    self._update_posterior(tag, action)

        # Tag-based only: treat own action as evidence about own-tag agents
        if self.condition == Condition.TAG_BASED and self.own_tag in self._posteriors:
            self._update_posterior(self.own_tag, observation["own_action"])

    def _expected_teammate_probs(self) -> list[float]:
        """E[P(j staggers)] for each teammate j, via full posterior integration."""
        if self.condition == Condition.ANONYMOUS:
            e = float((self._posteriors["pool"] * self._p_stag_grids["pool"]).sum())
            return [e] * (self.n_players - 1)

        elif self.condition == Condition.TAG_BASED:
            result = []
            for tid in self._teammate_ids:
                tag = self._teammate_tags[tid]
                e   = float((self._posteriors[tag] * self._p_stag_grids[tag]).sum())
                result.append(e)
            return result

        else:  # IDENTITY
            result = []
            for tid in self._teammate_ids:
                key = str(tid)
                e   = float((self._posteriors[key] * self._p_stag_grids[key]).sum())
                result.append(e)
            return result

    def decide(self) -> str:
        probs = self._expected_teammate_probs()
        ev_s  = _ev_stag(probs, self.n_players)
        ev_h  = _ev_hare(probs, self.n_players)
        m     = max(ev_s, ev_h) / self.tau
        z_s   = math.exp(ev_s / self.tau - m)
        z_h   = math.exp(ev_h / self.tau - m)
        return "stag" if random.random() < z_s / (z_s + z_h) else "hare"

    def complexity(self) -> float:
        """
        Normalized entropy of each posterior, averaged across tracked entities.
        Naturally captures dimensionality: identity N=5 normalises by log(20^4).
        """
        entropies = []
        for posterior in self._posteriors.values():
            p     = posterior.flatten()
            p     = p[p > 0]
            h     = float(-np.sum(p * np.log(p)))
            h_max = math.log(posterior.size)
            entropies.append(h / h_max if h_max > 0 else 0.0)
        return float(np.mean(entropies))

    def reset(self) -> None:
        for key in self._posteriors:
            d = len(self._posteriors[key].shape)
            self._posteriors[key] = self._uniform(d)
