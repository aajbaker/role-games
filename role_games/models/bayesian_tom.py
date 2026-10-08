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

decision_rule="group" (Tag-based only): every agent, including the ones i
models, assumes each tag acts as a block and chooses its group's action given
P(other tag hunts stag).  A tag's choice then depends only on its belief about
the other tag, so each tag's posterior is 1D (θ = that tag's belief about the
other tag's stag rate).

include_own_action (Tag-based only): if True, i's own action also updates the
posterior for i's tag.  i holds a posterior only for tags its teammates have,
so an agent alone in its tag never models its own tag.
"""
import math
import random

import numpy as np

from ..agents import DecisionModel
from ..conditions import Condition
from ._ev import _ev_stag, _ev_hare, _ev_stag_vec, _ev_hare_vec, _ev_group


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
        decision_rule: str = "individual",
        rng: random.Random | None = None,
        include_own_action: bool = False,
    ) -> None:
        self.condition     = condition
        self.include_own_action = include_own_action
        self._rng          = rng if rng is not None else random.Random()
        self.own_agent_id  = own_agent_id
        self.own_tag       = own_tag
        self.tau           = tau
        self.n_players     = n_players
        self.discount      = discount
        self.n_grid        = n_grid
        self.group_mode    = (condition == Condition.TAG_BASED
                              and decision_rule == "group")

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
        self._tag_sizes: dict[str | None, int] = {}
        for tag in self._all_tags.values():
            self._tag_sizes[tag] = self._tag_sizes.get(tag, 0) + 1

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
            # Individual: one 2D posterior per tag (dim-0 = θ_red, dim-1 = θ_blue)
            # Group: one 1D posterior per tag (θ about the other tag)
            # Only tags held by a teammate; an agent alone in its tag doesn't model it
            d = 1 if self.group_mode else 2
            teammate_tags = {t["tag"] for t in teammates_info}
            for tag in sorted(t for t in teammate_tags if t is not None):
                self._posteriors[tag] = self._uniform(d)

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

        elif self.group_mode:
            # 1D grid over the key tag's belief about the other tag
            for key_tag in self._posteriors:
                other = self._other_tag(key_tag)
                ev_s, ev_h = _ev_group(self._tag_sizes[key_tag],
                                       self._tag_sizes[other], tv, n)
                self._p_stag_grids[key_tag] = self._softmax_grid(ev_s, ev_h)

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

    def _other_tag(self, tag: str) -> str:
        return next(t for t in self._tag_sizes if t != tag)

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
        if (self.condition == Condition.TAG_BASED and self.include_own_action
                and self.own_tag in self._posteriors):
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
        if self.group_mode:
            other = self._other_tag(self.own_tag)
            q     = float((self._posteriors[other] * self._p_stag_grids[other]).sum())
            ev_s, ev_h = _ev_group(self._tag_sizes[self.own_tag],
                                   self._tag_sizes[other], q, self.n_players)
        else:
            probs = self._expected_teammate_probs()
            ev_s  = _ev_stag(probs, self.n_players)
            ev_h  = _ev_hare(probs, self.n_players)
        m     = max(ev_s, ev_h) / self.tau
        z_s   = math.exp(ev_s / self.tau - m)
        z_h   = math.exp(ev_h / self.tau - m)
        return "stag" if self._rng.random() < z_s / (z_s + z_h) else "hare"

    def complexity(self) -> float:
        """
        Normalized entropy of each posterior, averaged across tracked entities.
        Naturally captures dimensionality: identity N=5 normalises by log(20^4).
        Group mode counts only the other tag's posterior, the only one the
        decision uses.
        """
        if self.group_mode:
            posteriors = [self._posteriors[self._other_tag(self.own_tag)]]
        else:
            posteriors = list(self._posteriors.values())
        entropies = []
        for posterior in posteriors:
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

    def teammate_replaced(self, agent_id: int) -> None:
        """
        Identity only: reset the posterior about the replaced teammate.
        Posteriors about other teammates are left alone: they are symmetric
        across their axes (a teammate's choice depends only on the mix of its
        beliefs), so they hold no belief tied to the replaced agent.
        """
        key = str(agent_id)
        if self.condition == Condition.IDENTITY and key in self._posteriors:
            self._posteriors[key] = self._uniform(self._posteriors[key].ndim)
