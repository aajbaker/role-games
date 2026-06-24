import math
import random

from ..agents import DecisionModel
from ..conditions import Condition
from ._ev import _ev_stag, _ev_hare


# ------------------------------------------------------------------
# Entropy helpers
# ------------------------------------------------------------------

def _binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


_LOG2 = math.log(2)


def _normalized_entropy(p: float) -> float:
    return _binary_entropy(p) / _LOG2


# ------------------------------------------------------------------
# FictitiousPlay
# ------------------------------------------------------------------

class FictitiousPlay(DecisionModel):
    """
    Fictitious Play with softmax action selection.

    Belief representation varies by condition:
      Anonymous  — one pooled count over all teammates
      Identity   — one count per teammate agent_id
      Tag-based  — one count per tag label

    Works for any number of players via Poisson-Binomial EV.
    """

    def __init__(
        self,
        condition: Condition,
        own_tag: str | None,
        tau: float,
        n_players: int,
        teammates_info: list[dict],
        discount: float = 1.0,
    ):
        self.condition = condition
        self.own_tag = own_tag
        self.tau = tau
        self.n_players = n_players
        self.discount = discount

        self._teammate_ids: list[int] = [t["agent_id"] for t in teammates_info]
        self._teammate_tags: list[str | None] = [t["tag"] for t in teammates_info]

        self._counts: dict[str, dict[str, int]] = {}
        self._init_counts(teammates_info)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_counts(self, teammates_info: list[dict]) -> None:
        if self.condition == Condition.ANONYMOUS:
            self._counts["pool"] = {"stag": 1, "hare": 1}

        elif self.condition == Condition.IDENTITY:
            for t in teammates_info:
                self._counts[str(t["agent_id"])] = {"stag": 1, "hare": 1}

        else:  # TAG_BASED
            for tag in {t["tag"] for t in teammates_info}:
                self._counts[tag] = {"stag": 1, "hare": 1}

    # ------------------------------------------------------------------
    # Belief access
    # ------------------------------------------------------------------

    def _p(self, key: str) -> float:
        counts = self._counts.get(key, {"stag": 1, "hare": 1})
        return counts["stag"] / (counts["stag"] + counts["hare"])

    def _p_teammates(self) -> list[float]:
        """Return P(stag) belief for each of the N-1 teammates."""
        if self.condition == Condition.ANONYMOUS:
            p = self._p("pool")
            return [p] * (self.n_players - 1)

        elif self.condition == Condition.IDENTITY:
            return [self._p(str(tid)) for tid in self._teammate_ids]

        else:  # TAG_BASED — each teammate mapped to their tag's belief
            return [self._p(tag) for tag in self._teammate_tags]

    # ------------------------------------------------------------------
    # DecisionModel interface
    # ------------------------------------------------------------------

    def decide(self) -> str:
        probs = self._p_teammates()
        ev_s = _ev_stag(probs, self.n_players)
        ev_h = _ev_hare(probs, self.n_players)
        m = max(ev_s, ev_h) / self.tau
        z_s = math.exp(ev_s / self.tau - m)
        z_h = math.exp(ev_h / self.tau - m)
        return "stag" if random.random() < z_s / (z_s + z_h) else "hare"

    def _decay_counts(self) -> None:
        """Multiply every count by the discount factor (no-op when discount == 1.0)."""
        if self.discount == 1.0:
            return
        for key in self._counts:
            self._counts[key]["stag"] *= self.discount
            self._counts[key]["hare"] *= self.discount

    def update(self, observation: dict) -> None:
        """
        observation = {'teammates': [{'agent_id': int, 'tag': str|None, 'action': str}, ...]}
        Discount is applied once per call (= once per round) before processing observations.
        """
        self._decay_counts()
        for tm in observation["teammates"]:
            action = tm["action"]

            if self.condition == Condition.ANONYMOUS:
                self._counts["pool"][action] += 1

            elif self.condition == Condition.IDENTITY:
                key = str(tm["agent_id"])
                if key not in self._counts:
                    self._counts[key] = {"stag": 1, "hare": 1}
                self._counts[key][action] += 1

            else:  # TAG_BASED
                tag = tm["tag"]
                if tag not in self._counts:
                    self._counts[tag] = {"stag": 1, "hare": 1}
                self._counts[tag][action] += 1

        # Tag-based only: also count own action toward own tag's history
        if self.condition == Condition.TAG_BASED and self.own_tag is not None:
            own_action = observation["own_action"]
            if self.own_tag not in self._counts:
                self._counts[self.own_tag] = {"stag": 1, "hare": 1}
            self._counts[self.own_tag][own_action] += 1

    def complexity(self) -> float:
        """Normalized entropy averaged across distinct belief distributions."""
        if self.condition == Condition.ANONYMOUS:
            return _normalized_entropy(self._p("pool"))

        elif self.condition == Condition.IDENTITY:
            entropies = [_normalized_entropy(self._p(str(tid))) for tid in self._teammate_ids]
            return sum(entropies) / len(entropies)

        else:  # TAG_BASED
            distinct_tags = list({t for t in self._teammate_tags if t is not None})
            entropies = [_normalized_entropy(self._p(tag)) for tag in distinct_tags]
            return sum(entropies) / len(entropies)

    def reset(self) -> None:
        """Reset all belief counts to uniform prior (pseudocount of 1 each)."""
        for key in self._counts:
            self._counts[key] = {"stag": 1, "hare": 1}
