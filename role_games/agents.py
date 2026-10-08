from abc import ABC, abstractmethod


class DecisionModel(ABC):
    @abstractmethod
    def update(self, observation: dict) -> None:
        """Update internal beliefs given what happened last round."""

    @abstractmethod
    def decide(self) -> str:
        """Return 'stag' or 'hare'."""

    @abstractmethod
    def complexity(self) -> float:
        """Return normalized [0,1] entropy-based uncertainty before decision."""

    @abstractmethod
    def reset(self) -> None:
        """Reset beliefs to initial uniform prior (called on replacement)."""

    def teammate_replaced(self, agent_id: int) -> None:
        """Called on each remaining agent when a teammate is replaced. Default: no-op."""


class Agent:
    def __init__(self, agent_id: int, tag: str | None, model: DecisionModel):
        self.agent_id = agent_id
        self.tag = tag
        self.model = model

    def decide(self) -> str:
        return self.model.decide()

    def update(self, observation: dict) -> None:
        self.model.update(observation)

    def complexity(self) -> float:
        return self.model.complexity()

    def reset(self) -> None:
        self.model.reset()

    def teammate_replaced(self, agent_id: int) -> None:
        self.model.teammate_replaced(agent_id)
