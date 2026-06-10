import random
from collections import deque
from typing import Any

from .agents import Agent
from .conditions import Condition
from .models import FictitiousPlay


# ------------------------------------------------------------------
# Tag assignment
# ------------------------------------------------------------------

def _assign_tags(n_players: int, rng: random.Random) -> list[str | None]:
    """
    Assign tags for the Tag-based condition.
      N=3: 1 red, 2 blue
      N=4: 2 red, 2 blue
      N=5: 2 red, 3 blue
    Assignment is random across agents each run.
    """
    n_red = 1 if n_players == 3 else 2
    tags: list[str] = ["red"] * n_red + ["blue"] * (n_players - n_red)
    rng.shuffle(tags)
    return tags


# ------------------------------------------------------------------
# Agent setup
# ------------------------------------------------------------------

def _make_agents(
    condition: Condition, tau: float, n_players: int, rng: random.Random,
    discount: float = 1.0,
) -> list[Agent]:
    agent_ids = list(range(n_players))

    if condition == Condition.TAG_BASED:
        tags: list[str | None] = _assign_tags(n_players, rng)
    else:
        tags = [None] * n_players

    agents = []
    for i, (aid, tag) in enumerate(zip(agent_ids, tags)):
        teammates_info = [
            {"agent_id": agent_ids[j], "tag": tags[j]}
            for j in range(n_players) if j != i
        ]
        model = FictitiousPlay(condition, tag, tau, n_players, teammates_info,
                               discount=discount)
        agents.append(Agent(aid, tag, model))

    return agents


# ------------------------------------------------------------------
# Payoff
# ------------------------------------------------------------------

def _group_total(actions: list[str]) -> float:
    stag_count = actions.count("stag")
    hare_count = actions.count("hare")
    return (4.0 if stag_count >= 2 else 0.0) + hare_count


# ------------------------------------------------------------------
# Single simulation run
# ------------------------------------------------------------------

def run_simulation(
    condition: Condition,
    n_rounds: int = 50,
    k_convergence: int = 5,
    tau: float = 0.1,
    n_players: int = 3,
    replacement_rate: float = 0.0,
    discount: float = 1.0,
    sim_id: int = 0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run one simulation for n_rounds. Returns per-round, per-agent records.

    convergence — logged per round as a boolean: True if the last k_convergence
                  rounds all had stag_count == 2 (the structural optimum for any N).
    replaced    — True for an agent in the first round after their beliefs were reset.
    discount    — exponential decay factor applied to belief counts each round
                  before the new observation is added (1.0 = standard fictitious play).
    """
    rng = random.Random(seed)
    agents = _make_agents(condition, tau, n_players, rng, discount=discount)

    records: list[dict[str, Any]] = []
    recent_stag_counts: deque[int] = deque(maxlen=k_convergence)
    replaced_this_round: set[int] = set()

    for round_num in range(1, n_rounds + 1):
        # Measure complexity before deciding
        complexities = [a.complexity() for a in agents]

        # Simultaneous decisions
        actions = [a.decide() for a in agents]

        # Payoffs
        stag_count = actions.count("stag")
        total = _group_total(actions)
        per_player = total / n_players

        # Rolling convergence: optimal iff stag_count == 2 for last k rounds
        recent_stag_counts.append(stag_count)
        converged = (
            len(recent_stag_counts) == k_convergence
            and all(sc == 2 for sc in recent_stag_counts)
        )

        # Log
        for agent, action, comp in zip(agents, actions, complexities):
            records.append(
                {
                    "simulation_id": sim_id,
                    "condition": condition.value,
                    "n_players": n_players,
                    "replacement_rate": replacement_rate,
                    "round": round_num,
                    "agent_id": agent.agent_id,
                    "agent_tag": agent.tag,
                    "decision": action,
                    "stag_count": stag_count,
                    "group_total": total,
                    "agent_payoff": per_player,
                    "complexity": comp,
                    "replaced": agent.agent_id in replaced_this_round,
                    "converged": converged,
                }
            )

        # Belief updates
        action_map = {a.agent_id: act for a, act in zip(agents, actions)}
        for agent in agents:
            observation = {
                "teammates": [
                    {
                        "agent_id": other.agent_id,
                        "tag": other.tag,
                        "action": action_map[other.agent_id],
                    }
                    for other in agents if other.agent_id != agent.agent_id
                ]
            }
            agent.update(observation)

        # Replacement for next round (invisible to other agents)
        replaced_this_round = set()
        if replacement_rate > 0.0 and rng.random() < replacement_rate:
            chosen = rng.choice(agents)
            chosen.reset()
            replaced_this_round.add(chosen.agent_id)

    return records


# ------------------------------------------------------------------
# Multiple runs
# ------------------------------------------------------------------

def run_multiple(
    condition: Condition,
    n_simulations: int = 100,
    n_rounds: int = 50,
    k_convergence: int = 5,
    tau: float = 0.1,
    n_players: int = 3,
    replacement_rate: float = 0.0,
    discount: float = 1.0,
    base_seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run n_simulations independent simulations.
    Returns (round_records, summary_records).
    """
    all_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for sim_id in range(n_simulations):
        seed = None if base_seed is None else base_seed + sim_id
        records = run_simulation(
            condition=condition,
            n_rounds=n_rounds,
            k_convergence=k_convergence,
            tau=tau,
            n_players=n_players,
            replacement_rate=replacement_rate,
            discount=discount,
            sim_id=sim_id,
            seed=seed,
        )
        all_records.extend(records)

        # Summary: one row per simulation
        agent0 = [r for r in records if r["agent_id"] == 0]
        total_payoff = sum(r["agent_payoff"] for r in agent0)
        mean_complexity = sum(r["complexity"] for r in records) / len(records)
        summary_records.append(
            {
                "simulation_id": sim_id,
                "condition": condition.value,
                "n_players": n_players,
                "replacement_rate": replacement_rate,
                "total_payoff": total_payoff,
                "mean_complexity": mean_complexity,
            }
        )

    return all_records, summary_records


def run_all_conditions(
    n_simulations: int = 100,
    n_rounds: int = 50,
    k_convergence: int = 5,
    tau: float = 0.1,
    n_players: int = 3,
    replacement_rate: float = 0.0,
    discount: float = 1.0,
    base_seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all three conditions and return combined records."""
    all_round_records: list[dict[str, Any]] = []
    all_summary_records: list[dict[str, Any]] = []

    for condition in Condition:
        rr, sr = run_multiple(
            condition=condition,
            n_simulations=n_simulations,
            n_rounds=n_rounds,
            k_convergence=k_convergence,
            tau=tau,
            n_players=n_players,
            replacement_rate=replacement_rate,
            discount=discount,
            base_seed=base_seed,
        )
        all_round_records.extend(rr)
        all_summary_records.extend(sr)

    return all_round_records, all_summary_records
