# Role Games: Agent-Based Simulation — Model Specification

**Project:** Role Games  
**Author:** Aaron Baker  
**Status:** Active  
**Last Updated:** May 2026

---

## 1. Overview

This simulation models how small groups of agents solve an asymmetric coordination problem (a modified 3-player stag hunt) under three conditions that vary the informational markers available to agents. The primary goals are:

1. Establish a normative behavioral baseline for coordination under each condition
2. Measure convergence speed and group payoff as primary outcomes
3. Estimate cognitive complexity of individual decision-making per condition
4. Produce data comparable to human behavioral experiments

The model is explicitly a **computational-level model** in Marr's (1982) sense — it characterizes *what* agents are computing, not *how* the brain implements it. Algorithmic and implementational interpretations are left for future work.

---

## 2. Game Structure

### 2.1 The Modified Stag Hunt

Three players interact over multiple rounds. Each round, every player simultaneously and independently chooses one of two actions:

- **Hunt the stag** — requires 2 of 3 players to succeed; yields 4 points split evenly among all three players (≈1.33 per player)
- **Hunt a hare** — requires only 1 player; yields 1 point to the group, split evenly (≈0.33 per player)

There are 3 hares and 1 stag available each round. Payoffs are always split evenly among all three players regardless of individual choices.

### 2.2 Payoff Matrix

The social optimum is **2 stag, 1 hare = 5 points total**, requiring role differentiation rather than symmetric coordination. This is intentional — the paradigm is specifically designed to require complementary coordination, which is where role-based tags add the most theoretical value.

| Player 1 | Player 2 | Player 3 | Group Total | Per Player |
|----------|----------|----------|-------------|------------|
| Stag | Stag | Stag | 4 | 1.33 |
| Stag | Stag | Hare | 5 | 1.67 |
| Stag | Hare | Hare | 2 | 0.67 |
| Hare | Hare | Hare | 3 | 1.00 |

### 2.3 Round Structure

1. All agents simultaneously compute their decision
2. Choices are revealed simultaneously
3. Payoffs are calculated and distributed
4. Agents update their beliefs based on observed outcomes
5. Complexity is logged for each agent

### 2.4 Termination

Simulations always run for exactly **N rounds** (user-specified parameter). There is no early termination. Convergence is tracked as a logged outcome: a simulation is marked as converged if the group achieves the optimal outcome (5 points) for K consecutive rounds at any point during the run (K is a parameter, default: 5). The round at which convergence is first reached is recorded in the summary data.

---

## 3. Experimental Conditions

The condition is passed as a **runtime parameter** and controls what information agents have about their teammates. Tags are treated as immediately salient and prescriptive — agents use them from round 1 to organize beliefs and guide decisions. This reflects the theoretical claim that tags function as norms, not learned cues.

### 3.1 Anonymous
- All avatars are identical
- Agents cannot distinguish between teammates
- Belief representation: one shared distribution over actions, pooled across all teammates

### 3.2 Identity
- Each avatar has a unique appearance
- Agents can track individuals across rounds
- Belief representation: separate belief distribution per individual teammate

### 3.3 Tag-Based
- Avatars share features that denote group membership (e.g., hat color)
- In the 3-player game: 1 player has one tag (e.g., red hat), 2 players share another tag (e.g., blue hat)
- Agents maintain beliefs at the **tag level** — same-tag players are treated as exchangeable
- Tags are treated as **prescriptive**: agents use tag membership to determine what they should do, not just to track what others have done
- Belief representation: one distribution per tag group

---

## 4. Agent Architecture

### 4.1 Decision Model: Fictitious Play with Softmax

Agents use **Fictitious Play** as the baseline decision model. At each round, an agent:

1. Maintains a belief distribution over each teammate's likely action (stag or hare), organized according to condition (see Section 3)
2. Computes the expected value of each action given current beliefs
3. Selects an action via **softmax**:

```
P(action) = exp(EV(action) / τ) / Σ exp(EV(a) / τ)
```

Where **τ (temperature)** is a free parameter controlling rationality:
- τ → 0: deterministic best-response
- τ → ∞: random choice
- Default τ: 0.1 (near-rational; to be justified against human data)

Ties in expected value resolve naturally through the softmax distribution (i.e., randomly with equal probability).

#### Expected Value Calculation

The EV of each action is computed over the full payoff table. Because payoffs are always split evenly among all three players, the focal agent's payoff depends not just on their own action but on how many teammates also hunt stag. Let **p_1** and **p_2** denote the focal agent's current belief that teammate 1 and teammate 2 hunt stag, respectively. The three possible teammate outcome states are:

| Teammate outcome | Probability | Group total if focal hunts stag | Group total if focal hunts hare |
|---|---|---|---|
| Both hunt stag | p_1 × p_2 | 3 stag → 4 pts | 2 stag → 5 pts |
| Exactly one hunts stag | p_1(1−p_2) + p_2(1−p_1) | 2 stag → 5 pts | 1 stag → 2 pts |
| Neither hunts stag | (1−p_1)(1−p_2) | 1 stag → 2 pts | 0 stag → 3 pts |

Per-agent payoff is always group total / 3. The full EV expressions are:

```
EV(stag) = p_1·p_2 · (4/3)
         + (p_1(1−p_2) + p_2(1−p_1)) · (5/3)
         + (1−p_1)(1−p_2) · (2/3)

EV(hare) = p_1·p_2 · (5/3)
         + (p_1(1−p_2) + p_2(1−p_1)) · (2/3)
         + (1−p_1)(1−p_2) · (1)
```

Note that EV(hare) is not a fixed value — it depends on beliefs about teammates because the group total (and therefore the focal agent's share) varies with how many others hunt stag.

#### EV by Condition

The condition determines how p_1 and p_2 are derived from the agent's belief representation:

**Anonymous:** One pooled belief p applies to both teammates (treated as i.i.d.):
```
p_1 = p_2 = p

EV(stag) = p² · (4/3)  +  2p(1−p) · (5/3)  +  (1−p)² · (2/3)
EV(hare) = p² · (5/3)  +  2p(1−p) · (2/3)  +  (1−p)² · (1)
```

**Identity:** Separate beliefs per individual, so the full general form applies without simplification:
```
p_1 = belief about teammate 1 individually
p_2 = belief about teammate 2 individually
(use full EV expressions above)
```

**Tag-based (focal agent is red — unique tag):** Both teammates share the blue tag and are exchangeable:
```
p_1 = p_2 = p_blue
(mathematically identical to Anonymous, but p_blue is updated via
tag-level belief tracking and initialized under the prescriptive norm
that blue-tag agents hunt stag)
```

**Tag-based (focal agent is blue — shared tag):** One teammate shares the blue tag, one has the red tag:
```
p_1 = p_blue  (the other blue-tag teammate)
p_2 = p_red   (the red-tag teammate)
(use full EV expressions above with separate tag-level beliefs)
```

**Key property:** The condition does not change the reasoning process — it changes only the granularity of the belief representation that feeds into a common EV calculation. This is an intentional design choice that makes the conditions directly comparable and supports the theoretical claim that tags reduce coordination cost by structuring beliefs, not by changing how agents reason.

### 4.2 Initial Priors

Agents begin with a **uniform prior** over partner actions — each partner is equally likely to hunt stag or hare. With binary actions, uniform priors over actions and uniform priors over strategies are equivalent.

### 4.3 Belief Update Rule

After each round, agents update beliefs using **Fictitious Play counting**:

- In the **Anonymous** condition: the single pooled count is incremented based on observed teammate actions
- In the **Identity** condition: each individual teammate's count is updated separately
- In the **Tag-based** condition: the count for the relevant tag group is updated; observations about one member update beliefs about all members sharing that tag

### 4.4 Pluggable Decision Model Interface

The decision model is **modular**. Every decision model implements three methods:

```python
class DecisionModel:
    def update(self, observation: dict) -> None:
        """Update internal beliefs given what happened last round."""
        pass

    def decide(self) -> str:
        """Return 'stag' or 'hare'."""
        pass

    def complexity(self) -> float:
        """Return normalized [0,1] entropy-based uncertainty before decision."""
        pass
```

This interface allows Fictitious Play to be swapped for BToM or other models without modifying the game engine. The `complexity()` method returns entropy of the current belief/action distribution, normalized to [0,1], measured *before* the decision is made (mapping onto reaction time as a cognitive load proxy).

---

## 5. Simulation Execution

### 5.1 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `condition` | Anonymous / Identity / Tag-based | Required |
| `n_rounds` | Rounds per simulation | 50 |
| `k_convergence` | Consecutive optimal rounds required to mark a simulation as converged | 5 |
| `n_simulations` | Number of independent simulation runs to aggregate | 100 |
| `tau` | Softmax temperature | 0.1 |

### 5.2 Multiple Runs

Since N < 10 agents and group variance within a single run is high, simulations are run **n_simulations** times and results are aggregated. The default of 100 runs is chosen to provide stable estimates of convergence speed distributions while remaining computationally cheap.

### 5.3 Parameter Sweeps

The core simulation exposes a `run_all_conditions()` convenience function that runs all three conditions with shared parameters and returns combined records in a single call. Broader sweeps (e.g., varying τ across conditions) are not built in and will be handled via a separate script or in R.

---

## 6. Data Logging

### 6.1 Per-Round Data (Native)

Every round of every simulation run logs:

| Field | Description |
|-------|-------------|
| `simulation_id` | Unique ID for this simulation run |
| `condition` | Anonymous / Identity / Tag-based |
| `round` | Round number |
| `agent_id` | Agent identifier |
| `agent_tag` | Tag label (None in Anonymous condition) |
| `decision` | stag or hare |
| `group_total` | Total points earned by the group this round |
| `agent_payoff` | Points received by this agent this round |
| `complexity` | Normalized entropy of agent's belief distribution pre-decision |

### 6.2 Primary Unit of Analysis

The **trio** is the primary unit of analysis, reflecting the interdependence of payoffs. Individual agent data is preserved for secondary analyses.

### 6.3 Native Visualization

Per-round data is stored natively in the Python environment (as a pandas DataFrame) to support **in-notebook visualization** during development. The following plots are available in `visualize.py` and can be rendered individually or as a combined summary figure:

- **Group payoff over rounds** — mean group total by round, with reference lines for the optimal (5 pts) and all-hare (3 pts) baselines
- **Complexity over rounds** — mean normalized entropy of agent belief distributions by round
- **Convergence distribution** — histogram of the round at which each simulation first converged
- **Outcome proportions over rounds** — stacked area chart showing the proportion of each group outcome by round: All-Stag, Stag+Stag+Hare, Stag+Hare+Hare, All-Hare

All plots support multi-condition data, displaying each condition in a distinct color (group payoff, complexity, convergence) or as separate panels (outcome proportions).

### 6.4 CSV Export

When ready for full analysis in R, simulation data is exported to CSV. Output files:

- `simulation_rounds.csv` — full per-round, per-agent log (as per 6.1)
- `simulation_summary.csv` — trio-level summary per simulation run (convergence round, total payoff, mean complexity per condition)

---

## 7. Implementation Environment

- **Language:** Python
- **Working environment:** Jupyter Notebook (primary interface for running and visualizing simulations)
- **Core modules:** Organized as importable Python modules alongside the notebook
- **Visualization:** matplotlib / seaborn (in-notebook)
- **Data analysis:** CSV export → R
- **Version control / sharing:** GitHub + OSF

### 7.1 Module Structure

```
role_games/
│
├── game.py          # Game engine: round loop, payoff calculation, convergence tracking
├── agents.py        # Agent class + DecisionModel interface
├── models/
│   └── fictitious_play.py   # Fictitious Play implementation
├── conditions.py    # Condition enum (Anonymous, Identity, Tag-based)
├── logger.py        # Data logging and CSV export
├── visualize.py     # Plotting functions for in-notebook use
│
├── simulate.ipynb   # Main working notebook
└── README.md
```

### 7.2 Implementation Decisions

The following design choices were made during implementation and are not fully specified in the sections above:

- **Tag assignment:** In the Tag-based condition, the unique-tag (red) player is assigned randomly and uniformly across the three agents at the start of each simulation run. Tag assignment is independent across runs.
- **Belief initialization:** All belief counts are initialized with a pseudocount of 1 for each action (Laplace smoothing), yielding a uniform prior of p = 0.5. This applies to all conditions including Tag-based.
- **Complexity in the tag condition:** An agent's complexity is the average normalized entropy across its distinct belief distributions. A red-tag agent tracks one distribution (p_blue) and has complexity H(p_blue)/log(2). A blue-tag agent tracks two distributions (p_blue, p_red) and has complexity [H(p_blue) + H(p_red)] / (2·log(2)).
- **Softmax numerical stability:** The softmax is computed using the log-sum-exp trick (subtracting max(EV/τ) before exponentiating) to prevent overflow at low values of τ.

---

## 8. Future Work Log

The following features and extensions are explicitly deferred to later iterations. This log exists to ensure design decisions made now do not foreclose these options. Items are sorted by implementation difficulty.

### Straightforward — parameter additions or small changes to existing methods

| # | Feature | Notes |
|---|---------|-------|
| 1 | **Discounted / weighted Fictitious Play** | Add decay parameter δ ∈ (0,1] to `update()` in `fictitious_play.py`; counts updated as `counts[action] = δ · counts[action] + new_observation`; δ=1 recovers standard Fictitious Play. Psychologically motivated by human recency bias; natural second fitting parameter alongside τ. Generates testable prediction that optimal δ varies by condition |
| 2 | **Prediction error / surprise logging** | Log how surprised each agent is by observed outcomes each round; natural complement to complexity measure; maps onto neuroscience/cognitive science frameworks |
| 3 | **Learning rate logging** | Log posterior belief update magnitude as a secondary complexity measure |
| 4 | **Prior sensitivity analysis** | Systematic analysis of how initial pseudocount specification affects convergence results; vary starting counts across simulations |
| 5 | **Payoff structure variants** | Unequal splits, percentage contributions; payoff calculation is isolated in `game.py` making this straightforward to extend |

### Moderate — new scripts, additional parameters, or extensions to the game loop

| # | Feature | Notes |
|---|---------|-------|
| 6 | **Parameter sweep infrastructure** | Built-in support for sweeping τ, δ, conditions, group size; implemented as a separate script alongside the notebook |
| 7 | **Replacement rate manipulation** | Vary rate at which agents are swapped out mid-simulation; game.py round loop is structured with this addition in mind |
| 8 | **Softmax decomposition** | Decompose τ into separate parameters for risk sensitivity, social uncertainty, etc.; requires rethinking the decision model interface |
| 9 | **Hierarchy / scale extension** | Vary N players; introduce a coordinator role at larger N; requires extending tag assignment logic and EV calculation beyond 3-player case |
| 10 | **Accountability / punishment extension** | CPU agents defect; participant can costly punish; role markers affect punishment rates; requires new agent type and punishment mechanic in game loop |

### Substantial — new paradigms or major architectural additions

| # | Feature | Notes |
|---|---------|-------|
| 11 | **BToM decision model** | Slot into DecisionModel interface; will use probabilistic programming (possibly Julia/Gen); requires specifying prior over partner strategies and recursive reasoning depth |
| 12 | **Joint belief tracking in anonymous condition** | Replace independent pooled beliefs with a joint distribution over teammate action pairs; allows agent to detect group-level compositional patterns without individual identifiability. Testable prediction: if humans converge faster than model in anonymous condition, joint tracking is a candidate explanation |
| 13 | **Experience-based specialization** | Agents improve at tasks through repetition; requires adding a skill/performance component to the action outcomes |
| 14 | **Evolutionary modeling** | Agent strategies evolve across generations; separate paradigm from behavioral modeling; likely a distinct simulation module |
| 15 | **Web app visualization layer** | Interactive simulator; Python backend (Flask/FastAPI) serving the simulation; substantial frontend work |

---

## 9. Open Theoretical Commitments

These are explicit modeling choices that should be acknowledged in any paper:

1. **Computational-level framing:** The model characterizes what agents compute, not how the brain implements it (Marr, 1982)
2. **Tags are immediately prescriptive:** Agents use tags as norms from round 1, not learned cues
3. **Uniform initial priors:** Least-committal assumption; prior sensitivity analysis is deferred (see Future Work #12)
4. **Softmax as irrationality:** Temperature parameter τ absorbs all suboptimality; acknowledged as a simplification
5. **Complexity as pre-decision entropy:** Normalized entropy of belief distribution before decision; validated against reaction time data from human experiments
6. **Trio as unit of analysis:** Individual agent data preserved but trio is primary
7. **Independent belief pooling in anonymous condition:** Teammates are treated as i.i.d. in the anonymous condition. This is a conservative simplification — a more sophisticated agent could track joint distributions over teammate action pairs and detect group-level compositional patterns without individual identifiability. Current formulation may underpredict convergence speed in the anonymous condition (see Future Work #14)

