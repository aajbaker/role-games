# Role Games: Agent-Based Simulation — Model Specification

**Project:** Role Games  
**Author:** Aaron Baker  
**Status:** Active  
**Last Updated:** October 2026

---

## 1. Overview

This simulation models how small groups of agents solve an asymmetric coordination problem (a modified stag hunt; 3 players is the reference case, and groups of 3–5 are supported) under three conditions that vary the informational markers available to agents. The primary goals are:

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

**General N (3–5).** The same rule applies to larger groups. The stag is caught if at least 2 players hunt it, giving 4 points however many hunt it. Each hare hunter adds 1 point, and the group total is split evenly among all N players. The optimum is always exactly 2 stag hunters and N − 2 hare hunters (N + 2 points).

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

Simulations always run for exactly `n_rounds` rounds (user-specified parameter). There is no early termination. Convergence is logged per round: a round is marked `converged` if the last K rounds, including this one, all had exactly 2 stag hunters (the optimum for any N; K = `k_convergence`, default 5). The Simulate page plots the share of simulations in a converged state at each round.

---

## 3. Experimental Conditions

The condition is passed as a **runtime parameter** and controls what information agents have about their teammates. Tags are treated as immediately salient: agents use them from round 1 to organize beliefs, one belief per tag. Tags carry no built-in norm. Agents start with the same uniform prior about every tag and learn what each tag does from behaviour. Under the optional group decision rule (Section 4.5), tags also define who acts together.

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
- In the 3-player game: 1 player has one tag (e.g., red hat), 2 players share another tag (e.g., blue hat). Larger groups: 2 red / 2 blue (N=4), 2 red / 3 blue (N=5). Tags are assigned randomly each run.
- Agents maintain beliefs at the **tag level** — same-tag players are treated as exchangeable
- Tags are **descriptive**, not prescriptive: agents use tag membership to organize what they have seen others do, not as a rule for what they should do. Under the optional group decision rule (Section 4.5), tags also define who acts together
- Belief representation: one distribution per tag group

---

## 4. Agent Architecture

### 4.1 Decision Model: Fictitious Play with Softmax

Two decision models are implemented: **Fictitious Play** (described here) and depth-1 **Bayesian Theory of Mind** (Section 4.6). They differ in how beliefs are formed. Both choose using the same softmax-over-expected-value step below.

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
- Default τ: 0.1 in the Python functions, 0.05 in the Streamlit app (near-rational; to be justified against human data)

Ties in expected value resolve naturally through the softmax distribution (i.e., randomly with equal probability).

#### Expected Value Calculation

The EV of each action is computed over the full payoff table. Because payoffs are always split evenly among all three players, the focal agent's payoff depends not just on their own action but on how many teammates also hunt stag. Let **p_1** and **p_2** denote the focal agent's current belief that teammate 1 and teammate 2 hunt stag, respectively. The three possible teammate outcome states are:

| Teammate outcome | Probability | Group total if focal hunts stag | Group total if focal hunts hare |
|---|---|---|---|
| Both hunt stag | p_1 × p_2 | 3 stag → 4 pts | 2 stag → 5 pts |
| Exactly one hunts stag | p_1(1−p_2) + p_2(1−p_1) | 2 stag → 5 pts | 1 stag → 2 pts |
| Neither hunts stag | (1−p_1)(1−p_2) | 1 stag → 2 pts | 0 stag → 3 pts |

Per-agent payoff is always group total / 3. The full EV expressions are below. For N > 3, the number of teammates hunting stag follows a Poisson-binomial distribution over the N − 1 teammate beliefs. `role_games/models/_ev.py` computes it exactly by dynamic programming, and the 3-player expressions are the special case.

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
tag-level belief tracking; it starts from the same uniform prior
as every other belief)
```

**Tag-based (focal agent is blue — shared tag):** One teammate shares the blue tag, one has the red tag:
```
p_1 = p_blue  (the other blue-tag teammate)
p_2 = p_red   (the red-tag teammate)
(use full EV expressions above with separate tag-level beliefs)
```

**Key property:** The condition does not change the reasoning process — it changes only the granularity of the belief representation that feeds into a common EV calculation. This is an intentional design choice that makes the conditions directly comparable and supports the theoretical claim that tags reduce coordination cost by structuring beliefs, not by changing how agents reason. This holds under the default settings. Section 4.5 describes two optional Tag-based variants that deliberately depart from it.

### 4.2 Initial Priors

Agents begin with a **uniform prior** over partner actions — each partner is equally likely to hunt stag or hare. With binary actions, uniform priors over actions and uniform priors over strategies are equivalent.

### 4.3 Belief Update Rule

After each round, agents update beliefs using **Fictitious Play counting**:

- In the **Anonymous** condition: the single pooled count is incremented based on observed teammate actions
- In the **Identity** condition: each individual teammate's count is updated separately
- In the **Tag-based** condition: the count for the relevant tag group is updated; observations about one member update beliefs about all members sharing that tag. An agent holds a belief only about tags its teammates have, so an agent alone in its tag (red at N=3) holds no belief about its own tag.

Beliefs are formed from teammates' choices only; an agent's own choices are not counted (see Section 4.5 for the optional Tag-based exception). With a discount factor δ < 1 (`discount`), all counts are multiplied by δ once per round before that round's observations are added, so recent rounds weigh more; δ = 1 is standard Fictitious Play.

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

    def reset(self) -> None:
        """Reset beliefs to the uniform prior (called when an agent is replaced)."""
        pass

    def teammate_replaced(self, agent_id: int) -> None:
        """Called on each remaining agent when a teammate is replaced (default: no-op)."""
        pass
```

This interface allows Fictitious Play to be swapped for BToM or other models without modifying the game engine. The `complexity()` method returns entropy of the current belief/action distribution, normalized to [0,1], measured *before* the decision is made (mapping onto reaction time as a cognitive load proxy).

### 4.5 Tag-based Options (opt-in)

Both options are off by default. With the defaults, the Tag-based condition uses the same belief-formation and decision process as the other conditions and differs only in how beliefs are grouped. Each option adds one tag-specific assumption, so its effect can be tested on its own.

**Decision rule (`decision_rule`, default `"individual"`).**

- *Individual:* as in Section 4.1. The agent chooses its own action, treating each teammate as an independent draw with that teammate's tag-level stag probability.
- *Group:* the agent assumes each tag acts as a block and chooses its group's action, given q = P(other tag hunts stag). In Fictitious Play, q is the belief about the other tag. In BToM, q is predicted from the posterior about the other tag.

```
EV(stag) = q · pay(n_own + n_other) + (1 − q) · pay(n_own)
EV(hare) = q · pay(n_other)         + (1 − q) · pay(0)
```

Here n_own is the size of the agent's tag (including itself), n_other is the size of the other tag, and pay(k) is the per-player payoff when k players hunt stag. The choice is still a softmax over these EVs, and each agent still samples its own action, so a tag can split in practice. The thresholds that result:

| N (red / blue) | Red hunts stag when | Blue hunts stag when |
|---|---|---|
| 3 (1 / 2) | never (hare is better by 1/3 for any q) | always (stag is better by 2/3 for any q) |
| 4 (2 / 2) | q < 0.5 | q < 0.5 |
| 5 (2 / 3) | q < 0.5 | q < 0.25 |

At N = 3 the choice does not depend on beliefs. Under the group rule, complexity counts only the belief about the other tag, because that is the only belief the decision uses. In BToM, agents also model others as group reasoners, so each tag's posterior is 1D (see `btom_implementation_spec.md`).

**Count own choices toward own tag (`include_own_action`, default `False`).** When on, an agent's own action also updates its belief about its own tag, alongside its same-tag teammates' actions. It applies only under the individual rule, because the group rule does not use the own-tag belief. It also applies only to agents with a same-tag teammate.

### 4.6 Bayesian Theory of Mind

BToM agents model each teammate as a softmax expected-value maximizer with its own beliefs about the group. They infer those beliefs from observed choices by Bayesian inversion over a discretized belief grid, and predict each teammate's next choice from the posterior. The predicted probabilities feed the same EV and softmax step as Fictitious Play. Full design and implementation notes: `btom_implementation_spec.md`.

---

## 5. Simulation Execution

### 5.1 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `condition` | Anonymous / Identity / Tag-based | Required |
| `model_type` | `"fictitious_play"` or `"bayesian_tom"` | `"fictitious_play"` |
| `n_players` | Players per group (3–5) | 3 |
| `n_rounds` | Rounds per simulation | 50 |
| `k_convergence` | Consecutive optimal rounds (exactly 2 stag hunters) required to mark a round as converged | 5 |
| `n_simulations` | Number of independent simulation runs to aggregate | 100 |
| `tau` | Softmax temperature | 0.1 |
| `discount` | Belief decay δ applied once per round (1.0 = no forgetting) | 1.0 |
| `replacement_rate` | Probability each round that one randomly chosen agent is replaced by a newcomer starting from the uniform prior (Section 7.2) | 0.0 |
| `decision_rule` | Tag-based only: `"individual"` or `"group"` (Section 4.5) | `"individual"` |
| `include_own_action` | Tag-based only: count own choices toward own tag (Section 4.5) | `False` |
| `base_seed` | Seed for run *i* is `base_seed + i`; `None` = unseeded (Section 5.4) | `None` |

These are the defaults of the Python functions. The Streamlit Simulate page starts at τ = 0.05, δ = 0.90, 30 rounds, and 300 simulations.

### 5.2 Multiple Runs

Since N < 10 agents and group variance within a single run is high, simulations are run **n_simulations** times and results are aggregated. The default of 100 runs is chosen to provide stable estimates of convergence speed distributions while remaining computationally cheap.

### 5.3 Parameter Sweeps

The core simulation exposes a `run_all_conditions()` convenience function that runs all three conditions with shared parameters and returns combined records in a single call. Broader sweeps (e.g., varying τ across conditions) are not built in and will be handled via a separate script or in R.

### 5.4 Reproducibility

Each simulation run creates one random-number generator from its seed. That generator drives tag assignment, every agent's choices, and replacement, so the same seed reproduces a run exactly. Every setting uses one draw per agent per round, so runs with the same seed share tag assignments and replacement events across models and Tag-based options; this gives matched comparisons. `run_all_conditions()` passes the same `base_seed` to each condition.

---

## 6. Data Logging

### 6.1 Per-Round Data (Native)

Every round of every simulation run logs:

| Field | Description |
|-------|-------------|
| `simulation_id` | Run index within a batch |
| `condition` | Anonymous / Identity / Tag-based |
| `n_players` | Group size |
| `replacement_rate` | Replacement rate for the run |
| `round` | Round number |
| `agent_id` | Agent identifier |
| `agent_tag` | red / blue in the Tag-based condition; None otherwise |
| `decision` | stag or hare |
| `stag_count` | Number of stag hunters this round |
| `group_total` | Total points earned by the group this round |
| `agent_payoff` | Points received by this agent this round (group total / N) |
| `complexity` | Normalized entropy of agent's belief distribution pre-decision |
| `replaced` | True in the first round after this agent was reset |
| `model_type` | `fictitious_play` or `bayesian_tom` |
| `decision_rule` | Rule actually used (`individual` outside Tag-based) |
| `include_own_action` | Whether own choices were counted (`False` outside Tag-based) |
| `converged` | True if the last K rounds all had exactly 2 stag hunters |

### 6.2 Primary Unit of Analysis

The **trio** is the primary unit of analysis, reflecting the interdependence of payoffs. Individual agent data is preserved for secondary analyses.

### 6.3 Native Visualization

Per-round data is stored natively in the Python environment (as a pandas DataFrame) to support **in-notebook visualization** during development. The following plots are available in `visualize.py` and can be rendered individually or as a combined summary figure:

- **Group payoff over rounds** (`plot_group_total`) — mean group total by round
- **Total payoff** (`plot_total_payoff`) — mean cumulative group payoff per condition, ±1 SE
- **Convergence rate** (`plot_convergence_rate`) — share of simulations in a converged state at each round
- **Complexity over rounds** (`plot_complexity`) — mean normalized entropy of agent belief distributions by round
- **Outcome proportions over rounds** (`plot_outcome_proportions`) — stacked area chart by stag count: 2 Stag (optimal), 3+ Stag, 1 Stag, All-Hare
- **Summary figure** (`plot_summary`) — group payoff, complexity, and convergence rate, plus outcome proportions

All plots support multi-condition data, displaying each condition in a distinct color or as separate panels (outcome proportions).

### 6.4 CSV Export

When ready for full analysis in R, simulation data is exported to CSV. Output files:

- `simulation_rounds.csv` — full per-round, per-agent log (as per 6.1)
- `simulation_summary.csv` — one row per simulation run: condition, `n_players`, `replacement_rate`, `decision_rule`, `include_own_action`, `total_payoff` (cumulative per-player payoff), and `mean_complexity`

---

## 7. Implementation Environment

- **Language:** Python
- **Working environment:** Streamlit app (`app.py`, with Play and Simulate pages) and a Jupyter notebook (`simulate.ipynb`) for scripted runs
- **Core modules:** Organized as importable Python modules alongside the notebook
- **Visualization:** matplotlib / seaborn (in-notebook)
- **Data analysis:** CSV export → R
- **Version control / sharing:** GitHub + OSF

### 7.1 Module Structure

```
app.py               # Streamlit entry point (Play + Simulate pages)
Play.py              # Play against simulated agents
Simulate.py          # Batch simulation and plots
simulate.ipynb       # Notebook for scripted runs
role_games/
│
├── game.py          # Game engine: agent setup, round loop, payoffs, convergence tracking
├── agents.py        # Agent class + DecisionModel interface
├── models/
│   ├── _ev.py               # Expected-value helpers (individual and group rules)
│   ├── fictitious_play.py   # Fictitious Play
│   └── bayesian_tom.py      # Bayesian Theory of Mind
├── conditions.py    # Condition enum (Anonymous, Identity, Tag-based)
├── logger.py        # Data logging and CSV export
└── visualize.py     # Plotting functions
```

### 7.2 Implementation Decisions

The following design choices were made during implementation and are not fully specified in the sections above:

- **Tag assignment:** In the Tag-based condition, tags (1R/2B at N=3, 2R/2B at N=4, 2R/3B at N=5) are shuffled randomly across agents at the start of each simulation run. Tag assignment is independent across runs.
- **Belief initialization:** All Fictitious Play belief counts are initialized with a pseudocount of 1 for each action (Laplace smoothing), yielding a uniform prior of p = 0.5. BToM posteriors start uniform over the belief grid. This applies to all conditions including Tag-based.
- **Complexity in the tag condition:** An agent's complexity is the average normalized entropy across its distinct belief distributions about its teammates' tags. At N=3, a red-tag agent tracks one distribution (p_blue) and has complexity H(p_blue)/log(2). A blue-tag agent tracks two distributions (p_blue, p_red) and has complexity [H(p_blue) + H(p_red)] / (2·log(2)). Under the group decision rule, only the belief about the other tag counts.
- **No beliefs about a tag an agent is alone in:** An agent with no same-tag teammate keeps no belief about its own tag, in either model. Such a belief could only be updated from the agent's own choices and would never be used to predict anyone.
- **Replacement:** The replaced agent's beliefs reset to the uniform prior. The other agents know who was replaced (as the human player is told in Play). This only changes their beliefs in Identity, where each remaining agent resets its belief about that player; in Anonymous and Tag-based, beliefs are pooled across several players and are left unchanged. In BToM, a remaining agent's models of what *other* teammates believe are not adjusted: a teammate's choice depends only on the mix of its beliefs, so those models don't record which belief is about whom.
- **Random numbers:** One seeded generator per run drives tag assignment, agent choices, and replacement (Section 5.4).
- **Softmax numerical stability:** The softmax is computed using the log-sum-exp trick (subtracting max(EV/τ) before exponentiating) to prevent overflow at low values of τ.

---

## 8. Future Work Log

The following features and extensions are explicitly deferred to later iterations. This log exists to ensure design decisions made now do not foreclose these options. Items are sorted by implementation difficulty.

### Straightforward — parameter additions or small changes to existing methods

| # | Feature | Notes |
|---|---------|-------|
| 1 | **Discounted / weighted Fictitious Play** | **Implemented** (`discount`; also applies to BToM by decaying posteriors toward uniform). Original note: Add decay parameter δ ∈ (0,1] to `update()` in `fictitious_play.py`; counts updated as `counts[action] = δ · counts[action] + new_observation`; δ=1 recovers standard Fictitious Play. Psychologically motivated by human recency bias; natural second fitting parameter alongside τ. Generates testable prediction that optimal δ varies by condition |
| 2 | **Prediction error / surprise logging** | Log how surprised each agent is by observed outcomes each round; natural complement to complexity measure; maps onto neuroscience/cognitive science frameworks |
| 3 | **Learning rate logging** | Log posterior belief update magnitude as a secondary complexity measure |
| 4 | **Prior sensitivity analysis** | Systematic analysis of how initial pseudocount specification affects convergence results; vary starting counts across simulations |
| 5 | **Payoff structure variants** | Unequal splits, percentage contributions; payoff calculation is isolated in `game.py` making this straightforward to extend |

### Moderate — new scripts, additional parameters, or extensions to the game loop

| # | Feature | Notes |
|---|---------|-------|
| 6 | **Parameter sweep infrastructure** | Built-in support for sweeping τ, δ, conditions, group size; implemented as a separate script alongside the notebook |
| 7 | **Replacement rate manipulation** | **Implemented** (`replacement_rate`; see Section 7.2 for what remaining agents do). Original note: Vary rate at which agents are swapped out mid-simulation; game.py round loop is structured with this addition in mind |
| 8 | **Softmax decomposition** | Decompose τ into separate parameters for risk sensitivity, social uncertainty, etc.; requires rethinking the decision model interface |
| 9 | **Hierarchy / scale extension** | **Partly implemented**: N = 3–5 with tag assignment and Poisson-binomial EV; coordinator role not implemented. Original note: Vary N players; introduce a coordinator role at larger N; requires extending tag assignment logic and EV calculation beyond 3-player case |
| 10 | **Accountability / punishment extension** | CPU agents defect; participant can costly punish; role markers affect punishment rates; requires new agent type and punishment mechanic in game loop |

### Substantial — new paradigms or major architectural additions

| # | Feature | Notes |
|---|---------|-------|
| 11 | **BToM decision model** | **Implemented** (depth-1, NumPy grid posteriors; see `btom_implementation_spec.md`). Original note: Slot into DecisionModel interface; will use probabilistic programming (possibly Julia/Gen); requires specifying prior over partner strategies and recursive reasoning depth |
| 12 | **Joint belief tracking in anonymous condition** | Replace independent pooled beliefs with a joint distribution over teammate action pairs; allows agent to detect group-level compositional patterns without individual identifiability. Testable prediction: if humans converge faster than model in anonymous condition, joint tracking is a candidate explanation |
| 13 | **Experience-based specialization** | Agents improve at tasks through repetition; requires adding a skill/performance component to the action outcomes |
| 14 | **Evolutionary modeling** | Agent strategies evolve across generations; separate paradigm from behavioral modeling; likely a distinct simulation module |
| 15 | **Web app visualization layer** | **Implemented** as a Streamlit app (`app.py`). Original note: Interactive simulator; Python backend (Flask/FastAPI) serving the simulation; substantial frontend work |

---

## 9. Open Theoretical Commitments

These are explicit modeling choices that should be acknowledged in any paper:

1. **Computational-level framing:** The model characterizes what agents compute, not how the brain implements it (Marr, 1982)
2. **Tags are immediately salient but not prescriptive:** Agents use tags from round 1 to organize beliefs, but no tag-specific norm or prior is built in; what each tag does is learned from behaviour. Acting as a group by tag is an opt-in variant (Section 4.5)
3. **Uniform initial priors:** Least-committal assumption; prior sensitivity analysis is deferred (see Future Work #4)
4. **Softmax as irrationality:** Temperature parameter τ absorbs all suboptimality; acknowledged as a simplification
5. **Complexity as pre-decision entropy:** Normalized entropy of belief distribution before decision; validated against reaction time data from human experiments
6. **Trio as unit of analysis:** Individual agent data preserved but trio is primary
7. **Independent belief pooling in anonymous condition:** Teammates are treated as i.i.d. in the anonymous condition. This is a conservative simplification — a more sophisticated agent could track joint distributions over teammate action pairs and detect group-level compositional patterns without individual identifiability. Current formulation may underpredict convergence speed in the anonymous condition (see Future Work #12)
8. **Same process across conditions by default:** In every condition, agents form beliefs only from teammates' choices and use the same individual decision rule, so the Tag-based condition differs only in how beliefs are grouped. Counting one's own choices toward one's tag, and deciding as a group, are opt-in variants (Section 4.5)

