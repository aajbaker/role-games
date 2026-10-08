# Bayesian Theory of Mind — Implementation Specification

This document is a self-contained spec for implementing a Bayesian Theory of Mind (BToM) agent model alongside the existing Fictitious Play (FP) model. It includes all necessary context about the existing codebase, the full mathematical design, implementation instructions, and integration notes.

> **Status (October 2026):** Implemented in `role_games/models/bayesian_tom.py`. Sections 1–3, 5 and 6 describe the current implementation. Section 4 lists where each piece lives and how the build differs from the original plan. Section 8 is a change log.

---

## 1. Codebase Overview

### File structure (relevant files)

```
role-games-model/
├── app.py                          # Navigation router (st.navigation)
├── Simulate.py                     # Simulation page
├── Play.py                         # Human play page
├── role_games/
│   ├── __init__.py                 # Exports Condition, run_simulation, run_multiple, run_all_conditions, to_dataframes, export_csv
│   ├── conditions.py               # Condition enum
│   ├── agents.py                   # Agent class + DecisionModel interface
│   ├── game.py                     # _make_agents(), run_simulation(), etc.
│   └── models/
│       ├── __init__.py             # Exports FictitiousPlay, BayesianToM
│       ├── _ev.py                  # Shared EV helpers (scalar, vectorized, group rule)
│       ├── fictitious_play.py      # FP model
│       └── bayesian_tom.py         # BToM model
```

### Key classes and interfaces

**`DecisionModel` (agents.py)** — abstract base class all models must implement:
```python
class DecisionModel:
    def decide(self) -> str:          # returns "stag" or "hare"
    def update(self, observation: dict) -> None
    def complexity(self) -> float     # returns float in [0, 1]
    def reset(self) -> None           # resets to uniform prior
    def teammate_replaced(self, agent_id: int) -> None   # optional; default no-op
```

**Observation dict format** (passed to `update()`):
```python
{
    "teammates": [
        {"agent_id": int, "tag": str | None, "action": "stag" | "hare"},
        ...
    ],
    "own_action": "stag" | "hare",
    "own_tag": str | None,
}
```

**`Agent` (agents.py)** — wraps a DecisionModel:
```python
class Agent:
    agent_id: int
    tag: str | None
    def decide() -> str       # delegates to model.decide()
    def update(obs) -> None   # delegates to model.update()
    def complexity() -> float
    def reset() -> None
    def teammate_replaced(agent_id) -> None   # delegates to model
```

**`Condition` enum (conditions.py)**:
```python
class Condition(str, Enum):
    ANONYMOUS = "Anonymous"
    IDENTITY  = "Identity"
    TAG_BASED = "Tag-based"
```

**`_make_agents()` (game.py)** — current signature:
```python
def _make_agents(
    condition: Condition,
    tau: float,
    n_players: int,
    rng: random.Random,
    discount: float = 1.0,
    model_type: str = "fictitious_play",     # or "bayesian_tom"
    decision_rule: str = "individual",       # Tag-based only: or "group"
    include_own_action: bool = False,        # Tag-based only
) -> list[Agent]
```
The same `rng` (seeded per run) is passed to every agent's model and used for its choices.

**Tag assignment (game.py)**:
- N=3: 1 red, 2 blue
- N=4: 2 red, 2 blue
- N=5: 2 red, 3 blue
- Tags randomly shuffled each run

**Shared EV helpers (`models/_ev.py`)**, used by both models:
```python
def _poisson_binomial_pmf(probs: list[float]) -> list[float]
def _ev_stag(teammate_probs: list[float], n_players: int) -> float
def _ev_hare(teammate_probs: list[float], n_players: int) -> float
def _pb_pmf_vec(prob_grids: list[np.ndarray]) -> np.ndarray          # vectorized PB
def _ev_stag_vec(prob_grids, n_players) -> np.ndarray
def _ev_hare_vec(prob_grids, n_players) -> np.ndarray
def _ev_group(n_own, n_other, q, n_players) -> (ev_stag, ev_hare)   # group rule; q float or array
```

**Existing complexity helper (fictitious_play.py)**:
```python
def _normalized_entropy(p: float) -> float   # binary entropy / log(2)
```

**`run_simulation()` (game.py)** — current signature:
```python
def run_simulation(
    condition, n_rounds, k_convergence, tau, n_players,
    replacement_rate, discount, model_type, decision_rule,
    include_own_action, sim_id, seed
) -> list[dict]
```
`run_multiple()` and `run_all_conditions()` take the same parameters (with `base_seed` in place of `sim_id`/`seed`).

**Payoff function**:
```python
def _group_total(actions: list[str]) -> float:
    stag_count = actions.count("stag")
    hare_count = actions.count("hare")
    return (4.0 if stag_count >= 2 else 0.0) + hare_count
```
Individual EV = group_total / n_players.

---

## 2. BToM Model — Conceptual Design

### Core idea

FP asks: *what has agent j done historically?* BToM asks: *what does agent j believe others will do, and given those beliefs, what will j do?*

Agent i models each other agent j as a utility-maximizer with their own beliefs about the group. When j acts, i inverts j's decision process to infer what j must have believed. i then forward-simulates j using those inferred beliefs to predict j's next action.

**Depth-1 ToM only.** i models j's first-order beliefs about the world. i does NOT model j's beliefs about i's beliefs about j's beliefs. No infinite regress.

### The belief vector θ

For each tracked entity, the belief vector θ represents that entity's beliefs about the stag probability of each relevant group. The dimensionality of θ depends on the condition:

| Condition | θ represents | Dimensionality |
|-----------|-------------|----------------|
| Anonymous | Pool stag rate | 1D: θ ∈ [0,1] |
| Tag-based (individual rule) | Per-tag stag rates | 2D: (θ^red, θ^blue) ∈ [0,1]² |
| Tag-based (group rule) | The other tag's stag rate | 1D: θ^other ∈ [0,1] |
| Identity  | Per-individual stag rates | (n−1)D: (θ¹,…,θⁿ⁻¹) ∈ [0,1]^{n-1} |

### What i maintains (one posterior per tracked entity)

**Anonymous**: one posterior over θ (1D), representing the shared pool belief.
- All observations update the same posterior.

**Tag-based**: one posterior per tag that a teammate holds (usually two: "what red agents believe" and "what blue agents believe").
- Observations from a red agent update the red posterior.
- Observations from a blue agent update the blue posterior.
- Under the individual rule, each posterior lives in (θ^red, θ^blue) space because a tag-based agent tracks both tags.
- Under the group rule, an agent's choice depends only on its belief about the other tag. So each posterior is 1D: what that tag believes about the other tag's stag rate.
- An agent alone in its tag (red at N=3) keeps no posterior for its own tag. That posterior could only be updated from the agent's own choices, and it would never be used to predict a teammate.

**Identity**: (n−1) posteriors, each (n−1)D — one per other agent j.
- Observations from agent j update j's individual posterior.
- Each posterior lives in (θ^{id_0}, θ^{id_1}, ..., θ^{id_{n-2}}) space because an identity agent tracks each individual.
- This includes a dimension for i itself (j's belief about what i will do), which i treats as unknown and infers from j's actions.

### Why identity is genuinely more complex

- Anonymous: 1 posterior × 1 dimension = 1 total belief dimension
- Tag-based (individual rule): 2 posteriors × 2 dimensions = 4 total belief dimensions (an agent alone in its tag: 1 × 2 = 2)
- Tag-based (group rule): up to 2 posteriors × 1 dimension = 2
- Identity N=3: 2 posteriors × 2 dimensions = 4 total
- Identity N=4: 3 posteriors × 3 dimensions = 9 total
- Identity N=5: 4 posteriors × 4 dimensions = 16 total

Identity scales quadratically in n. This reflects a real cognitive cost: tracking what each individual believes about each other individual.

---

## 3. BToM Model — Mathematical Specification

### Grid representation

Discretize each belief dimension into `n_grid` evenly-spaced points:
```python
theta_vals = np.linspace(1/(2*n_grid), 1 - 1/(2*n_grid), n_grid)
# default n_grid=10: [0.05, 0.15, ..., 0.95]
# Avoids 0 and 1 for numerical stability in softmax
```

Each posterior is an n-dimensional numpy array of shape `(n_grid,)^d` where d is the belief dimensionality for that condition. The array stores probability masses (summing to 1) over the grid.

**Initial state**: uniform prior — all grid points have equal mass (1 / n_grid^d).

### Likelihood function

The likelihood of observing agent j (with tag `t_j`, facing teammates with tags/ids from group info) choose action `a` given belief vector θ is:

```
P(a | θ, j, group) = softmax(EV_stag(b_j(θ)), EV_hare(b_j(θ)))[a]
```

where `b_j(θ)` is j's teammate belief list derived from θ:

**Anonymous**: `b_j(θ) = [θ] * (n-1)` — j believes all n-1 teammates stag with probability θ.

**Tag-based**: `b_j(θ) = [θ^{tag_k} for each teammate k of j]` — j uses their tag-specific beliefs. Requires knowing j's tag and each teammate's tag (both known in tag-based condition).

**Identity**: `b_j(θ) = [θ^{id_k} for each teammate k of j]` — j uses their per-individual beliefs. The ordering of θ dimensions corresponds to j's teammates by agent_id.

Then:
```python
ev_s = _ev_stag(b_j(θ), n_players)
ev_h = _ev_hare(b_j(θ), n_players)
# Softmax with existing tau parameter:
p_stag = exp(ev_s / tau) / (exp(ev_s / tau) + exp(ev_h / tau))
P(stag | θ, j) = p_stag
P(hare | θ, j) = 1 - p_stag
```

This is vectorized over all grid points simultaneously using numpy broadcasting. The P(stag | θ) grid for each posterior is computed once at construction (`_p_stag_grids`) and reused for every update and prediction.

**Group rule (Tag-based):** j is modeled as a group reasoner too. With θ = j's belief that the other tag hunts stag:
```python
ev_s, ev_h = _ev_group(n_{t_j}, n_{other tag}, θ, n_players)   # linear in θ
```
At N=3 this EV gap does not depend on θ, so the likelihood is flat and the posteriors never move.

### Update rule

When i observes agent j choose action `a`:

1. Compute likelihood array `L` over the full grid: `L[grid_point] = P(a | θ=grid_point, j, group)`
2. Multiply the current posterior by `L` element-wise
3. Renormalize (divide by sum)

```python
posterior *= L  # element-wise
posterior /= posterior.sum()
```

**Which posterior to update:**
- Anonymous: always update `_posteriors["pool"]`
- Tag-based: update `_posteriors[j.tag]`
- Identity: update `_posteriors[j.agent_id]`

**Own action (Tag-based only, optional):** With `include_own_action=True`, i's own action also updates the posterior for i's tag, as in FP. It only applies if a teammate shares that tag. It is off by default, so beliefs come only from others' choices, as in the other conditions.

### Discount factor

Before each update (once per round, called via `_decay_posterior()`), dilute the posterior back toward the uniform prior:

```python
def _decay_posteriors(self) -> None:
    if self.discount == 1.0:
        return
    for key in self._posteriors:
        posterior = self._posteriors[key]
        log_p   = np.log(posterior + 1e-300)
        log_u   = -math.log(posterior.size)          # log of the uniform mass
        log_dec = self.discount * log_p + (1 - self.discount) * log_u
        p = np.exp(log_dec - log_dec.max())
        self._posteriors[key] = p / p.sum()
```

This mirrors FP's `_decay_counts()` behavior: recent observations matter more when δ < 1.

### Prediction: expected P(j staggers)

To predict what agent j will do next (used in own EV calculation), i computes the posterior-weighted expected stag probability:

```python
E[P(j staggers)] = sum over grid points of: posterior[θ] * P(stag | θ, j, group)
```

In numpy: `(posterior * p_stag_grid).sum()` where `p_stag_grid` is the precomputed softmax stag probability over the full grid.

This expected probability is what gets passed as j's stag probability into the Poisson-Binomial EV calculator for i's own decision.

### Decision rule

Same as FP — softmax over EV:

```python
def decide(self) -> str:
    if self.group_mode:                      # Tag-based, decision_rule="group"
        other = self._other_tag(self.own_tag)
        q = (self._posteriors[other] * self._p_stag_grids[other]).sum()
        ev_s, ev_h = _ev_group(n_own, n_other, q, self.n_players)
    else:
        probs = self._expected_teammate_probs()  # predicted P(stag) for each teammate
        ev_s = _ev_stag(probs, self.n_players)
        ev_h = _ev_hare(probs, self.n_players)
    # log-sum-exp trick for numerical stability
    m = max(ev_s, ev_h) / self.tau
    z_s = exp(ev_s / self.tau - m)
    z_h = exp(ev_h / self.tau - m)
    return "stag" if self._rng.random() < z_s / (z_s + z_h) else "hare"
```

**`_expected_teammate_probs()`** returns a list of length n-1, one predicted P(stag) per teammate, using the posterior for that teammate's key (`"pool"`, the teammate's tag, or the teammate's id):
```python
(self._posteriors[key] * self._p_stag_grids[key]).sum()
```
This is the posterior-weighted probability that the teammate hunts stag, i.e. the teammate's decision rule simulated forward. It is not the posterior mean of θ.

### Complexity measure

Entropy of each posterior, averaged across tracked entities, normalized to [0, 1]:

```python
def complexity(self) -> float:
    if self.group_mode:   # only the other tag's posterior drives the decision
        posteriors = [self._posteriors[self._other_tag(self.own_tag)]]
    else:
        posteriors = list(self._posteriors.values())
    entropies = []
    for posterior in posteriors:
        p = posterior.flatten()
        p = p[p > 0]
        h = -np.sum(p * np.log(p))              # raw entropy (nats)
        h_max = np.log(posterior.size)           # max entropy = log(n_grid^d)
        entropies.append(h / h_max if h_max > 0 else 0.0)
    return float(np.mean(entropies))
```

Because posteriors exist only for teammates' tags, this averages over beliefs the agent uses. It naturally captures the dimensionality difference:
- Anonymous: normalizes by log(n_grid) — 1D max entropy
- Tag-based (individual rule): normalizes by log(n_grid²) = 2*log(n_grid) — 2D max entropy
- Tag-based (group rule): normalizes by log(n_grid) — 1D max entropy
- Identity N=5: normalizes by log(n_grid⁴) = 4*log(n_grid) — 4D max entropy

The normalized complexity is still in [0, 1] and comparable across conditions.

### Reset

```python
def reset(self) -> None:
    for key in self._posteriors:
        d = len(self._posteriors[key].shape)
        self._posteriors[key] = self._uniform(d)
```

When a teammate j is replaced, every remaining agent's `teammate_replaced(j)` is called. In Identity, i resets its posterior about j to uniform. In Anonymous and Tag-based, posteriors are pooled across several players, so nothing changes. i's posteriors about other teammates k are not adjusted, even though they include k's beliefs about j. The likelihood is symmetric across a posterior's axes (k's choice depends only on the mix of its beliefs), so those posteriors stay symmetric and don't record which of k's beliefs is about j.

---

## 4. Implementation (completed)

The original build plan (shared EV helpers → `BayesianToM` class → exports → `model_type` in `game.py` → UI selectors) has been carried out. Where each piece lives:

| Piece | Location |
|-------|----------|
| Shared EV helpers (scalar, vectorized, group rule) | `role_games/models/_ev.py` |
| BToM model | `role_games/models/bayesian_tom.py` (`BayesianToM`) |
| Model exports | `role_games/models/__init__.py` |
| `model_type`, `decision_rule`, `include_own_action` passed down the call chain | `role_games/game.py`: `_make_agents` → `run_simulation` → `run_multiple` → `run_all_conditions` |
| UI selectors | `Simulate.py` sidebar; `Play.py` → Agent parameters (Tag-based options shown only for Tag-based games) |

**How the build differs from the original plan**
- Constructor signature: `BayesianToM(condition, own_agent_id, own_tag, tau, n_players, teammates_info, discount=1.0, n_grid=10, decision_rule="individual", rng=None, include_own_action=False)`.
- `n_grid` defaults to 10, not 20, for speed. Robustness at 20 and 50 is still to be checked.
- Instead of the planned `_b_j` / `_likelihood_grid` / `_expected_p_stag` helpers, P(stag | θ) grids are precomputed once per posterior (`_precompute_p_stag_grids`) and reused for both updating and prediction.
- Prediction uses the posterior-weighted P(stag | θ), not the marginal mean of θ (see Section 3).
- Posteriors exist only for teammates' tags. The own-action update is optional and off by default, and there is an optional group decision rule (Section 8).

---

## 5. Numpy Vectorization Notes

The trickiest part is computing the likelihood grid efficiently. Here is the conceptual approach for each condition.

### Anonymous (1D grid)

```python
# theta_vals: shape (n_grid,)
# For each theta, j believes all (n-1) teammates stag with prob theta
# Compute EV_stag and EV_hare for each theta value

ev_stag_grid = np.array([_ev_stag([t]*(n-1), n) for t in theta_vals])
ev_hare_grid = np.array([_ev_hare([t]*(n-1), n) for t in theta_vals])
# shape: (n_grid,)

# Softmax
m = np.maximum(ev_stag_grid, ev_hare_grid) / tau
p_stag = np.exp(ev_stag_grid/tau - m) / (np.exp(ev_stag_grid/tau - m) + np.exp(ev_hare_grid/tau - m))
# shape: (n_grid,)

likelihood = p_stag if action == "stag" else (1 - p_stag)
```

### Tag-based (2D grid)

```python
# theta_red_vals, theta_blue_vals: each shape (n_grid,)
# meshgrid: shape (n_grid, n_grid)
TR, TB = np.meshgrid(theta_vals, theta_vals, indexing="ij")
# TR[i,j] = theta_red at grid point (i,j)
# TB[i,j] = theta_blue at grid point (i,j)

# For j with tag "red" whose teammates are [red_teammate, blue_teammate, blue_teammate]:
# b_j = [TR, TB, TB]  (using TR because first teammate is red, TB for blue ones)
# EV_stag needs Poisson-Binomial with these beliefs

# Build teammate_probs as a list of 2D arrays aligned on the grid
teammate_probs = []
for tm_tag in j_teammate_tags:
    if tm_tag == "red":
        teammate_probs.append(TR)
    else:
        teammate_probs.append(TB)

# Then vectorize _poisson_binomial_pmf over the grid — this requires a
# vectorized PB implementation (see below)
```

### Identity ((n-1)D grid)

```python
# grids: (n_grid,)^(n-1) shaped via np.meshgrid or np.indices
# Each dimension corresponds to one teammate's stag probability in j's belief

# For N=3 (2D grid): same as tag-based but indexed by agent_id instead of tag
# For N=4 (3D grid): np.meshgrid with 3 arrays
# For N=5 (4D grid): np.meshgrid with 4 arrays

grids = np.meshgrid(*[theta_vals]*(n-1), indexing="ij")
# grids[k] has shape (n_grid,)^(n-1), represents j's belief about teammate k
```

### Tag-based, group rule (1D grid)

```python
# theta_vals: shape (n_grid,) = this tag's belief that the other tag hunts stag
ev_s, ev_h = _ev_group(n_this_tag, n_other_tag, theta_vals, n_players)
# _ev_group is linear in θ, so it works directly on the array; no Poisson-Binomial needed
```

### Vectorized Poisson-Binomial PMF

The scalar `_poisson_binomial_pmf` takes a list of scalar probabilities. The grid-based approach uses a vectorized version (`_pb_pmf_vec` in `_ev.py`) that takes an array of probability-arrays and returns a PMF array over a batch dimension:

```python
def _pb_pmf_vec(prob_grids: list[np.ndarray]) -> np.ndarray:
    """
    prob_grids: list of M arrays, each with the same shape S
    Returns: array of shape (*S, M+1) where result[..., k] = P(sum == k)
    """
    shape = prob_grids[0].shape
    M = len(prob_grids)
    dp = np.zeros((*shape, M + 1))
    dp[..., 0] = 1.0
    for p in prob_grids:
        for k in range(M, 0, -1):
            dp[..., k] = dp[..., k] * (1 - p) + dp[..., k-1] * p
        dp[..., 0] *= (1 - p)
    return dp  # shape (*S, M+1)
```

This is the most important performance-critical function. With numpy vectorization it handles the full grid in one pass.

---

## 6. Parameter Notes

- `n_grid = 10` is the current default (chosen for speed; robustness at 20 and 50 is still to be checked). Larger grids are more accurate but slower. Sizes at the default:
  - Anonymous: 10 points, negligible
  - Tag-based (individual rule): 100 points per posterior × 2 posteriors = 200 points total, fast
  - Tag-based (group rule): 10 points per posterior × 2 = 20, negligible
  - Identity N=3: 100 per posterior × 2 = 200, fast
  - Identity N=4: 1,000 per posterior × 3 = 3,000, fast
  - Identity N=5: 10,000 per posterior × 4 = 40,000 — the heaviest case
- `n_grid` is currently a constructor argument only. It is not passed through `run_simulation()` or exposed in the UI.
- `tau` (softmax temperature) applies to both i's own decision AND the likelihood model of j's decision. Use the same tau for both — this is the simplest assumption and means agents model others as having the same decision noise as themselves.
- `discount` behaves identically to FP: values < 1.0 decay the posterior toward uniform before each update.

---

## 7. Testing Suggestions

1. **Sanity check — anonymous BToM converges to FP**: With tau → 0 (deterministic), after many rounds of a fixed opponent playing stag every round, both FP and BToM should predict near-certainty of stag. Verify BToM's posterior concentrates on high θ values.

2. **Distinct from FP — early rounds**: With a player who switches strategy mid-game (plays hare for 5 rounds then stag for 5), compare FP vs BToM predictions. BToM should respond more sharply because it's fitting a belief model, not just averaging frequencies.

3. **Tag-based 2D posterior**: After observing red agents always stag and blue agents always hare, verify that the posterior over (θ^red, θ^blue) concentrates near (1, 0). Verify the marginal E[θ^red] ≈ 1.

4. **Complexity ordering**: Across conditions for the same agent and history, verify: identity > tag-based > anonymous (at least on average). This should hold because identity posteriors are higher-dimensional.

5. **Discount test**: With δ < 1 and an agent who switches from stag to hare, verify BToM adapts faster than with δ = 1.

6. **Backward compatibility**: `model_type="fictitious_play"` with new `_make_agents()` signature must produce identical results to the old signature. Add a test that runs a fixed-seed simulation with both and compares outputs.

---

## 8. Change Log

| Date | Change |
|------|--------|
| June 2026 | Initial build: `_ev.py` extracted from FP; `BayesianToM` added; `model_type` passed through `game.py`; model selectors in Simulate and Play. |
| October 2026 | **Group decision rule** (`decision_rule="group"`, Tag-based only): agents, and the agents they model, choose for their tag as a block. Tag posteriors become 1D, and complexity counts only the other tag's posterior. |
| October 2026 | **Own-action update made optional** (`include_own_action`, default `False`). Previously it was always on in Tag-based. The default now matches the other conditions: beliefs come only from others' choices. |
| October 2026 | **No posterior for a tag the agent is alone in.** This changed Tag-based complexity for red agents at N=3 under the individual rule; choices were unaffected. |
| October 2026 | **Replacement in Identity:** remaining agents now reset their posterior about the replaced agent (`teammate_replaced`). Before, they kept treating the newcomer as the person who left. This changes Identity results whenever `replacement_rate > 0`. |
| October 2026 | **Reproducible seeds:** choices now use the per-run seeded generator instead of Python's global `random`, so `base_seed` reproduces runs exactly. Seeded results from before this change will not match current output. |
