# Bayesian Theory of Mind — Implementation Specification

This document is a self-contained spec for implementing a Bayesian Theory of Mind (BToM) agent model alongside the existing Fictitious Play (FP) model. It includes all necessary context about the existing codebase, the full mathematical design, implementation instructions, and integration notes.

---

## 1. Codebase Overview

### File structure (relevant files)

```
role-games-model/
├── app.py                          # Navigation router (st.navigation)
├── Simulate.py                     # Simulation page
├── Play.py                         # Human play page
├── role_games/
│   ├── __init__.py                 # Exports run_all_conditions, to_dataframes
│   ├── conditions.py               # Condition enum
│   ├── agents.py                   # Agent class + DecisionModel interface
│   ├── game.py                     # _make_agents(), run_simulation(), etc.
│   └── models/
│       ├── __init__.py             # Exports FictitiousPlay
│       └── fictitious_play.py      # Existing FP model
```

### Key classes and interfaces

**`DecisionModel` (agents.py)** — abstract base class all models must implement:
```python
class DecisionModel:
    def decide(self) -> str:          # returns "stag" or "hare"
    def update(self, observation: dict) -> None
    def complexity(self) -> float     # returns float in [0, 1]
    def reset(self) -> None           # resets to uniform prior
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
```

**`Condition` enum (conditions.py)**:
```python
class Condition(Enum):
    ANONYMOUS = "anonymous"
    IDENTITY  = "identity"
    TAG_BASED = "tag_based"
```

**`_make_agents()` (game.py)** — current signature:
```python
def _make_agents(
    condition: Condition,
    tau: float,
    n_players: int,
    rng: random.Random,
    discount: float = 1.0,
) -> list[Agent]
```
Needs to be extended to accept `model_type`.

**Tag assignment (game.py)**:
- N=3: 1 red, 2 blue
- N=4: 2 red, 2 blue
- N=5: 2 red, 3 blue
- Tags randomly shuffled each run

**Existing EV helpers (fictitious_play.py)** — reuse these in BToM:
```python
def _poisson_binomial_pmf(probs: list[float]) -> list[float]
def _ev_stag(teammate_probs: list[float], n_players: int) -> float
def _ev_hare(teammate_probs: list[float], n_players: int) -> float
```
Consider moving these to a shared `role_games/models/_ev.py` file so both models can import them.

**Existing complexity helper (fictitious_play.py)**:
```python
def _normalized_entropy(p: float) -> float   # binary entropy / log(2)
```

**`run_simulation()` (game.py)** — current signature:
```python
def run_simulation(
    condition, n_rounds, k_convergence, tau, n_players,
    replacement_rate, discount, sim_id, seed
) -> list[dict]
```
Needs `model_type` parameter added and passed through to `_make_agents()`.

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
| Tag-based | Per-tag stag rates | 2D: (θ^red, θ^blue) ∈ [0,1]² |
| Identity  | Per-individual stag rates | (n−1)D: (θ¹,…,θⁿ⁻¹) ∈ [0,1]^{n-1} |

### What i maintains (one posterior per tracked entity)

**Anonymous**: one posterior over θ (1D), representing the shared pool belief.
- All observations update the same posterior.

**Tag-based**: two posteriors, each 2D — one for "what red agents believe" and one for "what blue agents believe."
- Observations from a red agent update the red posterior.
- Observations from a blue agent update the blue posterior.
- Each posterior lives in (θ^red, θ^blue) space because a tag-based agent tracks both tags.

**Identity**: (n−1) posteriors, each (n−1)D — one per other agent j.
- Observations from agent j update j's individual posterior.
- Each posterior lives in (θ^{id_0}, θ^{id_1}, ..., θ^{id_{n-2}}) space because an identity agent tracks each individual.
- This includes a dimension for i itself (j's belief about what i will do), which i treats as unknown and infers from j's actions.

### Why identity is genuinely more complex

- Anonymous: 1 posterior × 1 dimension = 1 total belief dimension
- Tag-based: 2 posteriors × 2 dimensions = 4 total belief dimensions
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
# e.g., n_grid=20: [0.025, 0.075, ..., 0.975]
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

This is vectorized over all grid points simultaneously using numpy broadcasting.

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

**Own action (tag-based only):** Same as FP — also update own tag's posterior using own action and own belief vector. This keeps the self-inclusive logic consistent with FP.

### Discount factor

Before each update (once per round, called via `_decay_posterior()`), dilute the posterior back toward the uniform prior:

```python
def _decay_posterior(self) -> None:
    if self.discount == 1.0:
        return
    n_points = self._posteriors[key].size
    uniform = np.ones(self._posteriors[key].shape) / n_points
    for key in self._posteriors:
        log_p = np.log(self._posteriors[key] + 1e-300)
        log_u = np.log(uniform)
        log_decayed = self.discount * log_p + (1 - self.discount) * log_u
        p = np.exp(log_decayed - log_decayed.max())
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
    probs = self._expected_teammate_probs()  # list of E[P(stag)] for each teammate
    ev_s = _ev_stag(probs, self.n_players)
    ev_h = _ev_hare(probs, self.n_players)
    # log-sum-exp trick for numerical stability
    m = max(ev_s, ev_h) / self.tau
    z_s = exp(ev_s / self.tau - m)
    z_h = exp(ev_h / self.tau - m)
    return "stag" if random.random() < z_s / (z_s + z_h) else "hare"
```

**`_expected_teammate_probs()`** returns a list of length n-1:
- Anonymous: `[E[θ_pool]] * (n-1)` where E[θ_pool] = sum(theta_vals * posterior_pool)
- Tag-based: `[E[θ^{tag_k}] for each teammate k]` — use the marginal expectation for each tag from the appropriate posterior
- Identity: `[E[θ^{id_k}] for each teammate k]` — use the marginal expectation for each individual

For tag-based and identity, the marginal expectation of dimension d from a multi-dimensional posterior is computed by summing/integrating out all other dimensions:
```python
# e.g., marginal for red from 2D posterior of shape (n_grid, n_grid):
# where dim 0 = theta_red, dim 1 = theta_blue
marginal_red = posterior.sum(axis=1)  # sum over theta_blue axis
E_theta_red = (marginal_red * theta_vals).sum()
```

### Complexity measure

Entropy of each posterior, averaged across tracked entities, normalized to [0, 1]:

```python
def complexity(self) -> float:
    entropies = []
    for key, posterior in self._posteriors.items():
        p = posterior.flatten()
        p = p[p > 0]
        h = -np.sum(p * np.log(p))              # raw entropy (nats)
        h_max = np.log(posterior.size)           # max entropy = log(n_grid^d)
        entropies.append(h / h_max if h_max > 0 else 0.0)
    return sum(entropies) / len(entropies)
```

This naturally captures the dimensionality difference:
- Anonymous: normalizes by log(n_grid) — 1D max entropy
- Tag-based: normalizes by log(n_grid²) = 2*log(n_grid) — 2D max entropy
- Identity N=5: normalizes by log(n_grid⁴) = 4*log(n_grid) — 4D max entropy

The normalized complexity is still in [0, 1] and comparable across conditions.

### Reset

```python
def reset(self) -> None:
    for key in self._posteriors:
        shape = self._posteriors[key].shape
        self._posteriors[key] = np.ones(shape) / np.prod(shape)
```

---

## 4. Implementation Plan

### Step 1 — Refactor shared EV helpers

Move `_poisson_binomial_pmf`, `_ev_stag`, `_ev_hare` from `fictitious_play.py` to a new file `role_games/models/_ev.py`. Update `fictitious_play.py` to import from there. `bayesian_tom.py` will also import from there.

### Step 2 — Create `role_games/models/bayesian_tom.py`

```python
import math
import random
import numpy as np
from ..agents import DecisionModel
from ..conditions import Condition
from ._ev import _ev_stag, _ev_hare

class BayesianToM(DecisionModel):
    def __init__(
        self,
        condition: Condition,
        own_tag: str | None,
        tau: float,
        n_players: int,
        teammates_info: list[dict],   # same format as FP
        discount: float = 1.0,
        n_grid: int = 20,
    ):
        self.condition = condition
        self.own_tag = own_tag
        self.tau = tau
        self.n_players = n_players
        self.discount = discount
        self.n_grid = n_grid

        self._theta_vals = np.linspace(
            1 / (2 * n_grid), 1 - 1 / (2 * n_grid), n_grid
        )
        self._teammate_ids  = [t["agent_id"] for t in teammates_info]
        self._teammate_tags = [t["tag"] for t in teammates_info]

        self._posteriors: dict[str, np.ndarray] = {}
        self._init_posteriors(teammates_info)

    def _init_posteriors(self, teammates_info):
        # Anonymous: one 1D posterior for the pool
        if self.condition == Condition.ANONYMOUS:
            self._posteriors["pool"] = self._uniform(1)

        # Tag-based: one 2D posterior per distinct tag (red + blue)
        elif self.condition == Condition.TAG_BASED:
            distinct_tags = sorted({t["tag"] for t in teammates_info})
            for tag in distinct_tags:
                self._posteriors[tag] = self._uniform(2)

        # Identity: one (n-1)D posterior per individual teammate
        else:  # IDENTITY
            d = len(teammates_info)   # = n_players - 1
            for t in teammates_info:
                self._posteriors[str(t["agent_id"])] = self._uniform(d)

    def _uniform(self, d: int) -> np.ndarray:
        shape = (self.n_grid,) * d
        n = self.n_grid ** d
        return np.ones(shape) / n
```

The remaining methods (`decide`, `update`, `complexity`, `reset`, `_decay_posteriors`) follow the mathematical spec in section 3.

**Key internal helpers to implement:**

`_b_j(theta_grid, j_tag, j_teammate_tags_or_ids)` — given a grid of θ values, compute the teammate belief list for agent j. This is the core vectorized operation and requires careful numpy broadcasting. See section 3 for the per-condition logic.

`_likelihood_grid(key, j_tag, j_teammate_info, action)` — compute P(action | θ) over the entire grid for entity `key`. Returns an array of the same shape as `_posteriors[key]`.

`_expected_p_stag(key, teammate_tag_or_id)` — compute the marginal expected P(stag) for a given teammate from the posterior stored at `key`. Uses marginal summation for multi-dimensional posteriors.

### Step 3 — Update `role_games/models/__init__.py`

```python
from .fictitious_play import FictitiousPlay
from .bayesian_tom import BayesianToM

__all__ = ["FictitiousPlay", "BayesianToM"]
```

### Step 4 — Update `role_games/game.py`

Add `model_type: str = "fictitious_play"` parameter to `_make_agents()`, `run_simulation()`, `run_multiple()`, and `run_all_conditions()`. Pass it through the call chain.

In `_make_agents()`:
```python
from .models import FictitiousPlay, BayesianToM

def _make_agents(condition, tau, n_players, rng, discount=1.0, model_type="fictitious_play"):
    ...
    for i, (aid, tag) in enumerate(zip(agent_ids, tags)):
        teammates_info = [...]
        if model_type == "fictitious_play":
            model = FictitiousPlay(condition, tag, tau, n_players, teammates_info, discount=discount)
        else:
            model = BayesianToM(condition, tag, tau, n_players, teammates_info, discount=discount)
        agents.append(Agent(aid, tag, model))
```

### Step 5 — Update `Simulate.py`

Add a model type selector to the sidebar (after the condition checkboxes or before parameters):
```python
model_type = st.selectbox(
    "Agent model",
    ["Fictitious Play", "Bayesian ToM"],
    key="model_type",
    help="FP tracks action frequencies. BToM models what other agents believe.",
)
model_type_key = "fictitious_play" if model_type == "Fictitious Play" else "bayesian_tom"
```
Pass `model_type=model_type_key` to `run_all_conditions()`.

Add `"model_type": "fictitious_play"` to `PARAM_DEFAULTS`.

### Step 6 — Update `Play.py`

Add model type selector to `_sidebar()`:
```python
st.selectbox("Agent model", ["Fictitious Play", "Bayesian ToM"],
             key="play_model_type")
```

In `_init_game()`, read `st.session_state.play_model_type` and pass the appropriate `model_type` string to `_make_agents()`.

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

### Vectorized Poisson-Binomial PMF

The existing `_poisson_binomial_pmf` takes a list of scalar probabilities. For the grid-based approach, you need a vectorized version that takes an array of probability-arrays and returns a PMF array over a batch dimension:

```python
def _poisson_binomial_pmf_vectorized(prob_grids: list[np.ndarray]) -> np.ndarray:
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

- `n_grid = 20` is the recommended default. Larger grids are more accurate but slower.
  - Anonymous: 20 points, negligible
  - Tag-based: 400 points per posterior × 2 posteriors = 800 points total, fast
  - Identity N=3: 400 per posterior × 2 = 800, fast
  - Identity N=4: 8,000 per posterior × 3 = 24,000, fast
  - Identity N=5: 160,000 per posterior × 4 = 640,000 — heavier but still feasible
- `n_grid` could be exposed as a configurable parameter in the UI (under "Agent parameters") if performance is a concern.
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

## 8. Summary of What Changes

| File | Change |
|------|--------|
| `role_games/models/_ev.py` | **New** — shared EV helpers extracted from FP |
| `role_games/models/bayesian_tom.py` | **New** — BToM model class |
| `role_games/models/__init__.py` | Export `BayesianToM` |
| `role_games/models/fictitious_play.py` | Import EV helpers from `_ev.py` |
| `role_games/game.py` | Add `model_type` param throughout call chain |
| `Simulate.py` | Add model type selector to sidebar + PARAM_DEFAULTS |
| `Play.py` | Add model type selector to sidebar |
