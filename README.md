# Role Games

Agent-based simulation of a modified stag hunt for small groups (3–5 players). Models how groups coordinate under three informational conditions — Anonymous, Identity, and Tag-based — using two agent models: Fictitious Play and depth-1 Bayesian Theory of Mind (BToM).

## Structure

```
app.py                       # Streamlit entry point (Play + Simulate pages)
Play.py                      # Play a game against simulated agents
Simulate.py                  # Run batches of simulations and plot results
simulate.ipynb               # Notebook for scripted runs
role_games/
├── game.py                  # Agent setup, round loop, payoffs, convergence tracking
├── agents.py                # Agent class + DecisionModel interface
├── conditions.py            # Condition enum
├── models/
│   ├── _ev.py               # Expected-value helpers (individual and group rules)
│   ├── fictitious_play.py   # Fictitious Play
│   └── bayesian_tom.py      # Bayesian Theory of Mind (depth-1, grid posteriors)
├── logger.py                # DataFrame construction and CSV export
└── visualize.py             # Plotting functions
role_games_spec.md           # Model specification
btom_implementation_spec.md  # BToM design and implementation reference
```

## Usage

### Streamlit app

```bash
streamlit run app.py
```

- **Play** — play one game against simulated agents in a chosen condition.
- **Simulate** — run many simulations of all three conditions side by side and plot payoff, convergence, complexity, and outcome proportions.

### Python / notebook

```python
from role_games import Condition, run_multiple, run_all_conditions, to_dataframes

# One condition
rounds, summary = run_multiple(Condition.TAG_BASED, n_simulations=100, n_players=4,
                               model_type="bayesian_tom", base_seed=0)

# All three conditions with shared parameters
rounds, summary = run_all_conditions(n_simulations=100, n_players=4, base_seed=0)
rounds_df, summary_df = to_dataframes(rounds, summary)
```

`simulate.ipynb` shows a full workflow with plots.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `condition` | Anonymous / Identity / Tag-based (`run_simulation`, `run_multiple`) | Required |
| `model_type` | `"fictitious_play"` or `"bayesian_tom"` | `"fictitious_play"` |
| `n_players` | Players per group (3–5). Tag-based split: 1 red / 2 blue (N=3), 2/2 (N=4), 2 red / 3 blue (N=5) | 3 |
| `n_rounds` | Rounds per simulation | 50 |
| `n_simulations` | Independent simulation runs | 100 |
| `k_convergence` | Consecutive optimal rounds (exactly 2 stag hunters) to mark convergence | 5 |
| `tau` | Softmax temperature (lower = more deterministic) | 0.1 |
| `discount` | Belief decay δ applied each round (1.0 = no forgetting) | 1.0 |
| `replacement_rate` | Probability each round that one random agent is replaced by a newcomer starting from the uniform prior. In Identity, the other agents also reset their belief about that player | 0.0 |
| `decision_rule` | Tag-based only: `"individual"` (choose own action, teammates independent) or `"group"` (each tag acts as a block; choose for my group given P(other tag hunts stag)) | `"individual"` |
| `include_own_action` | Tag-based, individual rule only: whether an agent's own choice also counts toward its own tag's belief (agents alone in their tag hold no belief about it) | `False` |
| `base_seed` | Run *i* uses seed `base_seed + i`. One seeded generator per run drives tag assignment, agent choices, and replacement, so the same seed reproduces a run exactly. `None` = unseeded | `None` |

These are the Python function defaults. The Simulate page starts at τ = 0.05, δ = 0.90, 30 rounds, and 300 simulations.

### Tag-based options

With the defaults (`decision_rule="individual"`, `include_own_action=False`), the Tag-based condition uses the same belief-formation and decision process as the other conditions. Beliefs come only from teammates' choices, and each agent chooses its own action. The only difference is that beliefs are grouped by tag. The two switches each add one tag-specific assumption, so their effects can be tested separately. See `role_games_spec.md` §4.5.

## Output

Each round record has one row per agent per round: condition and settings (`n_players`, `replacement_rate`, `model_type`, `decision_rule`, `include_own_action`), `round`, `agent_id`, `agent_tag`, `decision`, `stag_count`, `group_total`, `agent_payoff`, `complexity`, `replaced`, and `converged`. `decision_rule` and `include_own_action` record what was actually used, so non-Tag-based rows always show `"individual"` and `False`.

Each summary record has one row per simulation, with `total_payoff` (cumulative per-player payoff) and `mean_complexity`.

## Exporting data

```python
from role_games import export_csv
export_csv(rounds_df, summary_df, output_dir='data')
```

Writes `simulation_rounds.csv` and `simulation_summary.csv` for analysis in R.

## Dependencies

```
pip install -r requirements.txt
```
