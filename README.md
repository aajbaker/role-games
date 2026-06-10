# Role Games

Agent-based simulation of a 3-player modified stag hunt. Models how small groups coordinate under three informational conditions — Anonymous, Identity, and Tag-based — using Fictitious Play with softmax action selection.

## Structure

```
role_games/
├── game.py                  # Round loop, payoff calculation, convergence tracking
├── agents.py                # Agent class + DecisionModel interface
├── models/
│   └── fictitious_play.py   # Fictitious Play implementation
├── conditions.py            # Condition enum
├── logger.py                # DataFrame construction and CSV export
└── visualize.py             # Plotting functions

simulate.ipynb               # Main working notebook
```

## Usage

Open `simulate.ipynb` in VS Code or Jupyter. Set parameters at the top of the notebook, then either:

- Run a **single condition** using `run_multiple(condition=..., ...)`
- Run **all three conditions** at once using `run_all_conditions(...)`

Both return round-level and summary-level records that feed directly into the plotting functions.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `condition` | Anonymous / Identity / Tag-based | Required |
| `n_rounds` | Rounds per simulation | 50 |
| `k_convergence` | Consecutive optimal rounds to mark convergence | 5 |
| `n_simulations` | Independent simulation runs | 100 |
| `tau` | Softmax temperature (lower = more rational) | 0.1 |

## Exporting data

```python
from role_games import export_csv
export_csv(rounds_df, summary_df, output_dir='data')
```

Writes `simulation_rounds.csv` and `simulation_summary.csv` for analysis in R.

## Dependencies

```
pip install pandas matplotlib seaborn
```
