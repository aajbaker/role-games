import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CONDITION_ORDER = ["Anonymous", "Identity", "Tag-based"]
PALETTE = {"Anonymous": "#4878CF", "Identity": "#6ACC65", "Tag-based": "#D65F5F"}

# Outcome categories by stag count — generalises to any N
OUTCOME_ORDER = ["2 Stag", "3+ Stag", "1 Stag", "All-Hare"]
OUTCOME_COLORS = {
    "2 Stag":   "#4CAF50",   # optimal — green
    "3+ Stag":  "#1565C0",   # excess coordination — dark blue
    "1 Stag":   "#FF9800",   # stag fails — orange
    "All-Hare": "#9E9E9E",   # no coordination — gray
}


def _classify_outcome(stag_count: int) -> str:
    if stag_count == 0:
        return "All-Hare"
    elif stag_count == 1:
        return "1 Stag"
    elif stag_count == 2:
        return "2 Stag"
    else:
        return "3+ Stag"


def _multi(df: pd.DataFrame) -> bool:
    return df["condition"].nunique() > 1


def plot_group_total(rounds_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Mean group total by round, one line per condition."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    trio_rounds = (
        rounds_df.groupby(["condition", "simulation_id", "round"])["group_total"]
        .first()
        .reset_index()
    )
    mean_by_round = (
        trio_rounds.groupby(["condition", "round"])["group_total"]
        .mean()
        .reset_index()
    )

    if _multi(rounds_df):
        sns.lineplot(
            data=mean_by_round, x="round", y="group_total",
            hue="condition", hue_order=CONDITION_ORDER, palette=PALETTE,
            linewidth=2, ax=ax,
        )
    else:
        cond = rounds_df["condition"].iloc[0]
        ax.plot(
            mean_by_round["round"], mean_by_round["group_total"],
            color=PALETTE.get(cond, "steelblue"), linewidth=2, label=cond,
        )

    n_players = int(rounds_df["n_players"].iloc[0])
    optimal = n_players + 2   # 2 stag + rest hare
    all_hare = n_players      # all hare
    ax.axhline(optimal, color="green", linestyle="--", linewidth=1, label=f"Optimal ({optimal} pts)")
    ax.axhline(all_hare, color="gray", linestyle=":", linewidth=1, label=f"All-hare ({all_hare} pts)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean group total")
    ax.set_title("Group Payoff Over Rounds")
    ax.legend(title="Condition" if _multi(rounds_df) else None)
    return ax


def plot_complexity(rounds_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Mean agent complexity by round, one line per condition."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    mean_complexity = (
        rounds_df.groupby(["condition", "round"])["complexity"]
        .mean()
        .reset_index()
    )

    if _multi(rounds_df):
        sns.lineplot(
            data=mean_complexity, x="round", y="complexity",
            hue="condition", hue_order=CONDITION_ORDER, palette=PALETTE,
            linewidth=2, ax=ax,
        )
    else:
        cond = rounds_df["condition"].iloc[0]
        ax.plot(
            mean_complexity["round"], mean_complexity["complexity"],
            color=PALETTE.get(cond, "purple"), linewidth=2,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean complexity (normalized entropy)")
    ax.set_ylim(0, 1)
    ax.set_title("Agent Complexity Over Rounds")
    if _multi(rounds_df):
        ax.legend(title="Condition")
    return ax


def plot_convergence_rate(
    rounds_df: pd.DataFrame, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Proportion of simulations in a converged state at each round.
    Converged = last k_convergence rounds all had stag_count == 2.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    trio = (
        rounds_df.groupby(["condition", "simulation_id", "round"])["converged"]
        .first()
        .reset_index()
    )
    conv_rate = (
        trio.groupby(["condition", "round"])["converged"]
        .mean()
        .reset_index()
    )

    if _multi(rounds_df):
        sns.lineplot(
            data=conv_rate, x="round", y="converged",
            hue="condition", hue_order=CONDITION_ORDER, palette=PALETTE,
            linewidth=2, ax=ax,
        )
    else:
        cond = rounds_df["condition"].iloc[0]
        ax.plot(
            conv_rate["round"], conv_rate["converged"],
            color=PALETTE.get(cond, "steelblue"), linewidth=2,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Proportion converged")
    ax.set_ylim(0, 1)
    ax.set_title("Convergence Rate Over Rounds")
    if _multi(rounds_df):
        ax.legend(title="Condition")
    return ax


def plot_outcome_proportions(rounds_df: pd.DataFrame) -> plt.Figure:
    """
    Stacked area chart of outcome proportions by round.
    Outcomes classified by stag_count: 2 Stag (optimal), 3+ Stag, 1 Stag, All-Hare.
    One panel per condition when multiple conditions are present.
    """
    conditions = [c for c in CONDITION_ORDER if c in rounds_df["condition"].unique()]
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    trio = (
        rounds_df.groupby(["condition", "simulation_id", "round"])["stag_count"]
        .first()
        .reset_index()
    )
    trio["outcome"] = trio["stag_count"].map(_classify_outcome)

    for ax, cond in zip(axes, conditions):
        df = trio[trio["condition"] == cond]
        counts = (
            df.groupby(["round", "outcome"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=OUTCOME_ORDER, fill_value=0)
        )
        props = counts.div(counts.sum(axis=1), axis=0)

        ax.stackplot(
            props.index,
            [props[o] for o in OUTCOME_ORDER],
            labels=OUTCOME_ORDER,
            colors=[OUTCOME_COLORS[o] for o in OUTCOME_ORDER],
            alpha=0.85,
        )
        ax.set_xlim(props.index.min(), props.index.max())
        ax.set_ylim(0, 1)
        ax.set_xlabel("Round")
        ax.set_title(cond)
        if ax is axes[0]:
            ax.set_ylabel("Proportion of outcomes")
        if ax is axes[-1]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    fig.suptitle("Outcome Proportions Over Rounds", y=1.02)
    fig.tight_layout()
    return fig


def plot_total_payoff(rounds_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Bar chart of mean cumulative group payoff per condition, averaged across simulations.
    Error bars show ±1 standard error.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    # One total-payoff value per simulation: sum of group_total across all rounds
    sim_totals = (
        rounds_df.groupby(["condition", "simulation_id", "round"])["group_total"]
        .first()
        .groupby(["condition", "simulation_id"])
        .sum()
        .reset_index(name="total_payoff")
    )

    conditions = [c for c in CONDITION_ORDER if c in sim_totals["condition"].unique()]
    means = sim_totals.groupby("condition")["total_payoff"].mean()
    sems = sim_totals.groupby("condition")["total_payoff"].sem()

    bars = ax.bar(
        [conditions.index(c) for c in conditions],
        [means[c] for c in conditions],
        yerr=[sems[c] for c in conditions],
        color=[PALETTE[c] for c in conditions],
        capsize=5,
        width=0.5,
        error_kw={"linewidth": 1.5},
    )
    # Zoom y-axis to the range of the data so differences are visible
    vals = [means[c] for c in conditions]
    errs = [sems[c] for c in conditions]
    ymin = min(v - e for v, e in zip(vals, errs))
    ymax = max(v + e for v, e in zip(vals, errs))
    margin = (ymax - ymin) * 1.5 or ymax * 0.02
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean total group payoff")
    ax.set_title("Total Payoff by Condition")
    return ax


def plot_summary(rounds_df: pd.DataFrame) -> plt.Figure:
    """
    Four-panel summary figure:
      Top row: group payoff | complexity | convergence rate
      Bottom row: outcome proportions (one panel per condition)
    """
    conditions = [c for c in CONDITION_ORDER if c in rounds_df["condition"].unique()]
    n_conds = len(conditions)

    fig = plt.figure(figsize=(20, 9))

    # Top row: 4 panels
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 4, 3)
    ax4 = fig.add_subplot(2, 4, 4)
    plot_group_total(rounds_df, ax=ax1)
    plot_complexity(rounds_df, ax=ax2)
    plot_convergence_rate(rounds_df, ax=ax3)
    plot_total_payoff(rounds_df, ax=ax4)

    # Bottom row: one outcome-proportions panel per condition
    trio = (
        rounds_df.groupby(["condition", "simulation_id", "round"])["stag_count"]
        .first()
        .reset_index()
    )
    trio["outcome"] = trio["stag_count"].map(_classify_outcome)

    for i, cond in enumerate(conditions):
        ax = fig.add_subplot(2, max(n_conds, 4), max(n_conds, 4) + i + 1)
        df = trio[trio["condition"] == cond]
        counts = (
            df.groupby(["round", "outcome"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=OUTCOME_ORDER, fill_value=0)
        )
        props = counts.div(counts.sum(axis=1), axis=0)

        ax.stackplot(
            props.index,
            [props[o] for o in OUTCOME_ORDER],
            labels=OUTCOME_ORDER,
            colors=[OUTCOME_COLORS[o] for o in OUTCOME_ORDER],
            alpha=0.85,
        )
        ax.set_xlim(props.index.min(), props.index.max())
        ax.set_ylim(0, 1)
        ax.set_xlabel("Round")
        ax.set_title(f"Outcomes — {cond}")
        if i == 0:
            ax.set_ylabel("Proportion")
        if i == n_conds - 1:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    return fig
