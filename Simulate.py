import sys
import time

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, ".")
from role_games import run_all_conditions, to_dataframes
from role_games.visualize import (
    plot_complexity,
    plot_convergence_rate,
    plot_group_total,
    plot_outcome_proportions,
    plot_total_payoff,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

st.markdown("### Role Games Simulator")
st.caption("Modified stag hunt under three informational conditions.")

_desc = st.empty()
if "sim_has_run" not in st.session_state:
    _desc.markdown(
        '<div style="padding:14px 18px;border-radius:8px;background:#f8f8f8;border:1px solid #ddd;margin-bottom:4px;">'
        '<b>How the simulation works</b>'
        '<p style="color:#444;font-size:0.9em;margin:8px 0 6px 0;">'
        'Each run simulates <b>N independent groups</b> of agents playing the stag hunt over multiple rounds. '
        'All three informational conditions run in parallel so you can compare them directly.'
        '</p>'
        '<b style="font-size:0.9em;">Agent model — Fictitious Play</b>'
        '<p style="color:#444;font-size:0.9em;margin:6px 0 6px 0;">'
        'Agents track how often others have chosen stag in the past and use that to estimate what others will do next round. '
        'They pick the action with the higher expected payoff, with some randomness controlled by temperature (τ). '
        'A discount factor (δ) lets agents weight recent rounds more heavily.'
        '</p>'
        '<b style="font-size:0.9em;">Informational conditions</b>'
        '<ul style="color:#444;font-size:0.9em;margin:6px 0 0 0;padding-left:18px;line-height:1.8em;">'
        '<li><b>Anonymous</b> — agents see only the aggregate count of stag/hare choices each round</li>'
        '<li><b>Identity</b> — agents track each individual separately, building a belief per person</li>'
        '<li><b>Tag-based</b> — agents track choices by tag color (red / blue), one belief per tag</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

PLOT_REGISTRY = [
    ("Payoff by round",  plot_group_total),
    ("Total payoff",     plot_total_payoff),
    ("Convergence rate", plot_convergence_rate),
    ("Complexity",       plot_complexity),
]
ALL_NAMES = [name for name, _ in PLOT_REGISTRY] + ["Choices by round"]
PLOT_DEFAULTS = {"Payoff by round", "Total payoff"}

# ------------------------------------------------------------------
# Parameter defaults & session-state initialisation
# ------------------------------------------------------------------
PARAM_DEFAULTS: dict = {
    "n_players":        3,
    "replacement_rate": 0.0,
    "tau":              0.1,
    "discount":         1.0,
    "n_rounds":         30,
    "n_simulations":    300,
    "speed":            "Medium",
    "model_type":       "Fictitious Play",
}
for _k, _v in PARAM_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


st.caption("Show plots")
toggle_cols = st.columns(len(ALL_NAMES))
checks = {
    name: toggle_cols[i].checkbox(name, value=(name in PLOT_DEFAULTS), key=f"chk_{i}")
    for i, name in enumerate(ALL_NAMES)
}

selected_plots = [(name, fn) for name, fn in PLOT_REGISTRY if checks[name]]
show_outcomes = checks["Choices by round"]

st.divider()

# ------------------------------------------------------------------
# Sidebar — parameters only
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")

    model_type = st.selectbox(
        "Agent model",
        ["Fictitious Play", "Bayesian ToM"],
        key="model_type",
        help=(
            "Fictitious Play tracks action frequencies. "
            "Bayesian ToM models what other agents believe. "
            "⚠️ BToM with Identity + N=5 is compute-heavy."
        ),
    )
    model_type_key = "fictitious_play" if model_type == "Fictitious Play" else "bayesian_tom"

    n_players = st.number_input(
        "Players", min_value=3, max_value=5, step=1,
        key="n_players",
        help="Number of agents per group (3–5). Tags are assigned as 1R/2B (N=3), 2R/2B (N=4), or 2R/3B (N=5).",
    )
    replacement_rate = st.slider(
        "Replacement rate", 0.0, 0.5, step=0.05, format="%.2f",
        key="replacement_rate",
        help="Probability each round that one agent is replaced: its belief history is wiped and it restarts with a uniform prior. Other agents are told who was replaced.",
    )
    tau = st.slider(
        "Temperature (τ)", 0.0, 0.2, step=0.01, format="%.2f",
        key="tau",
        help="Softmax temperature controlling how deterministically agents choose. Lower τ → agents follow expected value more strictly; higher τ → more random exploration.",
    )
    discount = st.slider(
        "Discount factor (δ)", 0.5, 1.0, step=0.05, format="%.2f",
        key="discount",
        help="Exponential decay applied to belief counts each round before the new observation is added. δ = 1 (default) = standard fictitious play with full memory. Lower values make agents weight recent observations more heavily and forget the past faster.",
    )
    n_rounds = st.slider(
        "Rounds", 5, 50, step=5,
        key="n_rounds",
        help="Number of rounds each simulation runs. All simulations complete the full number of rounds regardless of convergence.",
    )
    n_simulations = st.slider(
        "Simulations", 10, 500, step=10,
        key="n_simulations",
        help="Number of independent simulation runs per condition. More simulations produce smoother, more reliable averages.",
    )

    st.divider()
    speed = st.select_slider(
        "Animation speed", ["Slow", "Medium", "Fast"],
        key="speed",
    )
    run_button = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    if st.button("↺  Reset defaults", use_container_width=True):
        st.components.v1.html("<script>window.location.reload()</script>", height=0)

DELAY = {"Slow": 0.25, "Medium": 0.08, "Fast": 0.01}[speed]

# ------------------------------------------------------------------
# Run + animate
# ------------------------------------------------------------------
if run_button:
    st.session_state["sim_has_run"] = True
    _desc.empty()

    if not selected_plots and not show_outcomes:
        st.warning("Select at least one plot to display.")
        st.stop()

    effective_tau = max(tau, 1e-4)

    with st.spinner(f"Running {n_simulations} simulations × {n_rounds} rounds…"):
        rr, sr = run_all_conditions(
            n_simulations=n_simulations,
            n_rounds=n_rounds,
            tau=effective_tau,
            n_players=n_players,
            replacement_rate=replacement_rate,
            discount=discount,
            model_type=model_type_key,
        )

    rounds_df, _ = to_dataframes(rr, sr)
    status = st.empty()

    # Always 2-column rows; single plots sit in the left column only
    plot_placeholders = []
    for chunk in [selected_plots[i:i + 2] for i in range(0, len(selected_plots), 2)]:
        col1, col2 = st.columns(2)
        plot_placeholders.append(col1.empty())
        if len(chunk) == 2:
            plot_placeholders.append(col2.empty())

    ph_outcomes = st.empty() if show_outcomes else None
    fig_height = 5 if len(selected_plots) <= 2 else 4.5

    for t in range(1, n_rounds + 1):
        df_t = rounds_df[rounds_df["round"] <= t]
        status.caption(f"Round {t} / {n_rounds}")

        for (_, fn), ph in zip(selected_plots, plot_placeholders):
            fig, ax = plt.subplots(figsize=(5, fig_height))
            fn(df_t, ax=ax)
            fig.tight_layout()
            ph.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if show_outcomes and ph_outcomes is not None:
            fig_out = plot_outcome_proportions(df_t)
            ph_outcomes.pyplot(fig_out, use_container_width=True)
            plt.close(fig_out)

        time.sleep(DELAY)

    status.success(f"Done — {n_rounds} rounds × {n_simulations} simulations")
