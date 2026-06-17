import sys
import random
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from role_games.conditions import Condition
from role_games.game import _make_agents, _group_total

# ─── Constants ───────────────────────────────────────────────────────────────

CONDITION_MAP = {
    "Anonymous":  Condition.ANONYMOUS,
    "Identity":   Condition.IDENTITY,
    "Tag-based":  Condition.TAG_BASED,
}

# Colors for identity / anonymous conditions (by player index 0..4)
PLAYER_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#e5a917", "#9b59b6"]
# Colors for tag-based condition (by tag name)
TAG_COLORS    = {"red": "#e57373", "blue": "#64b5f6"}

ACTION_EMOJI  = {"stag": "🦌", "hare": "🐇"}

# ─── Helpers that read game state ────────────────────────────────────────────

def _circle_color(agent) -> str:
    g = st.session_state.g
    if g["condition"] == Condition.TAG_BASED:
        return TAG_COLORS.get(agent.tag, "#aaa")
    return PLAYER_COLORS[g["player_idx"][agent.agent_id] % len(PLAYER_COLORS)]


def _player_label(agent, *, for_table=False) -> str:
    """Display label for an agent.
    Identity  → You / Player 1 / Player 2 …
    Tag/Anon  → You / Other Player (game) | You / Player 1 / Player 2 (table)
    """
    g     = st.session_state.g
    is_me = agent.agent_id == g["human_id"]
    idx   = g["player_idx"][agent.agent_id]
    if is_me:
        return "You"
    if for_table or g["condition"] == Condition.IDENTITY:
        return f"Player {idx}"
    return "Other Player"

# ─── Game initialisation ─────────────────────────────────────────────────────

def _init_game() -> None:
    cond_key    = st.session_state.play_condition
    condition   = CONDITION_MAP[cond_key]
    n_players   = int(st.session_state.play_n_players)
    n_rounds    = int(st.session_state.play_n_rounds)
    tau         = float(st.session_state.play_tau)
    discount    = float(st.session_state.play_discount)
    replacement = float(st.session_state.play_replacement)

    rng    = random.Random()
    agents = _make_agents(condition, tau, n_players, rng, discount=discount)
    human_id = agents[0].agent_id

    st.session_state.g = {
        "condition":   condition,
        "cond_name":   cond_key,
        "n_players":   n_players,
        "n_rounds":    n_rounds,
        "replacement": replacement,
        "agents":      agents,
        "human_id":    human_id,
        # stable per-player index (0 = human, 1..N-1 = simulated)
        "player_idx":  {a.agent_id: i for i, a in enumerate(agents)},
        "rng":         rng,
        "phase":       "choosing",
        "round":       1,
        "total":       0.0,      # cumulative group score
        "history":     [],
        "last":        None,
    }

# ─── Round logic ─────────────────────────────────────────────────────────────

def _play_round(human_action: str) -> None:
    g      = st.session_state.g
    agents = g["agents"]

    action_map = {
        a.agent_id: (human_action if a.agent_id == g["human_id"] else a.decide())
        for a in agents
    }
    actions    = list(action_map.values())
    stag_count = actions.count("stag")
    total      = _group_total(actions)

    g["last"] = {
        "round":       g["round"],
        "action_map":  action_map,
        "stag_count":  stag_count,
        "group_total": total,
    }
    g["total"]  += total
    g["history"].append(g["last"])
    g["phase"]   = "revealing"

    for agent in agents:
        if agent.agent_id == g["human_id"]:
            continue
        agent.update({
            "teammates": [
                {"agent_id": o.agent_id, "tag": o.tag, "action": action_map[o.agent_id]}
                for o in agents if o.agent_id != agent.agent_id
            ],
            "own_action": action_map[agent.agent_id],
            "own_tag":    agent.tag,
        })


def _next_round() -> None:
    g = st.session_state.g
    if g["replacement"] > 0 and g["rng"].random() < g["replacement"]:
        sims = [a for a in g["agents"] if a.agent_id != g["human_id"]]
        if sims:
            g["rng"].choice(sims).reset()
    g["round"] += 1
    g["phase"]  = "done" if g["round"] > g["n_rounds"] else "choosing"

# ─── Card HTML ───────────────────────────────────────────────────────────────

def _card(agent, *, action: str | None = None, show_action: bool = True) -> str:
    is_me  = agent.agent_id == st.session_state.g["human_id"]
    color  = _circle_color(agent)
    label  = _player_label(agent)
    border = "2px solid #333" if is_me else "2px solid #ddd"
    bg     = "#f0f4ff" if is_me else "#f8f8f8"

    # White inner ring on the circle marks "You"
    ring = f"box-shadow:0 0 0 3px #fff, 0 0 0 6px {color};" if is_me else ""
    circle = (
        f'<div style="width:52px;height:52px;border-radius:50%;'
        f'background:{color};margin:0 auto;{ring}"></div>'
    )

    action_html = ""
    if show_action and action:
        action_html = f'<div style="font-size:1.5em;margin-top:5px;">{ACTION_EMOJI[action]}</div>'

    return (
        f'<div style="text-align:center;padding:12px 8px;border-radius:10px;'
        f'background:{bg};border:{border};">'
        f'{circle}'
        f'<div style="font-size:0.78em;color:#555;margin-top:6px;">{label}</div>'
        f'{action_html}</div>'
    )

# ─── Shared UI ───────────────────────────────────────────────────────────────

def _header() -> None:
    g  = st.session_state.g
    rn = min(g["round"], g["n_rounds"])
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**Round {rn} / {g['n_rounds']}** &nbsp;·&nbsp; {g['cond_name']}",
                unsafe_allow_html=True)
    c2.markdown(
        f"<div style='text-align:right'><b>{g['total']:.1f} pts</b> group total</div>",
        unsafe_allow_html=True,
    )


def _show_players(action_map=None, *, show_actions: bool = True) -> None:
    g      = st.session_state.g
    agents = g["agents"]
    human  = next(a for a in agents if a.agent_id == g["human_id"])
    others = [a for a in agents if a.agent_id != g["human_id"]]

    cols = st.columns(len(agents))
    for col, agent in zip(cols, [human] + others):
        action = (action_map or {}).get(agent.agent_id)
        col.markdown(
            _card(agent, action=action, show_action=(show_actions and action is not None)),
            unsafe_allow_html=True,
        )

# ─── Phase renderers ─────────────────────────────────────────────────────────

def _render_choosing() -> None:
    _header()
    st.divider()
    _show_players()
    st.divider()

    st.markdown("**What do you choose?**")
    c1, c2, *_ = st.columns([1, 1, 3])
    if c1.button("🦌  Hunt Stag", use_container_width=True):
        _play_round("stag")
        st.rerun()
    if c2.button("🐇  Hunt Hare", use_container_width=True):
        _play_round("hare")
        st.rerun()


def _render_revealing() -> None:
    g          = st.session_state.g
    result     = g["last"]
    action_map = result["action_map"]
    anon       = g["condition"] == Condition.ANONYMOUS

    _header()
    st.divider()

    if anon:
        # Show circles but no per-player action labels; aggregate below
        _show_players(action_map=action_map, show_actions=False)
        sc           = result["stag_count"]
        hc           = g["n_players"] - sc
        human_action = action_map[g["human_id"]]
        st.markdown(
            f"**Outcome:** &nbsp; 🦌 ×{sc} &nbsp; 🐇 ×{hc} &nbsp;&nbsp;·&nbsp;&nbsp;"
            f"Your choice: **{ACTION_EMOJI[human_action]} {human_action.title()}**",
            unsafe_allow_html=True,
        )
    else:
        _show_players(action_map=action_map, show_actions=True)

    st.divider()
    mc1, mc2 = st.columns(2)
    mc1.metric("Group score this round", f"{result['group_total']:.1f} pts")
    mc2.metric("Group total so far",     f"{g['total']:.1f} pts")

    label = "▶  Next Round" if g["round"] < g["n_rounds"] else "▶  See Final Results"
    if st.button(label, type="primary"):
        _next_round()
        st.rerun()


def _render_done() -> None:
    g        = st.session_state.g
    n        = g["n_players"]
    nr       = g["n_rounds"]
    optimal  = n + 2        # 2 stag + rest hare
    minimum  = n - 1        # 1 lone stag, rest hare

    st.markdown("### Results")

    # Summary totals
    c1, c2, c3 = st.columns(3)
    c1.metric("Group total",    f"{g['total']:.0f} pts")
    c2.metric("Optimal total",  f"{optimal * nr} pts")
    c3.metric("Minimum total",  f"{minimum * nr} pts")

    # Per-round payoff chart
    history = g["history"]
    rounds  = [r["round"]       for r in history]
    payoffs = [r["group_total"] for r in history]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rounds, payoffs, marker="o", color="#4878CF", linewidth=2, label="Group score")
    ax.axhline(optimal, color="#4CAF50", linestyle="--", linewidth=1.2,
               label=f"Optimal ({optimal})")
    ax.axhline(minimum, color="#9E9E9E", linestyle=":",  linewidth=1.2,
               label=f"Minimum ({minimum})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Group score")
    ax.set_ylim(max(0, minimum - 0.5), optimal + 1.5)
    ax.set_xticks(rounds)
    ax.legend(fontsize=8)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Per-round table — all players' choices
    agents  = g["agents"]
    human   = next(a for a in agents if a.agent_id == g["human_id"])
    others  = [a for a in agents if a.agent_id != g["human_id"]]
    ordered = [human] + others

    rows = []
    for r in history:
        am  = r["action_map"]
        row = {"Round": r["round"]}
        for agent in ordered:
            col_name     = _player_label(agent, for_table=True)
            act          = am[agent.agent_id]
            row[col_name] = f"{ACTION_EMOJI[act]} {act.title()}"
        row["Group score"] = f"{r['group_total']:.0f}"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if st.button("▶  Play Again", type="primary"):
        del st.session_state.g
        st.rerun()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.header("Game settings")
        st.selectbox("Condition", list(CONDITION_MAP), key="play_condition",
                     help="Information available to all players about each other.")
        st.number_input("Players (including you)",
                        min_value=3, max_value=5, value=3, step=1,
                        key="play_n_players")
        st.slider("Rounds", 5, 50, 20, step=5, key="play_n_rounds")
        with st.expander("Agent parameters"):
            st.slider("Temperature (τ)", 0.0, 0.2, 0.1, step=0.01,
                      format="%.2f", key="play_tau",
                      help="How deterministically agents follow expected value.")
            st.slider("Discount factor (δ)", 0.5, 1.0, 1.0, step=0.05,
                      format="%.2f", key="play_discount",
                      help="How much agents downweight older observations.")
            st.slider("Replacement rate", 0.0, 0.5, 0.0, step=0.05,
                      format="%.2f", key="play_replacement",
                      help="Probability each round that one agent's memory resets.")
        st.divider()
        if st.button("▶  New Game", type="primary", use_container_width=True):
            _init_game()
            st.rerun()

# ─── Entry point ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Play",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_sidebar()

if "g" not in st.session_state:
    st.markdown("## Role Games — Play")
    st.caption("Make decisions alongside simulated agents in a modified stag hunt.")
    st.info("Open the **←** sidebar to configure and start a game.")
else:
    phase = st.session_state.g["phase"]
    if phase == "choosing":
        _render_choosing()
    elif phase == "revealing":
        _render_revealing()
    else:
        _render_done()
