import sys
import random
sys.path.insert(0, ".")

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

AVATAR_POOL  = ["🦊", "🐺", "🐻", "🦁", "🐯", "🐮", "🐸", "🐼"]
HUMAN_AVATAR = "🧑"
ACTION_EMOJI = {"stag": "🦌", "hare": "🐇"}
TAG_COLOR    = {"red": "#e57373", "blue": "#64b5f6"}

# ─── Game initialisation ─────────────────────────────────────────────────────

def _init_game() -> None:
    cond_key     = st.session_state.play_condition
    condition    = CONDITION_MAP[cond_key]
    n_players    = int(st.session_state.play_n_players)
    n_rounds     = int(st.session_state.play_n_rounds)
    tau          = float(st.session_state.play_tau)
    discount     = float(st.session_state.play_discount)
    replacement  = float(st.session_state.play_replacement)

    rng    = random.Random()
    agents = _make_agents(condition, tau, n_players, rng, discount=discount)

    human_id = agents[0].agent_id
    pool     = AVATAR_POOL.copy()
    avatars  = {human_id: HUMAN_AVATAR}
    for a in agents[1:]:
        avatars[a.agent_id] = pool.pop(0)

    st.session_state.g = {
        "condition":   condition,
        "cond_name":   cond_key,
        "n_players":   n_players,
        "n_rounds":    n_rounds,
        "replacement": replacement,
        "agents":      agents,
        "human_id":    human_id,
        "avatars":     avatars,
        "rng":         rng,
        "phase":       "choosing",   # choosing | revealing | done
        "round":       1,
        "total":       0.0,
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
    per_player = total / g["n_players"]

    g["last"] = {
        "round":       g["round"],
        "action_map":  action_map,
        "stag_count":  stag_count,
        "group_total": total,
        "payoff":      per_player,
    }
    g["total"]  += per_player
    g["history"].append(g["last"])
    g["phase"]   = "revealing"

    # Update simulated agents' beliefs (human's action is visible to them)
    for agent in agents:
        if agent.agent_id == g["human_id"]:
            continue
        observation = {
            "teammates": [
                {"agent_id": o.agent_id, "tag": o.tag, "action": action_map[o.agent_id]}
                for o in agents if o.agent_id != agent.agent_id
            ],
            "own_action": action_map[agent.agent_id],
            "own_tag":    agent.tag,
        }
        agent.update(observation)


def _next_round() -> None:
    g = st.session_state.g

    if g["replacement"] > 0 and g["rng"].random() < g["replacement"]:
        sims = [a for a in g["agents"] if a.agent_id != g["human_id"]]
        if sims:
            g["rng"].choice(sims).reset()

    g["round"] += 1
    g["phase"]  = "done" if g["round"] > g["n_rounds"] else "choosing"

# ─── HTML helpers ────────────────────────────────────────────────────────────

def _card(agent, *, action=None, show_id=True, show_tag=True) -> str:
    g      = st.session_state.g
    avatar = g["avatars"][agent.agent_id]
    is_me  = agent.agent_id == g["human_id"]
    border = "#4878CF" if is_me else "#ddd"
    bg     = "#f0f4ff" if is_me else "#f8f8f8"
    label  = "You" if is_me else (f"Agent {agent.agent_id}" if show_id else "·")

    tag_html = ""
    if show_tag and agent.tag:
        c = TAG_COLOR.get(agent.tag, "#aaa")
        tag_html = (
            f'<div style="margin:3px 0 0;">'
            f'<span style="background:{c};color:#fff;border-radius:4px;'
            f'padding:1px 7px;font-size:0.72em;">{agent.tag}</span></div>'
        )

    action_html = ""
    if action:
        action_html = f'<div style="font-size:1.6em;margin-top:4px;">{ACTION_EMOJI[action]}</div>'

    return (
        f'<div style="text-align:center;padding:12px 8px;border-radius:10px;'
        f'background:{bg};border:2px solid {border};">'
        f'<div style="font-size:2.4em;line-height:1;">{avatar}</div>'
        f'<div style="font-size:0.78em;color:#555;margin-top:3px;">{label}</div>'
        f'{tag_html}{action_html}</div>'
    )

# ─── Phase renderers ─────────────────────────────────────────────────────────

def _header() -> None:
    g   = st.session_state.g
    rn  = min(g["round"], g["n_rounds"])
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**Round {rn} / {g['n_rounds']}** &nbsp;·&nbsp; {g['cond_name']}",
                unsafe_allow_html=True)
    c2.markdown(f"<div style='text-align:right'><b>{g['total']:.2f} pts</b> total</div>",
                unsafe_allow_html=True)


def _render_choosing() -> None:
    g         = st.session_state.g
    agents    = g["agents"]
    condition = g["condition"]
    anon      = condition == Condition.ANONYMOUS

    _header()
    st.divider()

    if anon:
        n_others = g["n_players"] - 1
        st.markdown(
            f"You are playing anonymously with "
            f"**{n_others} other player{'s' if n_others > 1 else ''}**."
        )
    else:
        human  = next(a for a in agents if a.agent_id == g["human_id"])
        others = [a for a in agents if a.agent_id != g["human_id"]]
        show_id  = condition == Condition.IDENTITY
        show_tag = condition == Condition.TAG_BASED

        cols = st.columns(len(agents))
        for col, agent in zip(cols, [human] + others):
            col.markdown(_card(agent, show_id=show_id, show_tag=show_tag),
                         unsafe_allow_html=True)

    st.divider()
    st.markdown("**What do you choose?**")
    c1, c2, *_ = st.columns([1, 1, 3])
    if c1.button("🦌  Hunt Stag", use_container_width=True, type="primary"):
        _play_round("stag")
        st.rerun()
    if c2.button("🐇  Hunt Hare", use_container_width=True):
        _play_round("hare")
        st.rerun()


def _render_revealing() -> None:
    g          = st.session_state.g
    result     = g["last"]
    action_map = result["action_map"]
    agents     = g["agents"]
    condition  = g["condition"]
    anon       = condition == Condition.ANONYMOUS

    _header()
    st.divider()

    if anon:
        sc = result["stag_count"]
        hc = g["n_players"] - sc
        human_action = action_map[g["human_id"]]
        st.markdown(
            f"**Outcome:** &nbsp; 🦌 ×{sc} &nbsp; 🐇 ×{hc}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"Your choice: **{ACTION_EMOJI[human_action]} {human_action.title()}**"
        )
    else:
        human  = next(a for a in agents if a.agent_id == g["human_id"])
        others = [a for a in agents if a.agent_id != g["human_id"]]
        show_id  = condition == Condition.IDENTITY
        show_tag = condition == Condition.TAG_BASED

        cols = st.columns(len(agents))
        for col, agent in zip(cols, [human] + others):
            col.markdown(
                _card(agent, action=action_map[agent.agent_id],
                      show_id=show_id, show_tag=show_tag),
                unsafe_allow_html=True,
            )

    st.divider()

    mc1, mc2 = st.columns(2)
    mc1.metric("This round", f"+{result['payoff']:.2f} pts")
    mc2.metric("Total", f"{g['total']:.2f} pts")

    label = "▶  Next Round" if g["round"] < g["n_rounds"] else "▶  See Final Results"
    if st.button(label, type="primary"):
        _next_round()
        st.rerun()


def _render_done() -> None:
    g = st.session_state.g

    st.markdown("### Game over")
    avg = g["total"] / g["n_rounds"]
    st.markdown(
        f"You scored **{g['total']:.2f} pts** over {g['n_rounds']} rounds "
        f"({avg:.2f} pts / round on average)."
    )

    # Cumulative payoff chart
    running, rows = 0.0, []
    for r in g["history"]:
        running += r["payoff"]
        rows.append({"Round": r["round"], "Cumulative payoff": running})
    st.line_chart(pd.DataFrame(rows).set_index("Round"))

    # Round-by-round table
    table = []
    for r in g["history"]:
        act = r["action_map"][g["human_id"]]
        table.append({
            "Round":        r["round"],
            "Your choice":  f"{ACTION_EMOJI[act]} {act.title()}",
            "Stag players": r["stag_count"],
            "Payoff":       f"{r['payoff']:.2f} pts",
        })
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

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
    page_title="Play — Role Games",
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
