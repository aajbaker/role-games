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
# Uniform grey for all circles in the anonymous condition
ANON_COLOR    = "#9E9E9E"

ACTION_EMOJI  = {"stag": "🦌", "hare": "🐇"}

# ─── Helpers that read game state ────────────────────────────────────────────

def _circle_color(agent) -> str:
    g = st.session_state.g
    if g["condition"] == Condition.TAG_BASED:
        return TAG_COLORS.get(agent.tag, "#aaa")
    if g["condition"] == Condition.ANONYMOUS:
        return ANON_COLOR
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
    model_label = st.session_state.get("play_model_type", "Fictitious Play")
    model_type  = "fictitious_play" if model_label == "Fictitious Play" else "bayesian_tom"

    rng    = random.Random()
    agents = _make_agents(condition, tau, n_players, rng,
                          discount=discount, model_type=model_type)
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
        "history":          [],
        "last":             None,
        "last_replacement": None,
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
        "replacement": g.get("last_replacement"),
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
    g["last_replacement"] = None
    if g["replacement"] > 0 and g["rng"].random() < g["replacement"]:
        sims = [a for a in g["agents"] if a.agent_id != g["human_id"]]
        if sims:
            replaced = g["rng"].choice(sims)
            g["last_replacement"] = {
                "player_idx": g["player_idx"][replaced.agent_id],
                "tag":        replaced.tag,
            }
            replaced.reset()
    g["round"] += 1
    g["phase"]  = "done" if g["round"] > g["n_rounds"] else "choosing"

# ─── Card HTML ───────────────────────────────────────────────────────────────

def _card(agent, *, action: str | None = None, show_action: bool = True) -> str:
    is_me  = agent.agent_id == st.session_state.g["human_id"]
    color  = _circle_color(agent)
    label  = _player_label(agent)

    card_extra   = "box-shadow:0 2px 8px rgba(0,0,0,0.28);" if is_me else ""
    label_weight = "font-weight:600;" if is_me else ""
    ring = f"box-shadow:0 0 0 3px #fff, 0 0 0 5px {color};" if is_me else ""

    circle = (
        f'<div style="width:52px;height:52px;border-radius:50%;'
        f'background:{color};margin:0 auto;{ring}"></div>'
    )

    action_html = ""
    if show_action and action:
        action_html = f'<div style="font-size:1.5em;margin-top:5px;">{ACTION_EMOJI[action]}</div>'

    return (
        f'<div style="text-align:center;padding:12px 8px;border-radius:10px;'
        f'background:#f8f8f8;border:2px solid #ddd;{card_extra}">'
        f'{circle}'
        f'<div style="font-size:0.78em;color:#555;margin-top:6px;{label_weight}">{label}</div>'
        f'{action_html}</div>'
    )


def _mini_circle(agent, *, action: str | None = None, show_action: bool = False) -> str:
    """Single circle unit for use inside a group card."""
    is_me  = agent.agent_id == st.session_state.g["human_id"]
    color  = _circle_color(agent)
    label  = _player_label(agent)
    label_weight = "font-weight:600;" if is_me else ""
    ring = f"box-shadow:0 0 0 3px #fff, 0 0 0 5px {color};" if is_me else ""

    circle = (
        f'<div style="width:44px;height:44px;border-radius:50%;'
        f'background:{color};margin:0 auto;{ring}"></div>'
    )

    action_html = ""
    if show_action and action:
        action_html = f'<div style="font-size:1.2em;margin-top:4px;">{ACTION_EMOJI[action]}</div>'

    return (
        f'<div style="text-align:center;min-width:58px;padding:4px 6px;">'
        f'{circle}'
        f'<div style="font-size:0.70em;color:#555;margin-top:5px;{label_weight}">{label}</div>'
        f'{action_html}</div>'
    )


def _group_card(title: str, agents_ordered: list, *, action_map=None, show_actions: bool = False) -> str:
    """Card containing multiple mini-circles in a row, with aggregate actions below."""
    circles_html = "".join(_mini_circle(a) for a in agents_ordered)

    action_html = ""
    if show_actions and action_map is not None:
        stag_n = sum(1 for a in agents_ordered if action_map.get(a.agent_id) == "stag")
        hare_n = sum(1 for a in agents_ordered if action_map.get(a.agent_id) == "hare")
        emojis = ACTION_EMOJI["stag"] * stag_n + ACTION_EMOJI["hare"] * hare_n
        action_html = (
            f'<div style="font-size:1.3em;margin-top:10px;letter-spacing:2px;">{emojis}</div>'
            f'<div style="font-size:0.68em;color:#aaa;margin-top:6px;">order does not correspond to players</div>'
        )

    return (
        f'<div style="text-align:center;padding:12px 10px;border-radius:10px;'
        f'background:#f8f8f8;border:2px solid #ddd;">'
        f'<div style="font-size:0.78em;font-weight:600;color:#444;margin-bottom:10px;">{title}</div>'
        f'<div style="display:flex;flex-direction:row;justify-content:center;gap:40px;flex-wrap:wrap;">'
        f'{circles_html}'
        f'</div>'
        f'{action_html}'
        f'</div>'
    )

# ─── Shared UI ───────────────────────────────────────────────────────────────

def _header() -> None:
    g  = st.session_state.g
    rn = min(g["round"], g["n_rounds"])
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**Round {rn} / {g['n_rounds']}** &nbsp;·&nbsp; {g['cond_name']}",
                unsafe_allow_html=True)
    c2.markdown(
        f"<div style='text-align:right'><b>{g['total']:.0f} pts</b> group total</div>",
        unsafe_allow_html=True,
    )


def _show_players(action_map=None, *, show_actions: bool = True) -> None:
    g      = st.session_state.g
    agents = g["agents"]
    human  = next(a for a in agents if a.agent_id == g["human_id"])
    others = [a for a in agents if a.agent_id != g["human_id"]]

    if g["condition"] == Condition.IDENTITY:
        # N separate individual cards, human first
        cols = st.columns(len(agents))
        for col, agent in zip(cols, [human] + others):
            action = (action_map or {}).get(agent.agent_id)
            col.markdown(
                _card(agent, action=action,
                      show_action=(show_actions and action is not None)),
                unsafe_allow_html=True,
            )

    elif g["condition"] == Condition.ANONYMOUS:
        # All N circles in one "Players" card; actions shown as aggregate (stags first)
        pool = [human] + others
        st.markdown(
            _group_card("Players", pool, action_map=action_map, show_actions=show_actions),
            unsafe_allow_html=True,
        )

    else:  # TAG_BASED
        # One card per tag; human in their own tag's card; actions shown as aggregate per tag
        red_group  = sorted([a for a in agents if a.tag == "red"],  key=lambda a: a.agent_id != g["human_id"])
        blue_group = sorted([a for a in agents if a.tag == "blue"], key=lambda a: a.agent_id != g["human_id"])
        col_widths = [max(len(red_group), 1), max(len(blue_group), 1)]
        cols = st.columns(col_widths)
        for col, (tag, group) in zip(cols, [("Red", red_group), ("Blue", blue_group)]):
            title = f"{tag} Players" if len(group) != 1 else f"{tag} Player"
            col.markdown(
                _group_card(title, group, action_map=action_map, show_actions=show_actions),
                unsafe_allow_html=True,
            )

# ─── Phase renderers ─────────────────────────────────────────────────────────

def _render_choosing() -> None:
    g = st.session_state.g
    _header()
    st.divider()
    _show_players()

    rep = g.get("last_replacement")
    if rep:
        if g["condition"] == Condition.ANONYMOUS:
            msg = "One of the players has been replaced!"
        elif g["condition"] == Condition.IDENTITY:
            msg = f"Player {rep['player_idx']} has been replaced!"
        else:
            msg = f"One of the {rep['tag'].title()} players has been replaced!"
        st.markdown(
            f'<div style="margin-top:10px;margin-bottom:-10px;padding:8px 14px;'
            f'border-radius:6px;background:#e8f4fd;border:1px solid #b3d4f0;'
            f'color:#1a5276;font-size:0.9em;">ℹ️ {msg}</div>',
            unsafe_allow_html=True,
        )
    st.divider()

    n       = st.session_state.g["n_players"]
    optimal = n + 2
    minimum = n - 1

    st.markdown("**What do you choose?**")
    st.caption("🦌 Stag needs ≥ 2 players (+4 pts) · 🐇 Hare always pays (+1 pt)")
    st.caption(f"Minimum score: {minimum} pts  \nMaximum score: {optimal} pts")
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

    _header()
    st.divider()
    _show_players(action_map=action_map, show_actions=True)

    st.divider()
    mc1, mc2 = st.columns(2)
    mc1.metric("Group score this round", f"{result['group_total']:.0f} pts")
    mc2.metric("Group total so far",     f"{g['total']:.0f} pts")

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

    show_replacement = g["replacement"] > 0

    def _col_name(agent) -> str:
        label = _player_label(agent, for_table=True)
        if g["condition"] == Condition.TAG_BASED:
            return f"{label} ({agent.tag.title()})"
        return label

    rows = []
    for r in history:
        am  = r["action_map"]
        row = {"Round": r["round"]}
        for agent in ordered:
            act              = am[agent.agent_id]
            row[_col_name(agent)] = f"{ACTION_EMOJI[act]} {act.title()}"
        row["Group score"] = f"{r['group_total']:.0f}"
        if show_replacement:
            rep = r.get("replacement")
            row["Replaced"] = f"Player {rep['player_idx']}" if rep else None
        rows.append(row)

    display_df = pd.DataFrame(rows)
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Full export DataFrame — same rows plus game metadata columns
    meta = {
        "condition":       g["cond_name"],
        "n_players":       n,
        "n_rounds":        nr,
        "replacement_rate": g["replacement"],
        "model_type":      st.session_state.get("play_model_type", "Fictitious Play"),
        "tau":             st.session_state.get("play_tau", 0.05),
        "discount":        st.session_state.get("play_discount", 0.90),
    }
    export_df = display_df.copy()
    for col, val in meta.items():
        export_df[col] = val

    cond_slug = g["cond_name"].lower().replace("-", "").replace(" ", "")
    filename  = f"rolegames_{cond_slug}_{n}p_{nr}r.csv"

    dl_col, play_col = st.columns([1, 1])
    dl_col.download_button(
        "⬇  Download results",
        data=export_df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )
    if play_col.button("▶  Play Again", type="primary", use_container_width=True):
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
        st.slider("Replacement rate", 0.0, 0.5, 0.0, step=0.05,
                  format="%.2f", key="play_replacement",
                  help="Probability each round that one agent's memory resets.")
        st.slider("Rounds", 5, 20, 10, step=5, key="play_n_rounds")
        with st.expander("Agent parameters"):
            st.selectbox(
                "Agent model",
                ["Fictitious Play", "Bayesian ToM"],
                key="play_model_type",
                help=(
                    "Fictitious Play tracks action frequencies. "
                    "Bayesian ToM models what other agents believe."
                ),
            )
            st.slider("Temperature (τ)", 0.0, 0.2, 0.05, step=0.01,
                      format="%.2f", key="play_tau",
                      help="How deterministically agents follow expected value.")
            st.slider("Discount factor (δ)", 0.5, 1.0, 0.90, step=0.05,
                      format="%.2f", key="play_discount",
                      help="How much agents downweight older observations.")
        st.divider()
        if st.button("▶  New Game", type="primary", use_container_width=True):
            _init_game()
            st.rerun()

# ─── Entry point ─────────────────────────────────────────────────────────────

_sidebar()

if "g" not in st.session_state:
    st.markdown("## Role Games")
    st.markdown(
        "A coordination game. Each round you and a group of simulated players secretly choose "
        "an action, which combine for different payoffs. Your goal is to maximize your group's points."
    )
    st.divider()

    col_game, col_play = st.columns([3, 2], gap="large")

    with col_game:
        st.markdown("**Actions**")
        st.markdown(
            '<div style="display:flex;gap:40px;margin-bottom:14px;">'
            '<div style="flex:1;padding:10px 14px;border-radius:8px;'
            'background:#f0f4ff;border:1px solid #c5cae9;">'
            '<span style="font-size:1.3em;">🦌</span>&nbsp; <b>Hunt Stag</b><br>'
            '<span style="color:#555;font-size:0.88em;">'
            'Worth 4 points, but only 1 per round.<br>'
            'Risky: needs <b>≥ 2 players</b> to pay off.'
            '</span></div>'
            '<div style="flex:1;padding:10px 14px;border-radius:8px;'
            'background:#f0fff4;border:1px solid #c8e6c9;">'
            '<span style="font-size:1.3em;">🐇</span>&nbsp; <b>Hunt Hare</b><br>'
            '<span style="color:#555;font-size:0.88em;">'
            'Worth 1 point, but plenty available.<br>'
            'Safe: always pays off.'
            '</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Group score examples** *(3 players)*")
        st.markdown(
            '<table style="font-size:0.88em;width:100%;border-collapse:collapse;">'
            '<tr style="background:#efefef;">'
            '  <th style="padding:5px 10px;text-align:center;">🦌 Stag</th>'
            '  <th style="padding:5px 10px;text-align:center;">🐇 Hare</th>'
            '  <th style="padding:5px 10px;text-align:left;">Group score</th>'
            '</tr>'
            '<tr>'
            '  <td style="padding:5px 10px;text-align:center;">0</td>'
            '  <td style="padding:5px 10px;text-align:center;">3</td>'
            '  <td style="padding:5px 10px;">3 pts</td>'
            '</tr>'
            '<tr style="background:#fff3e0;">'
            '  <td style="padding:5px 10px;text-align:center;">1</td>'
            '  <td style="padding:5px 10px;text-align:center;">2</td>'
            '  <td style="padding:5px 10px;">2 pts &nbsp;<span style="color:#e65100;font-size:0.85em;">▼ worst outcome</span></td>'
            '</tr>'
            '<tr style="background:#e8f5e9;">'
            '  <td style="padding:5px 10px;text-align:center;"><b>2</b></td>'
            '  <td style="padding:5px 10px;text-align:center;"><b>1</b></td>'
            '  <td style="padding:5px 10px;"><b>5 pts</b> &nbsp;<span style="color:#2e7d32;font-size:0.85em;">★ optimal</span></td>'
            '</tr>'
            '<tr>'
            '  <td style="padding:5px 10px;text-align:center;">3</td>'
            '  <td style="padding:5px 10px;text-align:center;">0</td>'
            '  <td style="padding:5px 10px;">4 pts</td>'
            '</tr>'
            '</table>',
            unsafe_allow_html=True,
        )

    with col_play:
        st.markdown(
            '<div style="padding:14px 16px;border-radius:8px;background:#f8f8f8;border:1px solid #ddd;">'
            '<b>How to play</b>'
            '<ol style="margin:8px 0 0 0;padding-left:18px;color:#333;font-size:0.92em;line-height:1.8em;">'
            '<li>Open the <b>← sidebar</b> to choose a condition, players, and rounds</li>'
            '<li>Click <b>▶ New Game</b> to start</li>'
            '<li>Each round: choose <b>Stag</b> or <b>Hare</b></li>'
            '<li>See what the group chose and your score</li>'
            '<li>After all rounds: review the full history and chart</li>'
            '</ol>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Informational conditions** — what you know about other players each round")

    cc1, cc2, cc3 = st.columns(3)
    for col, icon, name, desc in [
        (cc1, "🔒", "Anonymous",
         "You see only aggregate counts — how many chose stag and hare — not who."),
        (cc2, "🪪", "Identity",
         "You see each individual player's choice every round, building a full history per person."),
        (cc3, "🏷️", "Tag-based",
         "Players have colored tags (red / blue). You see choices grouped by tag, not by individual."),
    ]:
        col.markdown(
            f'<div style="padding:10px 14px;border-radius:8px;'
            f'background:#f8f8f8;border:1px solid #ddd;min-height:90px;">'
            f'<b>{icon}&nbsp;{name}</b><br>'
            f'<span style="color:#444;font-size:0.88em;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    phase = st.session_state.g["phase"]
    if phase == "choosing":
        _render_choosing()
    elif phase == "revealing":
        _render_revealing()
    else:
        _render_done()
