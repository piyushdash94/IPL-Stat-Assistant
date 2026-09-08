"""
Compact IPL match summaries computed from the cleaned ball-by-ball DataFrame.

The block-character chart primitives (`sparkline`, `hbar`) are ported from the
sibling `cricbot` repo so the text card reads the same, but every input here is
derived from our own cleaned `df` (see load_and_clean_df.py) rather than an
ESPNCricinfo payload.
"""

import pandas as pd

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values, lo=None, hi=None):
    """Render a sequence of numbers as a one-line block sparkline."""
    nums = [v for v in values if v is not None]
    if not nums:
        return ""
    lo = min(nums) if lo is None else lo
    hi = max(nums) if hi is None else hi
    span = hi - lo
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        elif span <= 0:
            out.append(BLOCKS[len(BLOCKS) // 2])
        else:
            idx = round((v - lo) / span * (len(BLOCKS) - 1))
            out.append(BLOCKS[max(0, min(len(BLOCKS) - 1, idx))])
    return "".join(out)


def hbar(value, max_value, width=20, fill="█", empty="·"):
    """Render a single horizontal bar of exactly `width` characters."""
    if not max_value or max_value <= 0 or value is None:
        return empty * width
    filled = max(0, min(width, int(round((value / max_value) * width))))
    return fill * filled + empty * (width - filled)


def _is_wicket(match_df):
    """Boolean Series: deliveries on which a batter was dismissed."""
    return match_df["player_out"].astype("string").fillna("N/A") != "N/A"


def _shift_points(over_runs, over_wickets):
    """
    Identify up to 3 momentum-shift overs from per-over runs and wickets:
    'collapse' overs (2+ wickets) and the single biggest 'surge' over
    (well above the innings run rate). Modelless and cheap.

    Returns a list of {over (1-indexed), kind, runs, wickets}, ordered by over.
    """
    if not over_runs:
        return []
    n = len(over_runs)
    par = sum(over_runs) / n
    shifts = {}
    for i, w in enumerate(over_wickets):
        if w >= 2:
            shifts[i] = {"over": i + 1, "kind": "collapse",
                         "runs": over_runs[i], "wickets": w}
    top_i = max(range(n), key=lambda i: over_runs[i])
    if over_runs[top_i] >= max(12, par * 1.6) and top_i not in shifts:
        shifts[top_i] = {"over": top_i + 1, "kind": "surge",
                         "runs": over_runs[top_i], "wickets": over_wickets[top_i]}
    return [shifts[k] for k in sorted(shifts)][:3]


def list_matches(df):
    """
    Return a DataFrame of selectable matches, newest first.

    Columns: match_id, label — where label is a human-readable picker string
    like "S18 · Final · CSK vs MI · 2025-05-26".
    """
    grp = df.groupby("match_id", sort=False)
    rows = []
    for match_id, m in grp:
        teams = pd.unique(m["batting_team"].dropna())
        matchup = " vs ".join(map(str, teams[:2])) if len(teams) else "Unknown"
        season = m["ipl_season_no"].iloc[0]
        stage = str(m["match_no_this_season"].iloc[0])
        date = pd.to_datetime(m["date"].iloc[0]).date()
        rows.append(
            {
                "match_id": match_id,
                "date": date,
                "label": f"S{season} · {stage} · {matchup} · {date}",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("date", ascending=False).reset_index(drop=True)


def _innings_summary(inn_df, top_n=3):
    """Compute totals, boundaries, top batters and phase splits for one innings."""
    legal = int(inn_df["is_legal_ball"].sum())
    wickets = int(_is_wicket(inn_df).sum())
    totals = {
        "team": str(inn_df["batting_team"].iloc[0]),
        "runs": int(inn_df["total_runs_this_ball"].sum()),
        "wickets": wickets,
        "overs": f"{legal // 6}.{legal % 6}",
        "extras": int(inn_df["extra_runs_this_ball"].sum()),
        "fours": int((inn_df["runs_counted_for_batter"] == 4).sum()),
        "sixes": int((inn_df["runs_counted_for_batter"] == 6).sum()),
    }

    # Top batters by runs scored.
    bat = inn_df.groupby("batter", sort=False).agg(
        runs=("runs_counted_for_batter", "sum"),
        balls=("is_ball_counted_towards_batter", "sum"),
    )
    bat = bat[bat["balls"] > 0].sort_values("runs", ascending=False).head(top_n)
    top_batters = [
        {
            "name": str(name),
            "runs": int(r["runs"]),
            "balls": int(r["balls"]),
            "sr": round(r["runs"] / r["balls"] * 100, 2) if r["balls"] else 0.0,
        }
        for name, r in bat.iterrows()
    ]

    # Runs/wickets per over, ordered by over number (for sparkline + charts).
    per_over = (
        inn_df.assign(_w=_is_wicket(inn_df).astype(int))
        .groupby("over", sort=True)
        .agg(runs=("total_runs_this_ball", "sum"), wickets=("_w", "sum"))
    )
    over_runs = [int(x) for x in per_over["runs"].tolist()]
    over_wkts = [int(x) for x in per_over["wickets"].tolist()]

    # Cumulative worm + momentum-shift overs (for the plot and headline).
    cum_runs, running = [], 0
    for r in over_runs:
        running += r
        cum_runs.append(running)
    shifts = _shift_points(over_runs, over_wkts)

    # Powerplay / Middle / Death splits from the precomputed match_phase column.
    phases = []
    if "match_phase" in inn_df.columns:
        pg = inn_df.assign(_w=_is_wicket(inn_df).astype(int)).groupby(
            "match_phase", observed=True, sort=False
        )
        agg = pg.agg(
            runs=("total_runs_this_ball", "sum"),
            wickets=("_w", "sum"),
            legal=("is_legal_ball", "sum"),
        )
        for phase in ["Powerplay", "Middle", "Death"]:
            if phase in agg.index:
                row = agg.loc[phase]
                balls = int(row["legal"]) or 1
                phases.append(
                    {
                        "phase": phase,
                        "runs": int(row["runs"]),
                        "wickets": int(row["wickets"]),
                        "rpo": round(int(row["runs"]) / balls * 6, 1),
                    }
                )

    return {**totals, "top_batters": top_batters,
            "over_runs": over_runs, "over_wickets": over_wkts,
            "cum_runs": cum_runs, "shifts": shifts, "phases": phases}


def build_match_summary(df, match_id, top_n=3):
    """
    Build a structured summary for one match_id.

    Returns a dict with title/result/ground/meta and a list of per-innings
    summaries (each including over_runs/over_wickets for charting).
    """
    m = df[df["match_id"] == match_id]
    if m.empty:
        raise ValueError(f"No rows for match_id {match_id!r}")

    winner = str(m["winning_team"].iloc[0])
    outcome = str(m["win_outcome"].iloc[0])
    result = f"{winner} won ({outcome})" if winner not in ("N/A", "") else "No result"
    ground = ", ".join(
        x for x in [str(m["stadium"].iloc[0]), str(m["city"].iloc[0])] if x and x != "N/A"
    )
    potm = str(m["player_of_match"].iloc[0]) if "player_of_match" in m.columns else "N/A"

    innings = [
        _innings_summary(m[m["innings"] == inn], top_n=top_n)
        for inn in sorted(m["innings"].unique())
    ]
    title = " vs ".join(i["team"] for i in innings[:2]) if innings else "Match"

    return {
        "match_id": match_id,
        "title": title,
        "result": result,
        "ground": ground,
        "season": int(m["ipl_season_no"].iloc[0]),
        "stage": str(m["match_no_this_season"].iloc[0]),
        "date": str(pd.to_datetime(m["date"].iloc[0]).date()),
        "player_of_match": potm,
        "innings": innings,
    }


def build_headline(summary):
    """
    A 2-3 line, rule-based headline for a match — no LLM tokens.
    Covers the result + totals, the standout performer, and a turning point.
    """
    inns = summary.get("innings", [])
    if not inns:
        return ""
    lines = []

    if len(inns) >= 2:
        a, b = inns[0], inns[1]
        lines.append(
            f"{summary['result']} — {a['team']} {a['runs']}/{a['wickets']} vs "
            f"{b['team']} {b['runs']}/{b['wickets']}."
        )
    else:
        a = inns[0]
        lines.append(f"{summary['result']} — {a['team']} {a['runs']}/{a['wickets']}.")

    # Standout performer: Player of the Match if they batted, else top scorer.
    potm = summary.get("player_of_match", "N/A")
    perf = None
    for inn in inns:
        for bat in inn["top_batters"]:
            if bat["name"] == potm:
                perf = f"{potm} top-scored with {bat['runs']}({bat['balls']})"
                break
        if perf:
            break
    if not perf and potm not in ("N/A", ""):
        perf = f"{potm} was Player of the Match"
    if not perf:
        allb = [bat for inn in inns for bat in inn["top_batters"]]
        if allb:
            top = max(allb, key=lambda x: x["runs"])
            perf = f"{top['name']} top-scored with {top['runs']}({top['balls']})"
    if perf:
        lines.append(perf + ".")

    # Turning point from the shift overs (collapse preferred, else surge).
    turning = None
    for inn in inns:
        collapse = next((s for s in inn.get("shifts", []) if s["kind"] == "collapse"), None)
        if collapse:
            turning = (f"{inn['team']} lost {collapse['wickets']} wickets in over "
                       f"{collapse['over']} — the key swing.")
            break
    if not turning:
        for inn in inns:
            surge = next((s for s in inn.get("shifts", []) if s["kind"] == "surge"), None)
            if surge:
                turning = f"{inn['team']} broke away with {surge['runs']} off over {surge['over']}."
                break
    if turning:
        lines.append(turning)

    return "\n".join(lines[:3])


def chart_frame_from_markdown(text):
    """
    If `text` contains a GitHub-style markdown table with at least one numeric
    column, return a DataFrame indexed by the first column and holding only the
    numeric columns (ready for st.bar_chart). Otherwise return None.

    Pure and defensive so the caller can render a chart from an LLM answer
    without risking an exception.
    """
    import re

    rows, sep_seen = [], False
    sep_re = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
    for line in text.splitlines():
        if "|" not in line:
            continue
        if sep_re.match(line):
            sep_seen = True
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2:
            rows.append(cells)

    if not sep_seen or len(rows) < 2:
        return None

    header = rows[0]
    width = len(header)
    data = [r for r in rows[1:] if len(r) == width]
    if not data:
        return None

    tdf = pd.DataFrame(data, columns=header)
    numeric = tdf.apply(lambda s: pd.to_numeric(s.str.replace(",", "", regex=False),
                                                errors="coerce"))
    numeric_cols = [c for c in numeric.columns[1:] if numeric[c].notna().any()]
    if not numeric_cols:
        return None

    out = numeric[numeric_cols].copy()
    out.index = tdf[header[0]]
    return out.dropna(how="all")


def render_card(summary):
    """Render a summary dict as a cricbot-style plain-text card."""
    lines = ["🏏 MATCH", summary["title"], summary["result"], summary["ground"],
             f"S{summary['season']} · {summary['stage']} · {summary['date']}", ""]

    for inn in summary["innings"]:
        lines.append(f"{inn['team']}: {inn['runs']}/{inn['wickets']} in {inn['overs']} overs")
        lines.append(f"Extras: {inn['extras']} | 4s: {inn['fours']} | 6s: {inn['sixes']}")
        if inn["top_batters"]:
            lines.append("Top batters:")
            for b in inn["top_batters"]:
                lines.append(f"- {b['name']}: {b['runs']}({b['balls']}) SR:{b['sr']}")
        if inn["phases"]:
            lines.append("Phases:")
            for p in inn["phases"]:
                lines.append(f"{p['phase']:<10} {p['runs']:>3} runs  {p['wickets']}w  RPO {p['rpo']}")
        if inn["over_runs"]:
            lines.append("Runs/over:")
            lines.append(sparkline(inn["over_runs"], lo=0))
            lines.append("".join("W" if w else "·" for w in inn["over_wickets"]))
        lines.append("")

    if summary["player_of_match"] not in ("N/A", ""):
        lines.append(f"Player of the Match: {summary['player_of_match']}")

    return "\n".join(lines).rstrip()
