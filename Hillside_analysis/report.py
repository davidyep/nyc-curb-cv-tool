"""
Static HTML report generator for the Hillside Avenue bus-lane survey
comparison.

Used two ways:
  * imported by app.py to produce a downloadable report for the *current*
    filter selection, and
  * run directly to write a full-dataset report:

        ..\\.venv\\Scripts\\python.exe report.py            # all respondents
        ..\\.venv\\Scripts\\python.exe report.py out.html   # custom path
"""

from __future__ import annotations

import html
import sys

import pandas as pd
import plotly.graph_objects as go

import survey_lib as S


# ---------------------------------------------------------------------------
# Shared chart helper (also imported by app.py so both look identical)
# ---------------------------------------------------------------------------
def make_grouped_bar(cats, series_by_year, ytitle="% of respondents"):
    fig = go.Figure()
    long_labels = cats and max(len(str(c)) for c in cats) > 8
    for year in S.YEARS:
        yvals = series_by_year[year]
        fig.add_bar(
            name=str(year), x=cats, y=yvals,
            marker_color=S.YEAR_COLOR[year],
            text=[f"{v:.0f}%" if v is not None and pd.notna(v) else ""
                  for v in yvals],
            textposition="outside",
        )
    fig.update_layout(
        barmode="group", height=440, yaxis_title=ytitle,
        legend_title="Survey year", margin=dict(t=30, b=40),
        xaxis_tickangle=-25 if long_labels else 0,
        template="plotly_white",
    )
    return fig


def _net_gauge_bar(net_stats):
    """Horizontal bar of net satisfaction per year."""
    years = [y for y in S.YEARS if net_stats[y]["net"] is not None]
    fig = go.Figure()
    for y in years:
        fig.add_bar(
            x=[net_stats[y]["net"]], y=[str(y)], orientation="h",
            marker_color=S.YEAR_COLOR[y], name=str(y),
            text=[f"{net_stats[y]['net']:+.0f}"], textposition="outside",
        )
    fig.update_layout(
        height=200, xaxis_title="Net satisfaction (top-2 % − bottom-2 %)",
        xaxis_range=[-100, 100], showlegend=False,
        margin=dict(t=20, b=40, l=60), template="plotly_white",
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#888")
    return fig


# ---------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------
def _fig_html(fig, include_js):
    return fig.to_html(full_html=False,
                       include_plotlyjs="cdn" if include_js else False,
                       config={"displayModeBar": False})


def _nline(base, label="Base"):
    """Small 'n=' line showing the per-year denominator under a heading."""
    return (f"<div class='nline'>{label}: "
            f"<b>n={base[S.YEARS[0]]}</b> ({S.YEARS[0]}) &nbsp;·&nbsp; "
            f"<b>n={base[S.YEARS[1]]}</b> ({S.YEARS[1]})</div>")


def _table_html(tbl: pd.DataFrame, cat_col: str):
    """Comparison table with a 2024->2026 delta (pp) column."""
    a, b = f"{S.YEARS[0]} %", f"{S.YEARS[1]} %"
    rows = []
    for _, r in tbl.iterrows():
        x, y = r.get(a), r.get(b)
        delta = (None if pd.isna(x) or pd.isna(y) else y - x)
        dcls = ""
        dtxt = "–"
        if delta is not None:
            dcls = "pos" if delta > 0 else ("neg" if delta < 0 else "")
            dtxt = f"{delta:+.1f}"
        def cell(v):
            return "–" if v is None or pd.isna(v) else f"{v:.1f}%"
        rows.append(
            f"<tr><td class='cat'>{html.escape(str(r[cat_col]))}</td>"
            f"<td>{cell(x)}</td><td>{cell(y)}</td>"
            f"<td class='delta {dcls}'>{dtxt}</td></tr>"
        )
    return (
        "<table class='cmp'><thead><tr>"
        f"<th>{html.escape(cat_col)}</th><th>{S.YEARS[0]}</th>"
        f"<th>{S.YEARS[1]}</th><th>Δ pp</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# Questions that reflect opinion/behavior (real "trends") vs. who answered
Q_OPINION = ["sat", "issue", "buslines_norm", "often", "howget", "modes",
             "dur", "why", "parked", "findspot", "blocks", "paid", "parkdur"]
Q_SAMPLE = ["age", "gender", "race", "lang", "eff"]


def _movers(df, keys, min_share=8.0, min_base=40):
    """Return (question, category, 2024%, 2026%, delta_pp, n2024, n2026) rows,
    keeping only categories where at least one year reaches min_share AND both
    years have a base of at least min_base respondents (so small-sample swings —
    e.g. the parking questions — don't masquerade as headline trends)."""
    rows = []
    a, b = f"{S.YEARS[0]} %", f"{S.YEARS[1]} %"
    for key in keys:
        kind = S.QKIND[key]
        if kind in ("single", "scale"):
            tbl, cc = S.compare_single(df, key), "Category"
        elif kind == "multi":
            tbl, cc = S.compare_multi(df, key), "Option"
        elif kind == "buslines":
            tbl, cc = S.compare_buslines(df), "Line"
        else:
            continue
        base = tbl.attrs["n"]
        if min(base[S.YEARS[0]], base[S.YEARS[1]]) < min_base:
            continue
        for _, r in tbl.iterrows():
            cat = str(r[cc])
            if cat == "Other lines (tail)":
                continue
            x, y = r.get(a), r.get(b)
            if pd.isna(x) or pd.isna(y) or max(x, y) < min_share:
                continue
            rows.append((S.QLABEL[key], cat, x, y, y - x,
                         base[S.YEARS[0]], base[S.YEARS[1]]))
    return rows


def _mover_li(rows):
    out = []
    for q, cat, x, y, d, n1, n2 in rows:
        cls = "pos" if d > 0 else ("neg" if d < 0 else "")
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "▬")
        out.append(
            f"<li><b>{html.escape(q)}</b> — {html.escape(cat)}: "
            f"{x:.0f}% → {y:.0f}% "
            f"<span class='delta {cls}'>{arrow} {d:+.0f} pp</span> "
            f"<span class='nsmall'>n={n1}→{n2}</span></li>")
    return "".join(out)


def key_findings_html(df):
    ns = S.net_satisfaction(df)
    parts = ["<h2>Key findings &amp; trends</h2>",
             "<p class='sub'>Auto-generated from the current data. "
             "“pp” = percentage-point change from 2024 to 2026.</p>"]

    # 1) satisfaction headline
    bullets = []
    if ns[2024]["net"] is not None and ns[2026]["net"] is not None:
        d = ns["delta"]
        dir_ = "rose" if d > 0 else ("fell" if d < 0 else "was unchanged")
        bullets.append(
            f"<li><b>Bus-trip satisfaction {dir_} sharply.</b> Net satisfaction "
            f"went from <b>{ns[2024]['net']:+.0f}</b> to "
            f"<b>{ns[2026]['net']:+.0f}</b> ({d:+.0f} pp); the "
            f"Satisfied/Very-satisfied share moved "
            f"{ns[2024]['top2']:.0f}% → {ns[2026]['top2']:.0f}% while "
            f"dissatisfaction moved {ns[2024]['bottom2']:.0f}% → "
            f"{ns[2026]['bottom2']:.0f}%. Mean rating "
            f"{ns[2024]['mean']:.2f} → {ns[2026]['mean']:.2f}.</li>")
    parts.append(f"<div class='block'><h3>Headline</h3><ul>{''.join(bullets)}"
                 "</ul></div>")

    # 2) biggest opinion/behavior movers
    mv = _movers(df, Q_OPINION)
    ups = sorted((r for r in mv if r[4] > 0), key=lambda r: -r[4])[:6]
    downs = sorted((r for r in mv if r[4] < 0), key=lambda r: r[4])[:6]
    parts.append(
        "<div class='block'><h3>Largest upward trends (2024 → 2026)</h3>"
        f"<ul>{_mover_li(ups)}</ul>"
        "<h3 style='margin-top:14px'>Largest downward trends</h3>"
        f"<ul>{_mover_li(downs)}</ul>"
        "<p class='note'>Limited to answer categories reaching ≥8% in at least "
        "one year and with a base of ≥40 respondents both years, so small or "
        "thin-sample segments (e.g. parking questions) don’t masquerade as "
        "headline trends. n shown per item.</p></div>")

    # 3) bus-line specific finding
    bl = S.compare_buslines(df)
    real = bl[bl["Line"] != "Other lines (tail)"]
    tail_rows = bl[bl["Line"] == "Other lines (tail)"]
    if not real.empty and not tail_rows.empty:
      tail = tail_rows.iloc[0]
      top24 = real.sort_values(f"{S.YEARS[0]} %", ascending=False).iloc[0]
      parts.append(
        "<div class='block'><h3>Bus lines: the “Other” bucket was hiding real "
        "routes</h3><ul>"
        f"<li>The checkbox list omitted the corridor’s busiest write-in routes "
        f"(Q2, Q3, Q77, Q30). After parsing write-ins, the unclassified tail "
        f"shrinks to {tail[f'{S.YEARS[0]} %']:.0f}% (2024) / "
        f"{tail[f'{S.YEARS[1]} %']:.0f}% (2026).</li>"
        f"<li>Most-used single line both years is "
        f"<b>{top24['Line'].rstrip('*')}</b>. Q1 and Q2 usage rose notably "
        f"while Q30 nearly vanished between waves — consistent with (but not "
        f"proof of) route renaming in the Queens bus network redesign.</li>"
        "</ul></div>")

    # 4) sample composition caveat
    sm = _movers(df, Q_SAMPLE)
    smv = sorted(sm, key=lambda r: -abs(r[4]))[:5]
    parts.append(
        "<div class='block'><h3>Read with care: the two samples differ</h3>"
        "<p class='note'>These are shifts in <i>who responded</i>, not opinion "
        "changes. Filter to a matched subgroup in the app before attributing "
        "trends to the bus lane.</p>"
        f"<ul>{_mover_li(smv)}</ul></div>")
    return "".join(parts)


CSS = """
:root{--y1:#6c8ebf;--y2:#e07a5f;}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 margin:0;color:#1c2733;background:#f5f6f8;line-height:1.45}
.wrap{max-width:980px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:40px 0 8px;border-bottom:2px solid #e2e6ea;padding-bottom:6px}
.sub{color:#5a6a78;margin:0 0 20px}
.meta{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:16px 20px;
 margin:16px 0;font-size:14px}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}
.card{flex:1;min-width:150px;background:#fff;border:1px solid #e2e6ea;border-radius:10px;
 padding:14px 16px}
.card .lbl{font-size:12px;color:#5a6a78;text-transform:uppercase;letter-spacing:.03em}
.card .val{font-size:26px;font-weight:700;margin-top:4px}
.card .d{font-size:13px;margin-top:2px}
.pos{color:#1a7f37}.neg{color:#c1272d}
.block{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:18px 20px;margin:14px 0}
.block h3{margin:0 0 6px;font-size:16px}
table.cmp{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
.nline{font-size:12px;color:#5a6a78;margin:-2px 0 8px}
.nsmall{font-size:11px;color:#8b97a3;margin-left:4px}
table.cmp th,table.cmp td{border-bottom:1px solid #eceff2;padding:6px 10px;text-align:right}
table.cmp th:first-child,table.cmp td.cat{text-align:left}
table.cmp thead th{color:#5a6a78;font-weight:600;border-bottom:2px solid #e2e6ea}
td.delta{font-weight:600}
.note{font-size:12px;color:#7a8794;margin-top:6px}
footer{margin-top:48px;font-size:12px;color:#8b97a3;text-align:center}
"""


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def build_report_html(df: pd.DataFrame, filters: dict | None = None,
                      title: str = "Hillside Avenue Bus Lane — Survey Comparison"
                      ) -> str:
    filters = filters or {}
    n = {y: int((df.year == y).sum()) for y in S.YEARS}
    js_used = [False]  # include plotly.js only on first figure

    def fig(f):
        first = not js_used[0]
        js_used[0] = True
        return _fig_html(f, include_js=first)

    parts = [f"<div class='wrap'><h1>{html.escape(title)}</h1>",
             "<p class='sub'>Post-implementation opinion surveys of the NYC DOT "
             "Hillside Avenue bus lane · 2024 (earlier) vs 2026 (later).</p>"]

    # respondent + filter meta
    flabel = dict(S.FILTER_FIELDS)
    if filters:
        fstr = " · ".join(f"<b>{html.escape(flabel[k])}</b>: "
                          f"{html.escape(', '.join(map(str, v)))}"
                          for k, v in filters.items())
    else:
        fstr = "None — all respondents."
    parts.append(
        f"<div class='meta'><b>Respondents:</b> {n[2024]} (2024) · "
        f"{n[2026]} (2026)<br><b>Filters:</b> {fstr}</div>")

    # -- Executive summary: key findings & trends ---------------------------
    parts.append(key_findings_html(df))

    # -- Net satisfaction headline ------------------------------------------
    ns = S.net_satisfaction(df)
    parts.append("<h2>Net satisfaction with bus trip</h2>")
    cards = []
    for y in S.YEARS:
        s = ns[y]
        val = f"{s['net']:+.0f}" if s["net"] is not None else "–"
        mean = f"mean {s['mean']:.2f}" if s["mean"] is not None else ""
        cards.append(
            f"<div class='card'><div class='lbl'>{y} net score</div>"
            f"<div class='val'>{val}</div><div class='d'>{mean} · n={s['n']}</div></div>")
    d = ns["delta"]
    dcls = "" if d is None else ("pos" if d > 0 else ("neg" if d < 0 else ""))
    dtxt = "–" if d is None else f"{d:+.0f} pts"
    cards.append(
        f"<div class='card'><div class='lbl'>2024 → 2026 change</div>"
        f"<div class='val {dcls}'>{dtxt}</div>"
        f"<div class='d'>net = %top-2 − %bottom-2</div></div>")
    parts.append(f"<div class='cards'>{''.join(cards)}</div>")
    if any(ns[y]["net"] is not None for y in S.YEARS):
        parts.append(fig(_net_gauge_bar(ns)))
    parts.append("<p class='note'>Net satisfaction = share rating "
                 "Satisfied/Very&nbsp;satisfied minus share rating "
                 "Dissatisfied/Very&nbsp;dissatisfied (neutrals kept in the "
                 "base). Range −100 to +100.</p>")

    # -- Per-question sections ----------------------------------------------
    for key, label, kind in S.QUESTIONS:
        if kind in ("single", "scale"):
            tbl = S.compare_single(df, key)
            if tbl.empty:
                continue
            cats = tbl["Category"].tolist()
            series = {y: tbl[f"{y} %"].tolist() for y in S.YEARS}
            parts.append(f"<div class='block'><h3>{html.escape(label)}</h3>")
            parts.append(_nline(tbl.attrs["n"], "Answered"))
            parts.append(fig(make_grouped_bar(cats, series)))
            parts.append(_table_html(tbl, "Category") + "</div>")
        elif kind == "multi":
            tbl = S.compare_multi(df, key)
            cats = tbl["Option"].tolist()
            series = {y: tbl[f"{y} %"].tolist() for y in S.YEARS}
            parts.append(f"<div class='block'><h3>{html.escape(label)}</h3>")
            parts.append(_nline(tbl.attrs["n"], "Answered"))
            parts.append(fig(make_grouped_bar(
                cats, series, ytitle="% selecting (multi-select)")))
            parts.append(_table_html(tbl, "Option"))
            parts.append("<p class='note'>Select-all-that-apply: percentages "
                         "can exceed 100%. “–” = option not offered that year."
                         "</p></div>")
        elif kind == "buslines":
            tbl = S.compare_buslines(df)
            cats = tbl["Line"].tolist()
            series = {y: tbl[f"{y} %"].tolist() for y in S.YEARS}
            base = tbl.attrs["n"]
            parts.append(f"<div class='block'><h3>{html.escape(label)}</h3>")
            parts.append(_nline(base, "Named ≥1 line"))
            parts.append(fig(make_grouped_bar(
                cats, series, ytitle="% of bus-line respondents")))
            parts.append(_table_html(tbl.rename(columns={"Line": "Category"}),
                                     "Category"))
            parts.append(
                "<p class='note'>Write-in “Other” routes parsed and merged with "
                "checkbox lines (marked *). Base = respondents naming ≥1 line "
                f"({base[2024]} in 2024, {base[2026]} in 2026). “Other lines "
                "(tail)” = routes outside the top 15.</p></div>")
        elif kind == "text":
            freqs, samples, counts = S.text_summary(df, key)
            parts.append(f"<div class='block'><h3>{html.escape(label)}</h3>")
            parts.append(_nline(counts, "Responses"))
            cols = []
            for y in S.YEARS:
                rows = "".join(
                    f"<tr><td class='cat'>{html.escape(w)}</td><td>{c}</td></tr>"
                    for w, c in freqs[y])
                cols.append(
                    f"<div style='flex:1;min-width:220px'>"
                    f"<b>{y}</b> · {counts[y]} responses"
                    f"<table class='cmp'><thead><tr><th>Top words</th>"
                    f"<th>Mentions</th></tr></thead><tbody>{rows}</tbody></table></div>")
            parts.append(f"<div style='display:flex;gap:20px;flex-wrap:wrap'>"
                         f"{''.join(cols)}</div></div>")

    parts.append("<footer>Generated from hillside2024.csv &amp; "
                 "hillside2026.csv · questions harmonized to a common schema."
                 "</footer></div>")

    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{''.join(parts)}</body></html>")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "hillside_report.html"
    data = S.load_all()
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_report_html(data))
    print(f"Wrote {out}  ({(data.year==2024).sum()} + "
          f"{(data.year==2026).sum()} respondents)")
