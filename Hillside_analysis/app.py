"""
Hillside Avenue Bus Lane — Pre/Post Survey Comparison
=====================================================
Interactive comparison of the 2024 and 2026 post-implementation surveys.
Filters apply identically to both years so every comparison is drawn on the
same subgroup.

Run:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import survey_lib as S
from report import build_report_html, make_grouped_bar

st.set_page_config(page_title="Hillside Bus Lane Survey — 2024 vs 2026",
                   layout="wide")

YEAR_COLOR = S.YEAR_COLOR

df_all = S.load_all()

# ---------------------------------------------------------------------------
# Sidebar: unified filters (applied to both years)
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")
st.sidebar.caption("Applied identically to 2024 and 2026 for apples-to-apples "
                   "comparison.")

filters = {}
for field, label in S.FILTER_FIELDS:
    if field in S.MULTI_COLS:
        options = S.multi_options(field)  # all canonical options
    else:
        vals = df_all[field].dropna().unique().tolist()
        options = S.order_for(field, vals)
    if field == "zip":
        # too many zips for a clean multiselect ordering; sort numerically-ish
        options = sorted(o for o in df_all["zip"].dropna().unique() if o)
    chosen = st.sidebar.multiselect(label, options, key=f"flt_{field}")
    if chosen:
        filters[field] = chosen

df = S.apply_filters(df_all, filters)

n2024 = int((df.year == 2024).sum())
n2026 = int((df.year == 2026).sum())

if st.sidebar.button("Reset filters"):
    for field, _ in S.FILTER_FIELDS:
        st.session_state.pop(f"flt_{field}", None)
    st.rerun()

# ---------------------------------------------------------------------------
# Header + respondent counts
# ---------------------------------------------------------------------------
st.title("Hillside Avenue Bus Lane — Survey Comparison")
st.markdown("Post-implementation opinion surveys of the NYC DOT bus lane. "
            "**2024** = earlier post-install · **2026** = later post-install.")

c1, c2, c3 = st.columns(3)
c1.metric("2024 respondents (filtered)", f"{n2024}", f"of {(df_all.year==2024).sum()}")
c2.metric("2026 respondents (filtered)", f"{n2026}", f"of {(df_all.year==2026).sum()}")
c3.metric("Active filters", len(filters))

if filters:
    st.info("Filtering on: " + " · ".join(
        f"**{dict(S.FILTER_FIELDS)[f]}** = {', '.join(map(str, v))}"
        for f, v in filters.items()))

if n2024 == 0 or n2026 == 0:
    st.warning("One or both years have no respondents under the current "
               "filters. Loosen the filters to compare.")

tab_compare, tab_overview, tab_data = st.tabs(
    ["📊 Compare a question", "📋 All-question overview", "🔎 Data & notes"])


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def grouped_bar(cats, series_by_year, ytitle="% of respondents"):
    # shared with the static report so both render identically
    return make_grouped_bar(cats, series_by_year, ytitle)


def show_delta_table(tbl, pct_is_share=True):
    """Render a comparison table with a 2024->2026 delta column."""
    disp = tbl.copy()
    a, b = f"{S.YEARS[0]} %", f"{S.YEARS[1]} %"
    if a in disp and b in disp:
        disp["Δ (pp)"] = [
            None if (pd.isna(x) or pd.isna(y)) else round(y - x, 1)
            for x, y in zip(disp[a], disp[b])
        ]
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 1: single-question comparison
# ---------------------------------------------------------------------------
with tab_compare:
    labels = [S.QLABEL[k] for k, _, _ in S.QUESTIONS]
    pick = st.selectbox("Question to compare", labels, index=0)
    key = next(k for k, _, _ in S.QUESTIONS if S.QLABEL[k] == pick)
    kind = S.QKIND[key]

    if n2024 == 0 and n2026 == 0:
        st.stop()

    if kind in ("single", "scale"):
        tbl = S.compare_single(df, key)
        cats = tbl["Category"].tolist()
        series = {y: tbl[f"{y} %"].tolist() for y in S.YEARS}
        st.plotly_chart(grouped_bar(cats, series), use_container_width=True)
        bn = tbl.attrs["n"]
        st.caption(f"Answered this question — **n={bn[2024]}** (2024) · "
                   f"**n={bn[2026]}** (2026)")

        if kind == "scale":
            ns = S.net_satisfaction(df)
            st.markdown("##### Net satisfaction  ·  %top-2 − %bottom-2")
            m1, m2, m3, m4 = st.columns(4)
            n24, n26 = ns[2024], ns[2026]
            m1.metric(f"{S.YEARS[0]} net",
                      f"{n24['net']:+.0f}" if n24["net"] is not None else "—")
            m2.metric(f"{S.YEARS[1]} net",
                      f"{n26['net']:+.0f}" if n26["net"] is not None else "—",
                      f"{ns['delta']:+.0f} pts" if ns["delta"] is not None else None)
            dmean = (n26["mean"] - n24["mean"]
                     if n24["mean"] is not None and n26["mean"] is not None else None)
            m3.metric(f"{S.YEARS[0]} mean",
                      f"{n24['mean']:.2f}" if n24["mean"] is not None else "—")
            m4.metric(f"{S.YEARS[1]} mean",
                      f"{n26['mean']:.2f}" if n26["mean"] is not None else "—",
                      f"{dmean:+.2f}" if dmean is not None else None)
            st.caption("Net = share Satisfied/Very satisfied minus share "
                       "Dissatisfied/Very dissatisfied (neutrals kept in base). "
                       "Range −100 to +100.")
        show_delta_table(tbl)

    elif kind == "multi":
        tbl = S.compare_multi(df, key)
        cats = tbl["Option"].tolist()
        series = {y: tbl[f"{y} %"].tolist() for y in S.YEARS}
        st.plotly_chart(grouped_bar(cats, series,
                        ytitle="% selecting (multi-select)"),
                        use_container_width=True)
        bn = tbl.attrs["n"]
        st.caption(f"Base (made ≥1 selection) — **n={bn[2024]}** (2024) · "
                   f"**n={bn[2026]}** (2026). Percentages can exceed 100% "
                   "(select-all-that-apply); blank cells = option not offered "
                   "that year.")
        show_delta_table(tbl)

    elif kind == "buslines":
        tbl = S.compare_buslines(df)
        cats = tbl["Line"].tolist()
        series = {y: tbl[f"{y} %"].tolist() for y in S.YEARS}
        st.plotly_chart(grouped_bar(cats, series,
                        ytitle="% of bus-line respondents"),
                        use_container_width=True)
        base = tbl.attrs["n"]
        st.caption(
            f"Write-in routes from “Other” are parsed and merged with the "
            f"checkbox lines (marked *). Base = respondents naming ≥1 "
            f"identifiable line ({base[2024]} in 2024, {base[2026]} in 2026). "
            f"“Other lines (tail)” = respondents using only routes outside the "
            f"top 15. A rider can appear under multiple lines.")
        show_delta_table(tbl.rename(columns={"Line": "Category"}))

    elif kind == "text":
        freqs, samples, counts = S.text_summary(df, key)
        cols = st.columns(2)
        for col, year in zip(cols, S.YEARS):
            with col:
                st.subheader(f"{year}  ·  {counts[year]} responses")
                if freqs[year]:
                    wf = pd.DataFrame(freqs[year], columns=["word", "count"])
                    st.plotly_chart(
                        go.Figure(go.Bar(
                            x=wf["count"][::-1], y=wf["word"][::-1],
                            orientation="h",
                            marker_color=YEAR_COLOR[year])
                        ).update_layout(height=380, margin=dict(t=10, b=10),
                                        xaxis_title="mentions"),
                        use_container_width=True)
                with st.expander(f"Sample responses ({year})"):
                    for r in samples[year]:
                        st.write(f"- {r}")


# ---------------------------------------------------------------------------
# Tab 2: overview of every comparable single/scale question
# ---------------------------------------------------------------------------
with tab_overview:
    ns = S.net_satisfaction(df)
    o1, o2, o3 = st.columns(3)
    o1.metric(f"{S.YEARS[0]} net satisfaction",
              f"{ns[2024]['net']:+.0f}" if ns[2024]["net"] is not None else "—")
    o2.metric(f"{S.YEARS[1]} net satisfaction",
              f"{ns[2026]['net']:+.0f}" if ns[2026]["net"] is not None else "—",
              f"{ns['delta']:+.0f} pts" if ns["delta"] is not None else None)
    o3.metric("Bus-rating base (n)",
              f"{ns[2024]['n']} → {ns[2026]['n']}")
    st.markdown("Top answer and 2024→2026 shift for each comparable "
                "categorical question, on the current filter.")
    rows = []
    for key, label, kind in S.QUESTIONS:
        if kind == "scale":
            m = {}
            for y in S.YEARS:
                sc = df[df.year == y]["sat_score"].dropna()
                m[y] = sc.mean() if len(sc) else None
            rows.append({
                "Question": label,
                "2024": f"mean {m[2024]:.2f}" if m[2024] else "—",
                "2026": f"mean {m[2026]:.2f}" if m[2026] else "—",
                "Shift": (f"{m[2026]-m[2024]:+.2f}"
                          if m[2024] and m[2026] else "—"),
            })
        elif kind == "single":
            tbl = S.compare_single(df, key)
            if tbl.empty:
                continue
            top24 = tbl.sort_values("2024 %", ascending=False).iloc[0]
            top26 = tbl.sort_values("2026 %", ascending=False).iloc[0]
            rows.append({
                "Question": label,
                "2024": f"{top24['Category']} ({top24['2024 %']:.0f}%)",
                "2026": f"{top26['Category']} ({top26['2026 %']:.0f}%)",
                "Shift": "→ same top" if top24["Category"] == top26["Category"]
                         else "→ changed",
            })
        elif kind in ("multi", "buslines"):
            tbl = (S.compare_buslines(df) if kind == "buslines"
                   else S.compare_multi(df, key))
            optcol = "Line" if kind == "buslines" else "Option"
            def _top(colp):
                t = tbl.dropna(subset=[colp])
                t = t[t[optcol] != "Other lines (tail)"]
                if t.empty:
                    return "—"
                r = t.sort_values(colp, ascending=False).iloc[0]
                return f"{r[optcol]} ({r[colp]:.0f}%)"
            rows.append({
                "Question": label + " (top pick)",
                "2024": _top("2024 %"),
                "2026": _top("2026 %"),
                "Shift": "",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: data + methodology notes
# ---------------------------------------------------------------------------
with tab_data:
    st.markdown("""
### How the two surveys were aligned

The 2024 and 2026 SurveyMonkey exports do not use identical wording or the same
answer choices. This tool maps both onto a shared schema:

- **Satisfaction** — 2024 stored `1 - Very dissatisfied … 5 - Very satisfied`;
  2026 stored the labels only. Both are mapped to a 1–5 scale.
- **Travel time** — 2024 was free-text minutes; it is parsed and binned into
  the same buckets 2026 already uses (1–15, 16–30, 31–45, 46–60, 60+).
- **Trip reason / typical modes** — choices were renamed or split between years
  (e.g. 2024 *Social/Recreational/Religious* → 2026 *Social/Recreational* +
  *Religious services*, which are recombined here). Options offered in only one
  year (e.g. 2026 *Visit friends/family*, *Caregiving*) show blank for the
  other year.
- **Gender** — *Male/Female* (2024) mapped to *Man/Woman* (2026).
- **Age** — 2024 *65+* and 2026 *Over 65* merged to *65+*.
- Minor spelling differences (then/than, hour/hours) normalized.

Percentages for single-answer questions are of respondents who answered that
question that year. Percentages for multi-select questions are of respondents
who made at least one selection, and can sum past 100%.
""")
    st.markdown("#### Static report")
    st.caption("Generate a self-contained HTML report of every comparison for "
               "the current filter selection — openable in any browser, "
               "printable to PDF.")
    report_html = build_report_html(df, filters)
    st.download_button(
        "⬇ Download static HTML report (current filters)",
        report_html, file_name="hillside_report.html", mime="text/html",
        type="primary")

    st.markdown("#### Filtered respondent-level data")
    show_cols = ["year", "respondent_id", "age", "howget", "sat", "parked",
                 "often", "lang", "zip"]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data (CSV)",
        df.assign(**{q: df[q].apply(lambda s: "; ".join(sorted(s)))
                     for q in S.MULTI_COLS}).to_csv(index=False),
        file_name="hillside_filtered.csv", mime="text/csv")
