"""
Hillside Avenue bus-lane survey library.

Loads the two SurveyMonkey post-implementation surveys (2024 and 2026),
harmonizes their questions/answer choices onto a common schema, and exposes
helpers for filtering and pre/post comparison.

SurveyMonkey export format (both files):
    row 0  -> question text (only on the first sub-column of a matrix/multi)
    row 1  -> choice / sub-column labels
    row 2+ -> one respondent per row

Because question *text* differs slightly between the two years and some
choices were renamed or split, every mapping below is expressed with explicit
*column indices* per year, which are stable regardless of label wording.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    2024: os.path.join(DATA_DIR, "hillside2024.csv"),
    2026: os.path.join(DATA_DIR, "hillside2026.csv"),
}
YEARS = (2024, 2026)
YEAR_COLOR = {2024: "#6c8ebf", 2026: "#e07a5f"}

# ---------------------------------------------------------------------------
# Single-value column locations (0-based column index) per year
# ---------------------------------------------------------------------------
SINGLE_COLS = {
    "often":        {2024: 11, 2026: 11},   # How often do you visit Hillside
    "howget":       {2024: 32, 2026: 39},   # How did you get here today (single mode)
    "dur":          {2024: 34, 2026: 41},   # Travel time (2024 free-text mins, 2026 buckets)
    "sat":          {2024: 45, 2026: 52},   # Bus trip satisfaction 1-5
    "issue":        {2024: 46, 2026: 53},   # Biggest issue with bus ride
    "parked":       {2024: 48, 2026: 55},   # Did you park on Hillside
    "findspot":     {2024: 49, 2026: 56},   # Time to find a spot
    "blocks":       {2024: 50, 2026: 57},   # Blocks from destination
    "paid":         {2024: 51, 2026: 58},   # Did you pay to park
    "parkdur":      {2024: 52, 2026: 59},   # How long parked
    "issue_text":   {2024: 59, 2026: 61},   # Biggest transportation issue (open)
    "comments":     {2024: 60, 2026: 62},   # Additional comments (open)
    "age":          {2024: 75, 2026: 78},   # Age group
    "eff":          {2024: 76, 2026: 87},   # Survey effectiveness
    "lang":         {2024: 9,  2026: 10},   # Survey language
    "zip":          {2024: 10, 2026: 63},   # Home zip code
}

# ---------------------------------------------------------------------------
# Multi-select questions: canonical option -> {year: [column indices]}
# An option "exists" in a year only if it has at least one column there.
# ---------------------------------------------------------------------------
MULTI_COLS = {
    "why": {
        "Shopping/Dining":                  {2024: [12], 2026: [12]},
        "Medical":                          {2024: [13], 2026: [13]},
        "Live here":                        {2024: [14], 2026: [14]},
        "Work":                             {2024: [15], 2026: [16]},
        "School":                           {2024: [16], 2026: [17]},
        "Social/Recreational/Religious":    {2024: [17], 2026: [18, 19]},
        "Community resource":               {2024: [18], 2026: [22]},
        "Transit transfer/Passing through": {2024: [19], 2026: [23]},
        "Other":                            {2024: [20], 2026: [24]},
        "Visit friends/family":             {2024: [],   2026: [15]},
        "Childcare pickup/drop-off":        {2024: [],   2026: [20]},
        "Caregiving":                       {2024: [],   2026: [21]},
    },
    "modes": {
        "Walk":                             {2024: [24], 2026: [25]},
        "Subway":                           {2024: [27], 2026: [26]},
        "Bus":                              {2024: [26], 2026: [27]},
        "Personal car":                     {2024: [22], 2026: [31]},
        "Dropped off (family/friend)":      {2024: [23], 2026: [32]},
        "Bike/E-Scooter":                   {2024: [25], 2026: [35]},
        "LIRR":                             {2024: [28], 2026: [37]},
        "Access-A-Ride":                    {2024: [29], 2026: [29]},
        "For-hire vehicle (Uber/Taxi/Lyft)":{2024: [30], 2026: [33]},
        "Other":                            {2024: [31], 2026: [38]},
        "Mobility device":                  {2024: [],   2026: [28]},
        "Carpool (drive w/ someone)":       {2024: [],   2026: [30]},
        "Motorcycle":                       {2024: [],   2026: [34]},
        "Moped":                            {2024: [],   2026: [36]},
    },
    "buslines": {
        "Q1":    {2024: [35], 2026: [42]},
        "Q17":   {2024: [36], 2026: [43]},
        "Q36":   {2024: [37], 2026: [44]},
        "Q43":   {2024: [38], 2026: [45]},
        "Q76":   {2024: [39], 2026: [46]},
        "Q82":   {2024: [40], 2026: [47]},
        "N6":    {2024: [41], 2026: [48]},
        "N22":   {2024: [42], 2026: [49]},
        "N24":   {2024: [43], 2026: [50]},
        "Other": {2024: [44], 2026: [51]},
    },
    "gender": {
        "Man":                              {2024: [62], 2026: [65]},
        "Woman":                            {2024: [63], 2026: [66]},
        "Gender non-conforming/Non-binary": {2024: [64], 2026: [67]},
        "Different identity":               {2024: [65], 2026: [68]},
        "Prefer not to say":                {2024: [61], 2026: [64]},
    },
    "race": {
        "Asian":                              {2024: [68], 2026: [70]},
        "American Indian/Alaska Native":      {2024: [67], 2026: [71]},
        "Black or African American":          {2024: [69], 2026: [72]},
        "Hispanic or Latino":                 {2024: [71], 2026: [73]},
        "Middle Eastern/North African":       {2024: [70], 2026: [74]},
        "Native Hawaiian/Pacific Islander":   {2024: [72], 2026: [75]},
        "White":                              {2024: [73], 2026: [76]},
        "Some other race/ethnicity":          {2024: [74], 2026: [77]},
        "Prefer not to say":                  {2024: [66], 2026: [69]},
    },
}

# Checkbox bus lines (offered as their own column) and the free-text
# "Other (please specify)" column, used to normalize write-in routes.
BUSLINE_CHECKBOX = ["Q1", "Q17", "Q36", "Q43", "Q76", "Q82", "N6", "N22", "N24"]
BUSLINE_OTHER_COL = {2024: 44, 2026: 51}
_LINE_TOKEN = re.compile(r"[QNXS]\s?\d+")


def parse_lines(text: str) -> set:
    """Extract normalized bus-line codes (e.g. {'Q30','Q77'}) from free text."""
    if not text:
        return set()
    return {t.replace(" ", "").upper() for t in _LINE_TOKEN.findall(text.upper())}

# ---------------------------------------------------------------------------
# Value harmonization for single-value questions
# ---------------------------------------------------------------------------
MODE_CANON = {
    # 2024 labels
    "Personal Car": "Personal car",
    "Dropped Off (Family/Friend)": "Dropped off (family/friend)",
    "Bike/E-Scooter": "Bike/E-Scooter",
    "Uber/Taxi/Other For-Hire Vehicle": "For-hire vehicle (Uber/Taxi/Lyft)",
    "Walk": "Walk", "Bus": "Bus", "Subway": "Subway",
    "Access-A-Ride": "Access-A-Ride",
    # 2026 labels
    "Personal car": "Personal car",
    "Dropped off (Family/friend)": "Dropped off (family/friend)",
    "Personal bike/E-Scooter": "Bike/E-Scooter",
    "For hire vehicle taxi/Uber/Lyft etc.": "For-hire vehicle (Uber/Taxi/Lyft)",
    "Drive with someone (carpool)": "Carpool (drive w/ someone)",
    "Long Island Railroad (LIRR)": "LIRR",
    "Mobility device (wheelchair, crutch, cane, walker, etc.)": "Mobility device",
    "Motorcycle": "Motorcycle", "Moped": "Moped",
    "Other (please specify)": "Other",
}

SAT_SCORE = {
    "1 - Very dissatisfied": 1, "Very dissatisfied": 1,
    "2 - Dissatisfied": 2, "Dissatisfied": 2,
    "3 - Neutral": 3, "Neutral": 3,
    "4 - Satisfied": 4, "Satisfied": 4,
    "5 - Very satisfied": 5, "Very satisfied": 5,
}
SAT_LABEL = {1: "Very dissatisfied", 2: "Dissatisfied", 3: "Neutral",
             4: "Satisfied", 5: "Very satisfied"}

SIMPLE_FIX = {  # normalize minor spelling/typo differences between years
    "A couple of times a year": "A couple times a year",
    "Less then 1 block": "Less than 1 block",
    "More then 5 blocks": "More than 5 blocks",
    "1 hour 1 minutes - 2 hour": "1 hour 1 minutes - 2 hours",
    "Over 65": "65+",
}

# Preferred display order for ordinal single-value questions
ORDER = {
    "often": ["Daily", "Weekly", "Monthly", "A couple times a year",
              "This is my first time visiting the neighborhood"],
    "sat": ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied",
            "Very satisfied"],
    "dur": ["1 - 15 minutes", "16 - 30 minutes", "31 - 45 minutes",
            "46 - 60 minutes", "60+ minutes"],
    "findspot": ["5 minutes or less", "6 - 10 minutes", "11 - 15 minutes",
                 "More than 15 minutes"],
    "blocks": ["Less than 1 block", "1 - 2 blocks", "3 - 5 blocks",
               "More than 5 blocks"],
    "parkdur": ["5 minutes or less", "6 - 10 minutes", "11 - 15 minutes",
                "16 - 20 minutes", "21 - 30 minutes", "31 minutes - 1 hour",
                "1 hour 1 minutes - 2 hours", "More than 2 hours"],
    "age": ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+",
            "Prefer not to say"],
    "eff": ["Effective- Captured most of what I had to say", "Neutral/Unsure",
            "Ineffective- Did not capture most what I had to say"],
    "paid": ["No", "Yes, on street", "Yes, off street"],
    "parked": ["Yes", "No"],
    "howget": None, "issue": None, "lang": None,
}

# ---------------------------------------------------------------------------
# Registry of comparable questions surfaced in the UI
# ---------------------------------------------------------------------------
QUESTIONS = [
    # key, label, kind
    ("sat",       "Bus trip satisfaction (1-5)",              "scale"),
    ("issue",     "Biggest issue with bus ride",             "single"),
    ("buslines_norm", "Bus lines used — normalized (checkbox + write-ins)", "buslines"),
    ("buslines",  "Bus lines used today (raw checkboxes)",    "multi"),
    ("often",     "How often visit Hillside Ave",            "single"),
    ("howget",    "How you got here today (mode)",           "single"),
    ("modes",     "Modes typically used",                    "multi"),
    ("dur",       "Travel time to Hillside",                 "single"),
    ("why",       "Reason for visit today",                  "multi"),
    ("parked",    "Parked on Hillside Ave",                  "single"),
    ("findspot",  "Time to find a parking spot",             "single"),
    ("blocks",    "Blocks from destination when parked",     "single"),
    ("paid",      "Paid to park",                            "single"),
    ("parkdur",   "How long parked",                         "single"),
    ("age",       "Age group",                               "single"),
    ("gender",    "Gender",                                  "multi"),
    ("race",      "Race / ethnicity",                        "multi"),
    ("lang",      "Survey language",                         "single"),
    ("eff",       "Survey effectiveness (self-rated)",       "single"),
    ("issue_text","Biggest transportation issue (open text)","text"),
    ("comments",  "Additional comments (open text)",         "text"),
]
QKIND = {k: kind for k, _, kind in QUESTIONS}
QLABEL = {k: lbl for k, lbl, _ in QUESTIONS}

# Fields offered as cross-year filters (applied identically to both years)
FILTER_FIELDS = [
    ("age", "Age group"),
    ("gender", "Gender"),
    ("race", "Race / ethnicity"),
    ("often", "Visit frequency"),
    ("howget", "Mode used today"),
    ("modes", "Modes typically used"),
    ("buslines", "Bus line used"),
    ("parked", "Parked on Hillside"),
    ("lang", "Language"),
    ("zip", "Home zip code"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_MIN_RANGE = [(15, "1 - 15 minutes"), (30, "16 - 30 minutes"),
              (45, "31 - 45 minutes"), (60, "46 - 60 minutes")]


def _parse_minutes(text: str):
    """Best-effort parse of a free-text travel time into minutes (2024)."""
    if not text:
        return None
    t = text.strip().lower()
    hours = 0.0
    mins = 0.0
    matched = False
    hm = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hora|hr)", t)
    if "hour" in t or "hora" in t or "hr" in t:
        matched = True
        hours = float(hm.group(1)) if hm else 1.0
        after = t.split("hour", 1)[-1] if "hour" in t else t
        mm = re.search(r"(\d+)\s*(?:min|minute)", after)
        if mm:
            mins = float(mm.group(1))
    else:
        nums = re.findall(r"\d+(?:\.\d+)?", t)
        if nums:
            matched = True
            vals = [float(n) for n in nums]
            mins = sum(vals) / len(vals)  # average handles ranges like "10-15"
    if not matched:
        return None
    return hours * 60 + mins


def _bin_minutes(m):
    if m is None:
        return None
    for hi, lbl in _MIN_RANGE:
        if m <= hi:
            return lbl
    return "60+ minutes"


def _norm_single(key: str, raw: str):
    v = (raw or "").strip()
    if v == "":
        return None
    if key == "sat":
        return SAT_LABEL.get(SAT_SCORE.get(v))
    if key == "howget":
        return MODE_CANON.get(v, v)
    return SIMPLE_FIX.get(v, v)


@lru_cache(maxsize=None)
def load_year(year: int) -> pd.DataFrame:
    """Return a tidy, harmonized respondent-level frame for one survey year."""
    raw = pd.read_csv(FILES[year], header=None, dtype=str,
                      keep_default_na=False, skiprows=[0, 1])
    n = len(raw)
    out = {"year": [year] * n, "respondent_id": raw[0].tolist()}

    # single-value / text questions
    for key, cols in SINGLE_COLS.items():
        ci = cols[year]
        series = raw[ci]
        if key in ("issue_text", "comments", "zip"):
            out[key] = [s.strip() if s.strip() else None for s in series]
        elif key == "dur" and year == 2024:
            out[key] = [_bin_minutes(_parse_minutes(s)) for s in series]
        elif key == "dur":  # 2026 already bucketed
            out[key] = [s.strip() or None for s in series]
        else:
            out[key] = [_norm_single(key, s) for s in series]
        if key == "sat":
            out["sat_score"] = [SAT_SCORE.get((s or "").strip()) for s in series]

    df = pd.DataFrame(out)

    # multi-select questions -> frozenset of canonical options selected
    for q, opts in MULTI_COLS.items():
        selected = [set() for _ in range(n)]
        for opt, ymap in opts.items():
            for ci in ymap[year]:
                col = raw[ci]
                for i, val in enumerate(col):
                    if val.strip():
                        selected[i].add(opt)
        df[q] = [frozenset(s) for s in selected]

    # normalized bus lines: checkbox selections + parsed write-ins, folded
    # together (write-in Q43/Q17/etc. merge into their real line)
    other_col = raw[BUSLINE_OTHER_COL[year]]
    checkbox = {ln: MULTI_COLS["buslines"][ln][year][0] for ln in BUSLINE_CHECKBOX}
    norm = []
    for i in range(n):
        s = set()
        for ln, ci in checkbox.items():
            if raw.iat[i, ci].strip():
                s.add(ln)
        s |= parse_lines(other_col.iat[i])
        norm.append(frozenset(s))
    df["buslines_norm"] = norm
    return df


@lru_cache(maxsize=1)
def load_all() -> pd.DataFrame:
    return pd.concat([load_year(y) for y in YEARS], ignore_index=True)


# ---------------------------------------------------------------------------
# Option availability + ordering helpers
# ---------------------------------------------------------------------------
def multi_options(q: str, year: int | None = None):
    """Canonical options for a multi-select question, optionally limited to
    those present in a given year."""
    opts = MULTI_COLS[q]
    if year is None:
        return list(opts.keys())
    return [o for o, ym in opts.items() if ym[year]]


def option_in_year(q: str, opt: str, year: int) -> bool:
    return bool(MULTI_COLS[q][opt][year])


def order_for(key: str, present):
    """Order categories using the preferred ORDER, falling back to sorted."""
    pref = ORDER.get(key)
    present = list(present)
    if pref:
        ordered = [c for c in pref if c in present]
        ordered += sorted(c for c in present if c not in pref)
        return ordered
    return sorted(present)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """filters maps field -> list of accepted values. A respondent passes a
    filter if they match ANY selected value (membership for multi-selects)."""
    mask = pd.Series(True, index=df.index)
    for field, wanted in filters.items():
        if not wanted:
            continue
        wanted = set(wanted)
        if field in MULTI_COLS:
            mask &= df[field].apply(lambda s: bool(s & wanted))
        else:
            mask &= df[field].isin(wanted)
    return df[mask]


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------
def compare_single(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """% distribution of a single-value question by year (of respondents who
    answered that question that year)."""
    rows = []
    present = set()
    for year in YEARS:
        sub = df[df.year == year][key].dropna()
        vc = sub.value_counts()
        present |= set(vc.index)
        total = len(sub)
        rows.append((year, vc, total))
    cats = order_for(key, present)
    data = {"Category": cats}
    for year, vc, total in rows:
        data[f"{year} %"] = [round(100 * vc.get(c, 0) / total, 1) if total else 0.0
                             for c in cats]
        data[f"{year} n"] = [int(vc.get(c, 0)) for c in cats]
    out = pd.DataFrame(data)
    out.attrs["n"] = {year: total for year, _, total in rows}
    return out


def net_satisfaction(df: pd.DataFrame) -> dict:
    """Net satisfaction score per year.

    net = % rating Satisfied/Very satisfied (top-2)
        − % rating Dissatisfied/Very dissatisfied (bottom-2),
    over all respondents who gave a 1-5 rating (neutrals stay in the base).
    Ranges from -100 (all dissatisfied) to +100 (all satisfied).
    """
    res = {}
    for y in YEARS:
        sc = df[df.year == y]["sat_score"].dropna()
        n = len(sc)
        if n == 0:
            res[y] = dict(n=0, mean=None, top2=None, bottom2=None,
                          neutral=None, net=None)
            continue
        top2 = 100 * (sc >= 4).sum() / n
        bottom2 = 100 * (sc <= 2).sum() / n
        neutral = 100 * (sc == 3).sum() / n
        res[y] = dict(n=n, mean=float(sc.mean()), top2=top2, bottom2=bottom2,
                      neutral=neutral, net=top2 - bottom2)
    a, b = YEARS
    res["delta"] = (None if res[a]["net"] is None or res[b]["net"] is None
                    else res[b]["net"] - res[a]["net"])
    return res


def compare_multi(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """% of respondents selecting each option, by year. Options absent in a
    year are marked N/A rather than 0%."""
    counts = {}
    totals = {}
    for year in YEARS:
        sub = df[df.year == year]
        # respondents who answered the question at all (>=1 selection)
        answered = sub[sub[q].apply(len) > 0]
        totals[year] = len(answered)
        c = {}
        for s in answered[q]:
            for opt in s:
                c[opt] = c.get(opt, 0) + 1
        counts[year] = c
    all_opts = list(MULTI_COLS[q].keys())
    cats = order_for(q, all_opts)
    data = {"Option": cats}
    for year in YEARS:
        pcts, ns = [], []
        for opt in cats:
            if not option_in_year(q, opt, year):
                pcts.append(None)
                ns.append(None)
            else:
                cnt = counts[year].get(opt, 0)
                tot = totals[year]
                pcts.append(round(100 * cnt / tot, 1) if tot else 0.0)
                ns.append(cnt)
        data[f"{year} %"] = pcts
        data[f"{year} n"] = ns
    out = pd.DataFrame(data)
    out.attrs["n"] = totals
    return out


def compare_buslines(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top bus lines used, combining checkbox selections and parsed write-ins.

    Denominator each year = respondents who named at least one identifiable
    line. Lines beyond the top N are aggregated into one 'Other lines (tail)'
    row (respondent-level, no double counting). A '*' next to a line marks the
    ones that were offered as checkboxes; the rest were write-ins.
    """
    per_year = {y: df[df.year == y]["buslines_norm"] for y in YEARS}
    ident = {y: [s for s in per_year[y] if s] for y in YEARS}   # >=1 line
    totals = {y: len(ident[y]) for y in YEARS}

    combined = {}
    for y in YEARS:
        for s in ident[y]:
            for ln in s:
                combined[ln] = combined.get(ln, 0) + 1
    top = sorted(combined, key=lambda l: (-combined[l], l))[:top_n]
    top_set = set(top)

    rows = []
    per_line = {y: {ln: 0 for ln in top} for y in YEARS}
    tail = {y: 0 for y in YEARS}
    for y in YEARS:
        for s in ident[y]:
            for ln in s & top_set:
                per_line[y][ln] += 1
            if s - top_set:
                tail[y] += 1

    def mark(ln):
        return f"{ln}*" if ln in BUSLINE_CHECKBOX else ln

    data = {"Line": [mark(l) for l in top] + ["Other lines (tail)"]}
    for y in YEARS:
        pcts, ns = [], []
        for ln in top:
            c = per_line[y][ln]
            ns.append(c)
            pcts.append(round(100 * c / totals[y], 1) if totals[y] else 0.0)
        ns.append(tail[y])
        pcts.append(round(100 * tail[y] / totals[y], 1) if totals[y] else 0.0)
        data[f"{y} %"] = pcts
        data[f"{y} n"] = ns
    out = pd.DataFrame(data)
    out.attrs["n"] = totals
    return out


_WORD_RE = re.compile(r"[a-z']{3,}")
_STOP = set("""the and for you your are was were with have has that this they them but not
             from all any get got out too can will would there their here about into more
             just also than then when what which who whom why how our its it's don't didn't
             because these those over under some most much very lot able make made take
             hillside avenue ave bus lane""".split())


def text_summary(df: pd.DataFrame, key: str, top_n: int = 15):
    """Return (per-year word-frequency DataFrame, per-year sample responses)."""
    freqs, samples, counts = {}, {}, {}
    for year in YEARS:
        sub = df[df.year == year][key].dropna()
        sub = sub[sub.str.strip() != ""]
        counts[year] = len(sub)
        wc = {}
        for txt in sub:
            for w in _WORD_RE.findall(txt.lower()):
                if w in _STOP:
                    continue
                wc[w] = wc.get(w, 0) + 1
        freqs[year] = sorted(wc.items(), key=lambda kv: -kv[1])[:top_n]
        samples[year] = sub.head(30).tolist()
    return freqs, samples, counts
