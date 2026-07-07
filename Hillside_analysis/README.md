# Hillside Avenue Bus Lane — Survey Comparison App

Interactive comparison of two NYC DOT post-implementation surveys about the
Hillside Avenue bus lane: **2024** (earlier post-install) vs **2026** (later).

The two SurveyMonkey exports don't use identical wording or answer choices, so
the app harmonizes both onto a shared schema and only compares questions that
are genuinely pre/post comparable.

## Files
- `hillside2024.csv`, `hillside2026.csv` — raw SurveyMonkey exports (2-row header).
- `survey_lib.py` — schema, harmonization, loading, filtering, comparison logic.
- `report.py` — static HTML report generator (shared chart helper + CLI).
- `app.py` — Streamlit UI.

## Run
```bash
# from D:\Python files\Hillside_analysis
..\.venv\Scripts\python.exe -m streamlit run app.py
```
Then open http://localhost:8501.

## What it does
- **Unified filters** (age, gender, race, visit frequency, mode, bus line,
  parked, language, zip) apply to *both* years at once, so every comparison is
  drawn on the same subgroup.
- **Compare a question** tab — grouped 2024/2026 bar chart + a table with the
  percentage-point shift. Handles single-answer, 1–5 scale (with mean delta),
  multi-select (% selecting, blank where an option wasn't offered that year),
  and open-text (word frequencies + sample responses).
- **All-question overview** tab — top answer and 2024→2026 shift for every
  comparable question at a glance.
- **Normalized bus lines** — the survey's checkbox list only offered
  Q1/Q17/Q36/Q43/Q76/Q82/N6/N22/N24, so ~40% of riders wrote their route into
  "Other". The *"Bus lines used — normalized"* question parses those write-ins,
  folds redundant Q43/Q17/Q76 write-ins back into their real totals, and shows
  the top 15 lines (checkbox lines marked `*`) with everything else collapsed
  into one "Other lines (tail)" bar. The raw checkbox question is kept
  alongside it for transparency.
- **Net satisfaction score** — headline metric shown on the satisfaction view
  and the overview tab: `net = %(Satisfied + Very satisfied) − %(Dissatisfied +
  Very dissatisfied)`, over everyone who gave a 1–5 rating (neutrals stay in the
  base). Range −100 to +100. Updates live with the filters.
- **Data & notes** tab — the harmonization methodology, a filtered CSV
  download, and a **static HTML report** download that captures every
  comparison for the current filters (openable in any browser, printable to PDF).

## Static report
Generate a full-dataset report from the command line:
```bash
..\.venv\Scripts\python.exe report.py                 # -> hillside_report.html
..\.venv\Scripts\python.exe report.py my_report.html  # custom path
```
Or download a filtered report from the app's **Data & notes** tab.

## Harmonization highlights
- Satisfaction mapped to a common 1–5 scale.
- 2024 free-text travel times parsed and binned into 2026's time buckets.
- Renamed/split choices recombined (e.g. Social/Recreational + Religious).
- Male/Female → Man/Woman; `65+`/`Over 65` merged; spelling normalized.
- 2024-only / 2026-only questions and options are excluded or flagged as N/A.
