# Solution — Switzerland's Future Energy Mix

Solution to [`0. Casestudy.md`](<../Course Materials/0. Casestudy.md>) (see also [`SPEC.md`](<../Course Materials/SPEC.md>) for the full four-task brief).

## What's in here

| Path | Task it answers | Description |
|---|---|---|
| [`report.md`](report.md) | Tasks 2 & 4 | The full report: recommendation, per-aspect analysis, policy recommendations |
| [`docs/`](docs/) | Task 1 & 2 | Sourced research notes (with citations) and the structured dataset behind the dashboard |
| [`dashboard/`](dashboard/) | Task 3 | Interactive Streamlit + Plotly dashboard for exploring the data and different investment scenarios |

## Running the interactive dashboard

The dashboard is a [Streamlit](https://streamlit.io) app. You need Python 3.10+.

### 1. Set up a virtual environment and install dependencies

From the `dashboard/` folder:

```bash
cd "GenAI for Research and Prototyping/solution/dashboard"
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(If you're already using the repo-wide conda environment from [`environment.yml`](../../../environment.yml), `pandas` and `plotly` are already included — you only need to additionally `pip install streamlit`.)

### 2. Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (default `http://localhost:8501`) — open it in your browser. The app auto-reloads when you edit `app.py`.

### 3. What you can do in the dashboard

- **Current & Projected Mix** — grouped bar and pie charts of today's generation mix vs. illustrative 2035 ranges.
- **Compare Sources** — radar chart comparing up to 5 sources across all eight case-study aspects (cost, environmental impact, reliability, scalability, implementation time, land use, public acceptance, energy security).
- **Cost vs. Reliability** — bubble chart trading off approximate LCOE against reliability, sized by scalability.
- **Priority-Weighted Ranking** — set your own importance weight (0–5) for each of the eight aspects and see which source ranks best under your priorities; useful for role-playing different stakeholders (e.g. a cost-focused finance ministry vs. a reliability-focused grid operator).
- **2035 Investment Scenario** — allocate a target amount of new generation (default: the legal 35 TWh/year Mantelerlass target) across sources with sliders, and see the resulting winter-output share and blended reliability/acceptance scores.

### Data source

The dashboard reads [`docs/data/energy_mix.csv`](docs/data/energy_mix.csv) directly (path resolved relative to `app.py`, so it works regardless of your current working directory). See [`docs/data/README.md`](docs/data/README.md) for the data dictionary and an important caveat: the 2035 figures are illustrative estimates for scenario exploration, not an official forecast.

### Troubleshooting

- **`streamlit: command not found`** — make sure the virtual environment is activated (step 1) before running step 2.
- **Port already in use** — run `streamlit run app.py --server.port 8502` (or any free port).
- **Blank/error page on first load** — confirm you're running the command from inside the `dashboard/` folder, or that `docs/data/energy_mix.csv` still exists two directories up; the app resolves this path automatically but the file must exist on disk.
