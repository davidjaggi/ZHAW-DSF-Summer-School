# Solution — working notes

## Python environment

Always use the repository-local conda environment at the repo root — never the system
Python, and never create a new venv:

```
E:\PycharmProjects\ZHAW-DSF-Summer-School\.conda\python.exe
```

Run everything through it, e.g.:

```
"E:\PycharmProjects\ZHAW-DSF-Summer-School\.conda\python.exe" -m pip install -r dashboard/requirements.txt
"E:\PycharmProjects\ZHAW-DSF-Summer-School\.conda\python.exe" -m streamlit run dashboard/app.py
```

The env's `Scripts\` directory is not on PATH, so invoke tools as `python.exe -m <module>`
rather than relying on `streamlit`/`pip` being resolvable directly.

## Data

The case-study dataset lives in `data/` (`data/energy_mix.csv`, documented in
`data/README.md`). It was previously under `docs/data/` — keep paths pointing at `data/`.
