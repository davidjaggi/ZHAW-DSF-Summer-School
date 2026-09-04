# Solution — working notes

## Python environment

Always use the local conda environment — never the system Python, and never create a new
venv. It lives one level above the repo, not inside it:

```
C:\Users\jagg\Documents\Research\20 Projects\DIZH Summer School\.conda\python.exe
```

Run everything through it, e.g.:

```
"C:\Users\jagg\Documents\Research\20 Projects\DIZH Summer School\.conda\python.exe" -m pip install -r dashboard/requirements.txt
"C:\Users\jagg\Documents\Research\20 Projects\DIZH Summer School\.conda\python.exe" -m streamlit run dashboard/app.py --server.headless true
```

The env's `Scripts\` directory is not on PATH, so invoke tools as `python.exe -m <module>`
rather than relying on `streamlit`/`pip` being resolvable directly. `conda` itself is also
not on PATH — don't rely on `conda activate`; call `python.exe` by its full path instead.

Streamlit's first run prompts interactively for an email address and hangs waiting on
stdin in a non-interactive shell. Avoid it by writing an empty `~/.streamlit/credentials.toml`
(`[general]` / `email = ""`) before the first run, or always pass `--server.headless true`.

## Data

The case-study dataset lives in `data/` (`data/energy_mix.csv`, documented in
`data/README.md`). It was previously under `docs/data/` — keep paths pointing at `data/`.
