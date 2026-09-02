# GenAI for Research and Prototyping

**Summer School Workshop: Digital Sustainable Finance (UZH & ZHAW)**

This is the facilitator/instructor guide for the session. The student-facing brief lives in
[`case-study.md`](case-study.md) (short version) and [`SPEC.md`](SPEC.md) (full four-task brief);
a worked solution lives in [`solution/`](solution/).

---

## 1. Executive Summary & Workshop Architecture

- **Target Audience:** Undergraduate students (primarily Finance and Business Administration
  with minimal or no prior programming experience).
- **Workshop Duration:** 2 Hours (120 Minutes).
- **Core Philosophy:** *Architect before Coder.* Shifting students from unstructured,
  trial-and-error prompting to grounded research synthesis, linked knowledge management,
  planning-mode architecture, and rapid prototyping with Generative AI.
- **Hands-on Case Study:** Simulating the Swiss Electricity Mix (2026–2035) under the statutory
  mandates of the Swiss Federal Electricity Act (Stromgesetz / Mantelerlass 2024) and the Energy
  Perspectives 2050+ (EP 2050+ ZERO Basis) using an interactive Streamlit dashboard.

---

## 2. Minute-by-Minute 2-Hour Workshop Timetable

| Time Window | Segment | Focus & Content | Delivery Format | Learning Outcome |
|---|---|---|---|---|
| 0:00 – 0:25 (25 min) | Lecture Block 1 | AI-Assisted Research & Knowledge Architecture — Grounded research & citation synthesis with Gemini; Markdown fundamentals & Obsidian knowledge graphs | Interactive Lecture & Live Demo | Fact-checking policy data, separating statutory targets from technical constraints, and managing atomic notes. |
| 0:25 – 0:50 (25 min) | Hands-On Lab 1 | Swiss Energy Transition 2035: Data Extraction — Querying Gemini for baseline production & targets; Structuring an atomic Markdown policy brief in pairs | Student Pair Exercise | Transforming raw policy documents into structured numerical matrices. |
| 0:50 – 1:15 (25 min) | Lecture Block 2 | Prompting, Planning Mode & AI Coding Workflows — The 3-Stage Prompting Protocol (Spec → Logic → Code); Agent loops (ReAct) & error debugging with Streamlit | Interactive Lecture & Code Walkthrough | Understanding how LLMs write UI code and iteratively resolving runtime tracebacks. |
| 1:15 – 1:50 (35 min) | Hands-On Lab 2 | Streamlit Scenario Engine Prototyping — Prompting Gemini to architect and generate the app; Adding policy levers, charts, and statutory alerts | Student Prototyping Lab | Running a functional web application with dynamic sliders and charts. |
| 1:50 – 2:00 (10 min) | Wrap-Up | Showcase, Synthesis & Policy Discussion — Lightning demos of scenario outcomes; Nuclear phase-out trade-offs vs. winter import gaps | Group Debrief & Wrap-Up | Connecting AI prototyping directly back to sustainable finance and policy analysis. |

---

## 3. Deep-Dive Lecture Blocks

### Lecture Block 1: AI for Grounded Research & Knowledge Management (25 min)

**A. The Research Trilemma in Frontier AI**

- *The Synthesizer vs. The Oracle:*
  - LLMs are generative statistical reasoners, not static lookup databases.
  - Naive search queries ("Tell me about Swiss energy") yield generic, uncalibrated estimates.
  - Grounded extraction ("Extract the 2035 electricity production targets from the SFOE 2024
    Stromgesetz tables into a Markdown table with column headers: Carrier, 2024 Actual [TWh],
    2035 Target [TWh]") ensures deterministic, verifiable data.
- *Context Window Grounding:*
  - Providing primary policy documents (e.g., Swiss Federal Office of Energy executive
    summaries) directly to the model.
  - Requiring structured citations and cross-checking against official federal publications.

**B. Plaintext Markdown (.md) as the Lingua Franca of AI**

- *Why Markdown?*
  - Interoperable, human-readable, version-controllable, and zero proprietary lock-in.
  - LLMs produce native Markdown syntax (headings `#`, tables `|`, checklists `- [ ]`, code
    blocks ` ``` `).
- *Obsidian Demonstration (5–7 min):*
  - Concept of Atomic Notes: one idea per note.
  - Linking mechanism: `[[Swiss Nuclear Decommissioning Timeline]]` ↔
    `[[Winter Import Gap Strategy]]`.
  - Graph View: visualizing how technical constraints (solar seasonality) directly connect to
    legal boundaries (Stromgesetz 5 TWh winter import ceiling).
  - See [`Obsidian Guide/Markdown Guide.md`](../Obsidian%20Guide/Markdown%20Guide.md) for the
    reference syntax used in the demo, including the Obsidian-specific `[[link]]` conventions.

### Lecture Block 2: Prompting, Planning Mode & AI Coding Workflows (25 min)

**A. Why Direct Code Generation Fails for Non-Coders**

When students prompt "Build me an energy transition web app", LLMs generate monolithic, fragile
code that imports missing libraries, miscalculates dynamic arrays, or creates broken layouts.

The 3-Step Planning Protocol:

```
[ Step 1: System Spec Prompt ]   ➔ Define inputs, formulas, outputs in Markdown
               │
               ▼
[ Step 2: Architecture Review ]  ➔ Validate state variables & seasonal math
               │
               ▼
[ Step 3: Execution Prompt ]     ➔ Generate clean, modular Streamlit Python code
```

**B. Demystifying AI Agent Workflows & ReAct Loops**

The Agent Loop:

$$\text{User Objective} \longrightarrow \text{Thought (Reasoning)} \longrightarrow \text{Action (Tool/Code)} \longrightarrow \text{Observation (Output/Traceback)} \longrightarrow \text{Refinement}$$

*Live Debugging Demonstration:*

- Intentionally trigger a Streamlit runtime error (e.g.,
  `ValueError: All arrays must be of the same length`).
- Copy-paste the exact Python traceback into Gemini with the prompt: "Explain why this error
  occurred in line 42 and provide the corrected code snippet."
- Show students that debugging is an iterative conversation, not a fatal failure.

**C. Introduction to Streamlit Primitives**

Python UI mapping without HTML/CSS:

- `st.slider()` / `st.selectbox()` → User policy levers.
- `st.columns()` & `st.metric()` → Executive summary KPI cards.
- `st.plotly_chart()` → Interactive stacked generation mix vs. demand line.
- `st.error()` / `st.success()` → Automated statutory policy compliance checks.

---

## 4. Empirical Foundation: Swiss Energy Mix 2026–2035

### Ground-Truth Data Sheet (SFOE / BFE & Stromgesetz)

| Metric / Energy Carrier | Baseline (2024–2025 Actual) | 2035 Statutory Target (Stromgesetz) | Technical & Policy Constraints |
|---|---|---|---|
| Domestic Demand | 57.5 TWh | ~66.0–68.0 TWh | Increases at +1.0% to +1.4% p.a. due to heat pump and EV adoption. |
| Hydropower (Total) | 37.0–40.0 TWh (normalized baseline) | 37.9 TWh net generation | Run-of-river (16–18 TWh) + storage reservoirs (20–22 TWh). Includes +2 TWh guaranteed winter storage from 15 round-table hydro projects. |
| Nuclear Power | 23.0 TWh (~33% of domestic supply) | 0 to 17.0 TWh | 4 reactors: Beznau I/II (~6 TWh combined, scheduled retirement 2029–2031 under 60-yr lifespan), Gösgen (~8.5 TWh), Leibstadt (~9.5 TWh). |
| New Renewables (Solar PV, Wind, Biomass) | ~11.4 TWh (7.5 TWh PV, 3.9 TWh Biomass/Waste, 0.15 TWh Wind) | 35.0 TWh target (+23.6 TWh required) | Legally binding statutory target under the Swiss Stromgesetz. Requires adding ~2.5–2.75 TWh/year in solar PV capacity. |
| Winter Import Deficit | 3.0–5.0 TWh | ≤ 5.0 TWh (Legal Cap) | Winter represents 55% of annual electricity consumption, while solar produces only 25–30% of its annual yield during winter months. |

### Mathematical Logic for the Simulation Model

Demand Projection:

$$\text{Demand}_t = 57.5 \times (1 + g_{\text{demand}})^t \quad \text{where } t \in [1, 10]$$

Solar Capacity Expansion:

$$\text{Solar}_t = 7.5 + (\Delta_{\text{solar\_annual}} \times t)$$

Seasonal Winter Balance Check (Q1 + Q4):

$$\text{Winter Balance}_t = \Big(0.40 \cdot \text{Hydro}_t + 0.55 \cdot \text{Nuclear}_t + 0.25 \cdot \text{Solar}_t + 0.50 \cdot \text{Biomass}\Big) - \Big(0.55 \cdot \text{Demand}_t\Big)$$

If $\text{Winter Balance}_t < -5.0\text{ TWh}$, trigger a Security of Supply Alert.

---

## 5. Step-by-Step Student Prompting Guide

### Step 1: Research Prompt (Gemini / AI Web Interface)

```text
You are a senior energy policy researcher at a Swiss university.
Analyze the Swiss electricity supply targets under the Swiss Energy Perspectives 2050+ and the 2035 "Stromgesetz" framework.

Provide a Markdown summary table of Switzerland's electricity mix covering:
1. Current baseline (2024-2025): Hydro (Run-of-river & Storage), Nuclear, Solar PV, Biomass/Waste, and Total Demand.
2. The 2035 Target according to the new Electricity Act (35 TWh from non-hydro renewables).
3. The seasonal challenge: Explain why summer production differs from winter production, and calculate the estimated winter import gap if nuclear is retired without reaching the 35 TWh solar target.
```

### Step 2: Planning Mode Prompt (Architecture Specification)

```text
I need to build a single-page interactive Python Streamlit application that allows a user to model Switzerland's 10-year electricity transition (2026 to 2035).

Do NOT write the code yet. First, act as a software architect and produce:
1. State Variables & Sliders: What 4-5 interactive controls should the user have?
2. Math Equations: How will total annual generation and the winter import deficit be calculated per year?
3. UI Layout Plan: How to organize metrics at the top, a stacked area chart of the 10-year mix in the middle, and scenario summary alerts at the bottom.
```

### Step 3: Streamlit Code Generation Prompt

```text
Great plan. Now write the complete, clean, single-file Python code using `streamlit` and `plotly.graph_objects` (or `pandas`).

Requirements:
- Left Sidebar with sliders:
  * Annual Solar PV growth rate (TWh added per year, default target to hit 35 TWh total by 2035)
  * Nuclear retirement pace (Baseline 60-yr lifespan, Accelerated phase-out by 2035, or Lifetime Extension)
  * Hydro output efficiency factor (85% to 115% to simulate climate/rainfall variations)
  * Annual electricity demand growth rate (0.5% to 2.0% per year)
- Top Row: Key KPIs for the year 2035 (Total Production, Total Demand, Net Annual Balance, Winter Deficit).
- Main Area:
  * Stacked area/line chart showing generation mix from 2026 to 2035.
  * Line overlay of projected demand.
- Policy Compliance Banners:
  * Check if new renewables reach >= 35.0 TWh by 2035 (Stromgesetz target).
  * Display a warning if the 2035 Winter Import Gap exceeds the 5.0 TWh statutory resilience cap.
Ensure the code is self-contained and ready to execute.
```

---

## 6. Complete Streamlit Reference Code

This is the reference implementation that the 3-step prompting guide above should converge on.
A worked, repo-integrated version (reading from the shared case-study dataset rather than
hardcoded constants) lives in [`solution/dashboard/app.py`](solution/dashboard/app.py).

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Swiss Energy Mix 2026-2035 Scenario Engine",
    page_icon="🇨🇭",
    layout="wide"
)

st.title("🇨🇭 Swiss Electricity Transition Engine (2026–2035)")
st.markdown(
    "Calibrated with official **SFOE / BFE baseline data**, the **Stromgesetz 2035 Targets**, "
    "and the **Energy Perspectives 2050+** roadmap."
)

# --- SIDEBAR: POLICY CONTROLS ---
st.sidebar.header("Policy & Market Levers (2026–2035)")

solar_annual_add = st.sidebar.slider(
    "Solar PV Annual Addition (TWh/year)",
    min_value=1.0, max_value=4.0, value=2.75, step=0.25,
    help="Stromgesetz target requires adding ~2.75 TWh/year to hit 35 TWh total new renewables by 2035."
)

nuclear_scenario = st.sidebar.selectbox(
    "Nuclear Decommissioning Trajectory",
    options=[
        "Baseline (60-yr lifespan: Beznau decommissioned by 2031)",
        "Accelerated Phase-Out (Linear exit by 2035)",
        "Lifetime Extension (Operate all units through 2035)"
    ]
)

hydro_variability = st.sidebar.slider(
    "Hydropower Yield Index (Rainfall / Snowpack)",
    min_value=0.85, max_value=1.15, value=1.00, step=0.05,
    help="1.0 = Normal year (~37.0 TWh net). 1.15 = Wet year like 2024."
)

demand_growth = st.sidebar.slider(
    "Annual Demand Growth Rate (%)",
    min_value=0.5, max_value=2.0, value=1.2, step=0.1,
    help="Driven by EV electrification and heat pump installations (EP 2050+ estimates +1.0% to +1.4% p.a.)."
)

# --- SIMULATION ENGINE (2026 - 2035) ---
years = list(range(2026, 2036))
data = []

base_hydro = 37.0 * hydro_variability
base_biomass = 3.9  # Waste and biomass generation
base_wind = 0.15

for i, year in enumerate(years):
    t = i + 1  # 1 to 10 years ahead

    # 1. Total Demand Growth
    demand = 57.5 * ((1 + (demand_growth / 100)) ** t)

    # 2. Solar PV Progression (Starting from ~7.5 TWh baseline)
    solar = 7.5 + (solar_annual_add * t)
    new_renewables = solar + base_biomass + base_wind

    # 3. Nuclear Trajectory (23.0 TWh baseline)
    if nuclear_scenario == "Baseline (60-yr lifespan: Beznau decommissioned by 2031)":
        if year < 2029:
            nuclear = 23.0
        elif year == 2029:
            nuclear = 20.0
        elif year == 2030:
            nuclear = 18.5
        else:
            nuclear = 17.0
    elif nuclear_scenario == "Accelerated Phase-Out (Linear exit by 2035)":
        nuclear = max(0.0, 23.0 - (2.3 * t))
    else:  # Lifetime Extension
        nuclear = 23.0

    hydro = base_hydro
    total_production = hydro + nuclear + new_renewables
    annual_balance = total_production - demand

    # 4. Seasonal Winter Calculation (Q1 + Q4)
    # Winter represents 55% of annual demand; seasonal yields: Hydro 40%, Solar 25%, Nuclear 55%
    winter_demand = demand * 0.55
    winter_production = (hydro * 0.40) + (nuclear * 0.55) + (solar * 0.25) + ((base_biomass + base_wind) * 0.50)
    winter_balance = winter_production - winter_demand  # Negative = Net Import requirement

    data.append({
        "Year": year,
        "Hydro": hydro,
        "Nuclear": nuclear,
        "Solar PV": solar,
        "Biomass & Wind": base_biomass + base_wind,
        "Total Production": total_production,
        "Demand": demand,
        "Annual Balance": annual_balance,
        "Winter Balance": winter_balance
    })

df = pd.DataFrame(data)
final_2035 = df.iloc[-1]

# --- DASHBOARD METRICS DISPLAY ---
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "2035 Total Production",
    f"{final_2035['Total Production']:.1f} TWh",
    f"{final_2035['Total Production'] - 67.5:+.1f} vs 2025"
)
col2.metric(
    "2035 Projected Demand",
    f"{final_2035['Demand']:.1f} TWh",
    f"{final_2035['Demand'] - 57.5:+.1f} vs 2025"
)
col3.metric(
    "2035 Net Annual Balance",
    f"{final_2035['Annual Balance']:+.1f} TWh",
    "Net Export" if final_2035['Annual Balance'] >= 0 else "Net Import",
    delta_color="normal" if final_2035['Annual Balance'] >= 0 else "inverse"
)
col4.metric(
    "2035 Winter Balance",
    f"{final_2035['Winter Balance']:.1f} TWh",
    "Resilient" if final_2035['Winter Balance'] >= -5.0 else "Critical Deficit",
    delta_color="normal" if final_2035['Winter Balance'] >= -5.0 else "inverse"
)

# --- VISUALIZATION: 10-YEAR GENERATION TRAJECTORY ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Year'], y=df['Hydro'], name='Hydropower',
    mode='lines', stackgroup='one', line=dict(color='#1f77b4')
))
fig.add_trace(go.Scatter(
    x=df['Year'], y=df['Nuclear'], name='Nuclear Power',
    mode='lines', stackgroup='one', line=dict(color='#ff7f0e')
))
fig.add_trace(go.Scatter(
    x=df['Year'], y=df['Solar PV'], name='Solar PV',
    mode='lines', stackgroup='one', line=dict(color='#f1c40f')
))
fig.add_trace(go.Scatter(
    x=df['Year'], y=df['Biomass & Wind'], name='Biomass & Wind',
    mode='lines', stackgroup='one', line=dict(color='#2ca02c')
))
fig.add_trace(go.Scatter(
    x=df['Year'], y=df['Demand'], name='Total Electricity Demand',
    mode='lines+markers', line=dict(color='black', width=3, dash='dash')
))

fig.update_layout(
    title="Projected Swiss Electricity Mix vs. Demand (2026–2035)",
    xaxis_title="Year",
    yaxis_title="Terawatt-Hours (TWh)",
    hovermode="x unified",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- STATUTORY BENCHMARK ALERTS ---
st.markdown("### 📋 Statutory Policy Audit (2035 Benchmark)")
c1, c2 = st.columns(2)

with c1:
    new_renewables_2035 = final_2035['Solar PV'] + final_2035['Biomass & Wind']
    if new_renewables_2035 >= 35.0:
        st.success(f"✅ **Stromgesetz Target Met**: New renewables reach **{new_renewables_2035:.1f} TWh/a** (Target: ≥ 35.0 TWh).")
    else:
        st.warning(f"⚠️ **Target Missed**: New renewables reach **{new_renewables_2035:.1f} TWh/a** (Gap: {35.0 - new_renewables_2035:.1f} TWh).")

with c2:
    if final_2035['Winter Balance'] < -5.0:
        st.error(f"🚨 **Security of Supply Breach**: Winter import requirement is **{abs(final_2035['Winter Balance']):.1f} TWh**, exceeding the statutory **5.0 TWh safety cap**.")
    else:
        st.success(f"✅ **Winter Resilience Intact**: Winter import requirement is **{abs(final_2035['Winter Balance']):.1f} TWh** (within the 5.0 TWh safety cap).")
```

---

## 7. Fast-Track Google Colab Launch Guide (Zero-Install Setup)

To allow students with no local Python or terminal environment to run the app in their browser:

1. **Open Google Colab:** navigate to [colab.research.google.com](https://colab.research.google.com).
2. **Setup cell:**

   ```python
   !pip install -q streamlit pyngrok plotly pandas

   # Write the Streamlit application file
   %%writefile app.py
   # (Paste the reference Streamlit code here)
   ```

3. **Launch via localtunnel:**

   ```python
   # Launch Streamlit in the background
   import subprocess
   subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

   # Expose via localtunnel
   !npx localtunnel --port 8501
   ```

4. **Access the dashboard:** open the generated `localtunnel.me` URL and enter the external IP
   address returned by `!curl ipv4.icanhazip.com`.
