# Case Study: Switzerland's Future Energy Mix

**ZHAW Summer School — GenAI for Research and Prototyping**

*Time: no more than 2 hours · Work in small teams*

---

## Your task

You are part of the Swiss energy agency. Switzerland's energy consumption over the next
10 years is a given — your job is not to forecast demand, but to decide **where that
energy should come from**.

Draft a report and prototype an interactive dashboard that answer the following question:

> **What should Switzerland's energy mix look like over the next 10 years?**

Develop an evidence-based recommendation and prototype an interactive dashboard that
allows policymakers to understand the trade-offs behind your proposed mix.

---

## The constraint

Your proposed energy mix must **sum to 100%**. Instead of estimating absolute
volumes (TWh), express your recommendation as a share of total energy consumption per
source, e.g.:

| Energy source | Recommended share |
|---|---:|
| Hydropower | 45% |
| Solar | 30% |
| Nuclear | 15% |
| Wind | 7% |
| Biomass/other | 3% |
| **Total** | **100%** |

You then need to defend **why** this mix is preferable to today's mix.

---

## How to get there

Structure your work around three questions:

1. **Where are we today?** Use the [Swiss Energy Dashboard](https://www.energiedashboard.admin.ch/energie/energieverbrauch)
   and the [Federal Office of Energy's overall energy statistics](https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/overall-energy-statistics.html/)
   as your starting point to understand Switzerland's current energy consumption and mix.
2. **What are the trade-offs between technologies?** Research solar, hydro, wind,
   nuclear, and other relevant sources against the criteria below.
3. **What should Switzerland's future mix look like?** Recommend a share for each
   source (summing to 100%) and visualize your recommendation in a dashboard. As a
   reference point, the Federal Council's [Energy Perspectives 2050+](https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.html)
   already analyze one possible development of the energy system under its **Net-Zero
   (ZERO) scenario** — you can use it as a benchmark to compare your own recommendation
   against, or as a starting point to build on.

---

## Aspects to consider

Your report has to consider the following aspects:

- Cost
- CO₂ / environmental impact
- Reliability / intermittency
- Scalability
- Implementation time
- Land use
- Public acceptance
- Energy security / import dependence

Cost and CO₂ can be quantified with numbers. Aspects like public acceptance or
implementation complexity can be scored qualitatively (e.g. low/medium/high) — don't
let data cleaning eat into your research and prototyping time.

> **Do not evaluate technologies only in isolation.** Consider how they complement
> each other as part of an energy system — for example, how hydropower can balance
> intermittent solar and wind. The insight you're after is that a well-designed *mix*
> can outperform any single technology.

---

## Deliverable

### Report

Draft a report that:

- Recommends an energy mix for Switzerland, with shares summing to 100%.
- Considers each of the aspects listed above.
- Explains why the proposed mix is preferable to the current one.
- Covers **policy recommendations**.

### Dashboard

Prototype an interactive dashboard that, at minimum:

- Shows the **current mix vs. your recommended mix**.
- Compares energy sources across the main criteria above.
- Lets the user adjust the importance of 2–3 criteria and see how that changes the
  recommendation.
