---
title: "energy_mix.csv — Data Dictionary"
tags: [data, dataset]
---

# `energy_mix.csv` — data dictionary

Structured, source-by-source dataset that powers the dashboard's charts and scenario slider. It operationalizes the qualitative research in `../01-*.md` … `../06-*.md` into comparable numbers.

**Important caveat:** the `current_twh_2024` and `proj_twh_2035_*` columns are **illustrative, order-of-magnitude estimates** synthesized by the authors from the cited sources (official Swiss statistics are not published as a clean per-technology 2035 forecast at this granularity, except for wind, whose figures match the official interim targets — see `../02-policy-and-legal-context.md`). Treat them as a teaching/scenario-exploration aid, not an official forecast. The `*_score` columns (1 = worst, 5 = best) are the authors' qualitative scoring, explained in each corresponding research note, used only to drive the dashboard's radar/scatter visuals.

| Column | Meaning |
|---|---|
| `source` | Energy source / investment option |
| `category` | `Existing` (already built), `Expansion` (grows an existing technology), `New` (not yet deployed at scale in CH), `Reference` (imports — a comparison baseline, not a domestic investment option) |
| `current_twh_2024` | Approximate current annual generation (TWh) |
| `proj_twh_2035_low` / `proj_twh_2035_high` | Illustrative low/high 2035 range under a business-as-usual vs. accelerated buildout scenario |
| `lcoe_usd_kwh_low` / `lcoe_usd_kwh_high` | Approximate levelized cost of electricity range, USD/kWh (see `../03-cost-comparison-lcoe.md` for derivation) |
| `winter_output_share_pct` | Rough share of annual output typically generated in winter months (Oct–Mar) — the key energy-security metric for Switzerland |
| `cost_score`, `environmental_score`, `reliability_score`, `scalability_score`, `implementation_speed_score`, `land_use_score`, `public_acceptance_score`, `energy_security_score` | 1–5 qualitative scores (5 = most favorable) for each of the case study's required aspects, one score column per aspect |
