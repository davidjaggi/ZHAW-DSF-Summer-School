---
title: "Winter Import Gap"
tags: [theme, energy-security, winter-gap]
---

# Winter Import Gap

Switzerland is **not** short of electricity on an annual basis — it is short of *winter*
electricity. This is the central planning problem behind the whole case study.

- **The pattern:** in 8 of the last 10 winters, domestic production has not covered domestic
  demand. Average net winter imports over the last decade: **~4.5 TWh/winter (~15% of winter
  demand)**, peaking at 9.7 TWh in 2016/17.
- **The legal cap:** [[Mantelerlass 2024]] caps future net winter electricity imports at
  **5 TWh/year**, making "close the winter gap domestically" binding federal policy — see
  [[Imports and Grid Interconnection]].
- **Why it happens:** hydro reservoirs and the [[Nuclear Power]] fleet produce relatively
  evenly across the year, but [[Solar PV]] is heavily skewed to summer — lowland PV produces
  roughly 3–4x more in summer than winter.
- **What closes it:** [[Alpine Solar (Solarexpress)]] (winter-favorable), [[Wind Power]]
  (relatively stronger in winter), [[Hydropower]] flexibility and pumped storage (see
  [[Grimsel 4 Pumped Storage]]), and demand-side flexibility — ETH Zurich's stress tests
  converge on no single "silver bullet" source closing this gap alone.
- **In the simulation:** this is the `Winter Balance` calculation in the workshop's Streamlit
  dashboard (seasonal yields: Hydro 40%, Nuclear 55%, Solar 25%, Biomass 50%, against 55% of
  annual demand) — see [`../../workshop-guide.md`](<../../workshop-guide.md>), section 4.

## See also

[[Nuclear Power]] · [[Solar PV]] · [[Hydropower]] · [[Swiss Nuclear Decommissioning Timeline]]

Full sourced detail: [[Energy Security and Reliability]], [[Current Energy Mix]]
(`solution/docs/04`, `01`).
