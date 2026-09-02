---
title: "Switzerland's Future Energy Mix — Report & Recommendation"
tags: [report, recommendation, case-study]
aliases: ["Recommended Portfolio"]
---

# Switzerland's Future Energy Mix — Report & Recommendation

**Prepared for:** Swiss Federal Office of Energy (case-study role-play)
**Prepared by:** ZHAW DSF Summer School — GenAI for Research and Prototyping
**Date:** August 2026
**Companion materials:** [`docs/`](docs/) (sourced research notes and dataset) · [`dashboard/`](dashboard/) (interactive scenario explorer)

---

## Executive summary

> **Which energy source, or mix of energy sources, should Switzerland invest in?**

Switzerland should invest in a **diversified, winter-weighted renewables-plus-flexibility portfolio**, not a single "winner" technology:

1. **Maximize solar PV**, on both the Plateau (cheapest, fastest) and in the Alps (more expensive, but disproportionately valuable because it produces in winter).
2. **Expand hydro flexibility** (pumped storage, reservoir efficiency retrofits like Grimsel 4) rather than seeking new large reservoirs — there is little new large-hydro *volume* left to build, but a lot of *flexibility* value to unlock.
3. **Keep the existing nuclear fleet running** as long as it remains safe and economical — it is Switzerland's cheapest reliable winter baseload and shutting it early would only deepen the import/security problem.
4. **Grow wind modestly** where local acceptance allows — a small but genuinely winter-favorable contributor.
5. **Invest heavily in grid, storage, and demand-side flexibility** — the "fourth source" that lets more solar and wind be used without compromising reliability, and the fastest lever for closing the winter gap.
6. **Treat new nuclear as a 2040s-and-beyond option, not a 2030s answer.** Even if the pending referendum lifts the construction ban, no new plant can be designed, licensed, and built within the next 10 years. It should be kept as a strategic option (site studies, regulatory readiness) without being relied upon in this decade's supply plan.
7. **Actively reduce winter import dependence** toward the legal 5 TWh/year cap — not by refusing imports on principle, but by building enough winter-productive domestic capacity that imports become a genuine backstop rather than a structural necessity.

This is a **portfolio answer, not a single-source answer**, because Switzerland's core problem — a winter supply gap in an otherwise >90%-low-carbon system — is best solved by combining sources with complementary seasonal, cost, and risk profiles, not by picking the single "best" one on any one dimension. This mirrors the direction already set in Swiss law (the 2024 Mantelerlass), which this report treats as the baseline scenario to accelerate, not a starting point to reargue from zero.

---

## 1. The problem, precisely stated

Switzerland does **not** have an annual electricity shortage — it has a **seasonal one**. The country already gets roughly 90%+ of its electricity from low-carbon sources (hydro + nuclear + new renewables). The real decision is **which mix best closes the winter gap** (Switzerland has been a net winter importer in 8 of the last 10 years, ~4.5 TWh/winter on average) while meeting the Climate and Innovation Act's net-zero-2050 target, replacing an aging nuclear fleet, and doing so within a direct-democracy system where every large project can be — and often is — put to a public vote.

*(Full data and citations: [`docs/01-current-energy-mix.md`](docs/01-current-energy-mix.md), [`docs/04-energy-security-and-reliability.md`](docs/04-energy-security-and-reliability.md))*

---

## 2. Aspect-by-aspect analysis

### 2.1 Cost

New-build **solar PV** is the cheapest source available (global benchmark ~$0.04/kWh, competitive in the Plateau; alpine sites cost more but still moderate). **Existing hydro and nuclear** are the cheapest electricity Switzerland will ever have, being largely sunk investments with low marginal cost — the priority is to keep them running, not to think of them as "new spend." **New nuclear** is the most expensive and highest-risk option by a wide margin, with international precedent (Flamanville, Hinkley Point C) pointing to large cost and schedule overruns. **Wind** is cheap on a pure LCOE basis but Switzerland's usable resource is small. **Winter imports** are not free — they are priced at volatile European winter market rates, and every TWh of avoided import has hedging value beyond its own price.
→ *Detail: [`docs/03-cost-comparison-lcoe.md`](docs/03-cost-comparison-lcoe.md)*

### 2.2 Environmental impact

All the technologies under serious consideration (hydro, nuclear, solar, wind, biomass, geothermal) are low-carbon relative to fossil generation — this is not a fossil-vs-clean decision, it's a clean-vs-clean tradeoff. The real environmental differentiators are: nuclear waste and low-probability accident risk; hydro's legacy river-ecology impact (a sunk cost for existing dams, a hard constraint on new ones); solar's manufacturing footprint and land competition; and wind's localized forest-clearance and wildlife impact, which has become the specific flashpoint of Swiss wind opposition.
→ *Detail: [`docs/06-environmental-impact-scalability-implementation-time.md`](docs/06-environmental-impact-scalability-implementation-time.md)*

### 2.3 Reliability

**Hydro and nuclear are the most reliable, dispatchable sources** Switzerland has, and both types of plant complement solar and wind well. **Solar is weather- and season-dependent**, structurally weak in winter at low altitude, but meaningfully better at alpine sites. **Wind is complementary** to solar (relatively stronger in winter) but too small in current scale to move the reliability needle much on its own. **Storage and flexibility** (pumped hydro, batteries, demand response) are what let an increasingly renewable system stay reliable — underinvestment here is a bigger risk than underinvestment in any single generation technology.
→ *Detail: [`docs/04-energy-security-and-reliability.md`](docs/04-energy-security-and-reliability.md)*

### 2.4 Scalability

**Solar is by far the largest realistic growth lever** for the next decade (already tripling roughly every five years). **Wind and hydro-flexibility** can grow meaningfully in relative terms but are small/moderate in absolute terms (small resource base for wind; few remaining large hydro sites). **Nuclear and deep geothermal have essentially zero scalability within a 10-year window** — nuclear because of build time, geothermal-for-electricity because it remains pre-commercial in Switzerland. **Biomass has a low, largely fixed ceiling** (~10% of primary energy at full exploitation, already ~50% used).
→ *Detail: [`docs/06-environmental-impact-scalability-implementation-time.md`](docs/06-environmental-impact-scalability-implementation-time.md)*

### 2.5 Implementation time

This is the aspect that most directly rules **new nuclear out of the 2026–2036 window**: even under an optimistic legal and construction timeline, 15–20+ years is the realistic benchmark, versus months for rooftop solar, 2–4 years for large ground-mount/alpine solar under the Solarexpress fast-track, and ~7–8 years for hydro-flexibility retrofits like Grimsel 4 (permit filed late 2024, commissioning ~2031/32). Wind sits at the slow end for a renewable (5–10+ years) mainly because of local-opposition friction, not technology.
→ *Detail: [`docs/06-environmental-impact-scalability-implementation-time.md`](docs/06-environmental-impact-scalability-implementation-time.md)*

### 2.6 Land use

Switzerland's land constraint is less about total hectares and more about **which** land is used: Plateau land is valuable for agriculture and housing (constrains ground-mount solar); Alpine land is landscape- and tourism-sensitive (constrains alpine solar and wind visually, though the physical footprint is small); existing hydro reservoirs are large but already built, so *new* incremental land take from hydro is now small (flexibility retrofits reuse existing reservoirs); nuclear has the smallest physical footprint of any option, existing or new.
→ *Detail: [`docs/05-land-use-and-public-acceptance.md`](docs/05-land-use-and-public-acceptance.md)*

### 2.7 Public acceptance

This is arguably the **most decisive aspect in the Swiss context**, because direct democracy makes acceptance operationally binding, not just a sentiment. **Alpine solar** has mixed but workable acceptance (roughly half of projects approved locally). **Wind is the most contested technology today**, the target of two dedicated 2025 popular initiatives. **Nuclear's status is genuinely unresolved** — a 2017 vote banned new plants, but a 2024–2026 counter-campaign has reopened the question, with the outcome pending. **Hydro remains the most broadly accepted** source, helped by its long history and multi-purpose reservoirs.
→ *Detail: [`docs/05-land-use-and-public-acceptance.md`](docs/05-land-use-and-public-acceptance.md)*

### 2.8 Energy security

Switzerland's energy security question is specifically a **winter-supply** question, not a total-volume one. Sources that are most valuable for security are those that produce disproportionately in winter or on demand: dispatchable hydro, the existing (flat-output) nuclear fleet, alpine solar, and wind. Lowland solar is the weakest contributor to security despite being the cheapest and most scalable source — this tension is exactly why the recommended mix pairs it with winter-favorable technologies rather than relying on it alone. Imports remain a necessary backstop but are, by Swiss law itself (the 5 TWh/year cap), explicitly *not* meant to be the primary security mechanism going forward.
→ *Detail: [`docs/04-energy-security-and-reliability.md`](docs/04-energy-security-and-reliability.md)*

---

## 3. Recommended investment portfolio (10-year horizon, through ~2036)

| Priority | Action | Rationale |
|---|---|---|
| 1 | Accelerate Plateau + rooftop solar PV toward the top of the Mantelerlass's 35 TWh/2035 renewables target | Cheapest, fastest, most scalable source available today |
| 2 | Continue and expand Solarexpress-class alpine solar with mandatory local revenue-sharing | Disproportionate winter value; acceptance is achievable but not automatic — needs active community engagement |
| 3 | Fund hydro-flexibility retrofits (pumped storage, e.g. Grimsel 4-style projects) | Best available lever to make a renewables-heavy grid reliable; low incremental environmental/land impact |
| 4 | Keep the existing nuclear fleet operating at full safety-approved lifetime | Cheapest reliable winter baseload already built; early closure only worsens the winter gap it would need to be replaced |
| 5 | Grow wind where local votes support it; do not force it against organized local opposition | Genuine winter value, but the most acceptance-constrained technology — respect the outcome of direct democracy rather than trying to override it |
| 6 | Invest in grid reinforcement, storage, and demand-response markets as a parallel, equally funded track | Enables everything above; historically underfunded relative to generation |
| 7 | Maintain the existing biomass base; do not count on major growth | Small, largely fixed domestic potential |
| 8 | Track new nuclear and deep geothermal as **post-2036 strategic options**: fund site studies and regulatory readiness now, but do not include either in the 2026–2036 supply plan | Neither can deliver new capacity within this decade regardless of legal/technical progress |
| 9 | Manage winter imports down toward the 5 TWh/year legal cap as domestic winter-capacity comes online, rather than eliminating them abruptly | Imports remain a legitimate backstop and reflect real European market integration, not a failure in themselves |

The [interactive dashboard](dashboard/) lets you adjust the weight given to each aspect (cost, reliability, public acceptance, etc.) and each source's share of new investment, and see how the resulting portfolio's projected 2035 mix, winter-output share, and blended score change — reproducing the tradeoffs summarized above, in your own words to a room of stakeholders who may weight the aspects differently.

---

## 4. Policy recommendations

1. **Keep and enforce the 2024 Mantelerlass targets** (35 TWh/year new renewables by 2035, 5 TWh/year net winter import cap) as the binding baseline — this report's recommendation is an acceleration of the existing legal direction, not a departure from it.
2. **Make local acceptance a funded policy instrument, not an afterthought.** Mandate community revenue-sharing / local ownership stakes for alpine solar and wind projects, following the pattern of the roughly-50% approval rate already seen where projects go to local votes. Acceptance should be actively built, not assumed.
3. **Finish the single-cantonal-approval-process reform** (in force from April 2026) and monitor whether it meaningfully cuts permitting time for solar and wind; extend or adjust if the 2025 anti-wind initiatives succeed and change the legal landscape.
4. **Fund grid and storage on the same multi-year budget cycle as generation**, not as a residual — the ETH Zurich stress-test literature consistently identifies flexibility, not any single generation technology, as the binding constraint on reliability.
5. **Resolve the nuclear question through the pending referendum, then plan accordingly** — regardless of the outcome, no new plant affects the next decade's supply picture, so policy should decouple "keep the existing fleet running safely" (recommended regardless of the vote) from "permit new construction" (a longer-horizon strategic question that should not distract from the near-term renewables-plus-flexibility buildout).
6. **Set explicit, published interim milestones for winter-output share** (not just total TWh), so that solar buildout that is fast but winter-weak doesn't crowd out investment in genuinely winter-productive sources (alpine PV, wind, hydro flexibility).
7. **Pair generation policy with demand-side measures** (heat-pump and EV-charging flexibility, industrial demand response, continued efficiency gains) — the Energy Strategy 2050's per-capita consumption target remains a valid and underused lever for closing the winter gap from the demand side, not just the supply side.
8. **Maintain European grid interconnection and market access as a deliberate security asset**, alongside — not instead of — reducing structural import dependence; Switzerland's position at the center of the European grid is a genuine strength that a purely "self-sufficiency" framing would squander.

---

## 5. Limitations of this report

This report was produced as a 2-hour teaching case study using targeted web research (see [`docs/README.md`](docs/README.md) for methodology and full source list) rather than a full commissioned energy-system study. Several figures — particularly 2035 per-technology projections in the dashboard dataset — are **illustrative, order-of-magnitude estimates**, not official forecasts; they are clearly flagged as such in [`docs/data/README.md`](docs/data/README.md). The nuclear-ban referendum outcome was **still pending at the time of writing (August 2026)**; the recommendation above is designed to be robust to either outcome, but the reader should check the actual result before treating the nuclear-related conclusions as final.
