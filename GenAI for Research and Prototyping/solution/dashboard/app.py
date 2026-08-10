"""Interactive dashboard for the "Switzerland's Future Energy Mix" case study.

Run with: streamlit run app.py
Data source: ../docs/data/energy_mix.csv (see docs/data/README.md for the data dictionary
and the caveat that projections/scores are illustrative estimates, not official forecasts).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "energy_mix.csv"

SCORE_COLUMNS = {
    "cost_score": "Cost",
    "environmental_score": "Environmental impact",
    "reliability_score": "Reliability",
    "scalability_score": "Scalability",
    "implementation_speed_score": "Implementation time",
    "land_use_score": "Land use",
    "public_acceptance_score": "Public acceptance",
    "energy_security_score": "Energy security",
}

CATEGORY_COLORS = {
    "Existing": "#4C78A8",
    "Expansion": "#F58518",
    "New": "#E45756",
    "Reference": "#B0B0B0",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["lcoe_mid_usd_kwh"] = (df["lcoe_usd_kwh_low"] + df["lcoe_usd_kwh_high"]) / 2
    return df


def aspect_label(col: str) -> str:
    return SCORE_COLUMNS.get(col, col)


st.set_page_config(
    page_title="Switzerland's Future Energy Mix",
    page_icon="\U0001f1e8\U0001f1ed",
    layout="wide",
)

df = load_data()

st.title("Switzerland's Future Energy Mix — Scenario Explorer")
st.caption(
    "ZHAW DSF Summer School · GenAI for Research and Prototyping · case study dashboard. "
    "Backing research and full source citations: see `docs/`. Full written recommendation: see `report.md`."
)
st.info(
    "Figures on this dashboard are **illustrative, order-of-magnitude estimates** synthesized from the "
    "cited research (see `docs/data/README.md`), not an official government forecast. Use it to explore "
    "trade-offs, not as a precise prediction.",
    icon="ℹ️",
)

source_filter = st.sidebar.multiselect(
    "Filter sources shown across the dashboard",
    options=df["source"].tolist(),
    default=df["source"].tolist(),
)
view = df[df["source"].isin(source_filter)].copy()

tab_mix, tab_compare, tab_cost, tab_priorities, tab_scenario = st.tabs(
    [
        "Current & Projected Mix",
        "Compare Sources",
        "Cost vs. Reliability",
        "Priority-Weighted Ranking",
        "2035 Investment Scenario",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 — Current & projected mix
# ---------------------------------------------------------------------------
with tab_mix:
    st.subheader("Current generation vs. illustrative 2035 range")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_bar(
            name="2024 (current)",
            x=view["source"],
            y=view["current_twh_2024"],
            marker_color="#4C78A8",
        )
        fig.add_bar(
            name="2035 (low estimate)",
            x=view["source"],
            y=view["proj_twh_2035_low"],
            marker_color="#F58518",
        )
        fig.add_bar(
            name="2035 (high estimate)",
            x=view["source"],
            y=view["proj_twh_2035_high"],
            marker_color="#E45756",
        )
        fig.update_layout(
            barmode="group",
            yaxis_title="TWh / year",
            xaxis_title=None,
            legend_title=None,
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pie = px.pie(
            view,
            names="source",
            values="current_twh_2024",
            title="Share of current (2024) generation",
            hole=0.4,
        )
        pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(pie, use_container_width=True)

    st.caption(
        "Wind figures match Switzerland's official interim targets (0.3 TWh by 2025, 1.2 TWh by 2035, "
        "see docs/02-policy-and-legal-context.md); other 2035 ranges are illustrative low/high scenarios."
    )

# ---------------------------------------------------------------------------
# Tab 2 — Compare sources (radar chart)
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Compare sources across all eight case-study aspects")
    compare_sources = st.multiselect(
        "Choose 2-5 sources to compare",
        options=view["source"].tolist(),
        default=view["source"].tolist()[:3] if len(view) >= 3 else view["source"].tolist(),
        max_selections=5,
    )

    if compare_sources:
        radar = go.Figure()
        categories = list(SCORE_COLUMNS.values())
        for src in compare_sources:
            row = view[view["source"] == src].iloc[0]
            values = [row[c] for c in SCORE_COLUMNS] + [row[list(SCORE_COLUMNS)[0]]]
            radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=src,
                )
            )
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=560,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(radar, use_container_width=True)

        st.dataframe(
            view[view["source"].isin(compare_sources)]
            .set_index("source")[list(SCORE_COLUMNS)]
            .rename(columns=aspect_label),
            use_container_width=True,
        )
    else:
        st.warning("Select at least one source to compare.")

# ---------------------------------------------------------------------------
# Tab 3 — Cost vs reliability scatter
# ---------------------------------------------------------------------------
with tab_cost:
    st.subheader("Cost vs. reliability, sized by scalability")
    scatter = px.scatter(
        view,
        x="lcoe_mid_usd_kwh",
        y="reliability_score",
        size="scalability_score",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        text="source",
        size_max=40,
        labels={
            "lcoe_mid_usd_kwh": "Approx. LCOE (USD/kWh, midpoint)",
            "reliability_score": "Reliability score (1-5)",
            "category": "Category",
        },
    )
    scatter.update_traces(textposition="top center")
    scatter.update_layout(height=560, yaxis=dict(range=[0, 5.5]))
    st.plotly_chart(scatter, use_container_width=True)
    st.caption(
        "Bubble size = scalability score. Bubbles toward the bottom-left are cheap but unreliable "
        "(e.g. lowland solar); toward the top-left are cheap AND reliable (existing hydro/nuclear); "
        "toward the right are expensive, long-lead-time options (new nuclear, deep geothermal)."
    )

# ---------------------------------------------------------------------------
# Tab 4 — Priority-weighted ranking
# ---------------------------------------------------------------------------
with tab_priorities:
    st.subheader("Rank sources by your own priorities")
    st.write(
        "Different stakeholders weight the case study's eight aspects differently. Set how important "
        "each aspect is to you (0 = ignore, 5 = critical) and see which sources come out on top."
    )

    weight_cols = st.columns(4)
    weights = {}
    for i, (col, label) in enumerate(SCORE_COLUMNS.items()):
        with weight_cols[i % 4]:
            weights[col] = st.slider(label, min_value=0, max_value=5, value=3, key=f"w_{col}")

    total_weight = sum(weights.values()) or 1
    view["weighted_score"] = sum(view[col] * w for col, w in weights.items()) / total_weight

    ranked = view.sort_values("weighted_score", ascending=True)
    bar = px.bar(
        ranked,
        x="weighted_score",
        y="source",
        orientation="h",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        labels={"weighted_score": "Weighted score (0-5)", "source": ""},
    )
    bar.update_layout(height=480)
    st.plotly_chart(bar, use_container_width=True)

    top = ranked.iloc[-1]
    st.success(
        f"Under your current weighting, **{top['source']}** ranks highest "
        f"(weighted score {top['weighted_score']:.2f} / 5)."
    )

# ---------------------------------------------------------------------------
# Tab 5 — 2035 investment allocation scenario
# ---------------------------------------------------------------------------
with tab_scenario:
    st.subheader("Build your own 2035 investment scenario")
    st.write(
        "Allocate Switzerland's Mantelerlass target of **35 TWh/year of new renewables by 2035** "
        "(plus flexibility investment) across the sources below, and see the resulting winter-output "
        "share and blended reliability of your scenario."
    )

    target_new_twh = st.number_input(
        "Total new-generation target for 2035 (TWh/year)", min_value=5.0, max_value=60.0, value=35.0, step=1.0
    )

    allocable = view[view["category"].isin(["Expansion", "New"])].copy()
    if allocable.empty:
        st.warning("No expandable/new sources in the current filter — adjust the sidebar filter.")
    else:
        st.write("Allocate the target across the following sources (percent of total):")
        pct_cols = st.columns(len(allocable))
        pcts = {}
        default_pct = round(100 / len(allocable))
        for i, (_, row) in enumerate(allocable.iterrows()):
            with pct_cols[i]:
                pcts[row["source"]] = st.slider(
                    row["source"], min_value=0, max_value=100, value=default_pct, key=f"pct_{row['source']}"
                )

        total_pct = sum(pcts.values())
        if total_pct == 0:
            st.warning("Allocate at least some percentage to at least one source.")
        else:
            alloc_df = allocable.copy()
            alloc_df["allocated_twh"] = alloc_df["source"].map(pcts) / total_pct * target_new_twh
            alloc_df["winter_twh"] = alloc_df["allocated_twh"] * alloc_df["winter_output_share_pct"] / 100

            total_winter_share = (
                alloc_df["winter_twh"].sum() / alloc_df["allocated_twh"].sum() * 100
                if alloc_df["allocated_twh"].sum() > 0
                else 0
            )
            blended_reliability = (
                (alloc_df["allocated_twh"] * alloc_df["reliability_score"]).sum()
                / alloc_df["allocated_twh"].sum()
                if alloc_df["allocated_twh"].sum() > 0
                else 0
            )
            blended_acceptance = (
                (alloc_df["allocated_twh"] * alloc_df["public_acceptance_score"]).sum()
                / alloc_df["allocated_twh"].sum()
                if alloc_df["allocated_twh"].sum() > 0
                else 0
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Scenario winter-output share", f"{total_winter_share:.0f}%")
            m2.metric("Blended reliability score", f"{blended_reliability:.2f} / 5")
            m3.metric("Blended public-acceptance score", f"{blended_acceptance:.2f} / 5")

            if total_pct != 100:
                st.caption(
                    f"Note: sliders currently sum to {total_pct}%, so they are being renormalized to 100% "
                    "for this calculation."
                )

            fig = px.bar(
                alloc_df.sort_values("allocated_twh"),
                x="allocated_twh",
                y="source",
                orientation="h",
                labels={"allocated_twh": "Allocated new generation (TWh/year)", "source": ""},
                color="source",
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Compare your scenario's winter-output share against Switzerland's structural winter deficit "
                "(historically ~15% of winter demand met by imports, capped at 5 TWh/year net under the "
                "2024 law) — see docs/04-energy-security-and-reliability.md."
            )
