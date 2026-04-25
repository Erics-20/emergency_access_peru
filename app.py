"""
Streamlit dashboard — Emergency Healthcare Access Inequality in Peru (2025)

Four tabs:
  1  Data & Methodology   — problem context, sources, cleaning, index design
  2  Static Analysis      — 9 statistical charts with interpretations
  3  GeoSpatial Results   — static choropleth / analytical maps + district tables
  4  Interactive          — folium maps, district comparison, spec sensitivity
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit.components.v1 as components

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
FIGURES   = _HERE / "output/figures"
TABLES    = _HERE / "output/tables"
PROCESSED = _HERE / "data/processed"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Healthcare Access — Peru 2025",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_baseline() -> pd.DataFrame:
    return pd.read_csv(TABLES / "district_scores_baseline.csv", dtype={"ubigeo": str})

@st.cache_data
def load_alternative() -> pd.DataFrame:
    return pd.read_csv(TABLES / "district_scores_alternative.csv", dtype={"ubigeo": str})

@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(TABLES / "specification_comparison.csv", dtype={"ubigeo": str})

@st.cache_data
def load_consulta() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "consulta_clean.csv", dtype={"ubigeo": str})

@st.cache_data
def read_html(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

baseline    = load_baseline()
alternative = load_alternative()
comparison  = load_comparison()

# Pre-compute summary statistics used in multiple tabs
n_districts       = len(baseline)
n_zero_emerg      = (baseline["n_emergency"] == 0).sum()
pct_zero_emerg    = n_zero_emerg / n_districts * 100
median_dist_emerg = baseline["dist_km_nearest_emergency"].median()
max_dist_idx      = baseline["dist_km_nearest_emergency"].idxmax()
max_dist_row      = baseline.loc[max_dist_idx]
spearman_r        = (
    comparison["index_baseline"].rank()
    .corr(comparison["index_alternative"].rank())
)
n_tier_changed    = int(comparison["tier_change"].sum())
pct_tier_changed  = n_tier_changed / len(comparison) * 100

# ── Global header ──────────────────────────────────────────────────────────────
st.title("Emergency Healthcare Access Inequality in Peru")
st.caption(
    f"District-level geospatial analysis · MINSA IPRESS & HIS data, 2025 · "
    f"{n_districts:,} districts analysed"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Data & Methodology",
    "Static Analysis",
    "GeoSpatial Results",
    "Interactive Exploration",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA & METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Problem statement ──────────────────────────────────────────────────────
    st.header("Problem Statement")
    st.markdown(f"""
Geographic inequality in emergency healthcare access is a critical but
under-quantified policy problem across Peru's {n_districts:,} districts.
**{n_zero_emerg:,} of {n_districts:,} districts ({pct_zero_emerg:.0f}%) have
zero emergency-level health facilities** — MINSA categories II-1 through III-E —
within their administrative boundaries.

The median district is **{median_dist_emerg:.0f} km** from the nearest emergency
facility. The most remote district is **{max_dist_row['distrito']},
{max_dist_row['departamento']}**, at
**{max_dist_row['dist_km_nearest_emergency']:.0f} km** straight-line distance.

This analysis constructs a composite access index for every district to quantify,
map, and compare this inequality using geospatial methods and two competing
access definitions. The goal is to identify which districts are structurally
underserved and to test how sensitive that classification is to the choice of
access standard.
    """)

    st.divider()

    # ── Data sources ───────────────────────────────────────────────────────────
    st.header("Data Sources")
    sources = pd.DataFrame([
        {
            "Dataset":       "IPRESS",
            "Full name":     "Registry of Health Facilities (SUSALUD)",
            "Raw rows":      "~20,819",
            "Key variables": "Facility name, category (I-1 to III-E), coordinates (lat/lon), institution, status",
            "Format":        "CSV · encoding: latin-1",
        },
        {
            "Dataset":       "ConsultaC1",
            "Full name":     "Outpatient Consultations by Facility (MINSA HIS, 2025)",
            "Raw rows":      "~342,753",
            "Key variables": "UBIGEO, facility category, sex, age group, total consultations",
            "Format":        "CSV · encoding: latin-1 · delimiter: semicolon",
        },
        {
            "Dataset":       "DISTRITOS",
            "Full name":     "District Polygon Shapefile (IGN / INEI)",
            "Raw rows":      "1,873 polygons",
            "Key variables": "District boundaries (geometry only — .dbf absent from source)",
            "Format":        "Shapefile (.shp · .shx missing → regenerated)",
        },
        {
            "Dataset":       "CCPP",
            "Full name":     "Populated Places 1:100,000 (IGN / INEI)",
            "Raw rows":      "~136,587",
            "Key variables": "Settlement name, locality code (UBIGEO derivable), coordinates",
            "Format":        "Shapefile",
        },
    ])
    st.dataframe(sources, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cleaning decisions ─────────────────────────────────────────────────────
    st.header("Key Cleaning Decisions")
    st.markdown("""
| Dataset | Action | Reason | Rows affected |
|---------|--------|--------|---------------|
| IPRESS | Renamed `NORTE → longitud`, `ESTE → latitud` | Raw column labels are swapped — longitude stored under NORTE | All rows |
| IPRESS | Dropped 14 exact duplicates + 12 duplicate `codigo_unico` records | Verified duplicates confirmed by manual inspection | −26 rows |
| IPRESS | Filtered to Peru bounding box (lat −18.5 to 0, lon −82 to −68) | Coordinates outside Peru's extent are invalid | −2 rows |
| ConsultaC1 | Dropped 41,241 rows where ALL key columns are NE-coded | No analytical value when sex, age, and counts are all missing | −41,241 rows |
| ConsultaC1 | Dropped 19,458 exact duplicate rows | Row-level duplication confirmed | −19,458 rows |
| CCPP | Retained 72,388 points without `CÓDIGO` as ubigeo=null | Valid geometry is useful for spatial joins and distance calculations | 0 dropped |
| DISTRITOS | Dissolved 1,873 → 1,832 rows by UBIGEO | Shapefile has multiple polygon rows per district (exclaves, islands) | −41 rows |
| DISTRITOS | Assigned UBIGEO via spatial join from CCPP points (mode per polygon) | `.dbf` absent — no attribute data recoverable from source shapefile | 1,803/1,873 covered |
    """)

    st.divider()

    # ── Composite index methodology ────────────────────────────────────────────
    st.header("Composite Access Index — Methodology")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
The composite index aggregates three district-level components into a single
0–100 score (higher = better served).

**1. Facility availability**
Active-facility density = active facilities / km².  Density normalises by territory
area so that large but sparsely covered districts are not artificially elevated
relative to small, dense urban districts.  Active facilities only (EN FUNCIONAMIENTO)
are counted to exclude closed or restricted sites.

**2. Emergency activity**
Total outpatient consultations at emergency-level facilities (II-1 to III-E) in 2025,
from MINSA HIS.  This is a demand-side signal: districts with zero emergency-level
activity receive the minimum percentile rank.  It complements the supply-side density
count by capturing whether facilities are actually used.

**3. Spatial access**
Percentage of populated places (CCPP) within a distance threshold of the nearest
relevant facility.  A threshold share avoids the problem where a single extremely
remote village dominates the district mean distance, masking that most settlements
may be well covered.

Each component is converted to a percentile rank (0–100) before weighting so that
all three contribute on a comparable scale regardless of original units.
        """)

    with col_right:
        st.markdown("""
**Composite score formula:**

score = sum( w_c / sum(w) * pct_rank(X_c) )   for c in {fac, activity, access}

**Two specifications:**

| | Baseline | Alternative |
|--|--|--|
| Facility weights | 1/3 · 1/3 · 1/3 | 0.25 · 0.25 · 0.50 |
| Distance threshold | 30 km | 15 km |
| Facility type counted | Emergency (II–III) | Any facility |

The alternative doubles the weight on spatial access and lowers the distance
threshold while relaxing the facility type.  Comparing both specifications reveals
which districts are sensitive to the access definition and why.

**Service tier classification** (applied per specification):

| Tier | Score range |
|------|-------------|
| Best served | ≥ 75th percentile |
| Moderately served | 50th–75th percentile |
| Weakly served | 25th–50th percentile |
| Underserved | < 25th percentile |
        """)

    st.divider()

    # ── CRS strategy ───────────────────────────────────────────────────────────
    st.header("Coordinate Reference System (CRS) Strategy")
    st.markdown("""
All geospatial data is **stored** in **EPSG:4326** (WGS-84 geographic, degrees).
Distance and area computations are performed in **EPSG:32718** (WGS-84 / UTM Zone 18S,
metres) and results are stored back in EPSG:4326.

**Why UTM Zone 18S?** Peru spans UTM zones 17S–19S.  Zone 18S minimises linear
distortion for the Pacific coast and central Andes — the most densely populated
regions and the analytical focus of this study.  Distance errors at the periphery
(e.g. Iquitos, nominally in Zone 18S) remain below 1% for the distance ranges in
this analysis.

**Workflow:**
GeoDataFrame in EPSG:4326 → reproject to EPSG:32718 → compute distances / areas
→ store result → reproject back to EPSG:4326 for storage and display.
The projected CRS is applied immediately before computation and discarded afterwards.
    """)

    st.divider()

    # ── Limitations ────────────────────────────────────────────────────────────
    st.header("Limitations")
    st.markdown("""
- **Straight-line distances only.** Road network distances would better capture
  actual travel times, particularly in high-Andean and Amazonian districts where
  road topology diverges sharply from Euclidean geometry.

- **192 IPRESS facilities outside polygon boundaries.** These retain their
  administrative UBIGEO for counting but cannot participate in spatial distance
  computations.  They are most likely near coastlines or border zones where
  coordinate imprecision places them just outside the digitised boundary.

- **UTM Zone 18S covers ~1,600 km across Peru.** Facilities at the western or
  eastern extremes (Tumbes, Madre de Dios) incur systematic distance errors of
  up to ~0.5%.

- **ConsultaC1 captures only 2025 outpatient consultations.** Emergency department
  visit records are not separately itemised; the proxy — consultations at
  emergency-category facilities — may conflate elective and emergency demand.

- **DISTRITOS shapefile lacked attribute data (.dbf absent).** UBIGEO codes were
  recovered via spatial join from CCPP; 70 polygons (3.7%) remain without a
  confirmed code and are excluded from all indicator computations.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.header("Static Analysis")
    st.caption(
        "Charts generated with matplotlib + seaborn.  "
        "Organised by research question (Q1–Q4)."
    )

    # ── Q1 — Facility availability vs. emergency activity ─────────────────────
    st.subheader("Q1 — Facility Availability vs. Emergency Activity")

    p = FIGURES / "q1_fac_vs_activity_scatter.png"
    if p.exists():
        st.image(str(p), use_container_width=True)

    # Compute all-facility activity score and filter to districts with HIS records
    _consulta   = load_consulta()
    _all_agg    = (
        _consulta.groupby("ubigeo")["total_atenciones"]
        .sum()
        .reindex(baseline["ubigeo"], fill_value=0)
        .values
    )
    baseline_q1 = baseline.copy()
    baseline_q1["all_activity_score"] = (
        pd.Series(_all_agg).rank(pct=True, method="average") * 100
    ).values
    n_excl_q1   = int((_all_agg == 0).sum())
    q1_nonzero  = baseline_q1[_all_agg > 0].copy()

    st.markdown(f"""
Each point is one district.  The x-axis is the facility availability score
(percentile rank of active-facility density); the y-axis is the all-facility
activity score (percentile rank of total outpatient consultations across all
facility categories, I-1 to III-E).  All categories are included so that
districts with primary-care facilities but no emergency hospitals still receive a
distinct activity score.

**Why {n_excl_q1:,} districts are excluded:** The ConsultaC1 dataset comes from
MINSA's Health Information System (HIS), which only records consultations from
facilities that actively submit reports.  Districts with zero total consultations
across all {n_excl_q1:,} categories have no presence in HIS — they all receive the
same tied percentile rank, producing a meaningless horizontal cluster.  The scatter
shows only the {len(q1_nonzero):,} districts with at least one HIS consultation record.

**Interpretation:** The two components are positively correlated (Pearson r shown
in the corner), meaning districts with more facilities per km² also tend to record
higher consultation volumes.  Dispersion around the trend reveals districts that
are supply-rich but underutilised, and districts with high activity relative to
their infrastructure density.
    """)

    # Tables: top 5 / bottom 5 on both indicators combined (filtered data)
    q1_df = q1_nonzero[["distrito", "departamento", "fac_score", "all_activity_score"]].copy()
    q1_df["combined"] = (q1_df["fac_score"] + q1_df["all_activity_score"]) / 2

    top5_q1 = (
        q1_df.nlargest(5, "combined")
        [["distrito", "departamento", "fac_score", "all_activity_score", "combined"]]
        .rename(columns={
            "distrito": "District", "departamento": "Department",
            "fac_score": "Facility score", "all_activity_score": "Activity score (all categ.)",
            "combined": "Mean of both",
        })
        .round(1)
        .reset_index(drop=True)
    )
    top5_q1.index += 1

    bottom5_q1 = (
        q1_df.nsmallest(5, "combined")
        [["distrito", "departamento", "fac_score", "all_activity_score", "combined"]]
        .rename(columns={
            "distrito": "District", "departamento": "Department",
            "fac_score": "Facility score", "all_activity_score": "Activity score (all categ.)",
            "combined": "Mean of both",
        })
        .round(1)
        .reset_index(drop=True)
    )
    bottom5_q1.index += 1

    col_t5a, col_t5b = st.columns(2)
    with col_t5a:
        st.markdown("**Top 5 — highest on both indicators** *(among districts with HIS records)*")
        st.dataframe(top5_q1, use_container_width=True)
    with col_t5b:
        st.markdown("**Bottom 5 — lowest on both indicators** *(among districts with HIS records)*")
        st.dataframe(bottom5_q1, use_container_width=True)

    st.divider()

    # ── Q2 — Spatial access distribution ──────────────────────────────────────
    st.subheader("Q2 — Spatial Access from Populated Centres (CCPP)")

    p = FIGURES / "q2_access_score_kde.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    st.markdown("""
The KDE shows how the spatial access indicator is distributed across all districts.
The indicator is the **raw percentage** of populated places (CCPP) within 30 km of
the nearest emergency-level facility per district.

**Interpretation:** A strong concentration near 0% means the majority of Peru's
districts have almost no populated places within reach of emergency care.
A peak near 100% would indicate widespread coverage.  The shape of the KDE answers
directly whether the access deficit is the typical situation or the exception.
    """)

    # Table: top 10 districts by spatial access (tiebreaker: composite_score)
    top10_access = (
        baseline
        .sort_values(
            ["pct_ccpp_within_30km_emerg", "composite_score"],
            ascending=[False, False],
        )
        .head(10)
        [["distrito", "provincia", "departamento",
          "pct_ccpp_within_30km_emerg", "access_score", "composite_score", "tier"]]
        .rename(columns={
            "distrito":                   "District",
            "provincia":                  "Province",
            "departamento":               "Department",
            "pct_ccpp_within_30km_emerg": "% CCPP within 30 km",
            "access_score":               "Access score (0–100)",
            "composite_score":            "Composite score",
            "tier":                       "Tier",
        })
        .round({"% CCPP within 30 km": 1, "Access score (0–100)": 1, "Composite score": 1})
        .reset_index(drop=True)
    )
    top10_access.index += 1
    st.markdown(
        "**Top 10 districts — highest spatial access** "
        "(% CCPP within 30 km of nearest emergency facility; "
        "composite score used as tiebreaker among districts with equal coverage)"
    )
    st.dataframe(top10_access, use_container_width=True)

    st.divider()

    col_q2_dept, _ = st.columns([2, 1])
    with col_q2_dept:
        p = FIGURES / "q2_access_by_department.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
    st.markdown("""
**Interpretation:** Amazonian departments (Loreto, Ucayali, Madre de Dios) have
median coverage near 0% — most of their districts have almost no populated places
within 30 km of emergency care.  Lima and Callao approach 100%.  Andean departments
show high within-department variance: some valley districts are well connected while
highland ones remain isolated.
    """)

    st.divider()

    # ── Q3 — Multi-dimensional classification ─────────────────────────────────
    st.subheader("Q3 — Multi-dimensional Classification")

    p = FIGURES / "q3_components_by_tier.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    st.markdown("""
**Interpretation:** Tier separation is strongest for facility availability: Underserved
districts cluster near zero while Best served districts are clearly elevated.
Separation for emergency activity is weaker, reflecting that most districts across
all tiers have zero emergency-level consultations — this component becomes
discriminating only within the small subset of districts that have at least one
emergency-level facility.
    """)

    p = FIGURES / "q3_top_bottom_districts.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    st.markdown("""
**Interpretation:** The 20 most underserved districts are almost entirely concentrated
in Loreto and other Amazonian or high-Andean departments.  The 20 best-served districts
are in Lima, Arequipa, and the Pacific coast — reflecting decades of infrastructure
investment concentrated in urban coastal areas.
    """)

    st.divider()

    # ── Q4 — Specification sensitivity ────────────────────────────────────────
    st.subheader("Q4 — Specification Sensitivity")

    p = FIGURES / "q4_kde_comparison.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    st.markdown("""
Each curve shows the density of composite index scores across all districts under
one specification.  If the access definition change shifts the overall distribution,
one curve will sit to the right of the other.  If the shapes differ (one wider,
one more peaked), the alternative definition separates districts differently.
Strongly overlapping curves mean the overall score distribution is robust to the
specification choice even if individual district ranks change.

**Interpretation:** The medians shown in the legend quantify whether one specification
systematically assigns higher scores.  The spread of each curve indicates how
discriminating each weighting scheme is across Peru's districts.
    """)

    p = FIGURES / "q4_rank_shift_distribution.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    st.markdown("""
**Interpretation:** The distribution of rank shifts shows how much individual
districts move between specifications.  An approximately symmetric distribution
centred on 0 means neither specification systematically advantages a particular
type of district.  Fat tails represent districts with structural mismatches between
primary-care access and emergency-only access, causing large rank swings when the
definition changes.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEOSPATIAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.header("GeoSpatial Results")

    # ── Component score choropleths ────────────────────────────────────────────
    st.subheader("Component Score Choropleths")
    st.caption(
        "Each map encodes a percentile rank (0–100) by district.  "
        "Grey districts have no UBIGEO code and are excluded from the index."
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        p = FIGURES / "map_fac_score.png"
        if p.exists():
            st.image(str(p), caption="Q1 — Facility availability score", use_container_width=True)
    with col_m2:
        p = FIGURES / "map_activity_score.png"
        if p.exists():
            st.image(str(p), caption="Q1 — Emergency activity score", use_container_width=True)
    with col_m3:
        p = FIGURES / "map_access_score_baseline.png"
        if p.exists():
            st.image(str(p), caption="Q2 — Spatial access score (baseline)", use_container_width=True)

    st.markdown("""
The three component maps reveal distinct geographic patterns.  Facility availability
(left) is high along the Pacific coast and in provincial capitals.  Emergency activity
(centre) is highly concentrated in Lima and a handful of regional capitals, with most
of the country at near-zero levels.  Spatial access (right) shows the largest
territory of deprivation — vast Amazonian and high-Andean areas where no populated
place is within 30 km of emergency care.
    """)

    st.divider()

    # ── Service tier maps ──────────────────────────────────────────────────────
    st.subheader("Service Tier Maps")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        p = FIGURES / "map_baseline_tiers.png"
        if p.exists():
            st.image(
                str(p),
                caption="Baseline service tiers — equal weights (1/3 each), 30 km to emergency facility",
                use_container_width=True,
            )
    with col_t2:
        p = FIGURES / "map_alternative_tiers.png"
        if p.exists():
            st.image(
                str(p),
                caption="Alternative service tiers — access weight 0.50, 15 km to any facility",
                use_container_width=True,
            )

    st.markdown("""
Comparing the two tier maps shows which districts are sensitive to the access
definition.  Coastal and mid-Andes districts frequently improve a tier in the
alternative specification — they have dense primary-care networks but few emergency
hospitals, so they benefit when the facility-type bar is lowered.  Conversely, large
Amazonian districts show little change: even any facility is typically far beyond
15 km in those regions.
    """)

    st.divider()

    # ── Analytical maps ────────────────────────────────────────────────────────
    st.subheader("Analytical Maps")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        p = FIGURES / "geo_bivariate_choropleth.png"
        if p.exists():
            st.image(
                str(p),
                caption="Q1+Q3 — Bivariate: facility density x spatial access",
                use_container_width=True,
            )
        st.markdown("""
**Doubly underserved districts (red)** have both low facility density and low
spatial access — the most policy-critical quadrant, where neither supply nor
geography is favourable.  The orange quadrant (high facilities, low access) suggests
geography as the binding constraint: hospitals exist but many villages remain far
away.  The green quadrant (low facilities, high access) reflects districts near
large cities where surrounding populated places benefit from nearby facilities
despite the district itself having few.
        """)

    with col_g2:
        p = FIGURES / "geo_facility_desert.png"
        if p.exists():
            st.image(
                str(p),
                caption="Q1+Q2 — The emergency facility desert",
                use_container_width=True,
            )
        st.markdown("""
The overwhelming grey makes the structural problem immediate: 89% of Peru's districts
have no emergency-level facility within their boundaries.  The small number of red
points (emergency facilities) concentrate along the Pacific coast and in Lima,
while the Amazon basin and high Andes are near-total deserts.  This map provides
the geographic anchor for interpreting every metric in the analysis — the access
inequality is not gradual but categorical: a district either has emergency care or
it does not.
        """)

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        p = FIGURES / "geo_ccpp_access_map.png"
        if p.exists():
            st.image(
                str(p),
                caption="Q2 — Populated places by distance to nearest emergency facility",
                use_container_width=True,
            )
        st.markdown("""
74,577 of Peru's 136,587 populated places (55%) are more than 30 km from an
emergency facility; 23,546 (17%) are more than 60 km away.  This map shows those
places as individual points — not statistical abstractions.  Red points in the
Amazon and high Andes represent actual villages and towns with no realistic
overland path to emergency care.  The stratified sampling preserves all >60-km
points while showing a proportional sample of closer places.
        """)

    with col_g4:
        p = FIGURES / "geo_tier_divergence.png"
        if p.exists():
            st.image(
                str(p),
                caption="Q4 — Geographic pattern of specification sensitivity",
                use_container_width=True,
            )
        st.markdown("""
Districts whose tier changed between specifications are coloured by the magnitude
and direction of their rank shift.  Green districts (rose in alternative) concentrate
along the Pacific coast and mid-Andes valleys — dense primary care but few emergency
hospitals.  Red districts (fell in alternative) are often large rural areas where
even primary care is sparse, so the 15-km any-facility threshold remains out of
reach.  Grey districts (unchanged) account for ~64% of the country.
        """)

    st.divider()

    # ── District-level comparison table ───────────────────────────────────────
    st.subheader("District-Level Comparison Table")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        dept_options = ["All departments"] + sorted(
            baseline["departamento"].dropna().unique()
        )
        sel_dept = st.selectbox("Filter by department", dept_options, key="tab3_dept")
    with col_f2:
        tier_options = [
            "All tiers", "Underserved", "Weakly served", "Moderately served", "Best served"
        ]
        sel_tier = st.selectbox("Filter by baseline tier", tier_options, key="tab3_tier")

    display_cols = [
        "distrito", "provincia", "departamento", "tier",
        "composite_score", "fac_score", "activity_score", "access_score",
        "n_emergency", "dist_km_nearest_emergency", "pct_ccpp_within_30km_emerg",
    ]
    table_data = baseline[display_cols].copy()

    if sel_dept != "All departments":
        table_data = table_data[table_data["departamento"] == sel_dept]
    if sel_tier != "All tiers":
        table_data = table_data[table_data["tier"] == sel_tier]

    table_data = (
        table_data
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    table_data.index += 1

    st.dataframe(
        table_data.rename(columns={
            "distrito":                   "District",
            "provincia":                  "Province",
            "departamento":               "Department",
            "tier":                       "Tier",
            "composite_score":            "Score (0–100)",
            "fac_score":                  "Facility score",
            "activity_score":             "Activity score",
            "access_score":               "Access score",
            "n_emergency":                "Emerg. facilities",
            "dist_km_nearest_emergency":  "Dist. to emerg. (km)",
            "pct_ccpp_within_30km_emerg": "% CCPP within 30 km",
        }),
        use_container_width=True,
        height=400,
    )
    st.caption(
        f"Showing {len(table_data):,} districts · sorted by composite score (descending)"
    )

    st.divider()

    # ── Tier distribution summary ──────────────────────────────────────────────
    st.subheader("Tier Distribution Summary")

    tier_order = ["Underserved", "Weakly served", "Moderately served", "Best served"]
    tier_summary = (
        baseline
        .groupby("tier", observed=False)
        .agg(
            Districts=("composite_score", "count"),
            Median_score=("composite_score", "median"),
            Median_dist_emerg_km=("dist_km_nearest_emergency", "median"),
            Total_emerg_facilities=("n_emergency", "sum"),
            Pct_zero_emerg=("n_emergency", lambda x: round((x == 0).mean() * 100, 1)),
        )
        .reindex(tier_order)
        .round({"Median_score": 1, "Median_dist_emerg_km": 1})
    )
    tier_summary.index.name = "Tier"

    st.dataframe(
        tier_summary.rename(columns={
            "Median_score":          "Median composite score",
            "Median_dist_emerg_km":  "Median dist. to emerg. (km)",
            "Total_emerg_facilities":"Total emerg. facilities",
            "Pct_zero_emerg":        "% with zero emerg. facilities",
        }),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INTERACTIVE EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.header("Interactive Exploration")

    sub_maps, sub_compare, sub_spec = st.tabs([
        "Folium Maps",
        "District Comparison",
        "Specification Sensitivity",
    ])

    # ── Folium Maps ────────────────────────────────────────────────────────────
    with sub_maps:
        st.subheader("Interactive Maps")

        MAP_OPTIONS = {
            "Tier Explorer": {
                "file": FIGURES / "map_tiers_interactive.html",
                "description": (
                    "Hover any district to see its baseline tier and composite score.  "
                    "Click for a full-stats popup (all three component scores, distance to "
                    "nearest emergency facility, tier in both specifications).  "
                    "Use the **Layer Control** (top right) to toggle between baseline and "
                    "alternative specification tiers and to show or hide emergency facility markers."
                ),
            },
            "Access Score Choropleth": {
                "file": FIGURES / "map_access_explorer.html",
                "description": (
                    "Continuous spatial access score (0–100) coloured by a red-yellow-green scale.  "
                    "Hover a district to see its access score, percentage of CCPP within 30 km, "
                    "and distance to the nearest emergency facility.  "
                    "Toggle the **Emergency facilities** layer to see how facility locations "
                    "correlate with district access scores."
                ),
            },
            "CCPP Access-Deficit Heatmap": {
                "file": FIGURES / "map_ccpp_distances.html",
                "description": (
                    "Heatmap of the 74,577 populated places more than 30 km from the nearest "
                    "emergency facility, weighted by distance (farther = brighter).  "
                    "Zoom into any region to see the density and severity of isolated settlements.  "
                    "District tier polygons provide geographic orientation.  "
                    "Note: this map may take a moment to render due to the heatmap data volume."
                ),
            },
        }

        map_choice = st.radio(
            "Select map",
            list(MAP_OPTIONS.keys()),
            horizontal=True,
            key="map_radio",
        )

        chosen = MAP_OPTIONS[map_choice]
        st.info(chosen["description"])

        if chosen["file"].exists():
            with st.spinner("Loading map..."):
                html_content = read_html(str(chosen["file"]))
            components.html(html_content, height=650, scrolling=False)
        else:
            st.error(
                f"Map file not found: {chosen['file'].name}.  "
                "Run `python run_maps.py` to generate all interactive maps."
            )

    # ── District Comparison ────────────────────────────────────────────────────
    with sub_compare:
        st.subheader("District Comparison")
        st.markdown(
            "Select a department and two or more districts to compare their "
            "composite scores and underlying metrics side by side."
        )

        dept_list = sorted(baseline["departamento"].dropna().unique())
        sel_dept_comp = st.selectbox(
            "Department", dept_list, key="comp_dept"
        )

        # Build unique display labels — append (Provincia) when names clash within dept
        _dept_df = baseline[baseline["departamento"] == sel_dept_comp].copy()
        _name_counts = _dept_df["distrito"].value_counts()
        _dept_df["_label"] = _dept_df.apply(
            lambda r: f"{r['distrito']} ({r['provincia']})"
            if _name_counts.get(r["distrito"], 1) > 1 else r["distrito"],
            axis=1,
        )
        _label_to_ubigeo = _dept_df.set_index("_label")["ubigeo"].to_dict()

        dist_in_dept = sorted(_dept_df["_label"].dropna().unique())
        default_sel = dist_in_dept[:3] if len(dist_in_dept) >= 3 else dist_in_dept

        selected_labels = st.multiselect(
            "Districts to compare (select 2–6)",
            dist_in_dept,
            default=default_sel,
            key="comp_districts",
        )

        if len(selected_labels) < 2:
            st.warning("Select at least 2 districts to enable comparison.")
        else:
            selected_ubigeos = [_label_to_ubigeo[l] for l in selected_labels]
            _ubigeo_to_label = {v: k for k, v in _label_to_ubigeo.items()}

            sel_df = baseline[baseline["ubigeo"].isin(selected_ubigeos)].copy()
            sel_df["_label"] = sel_df["ubigeo"].map(_ubigeo_to_label)
            sel_df = sel_df.set_index("_label")

            # Score table
            score_cols = {
                "composite_score": "Composite score",
                "fac_score":        "Facility availability",
                "activity_score":   "Emergency activity",
                "access_score":     "Spatial access (baseline)",
            }
            score_table = (
                sel_df[[c for c in score_cols if c in sel_df.columns]]
                .rename(columns=score_cols)
                .T.round(1)
            )
            score_table.index.name = "Component (0–100 percentile rank)"
            st.markdown("**Score comparison**")
            st.dataframe(score_table, use_container_width=True)

            # Infrastructure metrics table
            infra_cols = {
                "tier":                       "Baseline tier",
                "n_emergency":                "Emergency facilities",
                "n_active":                   "Active facilities",
                "area_km2":                   "Area (km²)",
                "dist_km_nearest_emergency":  "Dist. to nearest emerg. (km)",
                "dist_km_nearest_any":        "Dist. to nearest any facility (km)",
                "pct_ccpp_within_30km_emerg": "% CCPP within 30 km (emerg.)",
                "emerg_atenciones":           "Emerg. consultations 2025",
            }
            avail = {k: v for k, v in infra_cols.items() if k in sel_df.columns}
            infra_table = (
                sel_df[list(avail.keys())]
                .rename(columns=avail)
                .T
                .astype(str)
            )
            infra_table.index.name = "Metric"
            st.markdown("**Infrastructure & access metrics**")
            st.dataframe(infra_table, use_container_width=True)

            # Tier in both specifications
            alt_sel = (
                alternative[alternative["ubigeo"].isin(selected_ubigeos)]
                .copy()
            )
            alt_sel["_label"] = alt_sel["ubigeo"].map(_ubigeo_to_label)
            alt_sel = alt_sel.set_index("_label")[["tier"]].rename(
                columns={"tier": "Alternative tier"}
            )
            baseline_tier = sel_df[["tier"]].rename(columns={"tier": "Baseline tier"})
            tier_comp = baseline_tier.join(alt_sel)
            tier_comp["Tier changed?"] = (
                tier_comp["Baseline tier"] != tier_comp["Alternative tier"]
            )
            tier_comp.index.name = "District"
            st.markdown("**Tier in each specification**")
            st.dataframe(tier_comp, use_container_width=True)

    # ── Specification Sensitivity ──────────────────────────────────────────────
    with sub_spec:
        st.subheader("Baseline vs. Alternative Specification")

        # Summary metrics
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Districts analysed", f"{len(comparison):,}")
        col_s2.metric(
            "Changed service tier",
            f"{n_tier_changed:,}",
            delta=f"{pct_tier_changed:.1f}% of all districts",
            delta_color="off",
        )
        col_s3.metric(
            "Spearman rank correlation",
            f"{spearman_r:.4f}",
            help="Rank correlation between baseline and alternative composite scores across all districts.",
        )

        st.markdown("""
A Spearman correlation close to 1.0 means both specifications rank districts in nearly
the same order overall.  The tier-change count shows how many districts sit close enough
to a tier boundary that the access-definition change flips their classification.
These are not necessarily the districts with the largest rank shifts — some districts
move many ranks while staying within the same tier, others move few ranks but cross
a boundary.
        """)

        st.divider()

        # Filter controls
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_change = st.selectbox(
                "Tier change filter",
                ["All districts", "Changed tier only", "Unchanged tier only"],
                key="spec_filter_change",
            )
        with col_f2:
            filter_dept_spec = st.selectbox(
                "Department",
                ["All departments"] + sorted(
                    comparison["departamento"].dropna().unique()
                ),
                key="spec_filter_dept",
            )
        with col_f3:
            sort_by = st.selectbox(
                "Sort by",
                [
                    "Rank shift — largest rise in alternative",
                    "Rank shift — largest fall in alternative",
                    "Baseline score — highest",
                    "Baseline score — lowest",
                ],
                key="spec_sort",
            )

        comp_view = comparison.copy()
        if filter_change == "Changed tier only":
            comp_view = comp_view[comp_view["tier_change"] == True]
        elif filter_change == "Unchanged tier only":
            comp_view = comp_view[comp_view["tier_change"] == False]
        if filter_dept_spec != "All departments":
            comp_view = comp_view[comp_view["departamento"] == filter_dept_spec]

        sort_map = {
            "Rank shift — largest rise in alternative":  ("rank_shift", False),
            "Rank shift — largest fall in alternative":  ("rank_shift", True),
            "Baseline score — highest":                  ("index_baseline", False),
            "Baseline score — lowest":                   ("index_baseline", True),
        }
        sort_col, sort_asc = sort_map[sort_by]
        comp_view = comp_view.sort_values(sort_col, ascending=sort_asc)

        show_cols = {
            "distrito":          "District",
            "provincia":         "Province",
            "departamento":      "Department",
            "tier_baseline":     "Baseline tier",
            "tier_alternative":  "Alternative tier",
            "tier_change":       "Tier changed?",
            "index_baseline":    "Baseline score",
            "index_alternative": "Alternative score",
            "rank_baseline":     "Baseline rank",
            "rank_alternative":  "Alternative rank",
            "rank_shift":        "Rank shift",
        }
        avail_cols = {k: v for k, v in show_cols.items() if k in comp_view.columns}

        st.dataframe(
            comp_view[list(avail_cols.keys())]
            .rename(columns=avail_cols)
            .reset_index(drop=True),
            use_container_width=True,
            height=420,
        )
        st.caption(
            f"Showing {len(comp_view):,} of {len(comparison):,} districts.  "
            "Rank shift > 0 means the district rose in the alternative specification."
        )

        st.divider()

        # Top movers
        st.subheader("Top 10 Districts — Largest Rise in Alternative Specification")
        top_movers = (
            comparison
            .nlargest(10, "rank_shift")[
                ["distrito", "departamento",
                 "tier_baseline", "tier_alternative",
                 "rank_baseline", "rank_alternative", "rank_shift"]
            ]
            .rename(columns={
                "distrito":         "District",
                "departamento":     "Department",
                "tier_baseline":    "Baseline tier",
                "tier_alternative": "Alternative tier",
                "rank_baseline":    "Baseline rank",
                "rank_alternative": "Alternative rank",
                "rank_shift":       "Rank shift",
            })
            .reset_index(drop=True)
        )
        top_movers.index += 1
        st.dataframe(top_movers, use_container_width=True)
        st.markdown("""
Districts that rise the most under the alternative specification share a structural
pattern: they have relatively good primary-care coverage (any facility within 15 km
for many of their populated places) but few or no emergency-level hospitals.  When
the access standard is relaxed from emergency-only to any facility, these districts
receive a much higher access component score, lifting their overall rank.
        """)
