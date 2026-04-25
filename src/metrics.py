"""
District-level indicators and composite healthcare access index.

Two specifications are built:

  Baseline    — Equal weights (1/3 each).  Spatial access measured as
                the share of CCPP populated places within 30 km of the
                nearest emergency-level facility (II-1 through III-E).

  Alternative — Access-weighted (25 / 25 / 50).  Spatial access measured
                as the share of CCPP places within 15 km of the nearest
                facility of *any* category.

The alternative isolates the effect of two simultaneous changes to the
access definition: (a) a stricter distance cut-off and (b) a lower-bar
service standard (any facility vs. emergency-only).  All other components
remain identical, so differences in district rankings are attributable
solely to the access definition.

Component rationale
-------------------
1. Facility availability
   Indicator : active-facility density = n_active / area_km²
   Why density: raw counts reward large districts; density normalises by
   territory so remote but sparsely covered districts are not artificially
   elevated.  We use *active* facilities (EN FUNCIONAMIENTO) to exclude
   closed or restricted sites.

2. Emergency activity
   Indicator : total outpatient consultations at emergency-level facilities
   (II-1, II-2, II-E, III-1, III-2, III-E) in 2025 per district.
   Why volume: consultation volume at hospital-grade facilities is a
   demand-side signal for actual emergency service utilisation, complementing
   the supply-side density count.  Districts with zero emergency-level
   activity receive the minimum percentile rank.

3. Spatial access
   Indicator : % of CCPP populated places within [threshold] km of the
   nearest [facility type].
   Why a threshold share: a district mean distance can be dominated by
   outlier remote villages.  The share within a clinically meaningful
   threshold captures how many populated centres have actionable geographic
   access to care.

Composite index
---------------
  score_i = Σ (w_c × pct_rank(indicator_c))    c ∈ {fac, activity, access}

Each component is ranked into 0–100 percentile space before weighting so
the three indicators are comparable regardless of their original units.

Service tier classification (applied independently per specification):
  Tier 4 — Best served       : score ≥ 75th percentile
  Tier 3 — Moderately served : 50th–75th percentile
  Tier 2 — Weakly served     : 25th–50th percentile
  Tier 1 — Underserved       : score < 25th percentile
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from src.data_loader import load_consulta_clean
from src.utils import PROCESSED_DIR

EMERGENCY_LEVELS = frozenset({"II-1", "II-2", "II-E", "III-1", "III-2", "III-E"})
OUTPUT_TABLES = Path("output/tables")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank (0–100); higher = better.  Ties use average rank."""
    return series.rank(pct=True, method="average") * 100


# ── Component functions ───────────────────────────────────────────────────────

def compute_facility_component(distritos: pd.DataFrame) -> pd.Series:
    """
    Percentile rank of active-facility density (n_active / area_km²).

    Returns
    -------
    pd.Series indexed by ubigeo, values 0–100.
    """
    density = distritos["n_active"] / distritos["area_km2"]
    return _pct_rank(density).rename("fac_score")


def compute_activity_component(
    consulta: pd.DataFrame,
    ubigeos: pd.Index,
) -> pd.Series:
    """
    Percentile rank of total consultations at emergency-level facilities.

    Districts with zero emergency activity receive the minimum rank (~0).

    Returns
    -------
    pd.Series indexed by ubigeo, values 0–100.
    """
    emerg = consulta[consulta["categoria"].isin(EMERGENCY_LEVELS)].copy()
    agg = (
        emerg.groupby("ubigeo")["total_atenciones"]
        .sum()
        .rename("emerg_atenciones")
    )
    # Align to all districts; 0 for those with no emergency-level activity
    filled = agg.reindex(ubigeos, fill_value=0)
    return _pct_rank(filled).rename("activity_score")


def compute_access_component(
    ccpp: gpd.GeoDataFrame,
    ubigeos: pd.Index,
    threshold_km: float,
    dist_col: str,
) -> pd.DataFrame:
    """
    Percentile rank of % of populated places within threshold_km of the
    nearest relevant facility.

    Parameters
    ----------
    ccpp         : CCPP layer with pre-computed distance columns.
    ubigeos      : target district index for alignment.
    threshold_km : distance cut-off in km.
    dist_col     : 'dist_km_nearest_emergency' or 'dist_km_nearest_any'.

    Returns
    -------
    DataFrame with 'access_score' (0–100) and 'pct_within' (raw %).
    """
    valid = ccpp[ccpp["ubigeo"].notna()].copy()
    valid["within"] = valid[dist_col] <= threshold_km

    agg = valid.groupby("ubigeo").agg(
        n_ccpp_district=("within", "count"),
        n_within=("within", "sum"),
    )
    agg["pct_within"] = (agg["n_within"] / agg["n_ccpp_district"] * 100).round(2)

    # Districts with no CCPP points get 0 % (worst case)
    pct = agg["pct_within"].reindex(ubigeos, fill_value=0)
    score = _pct_rank(pct).rename("access_score")

    return pd.DataFrame({"access_score": score, "pct_within": pct})


def compute_index(
    fac_score: pd.Series,
    activity_score: pd.Series,
    access_score: pd.Series,
    w_fac: float,
    w_activity: float,
    w_access: float,
) -> pd.Series:
    """Weighted composite index (0–100, higher = better served)."""
    total = w_fac + w_activity + w_access
    return (
        (w_fac      / total) * fac_score
        + (w_activity / total) * activity_score
        + (w_access   / total) * access_score
    ).rename("composite_score")


# ── Classification ────────────────────────────────────────────────────────────

_TIER_LABELS = {
    1: "Underserved",
    2: "Weakly served",
    3: "Moderately served",
    4: "Best served",
}


def classify_districts(score: pd.Series) -> pd.Series:
    """
    Four service tiers based on quartiles of the composite score.

      Tier 4 — Best served       : score ≥ 75th pct
      Tier 3 — Moderately served : 50th–75th pct
      Tier 2 — Weakly served     : 25th–50th pct
      Tier 1 — Underserved       : score < 25th pct
    """
    q25, q50, q75 = score.quantile([0.25, 0.50, 0.75])
    tiers = pd.cut(
        score,
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=[1, 2, 3, 4],
    ).astype(int)
    return tiers.map(_TIER_LABELS).rename("tier")


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_specifications(
    baseline: pd.DataFrame,
    alternative: pd.DataFrame,
) -> pd.DataFrame:
    """
    Side-by-side comparison of baseline and alternative indices.

    Returns a DataFrame (indexed by ubigeo) with:
      index_baseline, index_alternative, tier_baseline, tier_alternative,
      tier_change (bool), rank_baseline, rank_alternative, rank_shift.
    """
    comp = pd.DataFrame(
        {
            "index_baseline":    baseline["composite_score"],
            "tier_baseline":     baseline["tier"],
            "index_alternative": alternative["composite_score"],
            "tier_alternative":  alternative["tier"],
        }
    )
    comp["tier_change"] = comp["tier_baseline"] != comp["tier_alternative"]
    comp["rank_baseline"] = (
        comp["index_baseline"].rank(ascending=False, method="min").astype(int)
    )
    comp["rank_alternative"] = (
        comp["index_alternative"].rank(ascending=False, method="min").astype(int)
    )
    comp["rank_shift"] = comp["rank_baseline"] - comp["rank_alternative"]
    return comp


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_metrics_pipeline() -> dict:
    """
    Build and save district-level composite indices (baseline + alternative).

    Outputs
    -------
    output/tables/district_scores_baseline.csv
    output/tables/district_scores_alternative.csv
    output/tables/specification_comparison.csv
    """
    for d in ("output/tables", "output/figures"):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────
    print("Loading processed data...")
    distritos_geo = pd.read_csv(
        PROCESSED_DIR / "distritos_geo.csv",
        dtype={"ubigeo": str},
    )
    consulta = load_consulta_clean()
    ccpp = gpd.read_file(PROCESSED_DIR / "ccpp_with_distances.gpkg")

    print(f"  distritos_geo : {distritos_geo.shape[0]:,} rows")
    print(f"  consulta      : {consulta.shape[0]:,} rows")
    print(f"  ccpp          : {ccpp.shape[0]:,} rows")

    # Drop districts without ubigeo — cannot link to any other dataset
    distritos_geo = distritos_geo.dropna(subset=["ubigeo"]).set_index("ubigeo")
    ubigeos = distritos_geo.index
    print(f"  districts with valid ubigeo: {len(ubigeos):,}")

    # ── Shared components (identical across both specifications) ──────────
    print("\nComputing shared components...")

    fac_score      = compute_facility_component(distritos_geo)
    activity_score = compute_activity_component(consulta, ubigeos)

    # Aggregate emergency consultations for output table
    emerg_agg = (
        consulta[consulta["categoria"].isin(EMERGENCY_LEVELS)]
        .groupby("ubigeo")["total_atenciones"]
        .sum()
        .rename("emerg_atenciones")
        .reindex(ubigeos, fill_value=0)
    )

    for name, s in [("fac_score", fac_score), ("activity_score", activity_score)]:
        print(f"  {name:20s}: median={s.median():.1f}  "
              f"zero-count={( s < 1).sum():,}")

    # ── Access component — BASELINE (30 km, emergency facilities) ─────────
    print("\nAccess component — baseline (30 km to nearest emergency facility)...")
    access_b = compute_access_component(
        ccpp, ubigeos,
        threshold_km=30,
        dist_col="dist_km_nearest_emergency",
    )
    print(f"  pct_within_30km_emergency: "
          f"mean={access_b['pct_within'].mean():.1f}%  "
          f"median={access_b['pct_within'].median():.1f}%")

    # ── Access component — ALTERNATIVE (15 km, any facility) ─────────────
    print("Access component — alternative (15 km to nearest any facility)...")
    access_a = compute_access_component(
        ccpp, ubigeos,
        threshold_km=15,
        dist_col="dist_km_nearest_any",
    )
    print(f"  pct_within_15km_any      : "
          f"mean={access_a['pct_within'].mean():.1f}%  "
          f"median={access_a['pct_within'].median():.1f}%")

    # ── Build BASELINE index ──────────────────────────────────────────────
    print("\nBuilding BASELINE index (w=1/3, 1/3, 1/3 | 30 km emergency) ...")
    baseline_score = compute_index(
        fac_score, activity_score, access_b["access_score"],
        w_fac=1, w_activity=1, w_access=1,
    )
    baseline_df = distritos_geo[
        ["distrito", "provincia", "departamento",
         "n_active", "n_emergency", "area_km2",
         "dist_km_nearest_emergency", "dist_km_nearest_any"]
    ].copy()
    baseline_df["fac_score"]                   = fac_score
    baseline_df["activity_score"]              = activity_score
    baseline_df["emerg_atenciones"]            = emerg_agg
    baseline_df["access_score"]                = access_b["access_score"]
    baseline_df["pct_ccpp_within_30km_emerg"]  = access_b["pct_within"]
    baseline_df["composite_score"]             = baseline_score
    baseline_df["tier"]                        = classify_districts(baseline_score)

    # ── Build ALTERNATIVE index ───────────────────────────────────────────
    print("Building ALTERNATIVE index (w=0.25, 0.25, 0.50 | 15 km any) ...")
    alt_score = compute_index(
        fac_score, activity_score, access_a["access_score"],
        w_fac=0.25, w_activity=0.25, w_access=0.50,
    )
    alt_df = distritos_geo[
        ["distrito", "provincia", "departamento",
         "n_active", "n_emergency", "area_km2",
         "dist_km_nearest_emergency", "dist_km_nearest_any"]
    ].copy()
    alt_df["fac_score"]                 = fac_score
    alt_df["activity_score"]            = activity_score
    alt_df["emerg_atenciones"]          = emerg_agg
    alt_df["access_score"]              = access_a["access_score"]
    alt_df["pct_ccpp_within_15km_any"]  = access_a["pct_within"]
    alt_df["composite_score"]           = alt_score
    alt_df["tier"]                      = classify_districts(alt_score)

    # ── Comparison ────────────────────────────────────────────────────────
    print("\nComparing specifications...")
    comp = compare_specifications(baseline_df, alt_df)
    comp = comp.join(baseline_df[["distrito", "provincia", "departamento"]])

    # Spearman ρ = Pearson correlation of ranks (no scipy needed)
    spearman_r = comp["index_baseline"].rank().corr(comp["index_alternative"].rank())
    n_changed  = comp["tier_change"].sum()
    pct_changed = n_changed / len(comp) * 100

    print(f"  Spearman ρ (rank correlation)  : {spearman_r:.4f}")
    print(f"  Districts that changed tier    : {n_changed:,} / {len(comp):,} "
          f"({pct_changed:.1f}%)")

    # Districts with largest rank shifts
    movers = comp.nlargest(5, "rank_shift")[
        ["distrito", "departamento", "rank_baseline",
         "rank_alternative", "rank_shift", "tier_baseline", "tier_alternative"]
    ]
    print("\n  Top 5 districts that rose most in alternative spec:")
    print(movers.to_string(index=True))

    # ── Tier distributions ────────────────────────────────────────────────
    tier_order = ["Underserved", "Weakly served", "Moderately served", "Best served"]
    print("\nBaseline tier distribution:")
    for t in tier_order:
        n = (baseline_df["tier"] == t).sum()
        print(f"  {t:22s}: {n:4d} districts")

    print("\nAlternative tier distribution:")
    for t in tier_order:
        n = (alt_df["tier"] == t).sum()
        print(f"  {t:22s}: {n:4d} districts")

    # ── Save outputs ──────────────────────────────────────────────────────
    print("\nSaving analytical outputs...")
    baseline_df.reset_index().to_csv(
        OUTPUT_TABLES / "district_scores_baseline.csv", index=False
    )
    alt_df.reset_index().to_csv(
        OUTPUT_TABLES / "district_scores_alternative.csv", index=False
    )
    comp.reset_index().to_csv(
        OUTPUT_TABLES / "specification_comparison.csv", index=False
    )
    print(f"  district_scores_baseline.csv      ({len(baseline_df):,} rows)")
    print(f"  district_scores_alternative.csv   ({len(alt_df):,} rows)")
    print(f"  specification_comparison.csv      ({len(comp):,} rows)")

    return {
        "baseline":    baseline_df,
        "alternative": alt_df,
        "comparison":  comp,
        "spearman_r":  spearman_r,
        "n_changed":   n_changed,
    }


if __name__ == "__main__":
    from src.visualization import run_visualization_pipeline
    results = run_metrics_pipeline()
    run_visualization_pipeline(results)
