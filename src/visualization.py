"""
Static visualizations — matplotlib + seaborn.

Figure inventory (13 figures, all saved to output/figures/)
-----------------------------------------------------------
Maps (geographic context):
  map_fac_score.png               Q1  Facility density choropleth
  map_activity_score.png          Q1  Emergency activity choropleth
  map_access_score_baseline.png   Q2  Spatial access choropleth
  map_baseline_tiers.png          Q3  Baseline service-tier classification
  map_alternative_tiers.png       Q4  Alternative specification tiers

Statistical charts:
  q1_fac_vs_activity_scatter.png  Q1  Facility availability vs. emergency activity scatter
  q2_access_score_kde.png         Q2  KDE of spatial access distribution across districts
  q2_access_by_department.png     Q2  Department box plots sorted by median access
  q3_components_by_tier.png       Q3  Component box plots validating tier separation
  q3_top_bottom_districts.png     Q3  Horizontal bar chart — top/bottom 20 districts
  q4_kde_comparison.png           Q4  Overlaid KDE of composite scores by specification
  q4_rank_shift_distribution.png  Q4  Histogram of per-district rank shifts
  q4_score_ecdf_comparison.png    Q4  Overlaid ECDFs of both composite scores
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

FIGURES = Path("output/figures")

TIER_COLORS = {
    "Best served":       "#1a7c2e",
    "Moderately served": "#76b041",
    "Weakly served":     "#e8a020",
    "Underserved":       "#c0392b",
}
TIER_ORDER = ["Underserved", "Weakly served", "Moderately served", "Best served"]


# ── Map helpers ───────────────────────────────────────────────────────────────

def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    col: str,
    title: str,
    filename: str,
    cmap: str = "RdYlGn",
    legend_label: str = "Score (0–100)",
) -> None:
    """Continuous-variable choropleth map."""
    fig, ax = plt.subplots(figsize=(9, 11))
    gdf.plot(
        column=col, ax=ax, cmap=cmap, legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.55, "pad": 0.02},
        missing_kwds={"color": "#cccccc", "label": "No data"},
        linewidth=0.07, edgecolor="#aaaaaa",
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def plot_tier_map(
    gdf: gpd.GeoDataFrame,
    tier_col: str,
    title: str,
    filename: str,
) -> None:
    """Categorical service-tier choropleth map."""
    fig, ax = plt.subplots(figsize=(9, 11))
    for tier in TIER_ORDER:
        subset = gdf[gdf[tier_col] == tier]
        if len(subset):
            subset.plot(ax=ax, color=TIER_COLORS[tier],
                        linewidth=0.07, edgecolor="#aaaaaa")
    gdf[gdf[tier_col].isna()].plot(
        ax=ax, color="#cccccc", linewidth=0.07, edgecolor="#aaaaaa"
    )
    patches = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in TIER_ORDER]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              title="Service tier", title_fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Q1 — Facility availability vs. emergency activity ─────────────────────────

def plot_fac_vs_activity_scatter(
    baseline: pd.DataFrame,
    filename: str,
    n_excluded: int = 0,
) -> None:
    """
    Q1 — Scatter of facility availability score (x) vs. all-facility activity
    score (y), restricted to districts with at least one HIS consultation record.

    Districts with zero consultations across all categories are excluded because
    they all receive the same tied percentile rank, producing a meaningless
    horizontal cluster.  Their absence is noted in the chart footer.
    """
    df = baseline.copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df["fac_score"], df["all_activity_score"],
        color="#2c7bb6", alpha=0.45, s=14, linewidths=0,
    )

    r = df["fac_score"].corr(df["all_activity_score"])
    ax.text(
        0.98, 0.97, f"Pearson r = {r:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    if n_excluded > 0:
        ax.text(
            0.02, 0.02,
            f"{n_excluded:,} districts excluded — no HIS consultation records",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color="#888888",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.80),
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Facility availability score (0–100)", fontsize=11)
    ax.set_ylabel("All-facility activity score (0–100)", fontsize=11)
    ax.set_title(
        "Q1 — Facility availability vs. total outpatient activity by district\n"
        "(all facility categories I-1 to III-E; districts with no HIS records excluded)",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Q2 — Spatial access from populated centres ────────────────────────────────

def plot_access_score_kde(baseline: pd.DataFrame, filename: str) -> None:
    """
    Q2 — Kernel density estimate of the raw spatial access indicator across
    all districts: % of CCPP populated places within 30 km of the nearest
    emergency-level facility (baseline threshold).

    A concentration of districts near 0 % means most districts have almost
    no populated places within reach of emergency care; a concentration near
    100 % would mean broad coverage.  The KDE shows whether the distribution
    is bimodal (two populations of districts), skewed, or spread evenly.
    """
    col = "pct_ccpp_within_30km_emerg"
    data = baseline[col].dropna()

    pct_above_50 = (data > 50).mean() * 100
    pct_below_10 = (data < 10).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.kdeplot(
        x=data, ax=ax, fill=True, alpha=0.40,
        color="#2c7bb6", lw=2.2,
    )

    ax.axvline(50, color="#888888", ls="--", lw=1.3, alpha=0.8,
               label="50 % reference")
    ax.axvline(10, color="#c0392b", ls=":", lw=1.1, alpha=0.7,
               label="10 % reference")

    ax.text(
        0.98, 0.95,
        f"{pct_below_10:.0f}% of districts below 10%\n"
        f"{pct_above_50:.0f}% of districts above 50%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88),
    )

    ax.set_xlim(-2, 102)
    ax.set_xlabel(
        "% of populated places (CCPP) within 30 km of nearest emergency facility",
        fontsize=10,
    )
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "Q2 — Distribution of the spatial access indicator across districts\n"
        "(0 % = no populated place has access; 100 % = all populated places have access)",
        fontsize=11,
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def plot_access_by_department(baseline: pd.DataFrame, filename: str) -> None:
    """
    Q2 — Horizontal box plots of % CCPP places within 30 km of an emergency
    facility, one box per department, sorted ascending by median.
    """
    dept_medians = (baseline
                    .groupby("departamento")["pct_ccpp_within_30km_emerg"]
                    .median()
                    .sort_values())
    dept_order = dept_medians.index.tolist()
    n = len(dept_order)
    palette = sns.color_palette("RdYlGn", n)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(
        data=baseline,
        y="departamento", x="pct_ccpp_within_30km_emerg",
        order=dept_order, hue="departamento", palette=palette, legend=False,
        ax=ax, width=0.65, showfliers=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
        medianprops={"color": "black", "lw": 1.5},
        linewidth=0.7,
    )
    ax.axvline(50, color="#555555", ls="--", lw=1.0, alpha=0.7,
               label="50 % reference line")
    ax.set_xlabel("% of populated places (CCPP) within 30 km\nof nearest emergency facility",
                  fontsize=10)
    ax.set_ylabel("Department (sorted by median, worst → best)", fontsize=10)
    ax.set_title(
        "Q2 — Spatial access by department (baseline, 30 km to emergency facility)\n"
        "Each box = distribution across districts within the department",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(-5, 105)

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Q3 — Multi-dimensional classification ────────────────────────────────────

def plot_components_by_tier(baseline: pd.DataFrame, filename: str) -> None:
    """
    Q3 — Three-panel box plot: each panel shows one component score,
    boxes split by service tier.
    """
    components = ["fac_score", "activity_score", "access_score"]
    titles     = ["Facility availability score",
                  "Emergency activity score",
                  "Spatial access score (baseline)"]
    tier_palette = [TIER_COLORS[t] for t in TIER_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, col, title in zip(axes, components, titles):
        sns.boxplot(
            data=baseline, x="tier", y=col, order=TIER_ORDER,
            hue="tier", palette=tier_palette, legend=False,
            width=0.55, ax=ax, showfliers=False,
            medianprops={"color": "black", "lw": 1.8},
            linewidth=0.8,
        )
        ax.set_ylim(0, 100)
        ax.set_xlabel("")
        ax.set_ylabel("Score (0–100)" if ax is axes[0] else "")
        ax.set_title(title, fontsize=10, pad=6)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.axhline(50, color="#aaaaaa", ls=":", lw=0.8)

    fig.suptitle(
        "Q3 — Component scores by service tier\n"
        "(validates that tier classification captures multi-dimensional deprivation)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def plot_top_bottom_districts(
    baseline: pd.DataFrame,
    n: int = 20,
    filename: str = "q3_top_bottom_districts.png",
) -> None:
    """
    Q3 — Horizontal bar charts naming the 20 most underserved and 20 best-
    served districts by composite score.
    """
    df = baseline.reset_index(drop=True).sort_values("composite_score")
    bottom = df.head(n)
    top    = df.tail(n)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, subset, color, title_txt in [
        (axes[0], bottom, "#c0392b", f"Bottom {n} — Most underserved"),
        (axes[1], top,   "#1a7c2e", f"Top {n} — Best served"),
    ]:
        labels = (subset["distrito"].str[:18] + "\n"
                  + subset["departamento"].str[:12])
        scores = subset["composite_score"].values

        ax.barh(range(len(subset)), scores, color=color, alpha=0.80)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Composite score (0–100)", fontsize=9)
        ax.set_title(title_txt, fontsize=10, pad=8)
        ax.axvline(50, color="#888888", ls="--", lw=0.8)

        for i, v in enumerate(scores):
            ax.text(v + 1.2, i, f"{v:.1f}", va="center", fontsize=6.5)

    fig.suptitle(
        "Q3 — Baseline composite index: most and least served districts",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Q4 — Specification sensitivity ───────────────────────────────────────────

def plot_specification_kde(
    baseline: pd.DataFrame,
    alternative: pd.DataFrame,
    filename: str,
) -> None:
    """
    Q4 — Overlaid KDE of the composite index score for both specifications.

    Both curves use the same 0–100 score space.  If the access definition
    change shifts the overall distribution (e.g. one specification assigns
    systematically higher scores), the two KDE curves will diverge: one will
    sit to the right of the other.  If the shapes differ (one wider, one more
    peaked), the alternative definition is more or less discriminating in
    separating districts.  Overlapping curves indicate that the overall score
    distribution is robust to the specification choice, even if individual
    district ranks shift.
    """
    b_scores = baseline["composite_score"].dropna()
    a_scores = alternative["composite_score"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.kdeplot(
        x=b_scores, ax=ax, fill=True, alpha=0.35, lw=2.2,
        color="#2c7bb6",
        label=f"Baseline  (equal weights, 30 km emergency)  "
              f"median={b_scores.median():.1f}",
    )
    sns.kdeplot(
        x=a_scores, ax=ax, fill=True, alpha=0.35, lw=2.2,
        color="#d7191c",
        label=f"Alternative  (access×2, 15 km any facility)  "
              f"median={a_scores.median():.1f}",
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Composite index score (0–100)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "Q4 — Distribution of composite scores by specification\n"
        "Divergence between curves = effect of changing the access definition",
        fontsize=11,
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def plot_rank_shift_distribution(comparison: pd.DataFrame, filename: str) -> None:
    """
    Q4 — Histogram with KDE of the per-district rank shift between
    specifications (positive = rose in the alternative ranking).
    """
    df = comparison.copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(df["rank_shift"].min(), df["rank_shift"].max(), 70)
    ax.hist(df.loc[df["rank_shift"] <  0, "rank_shift"].values,
            bins=bins, color="#c0392b", alpha=0.70,
            label=f"Fell in alternative  (n={(df['rank_shift'] < 0).sum():,})")
    ax.hist(df.loc[df["rank_shift"] == 0, "rank_shift"].values,
            bins=3,   color="#888888", alpha=0.70,
            label=f"No change            (n={(df['rank_shift'] == 0).sum():,})")
    ax.hist(df.loc[df["rank_shift"] >  0, "rank_shift"].values,
            bins=bins, color="#1a7c2e", alpha=0.70,
            label=f"Rose in alternative  (n={(df['rank_shift'] > 0).sum():,})")

    sns.kdeplot(x=df["rank_shift"], ax=ax, color="black", lw=1.3, bw_adjust=0.8)
    ax.axvline(0, color="black", lw=1.4)

    n_changed = df["tier_change"].sum()
    ax.text(0.98, 0.95,
            f"{n_changed:,} districts changed tier ({n_changed/len(df)*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("Rank shift (baseline rank − alternative rank)\n"
                  "Positive = district rose in the alternative ranking", fontsize=10)
    ax.set_ylabel("Number of districts", fontsize=10)
    ax.set_title(
        "Q4 — Distribution of rank shifts between baseline and alternative specification\n"
        "KDE overlay shows overall shape; symmetric around 0 = specifications agree",
        fontsize=11,
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def plot_score_ecdf_comparison(comparison: pd.DataFrame, filename: str) -> None:
    """
    Q4 — Overlaid ECDFs of the two composite index scores across all districts.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.ecdfplot(data=comparison, x="index_baseline", ax=ax,
                 label="Baseline (equal weights, 30 km emergency)",
                 color="#2c7bb6", lw=2.2)
    sns.ecdfplot(data=comparison, x="index_alternative", ax=ax,
                 label="Alternative (access×2, 15 km any facility)",
                 color="#d7191c", lw=2.2)

    for q, lbl in [(0.25, "P25 — tier cut"), (0.50, "P50"), (0.75, "P75 — tier cut")]:
        ax.axhline(q, color="#aaaaaa", ls=":", lw=0.9)
        ax.text(101, q, lbl, va="center", fontsize=7, color="#666666")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Composite index score (0–100)", fontsize=10)
    ax.set_ylabel("Cumulative share of districts", fontsize=10)
    ax.set_title(
        "Q4 — Score distribution comparison: baseline vs. alternative\n"
        "Horizontal dashed lines = quartile tier boundaries",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_visualization_pipeline(results: dict) -> None:
    """
    Generate and save all 13 static figures.

    Expects `results` dict from metrics.run_metrics_pipeline().
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    baseline    = results["baseline"]
    alternative = results["alternative"]
    comparison  = results["comparison"]

    print("\nLoading district geometry for maps...")
    dist_geo = gpd.read_file("data/processed/distritos_geo.gpkg")

    # All-facility activity score (for Q1 scatter only — composite index unchanged)
    consulta = pd.read_csv(
        "data/processed/consulta_clean.csv", dtype={"ubigeo": str}
    )
    all_agg = (
        consulta.groupby("ubigeo")["total_atenciones"]
        .sum()
        .reindex(baseline.index, fill_value=0)
    )
    baseline = baseline.copy()
    baseline["all_activity_score"] = (
        all_agg.rank(pct=True, method="average") * 100
    )

    b_geo = dist_geo.merge(
        baseline.reset_index()[
            ["ubigeo", "composite_score", "fac_score",
             "activity_score", "access_score", "tier"]
        ],
        on="ubigeo", how="left",
    )
    a_geo = dist_geo.merge(
        alternative.reset_index()[["ubigeo", "tier"]],
        on="ubigeo", how="left",
    )

    print("Generating figures...")

    # ── Maps ──────────────────────────────────────────────────────────────
    plot_choropleth(
        b_geo, "fac_score",
        title="Q1 — Facility availability score: active-facility density (Peru, 2025)",
        filename="map_fac_score.png",
        legend_label="Percentile rank (0–100)",
    )
    plot_choropleth(
        b_geo, "activity_score",
        title="Q1 — Emergency activity score: consultations at II+III facilities (Peru, 2025)",
        filename="map_activity_score.png",
        legend_label="Percentile rank (0–100)",
    )
    plot_choropleth(
        b_geo, "access_score",
        title="Q2 — Spatial access score: % CCPP within 30 km of emergency facility (Peru, 2025)",
        filename="map_access_score_baseline.png",
        legend_label="Percentile rank (0–100)",
    )
    plot_tier_map(
        b_geo, "tier",
        title="Q3 — Baseline service tiers: district healthcare access (Peru, 2025)",
        filename="map_baseline_tiers.png",
    )
    plot_tier_map(
        a_geo, "tier",
        title="Q4 — Alternative service tiers: access×2 weight, 15 km any facility (Peru, 2025)",
        filename="map_alternative_tiers.png",
    )

    # ── Q1 ────────────────────────────────────────────────────────────────
    baseline_nonzero = baseline[all_agg > 0].copy()
    n_excluded = len(baseline) - len(baseline_nonzero)
    plot_fac_vs_activity_scatter(
        baseline_nonzero,
        filename="q1_fac_vs_activity_scatter.png",
        n_excluded=n_excluded,
    )

    # ── Q2 ────────────────────────────────────────────────────────────────
    plot_access_score_kde(baseline,
                          filename="q2_access_score_kde.png")
    plot_access_by_department(baseline,
                              filename="q2_access_by_department.png")

    # ── Q3 ────────────────────────────────────────────────────────────────
    plot_components_by_tier(baseline,
                            filename="q3_components_by_tier.png")
    plot_top_bottom_districts(baseline, n=20,
                              filename="q3_top_bottom_districts.png")

    # ── Q4 ────────────────────────────────────────────────────────────────
    plot_specification_kde(baseline, alternative,
                           filename="q4_kde_comparison.png")
    plot_rank_shift_distribution(comparison,
                                 filename="q4_rank_shift_distribution.png")
    plot_score_ecdf_comparison(comparison,
                               filename="q4_score_ecdf_comparison.png")

    print("\nVisualization pipeline complete.")
    print(f"  13 figures saved to {FIGURES}/")
