"""
Geospatial outputs: static multi-layer maps (GeoPandas + matplotlib) and
interactive exploration maps (Folium).

The maps are designed to explain the structure of emergency healthcare access
inequality in Peru — not simply to display raw locations.

Static maps  → output/figures/*.png
  geo_bivariate_choropleth.png  Q1+Q3  Doubly-disadvantaged districts (2×2 matrix)
  geo_facility_desert.png       Q1+Q2  Emergency facility deserts + facility locations
  geo_ccpp_access_map.png       Q2     Populated places by distance-to-emergency category
  geo_tier_divergence.png       Q4     Geographic pattern of specification sensitivity

Interactive maps → output/figures/*.html
  map_tiers_interactive.html    Q3     Tier explorer: hover tooltips + baseline/alt toggle
  map_access_explorer.html      Q1+Q2  Access score choropleth + emergency facility markers
  map_ccpp_distances.html       Q2     CCPP access-deficit heatmap for regional zoom-in
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MiniMap, Fullscreen
import branca.colormap as bcm
from pathlib import Path

FIGURES      = Path("output/figures")
PROCESSED    = Path("data/processed")
OUT_TABLES   = Path("output/tables")

EMERGENCY_LEVELS = frozenset({"II-1", "II-2", "II-E", "III-1", "III-2", "III-E"})

TIER_COLORS = {
    "Best served":       "#1a7c2e",
    "Moderately served": "#76b041",
    "Weakly served":     "#e8a020",
    "Underserved":       "#c0392b",
}
TIER_ORDER = ["Underserved", "Weakly served", "Moderately served", "Best served"]

# 2×2 bivariate scheme  (fac_high: 0=below median, 1=above)  ×  (acc_high: same)
BIVAR = {
    (0, 0): ("#d7191c", "Doubly underserved\n(low facilities, low access)"),
    (1, 0): ("#fdae61", "Facilities, isolated settlements\n(high facilities, low access)"),
    (0, 1): ("#1a9641", "Accessible, few facilities\n(low facilities, high access)"),
    (1, 1): ("#2c7bb6", "Well served\n(high facilities, high access)"),
}

# Distance-to-emergency categories
DIST_CATS = [
    (0,   15,  "#1a9641", "< 15 km"),
    (15,  30,  "#76b041", "15 – 30 km"),
    (30,  60,  "#e8a020", "30 – 60 km"),
    (60, 1e9,  "#c0392b", "> 60 km"),
]

PERU_CENTER = [-9.19, -75.0]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mapping_data() -> tuple:
    """
    Load and merge all datasets.  Returns (merged_gdf, ipress_emergency, ccpp).

    merged_gdf — district polygons with baseline scores, alternative tier,
                 rank shift, and tier-change flag.
    ipress_emergency — geolocated emergency-level facilities only.
    ccpp — populated places with pre-computed distances.
    """
    print("Loading data...")

    dist  = gpd.read_file(PROCESSED / "distritos_geo.gpkg")
    base  = pd.read_csv(OUT_TABLES / "district_scores_baseline.csv",
                        dtype={"ubigeo": str})
    alt   = pd.read_csv(OUT_TABLES / "district_scores_alternative.csv",
                        dtype={"ubigeo": str})
    comp  = pd.read_csv(OUT_TABLES / "specification_comparison.csv",
                        dtype={"ubigeo": str})
    ipress = gpd.read_file(PROCESSED / "ipress_geo.gpkg")
    ccpp   = gpd.read_file(PROCESSED / "ccpp_with_distances.gpkg")

    # Select score columns that don't duplicate dist's spatial columns
    score_cols = ["ubigeo", "fac_score", "activity_score", "emerg_atenciones",
                  "access_score", "pct_ccpp_within_30km_emerg",
                  "composite_score", "tier"]

    merged = dist.merge(base[score_cols], on="ubigeo", how="left")
    merged = merged.merge(
        alt[["ubigeo", "composite_score", "tier"]].rename(
            columns={"composite_score": "composite_score_alt", "tier": "tier_alt"}
        ),
        on="ubigeo", how="left",
    )
    merged = merged.merge(
        comp[["ubigeo", "tier_change", "rank_shift"]],
        on="ubigeo", how="left",
    )

    ipress_emerg = ipress[ipress["categoria"].isin(EMERGENCY_LEVELS)].copy()

    for name, ds in [("districts", merged), ("emerg. facilities", ipress_emerg),
                     ("CCPP", ccpp)]:
        print(f"  {name:20s}: {len(ds):>7,} rows")

    return merged, ipress_emerg, ccpp


# ── Shared helpers ────────────────────────────────────────────────────────────

def _ax_off(ax) -> None:
    ax.set_axis_off()


def _simplify(gdf: gpd.GeoDataFrame, tol: float = 0.008) -> gpd.GeoDataFrame:
    """Return a copy with simplified geometry for faster rendering."""
    out = gdf.copy()
    out["geometry"] = out.geometry.simplify(tolerance=tol, preserve_topology=True)
    return out


def _prep_folium(gdf: gpd.GeoDataFrame, num_cols: list, str_cols: list,
                 round_dec: int = 1) -> gpd.GeoDataFrame:
    """Fill NaN, round floats, and strip large unused columns for Folium."""
    g = gdf.copy()
    for c in num_cols:
        if c in g.columns:
            g[c] = g[c].round(round_dec).fillna(-999)
    for c in str_cols:
        if c in g.columns:
            g[c] = g[c].fillna("N/A")
    return g


def _folium_tier_legend(m: folium.Map) -> None:
    """Inject a fixed-position tier-colour legend into a Folium map."""
    items = "".join(
        f'<div><i style="background:{TIER_COLORS[t]};width:14px;height:14px;'
        f'float:left;margin-right:6px;margin-top:2px;display:inline-block;'
        f'border-radius:2px;"></i>{t}</div>'
        for t in reversed(TIER_ORDER)
    )
    html = (
        '<div style="position:fixed;bottom:60px;left:20px;width:200px;'
        'background:white;border:1px solid #aaa;z-index:9999;padding:10px;'
        'border-radius:6px;font-family:Arial;font-size:12px;">'
        f'<b>Service tier</b><br>{items}'
        '<div style="margin-top:6px">'
        '<i style="background:#cccccc;width:14px;height:14px;float:left;'
        'margin-right:6px;margin-top:2px;display:inline-block;border-radius:2px;"></i>'
        'No data</div></div>'
    )
    m.get_root().html.add_child(folium.Element(html))


# ── Static map 1 — bivariate choropleth (Q1 + Q3) ───────────────────────────

def plot_bivariate_choropleth(merged: gpd.GeoDataFrame, filename: str) -> None:
    """
    Q1 + Q3 — 2×2 bivariate choropleth: facility availability score (x-axis)
    vs. spatial access score (y-axis), each split at its median into high/low.

    This reveals four structurally distinct district types:
      • Doubly underserved (low–low): insufficient facilities AND most CCPP
        places >30 km from emergency care — the most policy-critical quadrant.
      • Well served (high–high): adequate density AND high geographic proximity.
      • Facilities with isolated settlements (high fac / low access): hospitals
        or clinics exist but many populated places remain far away — suggests
        geography rather than infrastructure as the binding constraint.
      • Accessible, few facilities (low fac / high access): communities are
        geographically close to emergency care but the facility count per km²
        is below the national median.

    Why 2×2 over a continuous bivariate scale: four named quadrants tell a
    direct policy story; a 3×3 or continuous scale adds precision at the cost
    of interpretability.
    """
    df = merged.copy()

    # Median splits (ignoring NaN districts)
    fac_med = df["fac_score"].median()
    acc_med = df["access_score"].median()
    df["fac_hi"]  = (df["fac_score"]    >= fac_med).astype("Int64")
    df["acc_hi"]  = (df["access_score"] >= acc_med).astype("Int64")
    df["bivar"] = list(zip(df["fac_hi"].fillna(-1).astype(int),
                           df["acc_hi"].fillna(-1).astype(int)))

    fig, ax = plt.subplots(figsize=(10, 12))

    # Draw full GDF as grey base first — establishes axes extent and
    # covers NaN-score districts (no ubigeo) without a separate empty-plot call
    df.plot(ax=ax, color="#cccccc", linewidth=0.07, edgecolor="#aaaaaa")

    # Overlay each quadrant (apply-based mask is safe for tuple Series)
    for key, (color, label) in BIVAR.items():
        sub = df[df["bivar"].apply(lambda b: b == key)]
        if len(sub):
            sub.plot(ax=ax, color=color, linewidth=0.07, edgecolor="#aaaaaa")

    # ── 2×2 legend inset ──
    ax_leg = ax.inset_axes([0.03, 0.03, 0.26, 0.26])
    grid = [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]  # [row: acc, col: fac]
    for ri, row in enumerate(grid):
        for ci, key in enumerate(row):
            color = BIVAR[key][0]
            rect = plt.Rectangle([ci, ri], 1, 1, color=color, ec="white", lw=0.8)
            ax_leg.add_patch(rect)
    ax_leg.set_xlim(0, 2)
    ax_leg.set_ylim(0, 2)
    ax_leg.set_xticks([0.5, 1.5])
    ax_leg.set_yticks([0.5, 1.5])
    ax_leg.set_xticklabels(["Low\nfacilities", "High\nfacilities"], fontsize=7)
    ax_leg.set_yticklabels(["Low\naccess", "High\naccess"], fontsize=7)
    ax_leg.tick_params(length=0)
    ax_leg.set_title("Legend", fontsize=7, pad=3)

    # Quadrant counts for annotation
    counts = {k: (df["bivar"] == k).sum() for k in BIVAR}
    ax.text(
        0.03, 0.33,
        "\n".join(f"{BIVAR[k][1].splitlines()[0]}: {counts[k]:,}" for k in BIVAR),
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
    )

    ax.set_title(
        "Q1 + Q3 — Bivariate map: facility density × spatial access\n"
        "Districts in red are doubly underserved on both dimensions",
        fontsize=12, pad=10,
    )
    _ax_off(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Static map 2 — emergency facility desert (Q1 + Q2) ──────────────────────

def plot_facility_desert(
    merged: gpd.GeoDataFrame,
    ipress_emerg: gpd.GeoDataFrame,
    filename: str,
) -> None:
    """
    Q1 + Q2 — Multi-layer map that makes the 'emergency facility desert'
    visually concrete.

    Layer 1 (base): district polygons coloured by whether the district has
    ANY emergency-level facility.
      • 0 facilities — light grey ('desert')
      • ≥1 facility  — shaded by count (light to dark green)
    Layer 2 (overlay): 268 geolocated emergency-level facilities as small
    red circles, clustered in coastal cities while vast inland areas are empty.

    The visual narrative: the overwhelming grey signals that 89 % of Peru's
    districts have no emergency-level facility within their boundaries; the
    handful of facility points are concentrated along the Pacific coast,
    leaving the Amazon basin and high Andes as near-total deserts.

    Why not just a choropleth of n_emergency: a choropleth conveys the count
    but not the spatial reality of isolated points surrounded by empty
    districts.  The point overlay makes the isolation tangible.
    """
    df = merged.copy()

    # Classify by emergency facility count
    bins   = [-1, 0, 1, 5, np.inf]
    labels = ["No emergency facility", "1 facility", "2–5 facilities", "6+ facilities"]
    colors = ["#e5e5e5", "#fdcc8a", "#fc8d59", "#b30000"]

    df["emerg_class"] = pd.cut(df["n_emergency"].fillna(0),
                               bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 12))

    for label, color in zip(labels, colors):
        sub = df[df["emerg_class"] == label]
        if len(sub):
            sub.plot(ax=ax, color=color, linewidth=0.06, edgecolor="#bbbbbb")

    # Emergency facility points
    ipress_emerg.plot(
        ax=ax, markersize=10, marker="o",
        color="#c0392b", edgecolor="#7b241c", linewidth=0.5, alpha=0.85,
        label="Emergency facility",
        zorder=5,
    )

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in zip(labels, colors)]
    patches.append(mpatches.Patch(color="#c0392b",
                                  label=f"Emergency facility (n={len(ipress_emerg):,})"))
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              title="Emergency facility count", title_fontsize=9, framealpha=0.9)

    # Annotation
    n_desert = (df["n_emergency"].fillna(0) == 0).sum()
    pct      = n_desert / len(df) * 100
    ax.text(
        0.03, 0.97,
        f"{n_desert:,} of {len(df):,} districts\n({pct:.0f}%) have zero\nemergency-level facilities",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
    )

    ax.set_title(
        "Q1 + Q2 — The emergency facility desert in Peru (2025)\n"
        "Grey = no emergency-level facility within district boundaries",
        fontsize=12, pad=10,
    )
    _ax_off(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Static map 3 — CCPP access map (Q2) ──────────────────────────────────────

def plot_ccpp_access_map(
    merged: gpd.GeoDataFrame,
    ccpp: gpd.GeoDataFrame,
    filename: str,
) -> None:
    """
    Q2 — CCPP populated-place points coloured by their straight-line distance
    to the nearest emergency-level facility (4 categories), layered on top of
    light district-tier boundaries.

    Showing individual settlement points rather than district-level aggregates
    makes the access problem concrete: 74,577 of Peru's 136,587 populated
    places (55 %) are more than 30 km from emergency care; 23,546 are more
    than 60 km away.  These are real villages and towns, not statistical
    abstractions.

    Sampling strategy: all >60 km points are shown (23,546); a proportional
    random sample of the closer categories fills in the remainder up to
    ~35,000 points total.  This ensures the most isolated settlements —
    the analytical focus — are fully visible rather than under-sampled.

    Why points over a density raster: a density map would show WHERE many
    isolated CCPP are clustered.  The point map shows both the density AND
    the distribution of severity levels simultaneously.
    """
    # ── Stratified sample preserving all extreme cases ──
    far    = ccpp[ccpp["dist_km_nearest_emergency"] > 60].copy()
    medium = ccpp[(ccpp["dist_km_nearest_emergency"] > 30) &
                  (ccpp["dist_km_nearest_emergency"] <= 60)].copy()
    close  = ccpp[ccpp["dist_km_nearest_emergency"] <= 30].copy()

    n_target = 35_000
    n_far    = len(far)                           # keep all
    n_med    = min(len(medium), 7_000)
    n_cl     = min(len(close),  n_target - n_far - n_med)

    sample = pd.concat([
        far,
        medium.sample(n_med, random_state=42),
        close.sample(n_cl,  random_state=42),
    ], ignore_index=True)
    sample_gdf = gpd.GeoDataFrame(sample, geometry="geometry", crs=ccpp.crs)

    fig, ax = plt.subplots(figsize=(10, 12))

    # Background: district tier boundaries (muted)
    for tier in TIER_ORDER:
        sub = merged[merged["tier"] == tier]
        if len(sub):
            sub.plot(ax=ax,
                     color=mcolors.to_rgba(TIER_COLORS[tier], 0.08),
                     linewidth=0.2, edgecolor="#cccccc")

    # CCPP points by category
    for lo, hi, color, label in DIST_CATS:
        mask = (sample_gdf["dist_km_nearest_emergency"] >= lo) & \
               (sample_gdf["dist_km_nearest_emergency"] <  hi)
        sub  = sample_gdf[mask]
        n    = (ccpp["dist_km_nearest_emergency"] >= lo) & \
               (ccpp["dist_km_nearest_emergency"] <  hi)
        if len(sub):
            sub.plot(ax=ax, color=color, markersize=1.2, alpha=0.55,
                     label=f"{label}  ({n.sum():,} total)")

    ax.legend(loc="lower right", fontsize=8, title="Distance to nearest\nemergency facility",
              title_fontsize=8, framealpha=0.9, markerscale=3)

    ax.text(
        0.03, 0.97,
        f"74,577 CCPP (55%) are > 30 km\nfrom emergency care\n"
        f"23,546 (17%) are > 60 km",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
    )

    ax.set_title(
        "Q2 — Populated places by distance to nearest emergency facility\n"
        "Red points = villages/towns >60 km from emergency care",
        fontsize=12, pad=10,
    )
    _ax_off(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Static map 4 — tier divergence (Q4) ──────────────────────────────────────

def plot_tier_divergence(merged: gpd.GeoDataFrame, filename: str) -> None:
    """
    Q4 — Districts whose service tier changed between baseline and alternative
    specification are coloured by the direction and magnitude of their rank
    shift (baseline rank − alternative rank).

    • Green  = rose in the alternative (benefited from the looser, any-facility
               15-km threshold).  Concentration along the coast and in mid-
               Andes valleys reveals that these districts have dense primary
               care but limited emergency-level facilities.
    • Red    = fell in the alternative.  Often large Amazonian and Andean
               districts where even primary-care facilities are sparse.
    • Grey   = tier unchanged (1,134 districts, 64.4 %).

    Why a diverging continuous scale over a categorical change map: the
    magnitude of rank shift (up to ±1,182 ranks) matters as much as the
    direction.  A binary changed/unchanged map would hide that some districts
    move marginally (±10 ranks) while others move dramatically (>500 ranks).
    """
    df = merged.copy()

    # Separate unchanged and changed
    unchanged = df[df["tier_change"] != True].copy()
    changed   = df[df["tier_change"] == True].copy()

    # Diverging normalisation centred at 0
    vabs = df["rank_shift"].abs().quantile(0.95)   # clip extreme outliers for colour scale
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    fig, ax = plt.subplots(figsize=(10, 12))

    # Unchanged base layer
    unchanged.plot(ax=ax, color="#e8e8e8", linewidth=0.06, edgecolor="#cccccc")

    # Changed districts with diverging colour
    changed.plot(
        column="rank_shift", ax=ax, cmap="RdYlGn", norm=norm,
        linewidth=0.06, edgecolor="#999999",
        legend=True,
        legend_kwds={
            "label": "Rank shift (positive = rose in alternative)",
            "shrink": 0.50, "pad": 0.02,
        },
        missing_kwds={"color": "#cccccc"},
    )

    # Annotation
    n_rose = (changed["rank_shift"] > 0).sum()
    n_fell = (changed["rank_shift"] < 0).sum()
    ax.text(
        0.03, 0.97,
        f"Tier changed: {len(changed):,} districts (35.6%)\n"
        f"  Rose in alternative: {n_rose:,}\n"
        f"  Fell in alternative: {n_fell:,}\n"
        f"Grey = tier unchanged ({len(unchanged):,} districts)",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
    )

    ax.set_title(
        "Q4 — Geographic pattern of specification sensitivity\n"
        "Green = rose in alternative spec; Red = fell; Grey = tier unchanged",
        fontsize=12, pad=10,
    )
    _ax_off(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


# ── Interactive map 1 — tier explorer (Q3) ───────────────────────────────────

def make_tier_explorer(
    merged: gpd.GeoDataFrame,
    ipress_emerg: gpd.GeoDataFrame,
    filepath: Path,
) -> None:
    """
    Q3 — Interactive tier explorer.

    • Base: district polygons coloured by baseline service tier.  Hover to see
      district name, department, baseline tier, composite score, and all three
      component scores.  Click for a full-stats popup.
    • Toggle: alternative specification tier layer (initially hidden).
      Switching between layers shows exactly which districts change colour —
      the interactive equivalent of Q4's static divergence map.
    • Toggle: emergency facility markers (red circles with popup).
    • Controls: minimap, fullscreen, layer control.
    """
    # Prepare geometry (simplified) and columns
    gdf = _simplify(merged)
    gdf = _prep_folium(
        gdf,
        num_cols=["composite_score", "fac_score", "activity_score", "access_score",
                  "pct_ccpp_within_30km_emerg", "dist_km_nearest_emergency",
                  "n_emergency", "composite_score_alt", "rank_shift"],
        str_cols=["tier", "tier_alt", "distrito", "provincia", "departamento",
                  "ubigeo"],
    )
    # Replace sentinel -999 with readable "N/A" in string representation for popup
    gdf["tier_change_label"] = gdf["tier_change"].map(
        {True: "YES", False: "no"}
    ).fillna("N/A")

    # Keep only needed columns before to_json (reduces file size)
    keep = ["ubigeo", "distrito", "provincia", "departamento",
            "tier", "tier_alt", "tier_change_label", "rank_shift",
            "composite_score", "composite_score_alt",
            "fac_score", "activity_score", "access_score",
            "pct_ccpp_within_30km_emerg", "dist_km_nearest_emergency",
            "n_emergency", "area_km2", "geometry"]
    gdf = gdf[[c for c in keep if c in gdf.columns]]

    geojson_str = gdf.to_json()

    m = folium.Map(location=PERU_CENTER, zoom_start=6, tiles="CartoDB positron",
                   prefer_canvas=True)

    # ── Baseline tier layer ────────────────────────────────────────────────
    folium.GeoJson(
        geojson_str,
        name="Baseline tiers",
        style_function=lambda f: {
            "fillColor": TIER_COLORS.get(
                (f["properties"].get("tier") or "").strip(), "#cccccc"
            ),
            "color": "#ffffff",
            "weight": 0.4,
            "fillOpacity": 0.78,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["distrito", "departamento", "tier", "composite_score",
                    "fac_score", "activity_score", "access_score"],
            aliases=["District:", "Dept:", "Tier:", "Composite (0–100):",
                     "Facility score:", "Activity score:", "Access score:"],
            style=(
                "background-color:white;color:#333;font-family:Arial;"
                "font-size:12px;padding:8px;"
            ),
            localize=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["ubigeo", "distrito", "provincia", "departamento",
                    "tier", "composite_score", "fac_score", "activity_score",
                    "access_score", "n_emergency",
                    "pct_ccpp_within_30km_emerg", "dist_km_nearest_emergency",
                    "tier_alt", "tier_change_label", "rank_shift"],
            aliases=["UBIGEO:", "District:", "Province:", "Department:",
                     "Baseline tier:", "Score (0–100):", "Facility score:",
                     "Activity score:", "Access score:", "Emergency facilities:",
                     "% CCPP within 30 km:", "Dist. to nearest emerg. (km):",
                     "Alternative tier:", "Tier changed?:", "Rank shift:"],
            max_width=320,
        ),
        show=True,
    ).add_to(m)

    # ── Alternative tier layer (hidden by default) ─────────────────────────
    folium.GeoJson(
        geojson_str,
        name="Alternative tiers (toggle me)",
        style_function=lambda f: {
            "fillColor": TIER_COLORS.get(
                (f["properties"].get("tier_alt") or "").strip(), "#cccccc"
            ),
            "color": "#ffffff",
            "weight": 0.4,
            "fillOpacity": 0.78,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["distrito", "departamento", "tier_alt", "composite_score_alt"],
            aliases=["District:", "Dept:", "Alternative tier:", "Score:"],
        ),
        show=False,
    ).add_to(m)

    # ── Emergency facility markers ─────────────────────────────────────────
    fg_fac = folium.FeatureGroup(name="Emergency facilities", show=True)
    for _, row in ipress_emerg.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="#7b241c",
            weight=1,
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.85,
            tooltip=(
                f"<b>{row.get('nombre_establecimiento', 'N/A')}</b><br>"
                f"Category: {row.get('categoria', 'N/A')}<br>"
                f"Institution: {row.get('institucion', 'N/A')}"
            ),
        ).add_to(fg_fac)
    fg_fac.add_to(m)

    # ── Controls ───────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
    Fullscreen().add_to(m)
    _folium_tier_legend(m)

    m.save(str(filepath))
    print(f"  {filepath.name}")


# ── Interactive map 2 — access explorer (Q1 + Q2) ────────────────────────────

def make_access_explorer(
    merged: gpd.GeoDataFrame,
    ipress_emerg: gpd.GeoDataFrame,
    filepath: Path,
) -> None:
    """
    Q1 + Q2 — Access score choropleth with emergency facility overlay.

    • Base: continuous spatial access score (0–100) coloured by RdYlGn scale.
      Hovering a district shows the access score, % of CCPP within 30 km,
      and distance from the district centroid to the nearest emergency facility.
    • Toggle: 268 emergency facility markers.  Zooming in reveals the spatial
      relationship between facility locations and district access scores —
      where facilities exist, surrounding districts tend to score high; empty
      regions are systematically red.
    • This map lets the user EXPLORE the mechanism behind Q2 findings: it is
      the geographic absence of emergency facilities, not just district area,
      that drives low access scores.
    """
    gdf = _simplify(merged)
    gdf = _prep_folium(
        gdf,
        num_cols=["access_score", "pct_ccpp_within_30km_emerg",
                  "dist_km_nearest_emergency", "composite_score",
                  "n_emergency", "area_km2"],
        str_cols=["tier", "distrito", "departamento"],
    )

    keep = ["ubigeo", "distrito", "departamento", "tier",
            "access_score", "pct_ccpp_within_30km_emerg",
            "dist_km_nearest_emergency", "composite_score",
            "n_emergency", "area_km2", "geometry"]
    gdf_slim = gdf[[c for c in keep if c in gdf.columns]]

    # Continuous colour map (branca)
    colormap = bcm.linear.RdYlGn_09.scale(0, 100)
    colormap.caption = "Spatial access score (0–100)"

    m = folium.Map(location=PERU_CENTER, zoom_start=6, tiles="CartoDB positron",
                   prefer_canvas=True)

    # Choropleth via GeoJson + style_function
    folium.GeoJson(
        gdf_slim.to_json(),
        name="Spatial access score",
        style_function=lambda f: {
            "fillColor": colormap(
                float(f["properties"].get("access_score") or 0)
                if (f["properties"].get("access_score") or -999) > -999
                else 0
            ),
            "color": "#aaaaaa",
            "weight": 0.3,
            "fillOpacity": 0.80,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["distrito", "departamento", "access_score",
                    "pct_ccpp_within_30km_emerg", "dist_km_nearest_emergency",
                    "n_emergency", "tier"],
            aliases=["District:", "Department:", "Access score (0–100):",
                     "% CCPP within 30 km:", "Dist. to nearest emerg. (km):",
                     "Emergency facilities:", "Baseline tier:"],
            style=(
                "background-color:white;color:#333;font-family:Arial;"
                "font-size:12px;padding:8px;"
            ),
        ),
        show=True,
    ).add_to(m)

    colormap.add_to(m)

    # Emergency facility markers
    fg_fac = folium.FeatureGroup(name="Emergency facilities", show=True)
    for _, row in ipress_emerg.iterrows():
        cat = row.get("categoria", "?")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="#1c3b5a",
            weight=1,
            fill=True,
            fill_color="#2c7bb6",
            fill_opacity=0.85,
            tooltip=(
                f"<b>{row.get('nombre_establecimiento', 'N/A')}</b><br>"
                f"Category: {cat}  |  "
                f"Dept: {row.get('departamento', 'N/A')}"
            ),
        ).add_to(fg_fac)
    fg_fac.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
    Fullscreen().add_to(m)

    m.save(str(filepath))
    print(f"  {filepath.name}")


# ── Interactive map 3 — CCPP access-deficit heatmap (Q2) ─────────────────────

def make_ccpp_heatmap(
    merged: gpd.GeoDataFrame,
    ccpp: gpd.GeoDataFrame,
    filepath: Path,
) -> None:
    """
    Q2 — Heatmap of populated places more than 30 km from the nearest
    emergency facility, weighted by distance (farther = more intense).

    55 % of Peru's 136,587 populated places (74,577 points) are beyond the
    30-km baseline threshold.  This interactive heatmap lets the user zoom
    into any region to see the density of isolated settlements: bright/hot
    areas have many communities far from emergency care; cool/dark areas
    either have nearby facilities or sparse settlement.

    The heatmap uses CCPP points > 30 km as input, weighted by
    min(distance / 200, 1.0), so very remote places (> 200 km) saturate at
    full intensity.  This is more analytically informative than an unweighted
    heatmap of all CCPP points, which would conflate well-served dense coastal
    areas with isolated Amazonian communities.

    District tier polygons are shown as a light background layer for
    geographic orientation.
    """
    gdf_back = _simplify(merged)
    gdf_back = _prep_folium(gdf_back, num_cols=["composite_score"],
                             str_cols=["tier", "distrito", "departamento"])

    keep_back = ["ubigeo", "distrito", "departamento", "tier",
                 "composite_score", "geometry"]
    gdf_back = gdf_back[[c for c in keep_back if c in gdf_back.columns]]

    # CCPP > 30 km, weighted by distance
    far = ccpp[ccpp["dist_km_nearest_emergency"] > 30].copy()
    far["weight"] = (far["dist_km_nearest_emergency"] / 200).clip(upper=1.0)

    heat_data = list(zip(
        far.geometry.y,
        far.geometry.x,
        far["weight"].round(3),
    ))

    m = folium.Map(location=PERU_CENTER, zoom_start=6,
                   tiles="CartoDB dark_matter", prefer_canvas=True)

    # District background (very transparent tier colors)
    folium.GeoJson(
        gdf_back.to_json(),
        name="District tiers (background)",
        style_function=lambda f: {
            "fillColor": TIER_COLORS.get(
                (f["properties"].get("tier") or "").strip(), "#333333"
            ),
            "color": "#555555",
            "weight": 0.4,
            "fillOpacity": 0.08,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["distrito", "departamento", "tier"],
            aliases=["District:", "Dept:", "Tier:"],
        ),
        show=True,
    ).add_to(m)

    # CCPP access-deficit heatmap
    HeatMap(
        heat_data,
        name=f"CCPP access deficit (n={len(far):,} places > 30 km)",
        min_opacity=0.25,
        max_zoom=14,
        radius=12,
        blur=10,
        gradient={0.2: "#ffffb2", 0.5: "#fd8d3c", 0.75: "#f03b20", 1.0: "#bd0026"},
    ).add_to(m)

    # Annotation via static HTML
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;top:20px;right:20px;width:250px;'
        'background:rgba(0,0,0,0.75);color:white;border-radius:6px;'
        'padding:10px;font-family:Arial;font-size:12px;z-index:9999;">'
        '<b>CCPP access-deficit heatmap</b><br>'
        f'Showing {len(far):,} populated places<br>'
        'that are &gt;30 km from the nearest<br>'
        'emergency-level facility.<br>'
        'Brightness = distance severity.<br><br>'
        '<span style="color:#bd0026;">■</span> Very distant (&gt;60 km)<br>'
        '<span style="color:#fd8d3c;">■</span> Distant (30–60 km)<br>'
        '</div>'
    ))

    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)

    m.save(str(filepath))
    print(f"  {filepath.name}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_mapping_pipeline() -> None:
    """
    Load all data and generate all 7 geospatial outputs (4 static + 3 interactive).
    """
    FIGURES.mkdir(parents=True, exist_ok=True)

    merged, ipress_emerg, ccpp = load_mapping_data()

    print("\nGenerating static maps (GeoPandas + matplotlib)...")
    plot_bivariate_choropleth(merged,
                              filename="geo_bivariate_choropleth.png")
    plot_facility_desert(merged, ipress_emerg,
                         filename="geo_facility_desert.png")
    plot_ccpp_access_map(merged, ccpp,
                         filename="geo_ccpp_access_map.png")
    plot_tier_divergence(merged,
                         filename="geo_tier_divergence.png")

    print("\nGenerating interactive maps (Folium)...")
    make_tier_explorer(merged, ipress_emerg,
                       filepath=FIGURES / "map_tiers_interactive.html")
    make_access_explorer(merged, ipress_emerg,
                         filepath=FIGURES / "map_access_explorer.html")
    make_ccpp_heatmap(merged, ccpp,
                      filepath=FIGURES / "map_ccpp_distances.html")

    print("\nMapping pipeline complete.")
    sizes = [(p.name, p.stat().st_size / 1024)
             for p in FIGURES.iterdir()
             if p.suffix in (".png", ".html")
             and ("geo_" in p.name or "map_" in p.name)]
    for name, kb in sorted(sizes):
        print(f"  {name:45s}: {kb:>7.0f} KB")


if __name__ == "__main__":
    run_mapping_pipeline()
