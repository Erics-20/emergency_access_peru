"""
Spatial joins, CRS management, distance calculations, and GeoDataFrame assembly.

CRS strategy
------------
EPSG:4326  (WGS-84, geographic)  – all data stored and displayed in degrees.
EPSG:32718 (WGS84 / UTM Zone 18S, projected) – used **only** during distance
and area computations (units = metres), then results are stored back in 4326.

Peru spans UTM zones 17S–19S; Zone 18S provides the best single-zone fit for
the heavily-populated Pacific coast and central Andes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from src.data_loader import (
    load_ipress_clean,
    load_ipress_geo,
    load_ccpp_clean,
    load_distritos_clean,
)
from src.utils import save_df, save_gdf

# ── CRS constants ─────────────────────────────────────────────────────────────

GEO_CRS    = "EPSG:4326"   # WGS-84 geographic — storage CRS for all outputs
METRIC_CRS = "EPSG:32718"  # WGS-84 / UTM Zone 18S — used for distances & areas

# ── Facility level sets ───────────────────────────────────────────────────────

# Hospital-level categories with formal emergency departments (MINSA resolution)
EMERGENCY_LEVELS = frozenset({"II-1", "II-2", "II-E", "III-1", "III-2", "III-E"})
PRIMARY_LEVELS   = frozenset({"I-1", "I-2", "I-3", "I-4"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedup_sjoin(result: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop extra rows produced when sjoin finds equidistant features."""
    return result[~result.index.duplicated(keep="first")]


def dissolve_districts(distritos: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Some districts appear as multiple polygon rows in the source shapefile
    (disconnected territories, river islands, exclaves).  Union them by
    ubigeo so every district has exactly one (Multi)Polygon row.

    Polygons without a ubigeo are kept as individual rows with a synthetic
    key so they are not silently dropped.

    Returns
    -------
    GeoDataFrame — one row per district (EPSG:4326)
    """
    print("[geospatial] Dissolving multi-part district polygons by ubigeo...")
    n_before = len(distritos)

    # Rows with known ubigeo → dissolve (union geometry, keep first admin names)
    with_id = distritos[distritos["ubigeo"].notna()].copy()
    name_cols = ["distrito", "provincia", "departamento"]
    first_names = with_id.groupby("ubigeo")[name_cols].first()

    dissolved = (
        with_id[["ubigeo", "geometry"]]
        .dissolve(by="ubigeo", as_index=False)
        .merge(first_names, on="ubigeo")
    )

    # Rows without ubigeo stay as-is
    without_id = distritos[distritos["ubigeo"].isna()].copy()

    result = pd.concat([dissolved, without_id], ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=GEO_CRS)

    n_removed = n_before - len(result)
    print(f"  Rows before dissolve : {n_before:,}")
    print(f"  Rows after dissolve  : {len(result):,}  (merged {n_removed} duplicate-polygon rows)")
    return result


# ── Spatial operations ────────────────────────────────────────────────────────

def assign_facilities_to_districts(
    ipress_geo: gpd.GeoDataFrame,
    distritos: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatial join: for each geolocated IPRESS facility, record which district
    polygon its coordinates fall inside.

    Parameters
    ----------
    ipress_geo : GeoDataFrame (EPSG:4326) — facilities with valid coordinates
    distritos  : GeoDataFrame (EPSG:4326) — district polygons

    Returns
    -------
    GeoDataFrame with added columns:
        ubigeo_spatial  – ubigeo of the district polygon the point falls in
        admin_match     – True when administrative ubigeo == spatial ubigeo
    """
    print("[geospatial] Assigning facilities → districts (spatial join)...")

    dist_cols = ["ubigeo", "distrito", "provincia", "departamento", "geometry"]
    dist_sub  = distritos[dist_cols].rename(
        columns={"ubigeo": "ubigeo_spatial",
                 "distrito": "distrito_spatial",
                 "provincia": "provincia_spatial",
                 "departamento": "departamento_spatial"}
    )

    joined = gpd.sjoin(ipress_geo, dist_sub, how="left", predicate="within")
    joined = _dedup_sjoin(joined)
    joined = joined.drop(columns=["index_right"], errors="ignore")

    joined["admin_match"] = joined["ubigeo"] == joined["ubigeo_spatial"]

    n_matched   = joined["ubigeo_spatial"].notna().sum()
    n_mismatch  = (~joined["admin_match"] & joined["ubigeo_spatial"].notna()).sum()
    print(f"  Facilities matched to a polygon : {n_matched:,}")
    print(f"  Admin ≠ spatial ubigeo (coord error / boundary): {n_mismatch:,}")
    print(f"  Outside all polygons             : {joined['ubigeo_spatial'].isna().sum():,}")

    return joined


def enrich_ccpp_districts(
    ccpp: gpd.GeoDataFrame,
    distritos: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Fill `ubigeo` for CCPP points that lack a locality CÓDIGO by running a
    spatial join against the district polygon layer.

    Points already carrying a CÓDIGO-derived ubigeo keep that value; only the
    ~72 K points without it are updated.

    Parameters
    ----------
    ccpp     : GeoDataFrame (EPSG:4326) — all populated places
    distritos: GeoDataFrame (EPSG:4326) — district polygons with ubigeo

    Returns
    -------
    GeoDataFrame with `ubigeo` filled as completely as possible.
    """
    print("[geospatial] Enriching CCPP with district ubigeo (spatial join)...")

    # Points that need ubigeo from the polygon
    needs_ubigeo = ccpp["ubigeo"].isna()
    ccpp_missing = ccpp[needs_ubigeo].copy()
    ccpp_ok      = ccpp[~needs_ubigeo].copy()

    dist_sub = distritos[["ubigeo", "geometry"]].rename(
        columns={"ubigeo": "ubigeo_spatial"}
    )
    joined = gpd.sjoin(ccpp_missing, dist_sub, how="left", predicate="within")
    joined = _dedup_sjoin(joined)
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined["ubigeo"] = joined["ubigeo_spatial"]
    joined = joined.drop(columns=["ubigeo_spatial"], errors="ignore")

    result = pd.concat([ccpp_ok, joined], ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=GEO_CRS)

    filled = joined["ubigeo"].notna().sum()
    print(f"  Previously missing ubigeo : {needs_ubigeo.sum():>7,}")
    print(f"  Filled via spatial join   : {filled:>7,}")
    print(f"  Still null (outside poly) : {needs_ubigeo.sum() - filled:>7,}")
    print(f"  Total CCPP rows           : {len(result):>7,}")

    return result


def compute_district_geometry(distritos: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add area (km²) and centroid coordinates (WGS-84) to the district layer.

    Area and centroids are computed in METRIC_CRS (EPSG:32718, metres) then
    stored in degrees so all outputs stay in GEO_CRS.

    Returns
    -------
    GeoDataFrame with new columns: area_km2, centroid_lon, centroid_lat
    """
    print("[geospatial] Computing district areas and centroids (EPSG:32718)...")

    dist_m = distritos.to_crs(METRIC_CRS)
    distritos = distritos.copy()
    distritos["area_km2"]     = (dist_m.geometry.area / 1e6).round(2)
    centroids_4326            = dist_m.geometry.centroid.to_crs(GEO_CRS)
    distritos["centroid_lon"] = centroids_4326.x.round(6)
    distritos["centroid_lat"] = centroids_4326.y.round(6)

    print(f"  Area range (km²): {distritos['area_km2'].min():.1f} – {distritos['area_km2'].max():,.0f}")
    print(f"  Centroid lon range: {distritos['centroid_lon'].min():.2f} – {distritos['centroid_lon'].max():.2f}")
    print(f"  Centroid lat range: {distritos['centroid_lat'].min():.2f} – {distritos['centroid_lat'].max():.2f}")

    return distritos


def compute_nearest_distances(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    label: str,
) -> pd.Series:
    """
    For each row in `origins`, find the nearest feature in `destinations`
    and return the distance in kilometres.

    Both GDFs are reprojected to METRIC_CRS (EPSG:32718) for the calculation.

    Parameters
    ----------
    origins      : GeoDataFrame whose index matches the calling GDF
    destinations : GeoDataFrame of target features
    label        : short label used in console output only

    Returns
    -------
    pd.Series (same index as origins) — distance in km, NaN where no match
    """
    if len(destinations) == 0:
        print(f"  [WARNING] No {label} destinations — returning NaN distances")
        return pd.Series(np.nan, index=origins.index)

    orig_m = origins.to_crs(METRIC_CRS)
    dest_m = destinations[["geometry"]].to_crs(METRIC_CRS)

    joined = gpd.sjoin_nearest(orig_m, dest_m, how="left", distance_col="_dist_m")
    joined = _dedup_sjoin(joined)

    dist_km = (joined["_dist_m"] / 1000).round(3)
    dist_km.index = origins.index  # re-align to original index

    print(f"  Nearest {label:22s} → median {dist_km.median():.1f} km, max {dist_km.max():.1f} km")
    return dist_km


def aggregate_facilities_to_districts(
    ipress_all: pd.DataFrame,
    ipress_geo: gpd.GeoDataFrame,
    distritos: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Compute per-district facility counts using **administrative** ubigeo
    (covers all 20 793 facilities, not just the 7 941 with GPS coordinates).

    Spatial ubigeo from `ipress_geo` is used to cross-check counts and detect
    districts whose geolocated facilities fall in a different polygon.

    Returns
    -------
    DataFrame indexed by ubigeo with columns:
        n_facilities, n_active, n_emergency, n_primary, n_uncategorised
    """
    print("[geospatial] Aggregating facility counts by district (admin ubigeo)...")

    df = ipress_all.copy()
    df["is_active"]       = df["condicion"] == "EN FUNCIONAMIENTO"
    df["is_emergency"]    = df["categoria"].isin(EMERGENCY_LEVELS)
    df["is_primary"]      = df["categoria"].isin(PRIMARY_LEVELS)
    df["is_uncategorised"] = ~df["is_emergency"] & ~df["is_primary"]

    agg = df.groupby("ubigeo").agg(
        n_facilities    =("codigo_unico", "count"),
        n_active        =("is_active",    "sum"),
        n_emergency     =("is_emergency", "sum"),
        n_primary       =("is_primary",   "sum"),
        n_uncategorised =("is_uncategorised", "sum"),
    ).astype(int)

    print(f"  Districts with ≥1 facility (admin): {len(agg):,}")
    print(f"  Total facilities aggregated        : {agg['n_facilities'].sum():,}")
    print(f"  Districts with ≥1 emergency fac.  : {(agg['n_emergency'] > 0).sum():,}")
    return agg


def aggregate_ccpp_to_districts(
    ccpp: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Count the number of CCPP populated places per district.

    Returns
    -------
    DataFrame indexed by ubigeo with column: n_ccpp
    """
    print("[geospatial] Aggregating CCPP counts by district...")
    valid = ccpp[ccpp["ubigeo"].notna()]
    agg = valid.groupby("ubigeo").size().rename("n_ccpp").to_frame()
    print(f"  Districts with ≥1 populated place: {len(agg):,}")
    return agg


# ── District-level GeoDataFrame builder ─────────────────────────────────────

def build_district_geodataframe(
    distritos: gpd.GeoDataFrame,
    ipress_geo: gpd.GeoDataFrame,
    ipress_all: pd.DataFrame,
    ccpp: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assemble the master district-level GeoDataFrame by combining:
      • district geometry + admin names (distritos_clean)
      • area and centroid (computed in UTM Zone 18S → stored in WGS-84)
      • facility counts by level (from full IPRESS registry)
      • nearest-facility distances (from georeferenced IPRESS subset)
      • populated-place counts (from CCPP)

    All spatial predicates and distance operations use METRIC_CRS (EPSG:32718)
    for accuracy; the final output is stored in GEO_CRS (EPSG:4326).

    Returns
    -------
    GeoDataFrame — one row per district polygon (EPSG:4326)
    """
    print("\n[geospatial] Building master district GeoDataFrame...")

    # 0. Dissolve multi-part polygons so every district = one row
    distritos = dissolve_districts(distritos)

    # 1. Geometry: areas + centroids
    gdf = compute_district_geometry(distritos)

    # 2. Facility counts (administrative ubigeo → all facilities)
    fac_counts = aggregate_facilities_to_districts(ipress_all, ipress_geo, gdf)
    gdf = gdf.merge(fac_counts, on="ubigeo", how="left")
    for col in ("n_facilities", "n_active", "n_emergency", "n_primary", "n_uncategorised"):
        gdf[col] = gdf[col].fillna(0).astype(int)

    # 3. CCPP counts
    ccpp_counts = aggregate_ccpp_to_districts(ccpp)
    gdf = gdf.merge(ccpp_counts, on="ubigeo", how="left")
    gdf["n_ccpp"] = gdf["n_ccpp"].fillna(0).astype(int)

    # 4. Distance to nearest ANY facility (uses geolocated subset only)
    print("[geospatial] Computing nearest-facility distances from district centroids...")

    centroid_gdf = gpd.GeoDataFrame(
        gdf[["ubigeo"]],
        geometry=gpd.points_from_xy(gdf["centroid_lon"], gdf["centroid_lat"]),
        crs=GEO_CRS,
    )

    gdf["dist_km_nearest_any"] = compute_nearest_distances(
        centroid_gdf, ipress_geo, "any facility"
    ).values

    # 5. Distance to nearest EMERGENCY facility
    ipress_emerg = ipress_geo[ipress_geo["categoria"].isin(EMERGENCY_LEVELS)].copy()
    gdf["dist_km_nearest_emergency"] = compute_nearest_distances(
        centroid_gdf, ipress_emerg, "emergency facility"
    ).values

    # 6. Facility density (per 100 km²) — useful normaliser
    gdf["facility_density_per100km2"] = (
        gdf["n_facilities"] / gdf["area_km2"] * 100
    ).round(4)

    # Ensure output CRS is correct
    gdf = gdf.set_crs(GEO_CRS, allow_override=True)

    print(f"\n  District GDF shape : {gdf.shape}")
    print(f"  CRS                : {gdf.crs}")
    print(f"  Districts with nearest-any distance    : {gdf['dist_km_nearest_any'].notna().sum():,}")
    print(f"  Districts with nearest-emerg distance  : {gdf['dist_km_nearest_emergency'].notna().sum():,}")

    return gdf


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_geospatial_pipeline() -> dict:
    """
    Full geospatial pipeline.

    Loads all cleaned datasets, executes spatial joins and distance logic,
    saves enriched outputs to data/processed/, and returns a dict of results.
    """
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("output/tables").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEOSPATIAL PIPELINE")
    print("=" * 60)

    # Load cleaned inputs
    print("\nLoading cleaned datasets...")
    ipress_all  = load_ipress_clean()
    ipress_geo  = load_ipress_geo()
    ccpp        = load_ccpp_clean()
    distritos   = load_distritos_clean()

    for name, ds in [
        ("ipress_all",  ipress_all),
        ("ipress_geo",  ipress_geo),
        ("ccpp",        ccpp),
        ("distritos",   distritos),
    ]:
        print(f"  {name:14s}: {ds.shape[0]:>7,} rows")

    # ── Step 1: Facility → district spatial join ──────────────────────────────
    print()
    ipress_districts = assign_facilities_to_districts(ipress_geo, distritos)
    save_gdf(ipress_districts, "ipress_districts")

    # ── Step 2: CCPP district enrichment (fill missing ubigeo via polygon) ────
    print()
    ccpp_districts = enrich_ccpp_districts(ccpp, distritos)
    save_gdf(ccpp_districts, "ccpp_districts")

    # ── Step 3: Master district GeoDataFrame ──────────────────────────────────
    print()
    distritos_geo = build_district_geodataframe(
        distritos, ipress_geo, ipress_all, ccpp_districts
    )
    save_gdf(distritos_geo, "distritos_geo")

    # Also export as CSV for downstream metric computation
    dist_csv = distritos_geo.drop(columns="geometry").copy()
    save_df(dist_csv, "distritos_geo")

    # ── Step 4: CCPP nearest-facility distance (point-level granularity) ──────
    print("\n[geospatial] Computing nearest-facility distances for CCPP points...")
    ccpp_valid = ccpp_districts[ccpp_districts.geometry.notna()].copy()

    ccpp_valid["dist_km_nearest_any"] = compute_nearest_distances(
        ccpp_valid, ipress_geo, "any facility"
    ).values

    ipress_emerg = ipress_geo[ipress_geo["categoria"].isin(EMERGENCY_LEVELS)].copy()
    ccpp_valid["dist_km_nearest_emergency"] = compute_nearest_distances(
        ccpp_valid, ipress_emerg, "emergency facility"
    ).values

    save_gdf(ccpp_valid, "ccpp_with_distances")

    print("\n" + "=" * 60)
    print("Geospatial pipeline complete.")
    print("=" * 60)

    return {
        "ipress_districts": ipress_districts,
        "ccpp_districts":   ccpp_districts,
        "ccpp_distances":   ccpp_valid,
        "distritos_geo":    distritos_geo,
    }


if __name__ == "__main__":
    run_geospatial_pipeline()
