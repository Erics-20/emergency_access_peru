"""
Cleaning and preprocessing for all raw datasets.

Key decisions documented here are also summarised in
output/tables/cleaning_summary.md.
"""
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.data_loader import load_all
from src.utils import PERU_LAT, PERU_LON, save_df, save_gdf

# ── Explicit column rename maps (raw label → snake_case name) ────────────────

_IPRESS_RENAME = {
    "Institución":                                                    "institucion",
    "Código Único":                                                   "codigo_unico",
    "Nombre del establecimiento":                                     "nombre_establecimiento",
    "Clasificación":                                                  "clasificacion",
    "Tipo":                                                           "tipo",
    "Departamento":                                                   "departamento",
    "Provincia":                                                      "provincia",
    "Distrito":                                                       "distrito",
    "UBIGEO":                                                         "ubigeo",
    "Dirección":                                                      "direccion",
    "Código DISA":                                                    "codigo_disa",
    "Código Red":                                                     "codigo_red",
    "Código Microrred":                                               "codigo_microrred",
    "DISA":                                                           "disa",
    "Red":                                                            "red",
    "Microrred":                                                      "microrred",
    "Código UE":                                                      "codigo_ue",
    "Unidad Ejecutora":                                               "unidad_ejecutora",
    "Categoria":                                                      "categoria",
    "Teléfono":                                                       "telefono",
    "Tipo Doc.Categorización":                                        "tipo_doc_categorizacion",
    "Nro.Doc.Categorización":                                         "nro_doc_categorizacion",
    "Horario":                                                        "horario",
    "Inicio de Actividad":                                            "inicio_actividad",
    "Director Médico y/o Responsable de la Atención de Salud":       "director",
    "Estado":                                                         "estado",
    "Situación":                                                      "situacion",
    "Condición":                                                      "condicion",
    "Inspección":                                                     "inspeccion",
    # Coordinate columns: raw labels are swapped — NORTE holds longitude
    # values and ESTE holds latitude values (confirmed from facility locations).
    "NORTE":                                                          "longitud",
    "ESTE":                                                           "latitud",
    "COTA":                                                           "altitud_msnm",
    "CAMAS":                                                          "camas",
}

_CONSULTA_RENAME = {
    "ANHO":                 "anho",
    "MES":                  "mes",
    "UBIGEO":               "ubigeo",
    "DEPARTAMENTO":         "departamento",
    "PROVINCIA":            "provincia",
    "DISTRITO":             "distrito",
    "SECTOR":               "sector",
    "CATEGORIA":            "categoria",
    "CO_IPRESS":            "codigo_ipress",
    "RAZON_SOC":            "razon_social",
    "SEXO":                 "sexo",
    "EDAD":                 "grupo_etario",
    "NRO_TOTAL_ATENCIONES": "total_atenciones",
    "NRO_TOTAL_ATENDIDOS":  "total_atendidos",
}

_CCPP_RENAME = {
    "OBJECTID":   "objectid",
    "NOM_POBLAD": "nombre_poblado",
    "FUENTE":     "fuente",
    "CÓDIGO":     "codigo",
    "CAT_POBLAD": "cat_poblado",
    "DIST":       "distrito",
    "PROV":       "provincia",
    "DEP":        "departamento",
    "CÓD_INT":    "cod_interno",
    "CATEGORIA":  "categoria",
    "X":          "lon_attr",   # attribute column (used for reference)
    "Y":          "lat_attr",   # attribute column (used for reference)
    "N_BUSQDA":   "nombre_busqueda",
}

# SEXO codes in raw ConsultaC1 → standardised label
_SEXO_MAP = {"1": "M", "01": "M", "2": "F", "02": "F", "NE_0001": None, "NE_0002": None}


# ── Individual dataset cleaners ──────────────────────────────────────────────

def clean_ipress(df: pd.DataFrame) -> tuple:
    """
    Returns
    -------
    ipress_clean : pd.DataFrame
        All facilities with standardised columns (20k+ rows).
    ipress_geo : gpd.GeoDataFrame
        Subset with valid WGS-84 coordinates as point layer.
    """
    print("\n[IPRESS] Cleaning ...")
    n_raw = len(df)

    df = df.rename(columns=_IPRESS_RENAME)

    # 1. Drop exact duplicate rows
    df = df.drop_duplicates()
    n_exact_dup = n_raw - len(df)

    # 2. Drop duplicate facility codes — keep the first occurrence
    n_before = len(df)
    df = df.drop_duplicates(subset="codigo_unico", keep="first")
    n_uid_dup = n_before - len(df)

    # 3. Zero-pad UBIGEO to 6 characters (e.g. 60101 → "060101")
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    # 4. Strip leading/trailing whitespace from string columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(
        lambda c: c.str.strip() if hasattr(c, "str") else c
    )

    # 5. Drop constant-value columns that add no analytical value
    #    'situacion' is always blank in the raw file
    df = df.drop(columns=["situacion"], errors="ignore")

    print(f"  Raw rows              : {n_raw:>8,}")
    print(f"  Exact duplicates      : {n_exact_dup:>8,}")
    print(f"  Duplicate codigo_unico: {n_uid_dup:>8,}")
    print(f"  Clean rows (all fac.) : {len(df):>8,}")

    # ── Build point GeoDataFrame for facilities with valid coordinates --------
    lat_lo, lat_hi = PERU_LAT
    lon_lo, lon_hi = PERU_LON

    mask_valid = (
        df["latitud"].notna()
        & df["longitud"].notna()
        & df["latitud"].between(lat_lo, lat_hi)
        & df["longitud"].between(lon_lo, lon_hi)
        # Reject exact-zero pairs used as missing-value proxies
        & ~((df["latitud"] == 0) & (df["longitud"] == 0))
    )
    geo_df = df[mask_valid].copy()
    n_invalid = len(df) - len(geo_df)

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(geo_df["longitud"], geo_df["latitud"])
    ]
    ipress_geo = gpd.GeoDataFrame(geo_df, geometry=geometry, crs="EPSG:4326")

    print(f"  Facilities w/ valid coords : {len(ipress_geo):>6,}")
    print(f"  Null / out-of-bounds coords: {n_invalid:>6,}")

    return df, ipress_geo


def clean_consulta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns
    -------
    consulta_clean : pd.DataFrame
        Deduplicated outpatient visit records with corrected dtypes.
    """
    print("\n[ConsultaC1] Cleaning ...")
    n_raw = len(df)

    df = df.rename(columns=_CONSULTA_RENAME)

    # 1. Drop exact duplicates (identical rows reported more than once)
    df = df.drop_duplicates()
    n_dup = n_raw - len(df)

    # 2. Drop fully "not-specified" rows: MINSA's HIS system encodes rows
    #    where both the patient demographics AND the visit counts are unknown
    #    as 'NE_0001'/'NE_0002'. These rows carry no analytical value.
    _ne_codes = {"NE_0001", "NE_0002"}
    mask_ne = (
        df["sexo"].isin(_ne_codes)
        & df["total_atenciones"].isin(_ne_codes)
    )
    n_ne = mask_ne.sum()
    df = df[~mask_ne].copy()

    # 3. Zero-pad UBIGEO to 6 characters
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    # 4. Standardise sexo: collapse '01'/'1'→'M', '02'/'2'→'F', NE*→NaN
    df["sexo"] = df["sexo"].astype(str).map(_SEXO_MAP)

    # 5. Convert visit/patient count columns to integer
    for col in ("total_atenciones", "total_atendidos"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # 6. Strip whitespace from string columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(
        lambda c: c.str.strip() if hasattr(c, "str") else c
    )

    null_counts = df["total_atenciones"].isna().sum()

    print(f"  Raw rows                     : {n_raw:>9,}")
    print(f"  Exact dups removed           : {n_dup:>9,}")
    print(f"  Fully-NE rows dropped        : {n_ne:>9,}")
    print(f"  Clean rows                   : {len(df):>9,}")
    print(f"  Rows w/ unknown sexo (NE)    : {df['sexo'].isna().sum():>9,}")
    print(f"  Rows w/ unreported count (NE): {null_counts:>9,}")

    return df


def clean_ccpp(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Returns
    -------
    ccpp_clean : gpd.GeoDataFrame
        Populated-place points within Peru bounds; ubigeo present only
        where the source CÓDIGO field is available (~53% of raw records).
    """
    print("\n[CCPP] Cleaning ...")
    n_raw = len(gdf)

    gdf = gdf.rename(columns=_CCPP_RENAME)

    # 1. Derive 6-digit UBIGEO from the first six chars of the locality code.
    #    ~72K records have no CÓDIGO (encoded as null in the raw shapefile);
    #    they retain valid geometry and admin names but ubigeo is left null.
    gdf["ubigeo"] = gdf["codigo"].str[:6].where(gdf["codigo"].notna())
    n_no_ubigeo = gdf["ubigeo"].isna().sum()

    # 2. Filter to points whose geometry falls inside Peru's bounding box.
    #    Geometry coordinates (geometry.x = lon, geometry.y = lat) are used
    #    instead of the lon_attr / lat_attr columns, which contain outliers.
    lat_lo, lat_hi = PERU_LAT
    lon_lo, lon_hi = PERU_LON
    mask_valid = (
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.x.between(lon_lo, lon_hi)
        & gdf.geometry.y.between(lat_lo, lat_hi)
    )
    n_invalid_geom = (~mask_valid).sum()
    gdf = gdf[mask_valid].copy()

    # 3. Drop redundant attribute coordinate columns (geometry is authoritative)
    gdf = gdf.drop(columns=["lon_attr", "lat_attr"], errors="ignore")

    print(f"  Raw rows                    : {n_raw:>8,}")
    print(f"  No CÓDIGO → ubigeo null     : {n_no_ubigeo:>8,}")
    print(f"  Out-of-bounds geometry      : {n_invalid_geom:>8,}")
    print(f"  Clean rows (all places)     : {len(gdf):>8,}")
    print(f"  w/ ubigeo                   : {gdf['ubigeo'].notna().sum():>8,}")
    print(f"  Unique district ubigeos     : {gdf['ubigeo'].nunique():>8,}")

    return gdf


def clean_distritos(
    gdf: gpd.GeoDataFrame,
    ccpp_clean: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign CRS and district identifiers to the polygon layer via spatial
    join with the CCPP populated-place points.

    Returns
    -------
    distritos_clean : gpd.GeoDataFrame
        District polygons with ubigeo, distrito, provincia, departamento.
    """
    print("\n[DISTRITOS] Cleaning ...")

    # 1. Assign CRS — confirmed EPSG:4326 from total bounds matching Peru
    gdf = gdf.set_crs("EPSG:4326")

    # 2. Spatial join: find which polygon each CCPP point falls inside.
    #    Use only the subset that has a known ubigeo (CÓDIGO-derived).
    print("  Running spatial join (CCPP points → district polygons) ...")
    ccpp_pts = ccpp_clean.loc[
        ccpp_clean["ubigeo"].notna(),
        ["ubigeo", "distrito", "provincia", "departamento", "geometry"],
    ].copy()

    joined = gpd.sjoin(ccpp_pts, gdf, how="inner", predicate="within")
    # joined["index_right"] references the row index in gdf

    def _mode(series: pd.Series):
        m = series.dropna().mode()
        return m.iloc[0] if len(m) else pd.NA

    # 3. Aggregate: most common ubigeo and admin names per polygon
    agg = joined.groupby("index_right").agg(
        ubigeo=("ubigeo", _mode),
        distrito=("distrito", _mode),
        provincia=("provincia", _mode),
        departamento=("departamento", _mode),
    )

    # 4. Map results back to the polygon layer (pandas aligns on index)
    gdf["ubigeo"]      = agg["ubigeo"]
    gdf["distrito"]    = agg["distrito"]
    gdf["provincia"]   = agg["provincia"]
    gdf["departamento"] = agg["departamento"]

    covered = gdf["ubigeo"].notna().sum()
    print(f"  Total district polygons        : {len(gdf):>6,}")
    print(f"  Polygons with UBIGEO assigned  : {covered:>6,}")
    print(f"  Polygons without match (remote): {len(gdf) - covered:>6,}")

    return gdf


# ── Main pipeline entry point ────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
    Full data-ingestion and cleaning pipeline.

    Loads all raw datasets, applies cleaning rules, saves cleaned outputs
    to data/processed/, and returns a dict of cleaned objects.
    """
    # Ensure output directories exist
    for d in ("data/processed", "output/tables", "output/figures", "video"):
        Path(d).mkdir(parents=True, exist_ok=True)

    raw = load_all()

    ipress_clean, ipress_geo = clean_ipress(raw["ipress"])
    consulta_clean            = clean_consulta(raw["consulta"])
    ccpp_clean                = clean_ccpp(raw["ccpp"])
    distritos_clean           = clean_distritos(raw["distritos"], ccpp_clean)

    print("\nSaving cleaned outputs to data/processed/ ...")
    save_df(ipress_clean,   "ipress_clean")
    save_gdf(ipress_geo,    "ipress_geo")
    save_df(consulta_clean, "consulta_clean")
    save_gdf(ccpp_clean,    "ccpp_clean")
    save_gdf(distritos_clean, "distritos_clean")

    print("\nPipeline complete.")
    return {
        "ipress":    ipress_clean,
        "ipress_geo": ipress_geo,
        "consulta":  consulta_clean,
        "ccpp":      ccpp_clean,
        "distritos": distritos_clean,
    }


if __name__ == "__main__":
    run_pipeline()
