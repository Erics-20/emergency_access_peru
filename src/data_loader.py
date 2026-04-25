"""Raw dataset loaders — one function per source file."""
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd

RAW = Path("data/raw")


def load_ipress() -> pd.DataFrame:
    """
    IPRESS: national registry of health facilities (SUSALUD).
    Source: data/raw/IPRESS.csv  |  Encoding: latin-1  |  Sep: comma
    Rows: ~20,819   Cols: 33
    """
    return pd.read_csv(RAW / "IPRESS.csv", encoding="latin-1")


def load_consulta() -> pd.DataFrame:
    """
    ConsultaC1: aggregated outpatient visits by facility, district,
    sex, and age group (MINSA HIS, year 2025).
    Source: data/raw/ConsultaC1_2025_v20.csv  |  Encoding: latin-1  |  Sep: semicolon
    Rows: ~342,753   Cols: 14
    """
    return pd.read_csv(
        RAW / "ConsultaC1_2025_v20.csv",
        encoding="latin-1",
        sep=";",
    )


def load_distritos() -> gpd.GeoDataFrame:
    """
    DISTRITOS: district-level polygon shapefile for all of Peru.
    Source: data/raw/DISTRITOS.shp
    NOTE: .dbf and .shx companion files are absent; geometry is intact
    and CRS is reassigned as EPSG:4326 during cleaning.
    Rows: 1,873 polygons
    """
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    return gpd.read_file(RAW / "DISTRITOS.shp")


def load_ccpp() -> gpd.GeoDataFrame:
    """
    CCPP_IGN100K: populated-place point layer (IGN / INEI, ~136K points).
    Source: data/raw/CCPP_0/CCPP_IGN100K.shp  |  CRS: EPSG:4326
    Rows: ~136,587   Cols: 13 + geometry
    """
    return gpd.read_file(RAW / "CCPP_0/CCPP_IGN100K.shp")


def load_ipress_clean() -> pd.DataFrame:
    """Load the cleaned IPRESS CSV, preserving leading-zero ubigeo codes."""
    return pd.read_csv(
        Path("data/processed/ipress_clean.csv"),
        dtype={"ubigeo": str},
        encoding="utf-8",
    )


def load_consulta_clean() -> pd.DataFrame:
    """Load the cleaned ConsultaC1 CSV, preserving leading-zero ubigeo codes."""
    return pd.read_csv(
        Path("data/processed/consulta_clean.csv"),
        dtype={"ubigeo": str},
        encoding="utf-8",
    )


def load_ipress_geo() -> gpd.GeoDataFrame:
    """Load georeferenced IPRESS facilities (valid-coordinate subset)."""
    return gpd.read_file(Path("data/processed/ipress_geo.gpkg"))


def load_ccpp_clean() -> gpd.GeoDataFrame:
    """Load the cleaned CCPP populated-place layer."""
    return gpd.read_file(Path("data/processed/ccpp_clean.gpkg"))


def load_distritos_clean() -> gpd.GeoDataFrame:
    """Load the cleaned district polygon layer (CRS EPSG:4326, with ubigeo)."""
    return gpd.read_file(Path("data/processed/distritos_clean.gpkg"))


def load_all() -> dict:
    """Load every raw dataset and return them in a named dict."""
    print("Loading raw datasets...")
    datasets = {
        "ipress":    load_ipress(),
        "consulta":  load_consulta(),
        "distritos": load_distritos(),
        "ccpp":      load_ccpp(),
    }
    for name, ds in datasets.items():
        print(f"  {name:12s}: {ds.shape[0]:>7,} rows × {ds.shape[1]} cols")
    return datasets
