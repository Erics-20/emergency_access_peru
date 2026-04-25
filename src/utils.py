"""Shared utility functions: column naming, file I/O, geographic constants."""
import unicodedata
import re
from pathlib import Path

import pandas as pd
import geopandas as gpd

PROCESSED_DIR = Path("data/processed")

# Bounding box for mainland Peru (WGS-84 decimal degrees)
PERU_LAT = (-18.5, 0.05)   # (min_lat, max_lat)
PERU_LON = (-82.0, -68.0)  # (min_lon, max_lon)


def remove_accents(text: str) -> str:
    """Strip diacritics from a unicode string (á→a, é→e, ñ→n, etc.)."""
    nfd = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def to_snake_case(name: str) -> str:
    """Convert an arbitrary column label to lowercase snake_case."""
    s = remove_accents(name).strip()
    s = re.sub(r"[\s.\-/\\]+", "_", s)   # whitespace/dots/slashes → _
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)  # strip remaining special chars
    s = re.sub(r"_+", "_", s)            # collapse consecutive underscores
    return s.strip("_").lower()


def save_df(df: pd.DataFrame, name: str) -> None:
    """Persist a cleaned DataFrame to data/processed/<name>.csv."""
    path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved {path}  ({len(df):,} rows × {df.shape[1]} cols)")


def save_gdf(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Persist a cleaned GeoDataFrame to data/processed/<name>.gpkg."""
    path = PROCESSED_DIR / f"{name}.gpkg"
    gdf.to_file(path, driver="GPKG")
    print(f"  Saved {path}  ({len(gdf):,} rows)")
