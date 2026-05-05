"""
Day 1: Initial exploration of the MassGIS assessor data.

Goal: Look at one city (Boston) and answer:
1. What columns are in the assessor table?
2. How many records?
3. What's the data look like for a few rows?
4. What does the parcel shapefile look like?
5. What's in the use code lookup table?

This is intentionally simple. Just SEE the data before doing anything with it.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from dbfread import DBF

# Path to the Boston extraction
BOSTON_DIR = Path("data/raw/assessor/L3_SHP_M035_Boston")

ASSESS_DBF = BOSTON_DIR / "M035Assess_CY22_FY23.dbf"
TAXPAR_SHP = BOSTON_DIR / "M035TaxPar_CY22_FY23.shp"
UC_LUT_DBF = BOSTON_DIR / "M035UC_LUT_CY22_FY23.dbf"


def explore_assessor_table() -> pd.DataFrame:
    """Load and summarize the main assessor table."""
    print("=" * 70)
    print("ASSESSOR TABLE: M035Assess_CY22_FY23.dbf")
    print("=" * 70)

    # DBF files are an old dBASE format. dbfread handles them cleanly.
    table = DBF(ASSESS_DBF, load=True, encoding="latin-1")
    df = pd.DataFrame(iter(table))

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    print("\nFirst 3 rows:")
    print(df.head(3).T)  # transpose so columns become rows — easier to read

    print("\nMissingness per column (top 10 most missing):")
    missing = df.isna().mean().sort_values(ascending=False).head(10)
    print(missing.to_string())

    return df


def explore_parcel_shapefile() -> gpd.GeoDataFrame:
    """Load and summarize the tax parcel polygons."""
    print("\n" + "=" * 70)
    print("TAX PARCEL SHAPEFILE: M035TaxPar_CY22_FY23.shp")
    print("=" * 70)

    gdf = gpd.read_file(TAXPAR_SHP)

    print(f"\nShape: {gdf.shape[0]:,} rows × {gdf.shape[1]} columns")
    print(f"\nCoordinate Reference System: {gdf.crs}")
    print(f"\nColumns: {list(gdf.columns)}")
    print("\nFirst 3 rows (geometry truncated):")
    print(gdf.head(3).drop(columns=["geometry"]).T)

    return gdf


def explore_use_code_lookup() -> pd.DataFrame:
    """Load the use code lookup so we know what 101, 102, etc. mean."""
    print("\n" + "=" * 70)
    print("USE CODE LOOKUP: M035UC_LUT_CY22_FY23.dbf")
    print("=" * 70)

    table = DBF(UC_LUT_DBF, load=True, encoding="latin-1")
    df = pd.DataFrame(iter(table))

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print("\nFirst 20 rows:")
    print(df.head(20).to_string(index=False))

    return df


def main() -> None:
    assess = explore_assessor_table()
    parcels = explore_parcel_shapefile()
    use_codes = explore_use_code_lookup()

    print("\n" + "=" * 70)
    print("DAY 1 SUMMARY")
    print("=" * 70)
    print(f"  Boston assessor records:  {len(assess):,}")
    print(f"  Boston parcel polygons:   {len(parcels):,}")
    print(f"  Use code categories:      {len(use_codes):,}")
    print("\nNext step: figure out which columns matter for the model.")


if __name__ == "__main__":
    main()