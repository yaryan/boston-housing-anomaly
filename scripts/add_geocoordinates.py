"""
Day 3: Geocode the clean assessor data via parcel shapefiles.

For each city:
1. Load the tax parcel shapefile (polygons with LOC_ID)
2. Reproject from MA State Plane (EPSG:26986) to WGS84 lat/long (EPSG:4326)
3. Compute the centroid of each parcel polygon

Then:

4. Join to the clean assessor data on LOC_ID
5. Save with new LATITUDE and LONGITUDE columns

Run from project root:
    python scripts/add_geocoordinates.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

CITY_DIRS = {
    "Boston":     Path("data/raw/assessor/L3_SHP_M035_Boston"),
    "Cambridge":  Path("data/raw/assessor/L3_SHP_M049_Cambridge"),
    "Somerville": Path("data/raw/assessor/L3_SHP_M274_Somerville"),
    "Brookline":  Path("data/raw/assessor/L3_SHP_M046_Brookline"),
}

TOWN_IDS = {"Boston": "035", "Cambridge": "049", "Somerville": "274", "Brookline": "046"}

INPUT_PATH = Path("data/processed/clean_assessor.parquet")
OUTPUT_PATH = Path("data/processed/geocoded_assessor.parquet")


def load_city_parcels(city: str, folder: Path) -> gpd.GeoDataFrame:
    """Load parcel polygons for one city, reproject to WGS84, compute centroids."""
    town_id = TOWN_IDS[city]

    # Auto-detect filename (different vintage per city)
    matches = list(folder.glob(f"M{town_id}TaxPar_*.shp"))
    if not matches:
        raise FileNotFoundError(f"No tax parcel shapefile in {folder}")
    shp_path = matches[0]

    print(f"\nLoading {city} parcels from {shp_path.name}...")
    gdf = gpd.read_file(shp_path)
    print(f"  Polygons: {len(gdf):,}")
    print(f"  Source CRS: {gdf.crs}")

    # Reproject to WGS84 (standard lat/long).
    # IMPORTANT: compute centroids in the *projected* CRS (meters), not lat/long,
    # because centroid math on lat/long is geometrically wrong.
    centroids_proj = gdf.geometry.centroid
    centroids_wgs84 = centroids_proj.to_crs("EPSG:4326")

    out = pd.DataFrame({
        "LOC_ID": gdf["LOC_ID"].values,
        "LONGITUDE": centroids_wgs84.x.values,
        "LATITUDE": centroids_wgs84.y.values,
    })

    # Some parcels appear multiple times in shapefiles (e.g., split parcels).
    # Keep just the first centroid per LOC_ID.
    out = out.drop_duplicates(subset="LOC_ID", keep="first")

    print(f"  Unique LOC_IDs: {len(out):,}")
    return out


def main() -> None:
    print(f"Loading clean assessor data from {INPUT_PATH}...")
    assessor = pd.read_parquet(INPUT_PATH)
    print(f"  Rows: {len(assessor):,}")

    print("\nLoading and reprojecting parcel polygons for all cities...")
    parcel_frames = [
        load_city_parcels(city, folder) for city, folder in CITY_DIRS.items()
    ]
    parcels = pd.concat(parcel_frames, ignore_index=True)
    print(f"\nTotal unique parcels across cities: {len(parcels):,}")

    print("\nJoining assessor data to parcel centroids on LOC_ID...")
    geocoded = assessor.merge(parcels, on="LOC_ID", how="left")

    matched = geocoded["LATITUDE"].notna().sum()
    unmatched = geocoded["LATITUDE"].isna().sum()
    print(f"  Matched (have lat/long): {matched:,} ({matched / len(geocoded):.1%})")
    print(f"  Unmatched: {unmatched:,}")

    if unmatched > 0:
        print("\nUnmatched rows by city:")
        print(geocoded[geocoded["LATITUDE"].isna()]["CITY_NAME"].value_counts().to_string())

    # Drop the rows we couldn't geocode — they'd break spatial features later.
    geocoded_clean = geocoded.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    print(f"\nFinal geocoded dataset: {len(geocoded_clean):,} rows")

    print("\nSanity check — coordinate ranges (should be Greater Boston):")
    print(f"  Latitude:  {geocoded_clean['LATITUDE'].min():.4f} to {geocoded_clean['LATITUDE'].max():.4f}")
    print(f"  Longitude: {geocoded_clean['LONGITUDE'].min():.4f} to {geocoded_clean['LONGITUDE'].max():.4f}")
    print("  (Expected: lat ~42.2-42.5, lng ~-71.2 to -70.9)")

    print(f"\nSaving to {OUTPUT_PATH}...")
    geocoded_clean.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()