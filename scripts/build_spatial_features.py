"""
Day 4: Add spatial features to the geocoded dataset.

Two features:
1. DIST_TO_T_KM       — distance (km) to the nearest MBTA rapid transit station
   (subway: Red, Blue, Orange, Green, Silver lines)
2. NBR_MEDIAN_PSF     — median price-per-sqft of recent sales within 0.5 miles
   (neighborhood price context, computed using historical sales only)

Run from project root:
    python scripts/build_spatial_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

INPUT = Path("data/processed/geocoded_assessor.parquet")
MBTA_STOPS = Path("data/external/stops.txt")
OUTPUT = Path("data/processed/features.parquet")

# Earth radius in km (for haversine and arc-length conversions)
EARTH_RADIUS_KM = 6371.0
NEIGHBORHOOD_RADIUS_KM = 0.8  # ~0.5 miles

# MBTA rapid transit route types in GTFS:
#   0 = Light rail / streetcar (Green, Mattapan)
#   1 = Subway / metro (Red, Blue, Orange)
#   2 = Commuter rail
#   3 = Bus
#   4 = Ferry
# We want rapid transit only — types 0 and 1.
RAPID_TRANSIT_TYPES = {0, 1}


def load_t_stations() -> pd.DataFrame:
    """Load MBTA stops, keep only rapid transit parent stations."""
    print(f"Loading MBTA stops from {MBTA_STOPS}...")
    stops = pd.read_csv(MBTA_STOPS)
    print(f"  Total stops in GTFS: {len(stops):,}")

    # GTFS structure: 'stops' includes platforms, entrances, parent stations, etc.
    # We want parent stations only (location_type=1) when available, otherwise
    # regular stops (location_type=0 or NaN).
    if "vehicle_type" in stops.columns:
        # MBTA includes a vehicle_type column on station rows
        rapid = stops[stops["vehicle_type"].isin(RAPID_TRANSIT_TYPES)].copy()
    else:
        # Fall back: filter via stop_id naming conventions for subway lines
        rapid = stops[stops["stop_id"].astype(str).str.startswith("place-")].copy()

    # Keep parent stations if location_type column exists
    if "location_type" in rapid.columns:
        parents = rapid[rapid["location_type"] == 1]
        if len(parents) > 0:
            rapid = parents

    rapid = rapid[["stop_id", "stop_name", "stop_lat", "stop_lon"]].drop_duplicates()
    rapid = rapid.dropna(subset=["stop_lat", "stop_lon"])

    print(f"  Rapid transit stations kept: {len(rapid):,}")
    print(f"  Sample stations: {', '.join(rapid['stop_name'].head(5).tolist())}")
    return rapid


def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/long to 3D unit-sphere coordinates for KDTree distance.

    Why: a KDTree on raw lat/long would give wrong distances near the poles
    or for any large area, and Boston is far enough north that 1 deg lng
    is shorter than 1 deg lat. Converting to 3D cartesian on the unit
    sphere lets us use Euclidean chord distance, which is monotonic with
    great-circle distance — so 'nearest neighbor' rankings are correct.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    """Convert chord distance (on unit sphere) to surface distance in km."""
    # Chord = 2 * sin(angle/2), so angle = 2 * arcsin(chord/2)
    chord = np.clip(chord, 0, 2)
    central_angle = 2 * np.arcsin(chord / 2)
    return central_angle * EARTH_RADIUS_KM


def add_distance_to_t(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """For each property, compute distance in km to the nearest T station."""
    print("\nComputing distance to nearest T station...")

    station_xyz = latlon_to_xyz(
        stations["stop_lat"].to_numpy(),
        stations["stop_lon"].to_numpy(),
    )
    property_xyz = latlon_to_xyz(
        df["LATITUDE"].to_numpy(),
        df["LONGITUDE"].to_numpy(),
    )

    tree = cKDTree(station_xyz)
    chord_distances, station_indices = tree.query(property_xyz, k=1)

    df = df.copy()
    df["DIST_TO_T_KM"] = chord_to_km(chord_distances)
    df["NEAREST_T_STATION"] = stations["stop_name"].to_numpy()[station_indices]

    print(f"  Distance stats (km): min={df['DIST_TO_T_KM'].min():.2f}, "
          f"median={df['DIST_TO_T_KM'].median():.2f}, "
          f"max={df['DIST_TO_T_KM'].max():.2f}")
    return df


def add_neighborhood_features(df: pd.DataFrame) -> pd.DataFrame:
    """For each property, compute median price-per-sqft of OTHER nearby sales.

    Critical detail: for each property X, we use only sales that happened
    BEFORE X's sale date. Otherwise we'd leak the future into our features
    (knowing prices that hadn't happened yet when X sold).
    """
    print(f"\nComputing neighborhood median PSF within {NEIGHBORHOOD_RADIUS_KM} km...")

    df = df.copy()
    df["PRICE_PER_SQFT"] = df["LS_PRICE"] / df["RES_AREA"]

    # Sort by sale year so for each row we can look at "all earlier sales"
    df_sorted = df.sort_values("LS_YEAR").reset_index(drop=True)

    # Convert chord radius to KDTree query radius on the unit sphere
    angle = NEIGHBORHOOD_RADIUS_KM / EARTH_RADIUS_KM
    chord_radius = 2 * np.sin(angle / 2)

    xyz = latlon_to_xyz(
        df_sorted["LATITUDE"].to_numpy(),
        df_sorted["LONGITUDE"].to_numpy(),
    )
    tree = cKDTree(xyz)

    # For each row, find all neighbors (including itself) within the radius
    print("  Querying KDTree (this is the slow step — about 1-2 minutes)...")
    neighbor_lists = tree.query_ball_tree(tree, r=chord_radius)

    psf = df_sorted["PRICE_PER_SQFT"].to_numpy()
    years = df_sorted["LS_YEAR"].to_numpy()

    nbr_median = np.full(len(df_sorted), np.nan)
    nbr_count = np.zeros(len(df_sorted), dtype=int)

    for i, neighbors in enumerate(neighbor_lists):
        # Use only neighbors that sold strictly before this row, exclude self
        my_year = years[i]
        valid = [j for j in neighbors if j != i and years[j] < my_year]
        if len(valid) >= 5:  # need at least 5 prior sales to be meaningful
            nbr_median[i] = np.median(psf[valid])
            nbr_count[i] = len(valid)

    df_sorted["NBR_MEDIAN_PSF"] = nbr_median
    df_sorted["NBR_SAMPLE_SIZE"] = nbr_count

    coverage = df_sorted["NBR_MEDIAN_PSF"].notna().mean()
    print(f"  Coverage (rows with neighborhood feature): {coverage:.1%}")
    print(f"  Mean neighbors used: {df_sorted['NBR_SAMPLE_SIZE'].mean():.1f}")

    return df_sorted


def main() -> None:
    print(f"Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    print(f"  Rows: {len(df):,}")

    stations = load_t_stations()
    df = add_distance_to_t(df, stations)
    df = add_neighborhood_features(df)

    print(f"\nFinal feature dataset: {len(df):,} rows × {len(df.columns)} columns")
    print("\nNew columns added:")
    for col in ["DIST_TO_T_KM", "NEAREST_T_STATION", "PRICE_PER_SQFT",
                "NBR_MEDIAN_PSF", "NBR_SAMPLE_SIZE"]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null:,} non-null ({non_null/len(df):.1%})")

    print(f"\nSaving to {OUTPUT}...")
    df.to_parquet(OUTPUT, index=False)
    print("Done.")


if __name__ == "__main__":
    main()