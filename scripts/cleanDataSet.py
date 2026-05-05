"""
Day 2: Load all 4 cities into one clean table.

Steps:
1. Read each city's assessor .dbf file (auto-detecting the year suffix)
2. Add CITY_NAME and FY_YEAR columns
3. Stack them all together
4. Harmonize USE_CODE across fiscal years (FY23 uses 0101, FY25/26 uses 1010)
5. Filter to residential only
6. Filter out obviously bad rows (zero price, zero area, very old sales)
7. Save the clean result as Parquet in data/processed/

Run from project root:
    python scripts/build_clean_dataset.py
"""

from pathlib import Path

import pandas as pd
from dbfread import DBF

# Paths to each city's extracted folder
CITY_DIRS = {
    "Boston":     Path("data/raw/assessor/L3_SHP_M035_Boston"),
    "Cambridge":  Path("data/raw/assessor/L3_SHP_M049_Cambridge"),
    "Somerville": Path("data/raw/assessor/L3_SHP_M274_Somerville"),
    "Brookline":  Path("data/raw/assessor/L3_SHP_M046_Brookline"),
}

# Each city has a TOWN_ID embedded in filenames: M035, M049, M274, M046
TOWN_IDS = {"Boston": "035", "Cambridge": "049", "Somerville": "274", "Brookline": "046"}

OUTPUT_PATH = Path("data/processed/clean_assessor.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_city(city: str, folder: Path) -> pd.DataFrame:
    """Load one city's assessor table and use-code lookup; return joined dataframe.

    Uses glob to handle the fact that each city's data is from a different
    fiscal year (e.g., Boston is CY22_FY23, Cambridge is CY25_FY26).
    """
    town_id = TOWN_IDS[city]

    print(f"\nLoading {city}...")

    assess_matches = list(folder.glob(f"M{town_id}Assess_*.dbf"))
    uc_lut_matches = list(folder.glob(f"M{town_id}UC_LUT_*.dbf"))

    if not assess_matches:
        raise FileNotFoundError(f"No assessor file found in {folder}")
    if not uc_lut_matches:
        raise FileNotFoundError(f"No use-code lookup found in {folder}")

    assess_path = assess_matches[0]
    uc_lut_path = uc_lut_matches[0]

    print(f"  Using: {assess_path.name}")

    # Main assessor table
    assess = pd.DataFrame(iter(DBF(assess_path, load=True, encoding="latin-1")))
    print(f"  Assessor rows: {len(assess):,}")

    # Use-code lookup (decodes USE_CODE -> human-readable description)
    uc_lut = pd.DataFrame(iter(DBF(uc_lut_path, load=True, encoding="latin-1")))
    uc_lut = uc_lut[["USE_CODE", "USE_DESC"]]

    # Join — every row gets a USE_DESC
    df = assess.merge(uc_lut, on="USE_CODE", how="left")
    df["CITY_NAME"] = city

    # Extract the fiscal year from the filename (e.g., "FY23" -> 2023)
    fy_str = assess_path.stem.split("FY")[-1][:2]
    df["FY_YEAR"] = 2000 + int(fy_str)

    return df


def harmonize_use_code(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize USE_CODE across MassGIS fiscal-year vintages.

    The format changed between FY23 and FY25:
      - FY23 (Boston):    "0101" (single fam), "0102" (condo), "0104" (2-fam) ...
      - FY25/26 (Cambridge, Somerville): "1010", "1020", "1040" ...

    Both encode the same property types — the new format just shifted the
    digits. We harmonize to the FY23 form (101, 102, 104, ...) so a single
    filter works across all cities.
    """
    df = df.copy()
    df["USE_CODE_NUM"] = pd.to_numeric(df["USE_CODE"], errors="coerce")

    # If the code is a 4-digit number ending in 0 (like 1010, 1020),
    # it's the new format — divide by 10 to get the canonical 3-digit code.
    new_format = (df["USE_CODE_NUM"] >= 1000) & (df["USE_CODE_NUM"] % 10 == 0)
    df.loc[new_format, "USE_CODE_NUM"] = df.loc[new_format, "USE_CODE_NUM"] // 10

    return df


def filter_residential(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only residential properties (use codes 101-199 in canonical form)."""
    mask = (df["USE_CODE_NUM"] >= 101) & (df["USE_CODE_NUM"] <= 199)
    return df[mask].copy()


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with obviously placeholder or unusable values."""
    n0 = len(df)
    df = df.copy()

    for col in ["LS_PRICE", "TOTAL_VAL", "BLDG_VAL", "LAND_VAL", "RES_AREA",
                "BLD_AREA", "YEAR_BUILT", "UNITS", "NUM_ROOMS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sale date is YYYYMMDD as int — convert to year for filtering
    df["LS_YEAR"] = pd.to_numeric(df["LS_DATE"].astype(str).str[:4], errors="coerce")

    df = df[df["LS_PRICE"] >= 50_000]      # arms-length sale, not gift/transfer
    df = df[df["LS_PRICE"] <= 10_000_000]  # filter outliers
    df = df[df["RES_AREA"] > 0]            # has real building area
    df = df[df["YEAR_BUILT"].between(1700, 2025)]  # real year, not "1900" sentinel
    df = df[df["LS_YEAR"] >= 2015]         # recent sales only

    print(f"  After quality filter: {len(df):,} (dropped {n0 - len(df):,})")
    return df


def main() -> None:
    frames = [load_city(city, folder) for city, folder in CITY_DIRS.items()]
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined raw rows: {len(combined):,}")

    print("\nHarmonizing use codes across fiscal years...")
    combined = harmonize_use_code(combined)

    print("\nFiltering to residential properties...")
    residential = filter_residential(combined)
    print(f"Residential rows: {len(residential):,}")

    print("\nApplying quality filters...")
    clean = filter_quality(residential)

    print(f"\nFinal clean dataset: {len(clean):,} rows × {len(clean.columns)} columns")
    print("\nRows per city:")
    print(clean["CITY_NAME"].value_counts().to_string())

    print("\nFiscal year per city:")
    print(clean.groupby("CITY_NAME")["FY_YEAR"].first().to_string())

    print("\nProperty type distribution (top 10):")
    print(clean["USE_CODE_NUM"].astype(int).value_counts().head(10).to_string())

    print(f"\nSaving to {OUTPUT_PATH}...")
    clean.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()