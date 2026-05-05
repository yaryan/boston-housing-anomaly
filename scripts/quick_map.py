"""
Day 3 visual payoff: map every geocoded property as a colored point.

Color = log(sale price). Hover = address + price.

Saves an interactive HTML map you open in your browser.

Run from project root:
    python scripts/quick_map.py
"""

from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

INPUT = Path("data/processed/geocoded_assessor.parquet")
OUTPUT = Path("notebooks/quick_map.html")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Plotting all 60K points crashes browsers — sample down for the quick view.
# When we build the real UI on Day 10 we'll handle the full dataset properly.
SAMPLE_SIZE = 5000


def price_to_color(price: float, vmin: float, vmax: float) -> str:
    """Map a log-price to a hex color from green (cheap) -> yellow -> red (expensive)."""
    log_price = np.log10(price)
    t = np.clip((log_price - vmin) / (vmax - vmin), 0, 1)
    if t < 0.5:
        # green to yellow
        r = int(2 * t * 255)
        g = 200
        b = 50
    else:
        # yellow to red
        r = 230
        g = int((1 - 2 * (t - 0.5)) * 200)
        b = 50
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    print(f"Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} rows")

    print(f"\nSampling {SAMPLE_SIZE:,} for a fast map...")
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)

    # Color scale based on the FULL dataset's log-price range
    vmin = np.log10(df["LS_PRICE"]).quantile(0.02)
    vmax = np.log10(df["LS_PRICE"]).quantile(0.98)

    # Center map on mean of all points
    center = [sample["LATITUDE"].mean(), sample["LONGITUDE"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    print("Adding pins...")
    for _, row in sample.iterrows():
        color = price_to_color(row["LS_PRICE"], vmin, vmax)
        popup = (
            f"<b>{row.get('SITE_ADDR', '?')}</b><br>"
            f"{row['CITY_NAME']}, {row.get('ZIP', '?')}<br>"
            f"Sold {int(row['LS_YEAR'])}: ${int(row['LS_PRICE']):,}<br>"
            f"{int(row['RES_AREA']):,} sqft, built {int(row['YEAR_BUILT'])}<br>"
            f"Type: {row.get('USE_DESC', '?')}"
        )
        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=0,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(m)

    print(f"Saving to {OUTPUT}...")
    m.save(str(OUTPUT))
    print(f"\nDone. Open it with:")
    print(f"  open {OUTPUT}")


if __name__ == "__main__":
    main()