# Boston Housing Anomaly Detector

> A machine learning system that flags over- and under-priced real estate listings across Greater Boston, with explainable predictions powered by SHAP.

**Live demo:** _coming soon_ → will be hosted on Hugging Face Spaces

---

## What this does

Given an active real estate listing in Greater Boston, this system:
1. Predicts the property's fair market value using a gradient-boosted model trained on ~60K historical sales from MA assessor records
2. Compares the asking price to the prediction and flags significant deviations
3. Explains *why* the model thinks a listing is over- or under-priced, using SHAP feature attributions

## Why I built this

Portfolio project for summer 2026 data science internship applications. Goals:
- Demonstrate end-to-end ML pipeline ownership: data acquisition → feature engineering → modeling → deployment
- Show real-world judgment: messy data, target leakage avoidance, time-aware validation
- Produce something tangible and demoable, not just a notebook

## Tech stack

- **Data:** MA municipal assessor records + scraped Realtor.com active listings
- **Storage:** DuckDB (analytical SQL), Parquet for processed data
- **ML:** LightGBM with SHAP explanations, scikit-learn baselines
- **Features:** Spatial (H3 hexagons, distance to amenities), temporal, text embeddings on listing descriptions
- **UI:** Streamlit + Folium for the map view
- **Deployment:** Hugging Face Spaces

## Project structure

```
boston-housing-anomaly/
├── app/                    # Streamlit application
│   ├── streamlit_app.py    # Main entry point
│   └── pages/              # Multi-page app: map, analyzer, model card
├── data/
│   ├── raw/                # Untouched scraped HTML, CSV downloads
│   ├── external/           # Reference data (T-stop locations, school scores)
│   └── processed/          # DuckDB database, Parquet files
├── notebooks/              # EDA, experimentation
├── src/
│   ├── scraping/           # Realtor.com scraper, rate limiting, parsers
│   ├── data/               # Loading, cleaning, joining assessor + listings
│   ├── features/           # Feature engineering pipelines
│   ├── models/             # Training, evaluation, SHAP wrappers
│   └── utils/              # Logging, config, shared helpers
├── scripts/                # CLI entry points (run_scraper.py, train.py)
├── tests/                  # Unit tests for parsers and feature logic
├── requirements.txt
└── README.md
```

## Getting started

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/boston-housing-anomaly.git
cd boston-housing-anomaly
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Download MA assessor data (see scripts/download_assessor.py)
python scripts/download_assessor.py

# 3. Run scraper for active listings (will take ~3 hours, respect rate limits)
python scripts/run_scraper.py --cities boston cambridge somerville brookline newton

# 4. Build features and train model
python scripts/build_features.py
python scripts/train.py

# 5. Launch the app
streamlit run app/streamlit_app.py
```

## Roadmap

- [ ] Day 1-4: Data foundation (assessor download, scraping, geocoding, joining)
- [ ] Day 5-9: Modeling (baseline → LightGBM → SHAP → evaluation)
- [ ] Day 10-14: UI and deployment

## Honest limitations

_To be filled in as I discover them._

## Author

Yashaswi — MS Data Science, Northeastern University
