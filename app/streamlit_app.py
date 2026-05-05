"""
Boston Housing Anomaly Detector — Streamlit app.

Three pages:
1. Map view: see all listings, color-coded by predicted vs. asking
2. Listing analyzer: deep dive on a single listing with SHAP explanation
3. Model card: methodology, performance, limitations
"""

import streamlit as st

st.set_page_config(
    page_title="Boston Housing Anomaly Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("🏠 Boston Housing Anomaly Detector")
    st.markdown(
        "Spotting over- and under-priced listings across Greater Boston "
        "using ~60K historical sales and explainable ML."
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Map view", "Listing analyzer", "Model card"],
    )

    if page == "Map view":
        render_map_view()
    elif page == "Listing analyzer":
        render_analyzer()
    else:
        render_model_card()


def render_map_view() -> None:
    st.header("Greater Boston: Listings vs. Predicted Fair Value")
    st.info("Coming soon: Folium map with red/green pins for over/under priced listings.")
    # Day 10 deliverable


def render_analyzer() -> None:
    st.header("Listing Analyzer")
    st.info("Coming soon: paste a listing URL or pick from the database to see SHAP breakdown.")
    # Day 10 deliverable


def render_model_card() -> None:
    st.header("Model Card")
    st.markdown(
        """
        ### Methodology
        _To be filled in._

        ### Performance
        _To be filled in._

        ### Known limitations
        _To be filled in._
        """
    )


if __name__ == "__main__":
    main()
