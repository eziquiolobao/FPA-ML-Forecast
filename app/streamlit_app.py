"""FP&A Rolling Forecast & Variance Radar - Streamlit entry point."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from common import sidebar  # noqa: E402

st.set_page_config(page_title="FP&A Rolling Forecast", page_icon="📈", layout="wide")

pages = [
    st.Page("views/summary.py", title="Executive summary", icon="🏠", url_path="summary", default=True),
    st.Page("views/forecast.py", title="Rolling forecast", icon="📈", url_path="forecast"),
    st.Page("views/variance.py", title="Variance radar", icon="🚦", url_path="variance"),
    st.Page("views/models.py", title="Model performance", icon="🏆", url_path="models"),
    st.Page("views/method.py", title="How it works", icon="🧭", url_path="method"),
]
nav = st.navigation(pages)
sidebar()
nav.run()
