"""Streamlit demo for StockPredictor."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from src.data_loader import load_ticker_data
from src.features import add_features


@st.cache_data
def load_history(ticker: str) -> pd.DataFrame:
    """Load cached local history for the selected ticker."""
    return load_ticker_data(ticker)


st.set_page_config(page_title="StockPredictor", layout="wide")
st.title("StockPredictor")
st.caption("Educational demo only — not financial advice.")
config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
ticker = st.selectbox("Ticker", config["tickers"])
try:
    history = load_history(ticker)
    featured = add_features(history)
    st.line_chart(history["Close"])
    latest = featured.iloc[-1]
    st.metric("Latest close", f"{latest['Close']:.2f}")
    st.info("Train a model with src/train.py and place its exported predictions/metrics in artifacts/ for live predictions.")
    metrics_path = Path("artifacts") / f"{ticker.replace('.', '_')}_metrics.csv"
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
    else:
        st.warning("No saved metrics found yet. Run the training workflow first.")
except FileNotFoundError:
    st.error("Data is not downloaded. Run the commands in README.md first.")

