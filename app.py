"""Streamlit dashboard for trained return models and saved evaluation plots."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_ticker_data
from src.dataset import chronological_split
from src.features import FEATURE_COLUMNS, add_features
from src.models import GRUModel, LSTMModel, TransformerModel


@st.cache_data
def load_history(ticker: str) -> pd.DataFrame:
    return load_ticker_data(ticker)


@st.cache_resource
def load_model(ticker: str, model_name: str, config: dict, feature_count: int):
    results_dir = Path(config["results_dir"]) / ticker.replace(".", "_")
    checkpoint = torch.load(results_dir / f"{model_name.lower()}.pt", map_location="cpu")
    training = config["training"]
    constructors = {
        "LSTM": LSTMModel(feature_count, training["hidden_size"], training["num_layers"], training["dropout"]),
        "GRU": GRUModel(feature_count, training["hidden_size"], training["num_layers"], training["dropout"]),
        "Transformer": TransformerModel(
            feature_count, training["hidden_size"], training["transformer_heads"],
            training["transformer_layers"], training["dropout"], config["window_size"],
        ),
    }
    model = constructors[model_name]
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["target_mean"], checkpoint["target_std"]


st.set_page_config(page_title="StockPredictor", layout="wide")
st.title("StockPredictor")
st.caption("Educational return-forecasting demo — not financial advice.")
config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
ticker = st.selectbox("Ticker", config["tickers"])
results_dir = Path(config["results_dir"]) / ticker.replace(".", "_")

try:
    history = load_history(ticker)
    featured = add_features(history)
    train_frame, _, _ = chronological_split(featured, tuple(config["split_ratios"]))
    feature_scaler = StandardScaler().fit(train_frame[FEATURE_COLUMNS])
    model_name = st.selectbox("Trained model", ["LSTM", "GRU", "Transformer"])
    model, target_mean, target_std = load_model(
        ticker, model_name, config, len(FEATURE_COLUMNS)
    )
    window = config["window_size"]
    values = feature_scaler.transform(featured[FEATURE_COLUMNS].tail(window))
    with torch.no_grad():
        predicted_scaled_return, _ = model(torch.tensor(values[None], dtype=torch.float32))
    predicted_return = float(predicted_scaled_return.item() * target_std + target_mean)
    latest_close = float(featured["Close"].iloc[-1])
    predicted_close = latest_close * np.exp(predicted_return)
    st.metric("Latest close", f"{latest_close:.2f}")
    st.metric("Predicted next-day close", f"{predicted_close:.2f}")
    st.metric("Predicted next-day log return", f"{predicted_return:.6f}")
    st.line_chart(history["Close"])
    prediction_plot = results_dir / "predictions.png"
    if prediction_plot.exists():
        st.subheader("Held-out prediction versus actual")
        st.image(str(prediction_plot), caption="Saved test-period predictions")
    else:
        st.info("Run the training workflow to create the held-out prediction plot.")
    metrics_path = results_dir / "test_metrics.csv"
    if metrics_path.exists():
        st.subheader("Saved test metrics")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
    else:
        st.info("No saved metrics found. Run python -m src.train first.")
except FileNotFoundError:
    st.error("Data or trained results are missing. Run the downloader and training workflow first.")
