"""Download and clean daily market data."""

from pathlib import Path
from typing import Optional

import pandas as pd


def download_ticker_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str],
    data_dir: str = "data",
) -> pd.DataFrame:
    """Download one ticker's OHLCV history and save it as a CSV file."""
    import yfinance as yf

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{ticker} is missing columns: {missing}")
    cleaned = data[required].copy().sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")].ffill().dropna()
    cleaned.index.name = "Date"
    cleaned.to_csv(destination / f"{ticker.replace('.', '_')}.csv")
    return cleaned


def load_ticker_data(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """Load a previously downloaded ticker CSV."""
    path = Path(data_dir) / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run the downloader first.")
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()


def download_all(
    tickers: list[str], start_date: str, end_date: Optional[str], data_dir: str
) -> dict[str, pd.DataFrame]:
    """Download data for all configured tickers."""
    return {
        ticker: download_ticker_data(ticker, start_date, end_date, data_dir)
        for ticker in tickers
    }


if __name__ == "__main__":
    download_all(["AAPL", "MSFT", "RELIANCE.NS"], "2018-01-01", None, "data")
