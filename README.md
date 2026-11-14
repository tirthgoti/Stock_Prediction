# StockPredictor — Deep Learning for Stock Price Forecasting

StockPredictor is an educational end-to-end machine-learning project that forecasts the next trading day's closing price and direction using historical OHLCV data. It compares PyTorch deep-learning models with classical baselines and provides an exploratory notebook, saved plots, checkpoints, and a Streamlit dashboard.

> **Disclaimer:** This project is for learning and portfolio demonstration only. It is not financial advice, does not guarantee returns, and must not be used as an automated trading system without substantial additional validation.

## Project goals

- Download at least five years of daily market data.
- Engineer commonly used technical indicators.
- Predict both:
  - next-day closing price, as a regression task;
  - next-day up/down direction, as a secondary classification target.
- Compare LSTM, GRU, and Transformer encoder models with naive, ARIMA, and XGBoost baselines.
- Demonstrate time-series-specific practices: chronological splits, training-only scaling, sliding windows, early stopping, and walk-forward validation utilities.
- Produce reproducible artifacts that are easy to explain in an AI/ML Engineer interview.

## Models

### Deep-learning models

All neural models use PyTorch and have two output heads:

1. A regression head for the next closing price.
2. A direction head for the probability/logit that the next close is higher than today's close.

The available models are:

- **LSTM:** recurrent architecture with gated memory, useful for learning sequential dependencies.
- **GRU:** a smaller recurrent architecture with fewer gates and parameters.
- **Transformer encoder:** self-attention-based sequence encoder using the final time step as the forecasting representation.

Regularization and training controls include dropout, Adam, weight decay, gradient clipping, a learning-rate scheduler, early stopping, and best-checkpoint restoration.

### Baselines

- **Naive:** predicts tomorrow's close as today's close.
- **ARIMA:** classical autoregressive integrated moving-average model.
- **XGBoost:** gradient-boosted decision trees using engineered features from the previous day.

The naive baseline is especially important: financial prices are highly autocorrelated, so a complex model should not be considered useful unless it beats this simple reference on a fair holdout set.

## Data and features

Configured tickers:

- `AAPL`
- `MSFT`
- `RELIANCE.NS`

The downloader uses `yfinance` and saves cleaned CSV files under `data/`. The data directory is Git-ignored because downloaded market data should be regenerated rather than committed.

Engineered features:

- daily percentage return;
- daily log return;
- 5-day, 20-day, and 50-day moving averages;
- 14-period RSI;
- MACD and MACD signal;
- 20-day Bollinger middle, upper, and lower bands;
- rolling 20-day volatility;
- daily volume change.

Rows with incomplete rolling-window values are removed. Missing values are forward-filled where appropriate before feature engineering.

## Leakage prevention

This project uses chronological splitting:

- 70% training;
- 15% validation;
- 15% test.

The data is never randomly shuffled across time. Random splitting can put future market regimes into the training set and produce unrealistically optimistic results.

Feature scaling is fitted on training data only:

```text
scaler.fit(training_features)
scaler.transform(validation_features)
scaler.transform(test_features)
```

The close-price regression target is also standardized using training close-price statistics only, then converted back to real price units before calculating metrics. This improves neural-network optimization while avoiding future information leakage.

## Sliding-window formulation

With a configured window size of 60:

- input: 60 consecutive trading days of engineered features;
- regression target: closing price on day 61;
- direction target: whether day 61's close is greater than day 60's close.

This creates causal examples in which each prediction uses only information available before its target day.

## Repository structure

```text
StockPredictor/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
├── data/                       # downloaded CSV files; ignored by Git
├── artifacts/                  # checkpoints, metrics, and plots; ignored by Git
├── notebooks/
│   └── 01_eda.ipynb            # exploratory analysis and plots
└── src/
    ├── __init__.py
    ├── data_loader.py          # download and load OHLCV data
    ├── features.py             # technical feature engineering
    ├── dataset.py              # splits, scaling, and PyTorch Dataset
    ├── models.py               # LSTM, GRU, Transformer
    ├── baselines.py            # naive, ARIMA, XGBoost
    ├── train.py                # training and experiment entry point
    ├── evaluate.py             # metrics and prediction plots
    └── backtest.py             # long/flat backtesting utilities
```

## Installation

From PowerShell:

```powershell
cd C:\Users\tirth\.copilot\chats\f4361b05-0289-414b-a71f-8a4ae6a50234

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for the current user, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run from start to finish

### 1. Download data

```powershell
python -m src.data_loader
```

This creates files similar to:

```text
data/AAPL.csv
data/MSFT.csv
data/RELIANCE_NS.csv
```

### 2. Run exploratory analysis

```powershell
jupyter notebook notebooks\01_eda.ipynb
```

Open the notebook, select the project virtual environment as the kernel, and choose **Kernel > Restart Kernel and Run All Cells**.

The notebook adds the project root to Python's import path automatically, so `from src...` imports work even though Jupyter starts in the `notebooks` directory.

### 3. Train all configured models

```powershell
python -m src.train
```

This trains LSTM, GRU, and Transformer models for every configured ticker. It also evaluates naive, ARIMA, and XGBoost baselines.

Training uses:

- Adam optimizer;
- MSE regression loss;
- binary cross-entropy direction loss;
- dropout;
- weight decay;
- gradient clipping;
- `ReduceLROnPlateau` learning-rate scheduling;
- validation-based early stopping;
- best-model checkpointing;
- deterministic random seeds where supported.

### 4. Inspect generated artifacts

```powershell
Get-ChildItem artifacts
Import-Csv artifacts\AAPL_metrics.csv
Import-Csv artifacts\MSFT_metrics.csv
Import-Csv artifacts\RELIANCE_NS_metrics.csv
```

Generated artifacts include:

- `*_lstm.pt`, `*_gru.pt`, `*_transformer.pt`: best model checkpoints;
- `*_loss.png`: training and validation loss curves;
- `*_predictions.png`: actual-versus-predicted price plots;
- `*_metrics.csv`: comparison tables.

Open the artifact folder graphically:

```powershell
explorer artifacts
```

### 5. Launch the Streamlit app

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The app lets the user select a configured ticker, view its historical close chart, inspect the latest close, and view saved metrics.

## Evaluation metrics

- **RMSE:** square root of mean squared error; penalizes larger errors more strongly.
- **MAE:** average absolute price error.
- **MAPE:** average absolute percentage error.
- **Directional accuracy:** percentage of days on which the predicted movement direction matches the actual movement direction.

Metrics are calculated on the chronological test set, which is not used for fitting model parameters or selecting the best checkpoint.

## Results from the completed run

These are the metrics generated by the current local run. They are reported without modification; market data and results may change when the project is rerun.

### AAPL

| Model | RMSE | MAE | MAPE (%) | Directional accuracy (%) |
|---|---:|---:|---:|---:|
| LSTM | 76.69 | 71.63 | 23.75 | 47.39 |
| GRU | 60.59 | 55.49 | 18.29 | 47.39 |
| Transformer | 94.73 | 89.94 | 29.99 | 47.39 |
| Naive | 4.58 | 3.26 | 1.11 | 47.39 |
| ARIMA | 118.84 | 113.31 | 39.10 | 47.56 |
| XGBoost | 97.82 | 91.25 | 31.18 | 47.40 |

### MSFT

| Model | RMSE | MAE | MAPE (%) | Directional accuracy (%) |
|---|---:|---:|---:|---:|
| LSTM | 40.41 | 34.09 | 7.16 | 49.25 |
| GRU | 39.08 | 33.14 | 6.94 | 50.37 |
| Transformer | 93.83 | 79.24 | 16.38 | 51.49 |
| Naive | 9.22 | 6.46 | 1.44 | 50.37 |
| ARIMA | 67.52 | 59.32 | 12.29 | 48.17 |
| XGBoost | 70.06 | 58.25 | 11.76 | 48.01 |

### RELIANCE.NS

| Model | RMSE | MAE | MAPE (%) | Directional accuracy (%) |
|---|---:|---:|---:|---:|
| LSTM | 30.45 | 23.48 | 1.69 | 52.29 |
| GRU | 25.46 | 19.75 | 1.42 | 51.91 |
| Transformer | 106.35 | 82.48 | 5.79 | 54.58 |
| Naive | 19.85 | 14.03 | 1.04 | 51.53 |
| ARIMA | 131.12 | 111.26 | 8.45 | 51.24 |
| XGBoost | 65.89 | 58.03 | 4.27 | 48.91 |

### Interpretation

- The GRU was the best neural model by error metrics for all three tickers in this run.
- The GRU beat the naive baseline on error metrics for RELIANCE.NS, but not for AAPL or MSFT.
- Directional accuracy stayed close to 50%, which is expected for a difficult next-day direction task and should not be presented as reliable trading performance.
- A complex model is not automatically better than a naive persistence forecast. This is a valuable result to discuss in an interview.
- These results are one test-period experiment, not evidence of investable predictive power.

## Walk-forward validation

`src/train.py` includes `walk_forward_splits`, which creates expanding-window validation indices:

```text
fold 1: train [0 ... t1], validate [t1 ... t2]
fold 2: train [0 ... t2], validate [t2 ... t3]
fold 3: train [0 ... t3], validate [t3 ... t4]
```

This better reflects deployment because training data grows over time and validation always occurs after training data. It is intentionally provided as a reusable building block rather than silently multiplying the runtime of the default experiment.

## Backtesting

`src/backtest.py` provides a simple educational long/flat strategy:

- buy/hold the asset when the model predicts an up day;
- stay in cash when the model predicts a down day;
- charge configurable transaction costs;
- compare against buy-and-hold;
- calculate total return, annualized Sharpe ratio, and maximum drawdown.

This backtest is deliberately simple. A serious evaluation would need realistic execution assumptions, slippage, liquidity, corporate actions, portfolio constraints, multiple market regimes, and strict prevention of selection bias.

## Configuration

Edit [config.yaml](config.yaml) to change:

- ticker symbols;
- date range;
- feature-window length;
- train/validation/test ratios;
- batch size and epoch count;
- learning rate;
- weight decay;
- dropout;
- recurrent hidden size;
- Transformer heads and layers;
- early-stopping patience;
- transaction cost assumptions.

## Common troubleshooting

### `ModuleNotFoundError: No module named 'src'` in the notebook

Use the latest notebook version and restart the kernel:

```text
Kernel > Restart Kernel and Run All Cells
```

The first code cell adds the project root to `sys.path`.

### ARIMA date-index warnings

The implementation converts the irregular trading-date index to a numeric index before forecasting because weekends and market holidays do not form a fixed daily frequency. Warnings from old already-running processes can be ignored after restarting with the latest code.

### Matplotlib font-cache message

This is normal the first time Matplotlib runs. It should not be treated as an error.

### No data found

Check the ticker symbol and internet connection, then run:

```powershell
python -m src.data_loader
```

## Interview explanation

A concise explanation of the project:

> I built a time-series forecasting pipeline that downloads OHLCV data, creates technical indicators, and converts the series into causal 60-day sliding windows. I split the data chronologically and fit all scalers only on the training period to prevent leakage. I compared LSTM, GRU, and Transformer encoders against naive, ARIMA, and XGBoost baselines. The neural models use a combined regression and direction objective with Adam, dropout, weight decay, a learning-rate scheduler, early stopping, and checkpointing. I evaluated on a held-out future period using RMSE, MAE, MAPE, and directional accuracy. The key lesson was that complex models do not always beat a naive price-persistence baseline, especially on noisy financial data.

Good follow-up topics:

1. Why random train/test splitting is invalid for time series.
2. Why fitting a scaler on all data causes leakage.
3. Why a naive baseline is mandatory.
4. Why next-day price prediction is difficult.
5. How walk-forward validation better simulates deployment.
6. Why backtest results are not the same as live trading performance.
7. How transaction costs and slippage change a strategy's apparent profitability.
8. How you would improve the project: probabilistic forecasts, richer exogenous data, purged validation, hyperparameter search, calibrated classification, and realistic portfolio simulation.

## Limitations and future improvements

- The default experiment uses a small set of tickers.
- The current models use only daily OHLCV-derived features.
- Hyperparameters are illustrative rather than extensively tuned.
- The Transformer is intentionally small for a portfolio project.
- The dashboard displays saved metrics and history; it is not a live brokerage or production inference system.
- Walk-forward utilities are included but should be integrated into a larger experiment protocol for publication-quality validation.
- The current backtest is educational and does not model slippage, order execution, liquidity, taxes, or portfolio sizing.

Potential future work:

- add probabilistic prediction intervals;
- predict returns instead of raw prices;
- add market index, sector, macroeconomic, and sentiment features;
- use purged and embargoed time-series cross-validation;
- tune models using validation-only data;
- add a formal backtest report;
- containerize the app;
- add automated tests and CI;
- deploy the Streamlit dashboard.
