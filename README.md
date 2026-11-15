# StockPredictor

StockPredictor is an educational time-series pipeline for predicting next-day stock log returns and direction from daily OHLCV data. It compares LSTM, GRU, and a small positional Transformer against zero-return, yesterday's-price, ARIMA, and XGBoost baselines. It uses chronological splits, training-only scaling, walk-forward validation, saved checkpoints, backtesting, and a Streamlit dashboard.

This project is for learning and portfolio demonstration only. It is not financial advice.

## Data and features

Configured tickers are `AAPL`, `MSFT`, and `RELIANCE.NS`. Data starts at `2018-01-01` and uses the latest available date. The downloader saves cleaned files under `data/`, which is intentionally ignored.

The 13 engineered features are daily percentage return, daily log return, 5/20/50-day moving averages, 14-period RSI, MACD, MACD signal, 20-day Bollinger middle/upper/lower bands, 20-day log-return volatility, and daily volume change.

Each example uses a 60-trading-day window. The regression target is:

```text
log(Close[t+1] / Close[t])
```

Returns are closer to stationary than raw prices and do not grow with the price level, reducing non-stationarity and out-of-range forecasts. Predicted returns are converted back to prices with:

```text
predicted_close = previous_close * exp(predicted_log_return)
```

The direction target is whether the next log return is positive. Data is split chronologically into 70% training, 15% validation, and 15% test data. Feature and target scalers are fitted only on the training portion of each fold.

## Models

Neural models use a combined MSE return loss and BCE-with-logits direction loss, Adam, learning rate `0.001`, weight decay `0.0001`, gradient clipping, ReduceLROnPlateau, dropout `0.2`, and validation early stopping.

- LSTM: two recurrent layers, hidden size 64.
- GRU: two recurrent layers, hidden size 64.
- Transformer: two encoder layers, hidden size 64, four heads, positional encoding, dropout 0.2, and feed-forward dimension 128.
- ZeroReturn: predicts a zero log return.
- YesterdayPrice: predicts the previous close.
- ARIMA: order `(5, 1, 0)` on log returns.
- XGBoost: 300 trees, depth 3, learning rate 0.03, subsample 0.8, and column subsample 0.8.

## Folder structure

```text
Stock_Prediction/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── data/                       # downloaded CSV files; ignored
├── results/                    # committed metrics, plots, logs, and checkpoints
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── data_loader.py
    ├── features.py
    ├── dataset.py
    ├── models.py
    ├── baselines.py
    ├── train.py
    ├── evaluate.py
    ├── backtest.py
    └── audit_backtest.py
```

## Running

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data_loader
python -m src.train
streamlit run app.py
```

The training run writes committed artifacts under `results/`. Each ticker directory contains test metrics, walk-forward averages, loss plots, prediction plots, checkpoints, equity curves, and backtest metrics. `results/run_log.txt` records row counts, date ranges, epochs trained, early stopping, and final train/validation losses.

The authoritative random-control output is `results/matched_random_baseline.csv`, which matches each selected strategy's exposure and trade count. `results/random_signal_baseline.csv` is an earlier unmatched random-signal audit retained as a historical artifact and is not used for the conclusions below.

## Screenshots

<!-- Add Streamlit screenshots here after launching `streamlit run app.py`. -->

## Results from the executed run

The following values are copied from `results/*/test_metrics.csv`. Return metrics are log-return units. Direction accuracy is a percentage; precision, recall, and F1 are proportions.

### AAPL

| Model | ReturnRMSE | ReturnMAE | PriceRMSE | PriceMAE | DirectionAccuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.015416190959513187 | 0.011066224426031113 | 4.569839224611589 | 3.2543689278068664 | 51.49253731343284 | 0.5207547169811321 | 0.9787234042553191 | 0.6798029556650246 |
| GRU | 0.015688901767134666 | 0.011307114735245705 | 4.655393990699512 | 3.323807773168028 | 47.38805970149254 | 0.0 | 0.0 | 0.0 |
| Transformer | 0.015547560527920723 | 0.011165465228259563 | 4.610895929802089 | 3.2837257619125886 | 49.25373134328358 | 0.6190476190476191 | 0.09219858156028368 | 0.16049382716049382 |
| ZeroReturn | 0.01544077255060336 | 0.011079283210796846 | 4.577609941691184 | 3.258134414900605 | 47.38805970149254 | 0.0 | 0.0 | 0.0 |
| YesterdayPrice | 0.01544077255060336 | 0.011079283210796846 | 4.577609941691184 | 3.258134414900605 | 47.38805970149254 | 0.0 | 0.0 | 0.0 |
| ARIMA | 0.015591103636307608 | 0.01118819131026415 | 4.621938420843162 | 3.2874329111780254 | 47.38805970149254 | 0.0 | 0.0 | 0.0 |
| XGBoost | 0.01826310898410991 | 0.013845777962751715 | 5.430556460015915 | 4.066489043443458 | 48.88059701492538 | 0.75 | 0.0425531914893617 | 0.08053691275167785 |

### MSFT

| Model | ReturnRMSE | ReturnMAE | PriceRMSE | PriceMAE | DirectionAccuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.020766710862517357 | 0.014485019259154797 | 9.236262738275963 | 6.4844506243900275 | 50.74626865671642 | 0.6666666666666666 | 0.015037593984962405 | 0.029411764705882353 |
| GRU | 0.020823901519179344 | 0.014552336186170578 | 9.255340961276962 | 6.512627758751627 | 49.25373134328358 | 0.46511627906976744 | 0.15037593984962405 | 0.22727272727272727 |
| Transformer | 0.02076626755297184 | 0.014451690949499607 | 9.23599596232935 | 6.471217962302279 | 47.76119402985074 | 0.48 | 0.631578947368421 | 0.5454545454545454 |
| ZeroReturn | 0.020725294095996594 | 0.014430901480195324 | 9.215090495023624 | 6.4592553893131965 | 50.373134328358205 | 0.0 | 0.0 | 0.0 |
| YesterdayPrice | 0.020725294095996594 | 0.014430901480195324 | 9.215090495023624 | 6.4592553893131965 | 50.373134328358205 | 0.0 | 0.0 | 0.0 |
| ARIMA | 0.020846794373811175 | 0.014590464601253714 | 9.281181384793243 | 6.534121601129816 | 49.25373134328358 | 0.4943820224719101 | 0.9924812030075187 | 0.66 |
| XGBoost | 0.02236062654833066 | 0.015742673119963035 | 9.912671913701242 | 7.026246775485083 | 47.01492537313433 | 0.4716981132075472 | 0.5639097744360902 | 0.5136986301369864 |

### RELIANCE.NS

| Model | ReturnRMSE | ReturnMAE | PriceRMSE | PriceMAE | DirectionAccuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.014811295084655285 | 0.010335239581763744 | 19.762666768782047 | 13.982369127299771 | 51.908396946564885 | 0.625 | 0.07692307692307693 | 0.136986301369863 |
| GRU | 0.014782343991100788 | 0.010318242013454437 | 19.729543668696916 | 13.963441111729425 | 52.67175572519084 | 0.6875 | 0.08461538461538462 | 0.1506849315068493 |
| Transformer | 0.01502708438783884 | 0.010654276236891747 | 20.062845673656316 | 14.434533946849143 | 48.473282442748086 | 0.43243243243243246 | 0.12307692307692308 | 0.19161676646706588 |
| ZeroReturn | 0.01487151910442853 | 0.010369171262124282 | 19.844834287735388 | 14.026817746921434 | 51.526717557251914 | 0.0 | 0.0 | 0.0 |
| YesterdayPrice | 0.01487151910442853 | 0.010369171262124282 | 19.844834287735388 | 14.026817746921434 | 51.526717557251914 | 0.0 | 0.0 | 0.0 |
| ARIMA | 0.015315893780715333 | 0.010896577169091294 | 20.428047066420866 | 14.760807021803892 | 51.526717557251914 | 0.0 | 0.0 | 0.0 |
| XGBoost | 0.015389535603634916 | 0.010965747675028961 | 20.59314653133974 | 14.845935745510502 | 51.14503816793893 | 0.4957983193277311 | 0.4645669291338583 | 0.4796747967479675 |

## Walk-forward averages

Five expanding-window folds were executed. The baseline models are included in the same fold averages.

| Ticker | Model | ReturnRMSE | DirectionAccuracy |
|---|---|---:|---:|
| AAPL | LSTM | 0.018806840106844903 | 49.760000000000005 |
| AAPL | GRU | 0.018790226057171823 | 50.239999999999995 |
| AAPL | Transformer | 0.01888214498758316 | 49.44000000000001 |
| AAPL | ZeroReturn | 0.01880924532410926 | 45.440000000000005 |
| AAPL | XGBoost | 0.020425902808683345 | 49.28 |
| MSFT | LSTM | 0.01680250857025385 | 50.4 |
| MSFT | GRU | 0.016815238818526267 | 48.64 |
| MSFT | Transformer | 0.017031246423721315 | 47.04 |
| MSFT | ZeroReturn | 0.016879573790282442 | 46.4 |
| MSFT | XGBoost | 0.01896557641358779 | 46.879999999999995 |
| RELIANCE.NS | LSTM | 0.014538205415010452 | 49.25619834710744 |
| RELIANCE.NS | GRU | 0.014504799805581569 | 50.909090909090914 |
| RELIANCE.NS | Transformer | 0.014546459168195724 | 50.247933884297524 |
| RELIANCE.NS | ZeroReturn | 0.014541603430115551 | 49.752066115702476 |
| RELIANCE.NS | XGBoost | 0.01607279112585673 | 50.41322314049587 |

## Backtest verification

The signal alignment is explicit in `src/backtest.py`: the prediction for close `t + 1` is compared with close `t`, and the resulting signal is stored at position `t + 1`, where it is multiplied by the return from close `t` to close `t + 1`.

Strategy and buy-and-hold use exactly the same price series and exactly the same equity-curve dates.

| Ticker | First date | Last date | Strategy equity days | Buy-and-hold equity days |
|---|---|---|---:|---:|
| AAPL | 2025-10-21 | 2026-11-13 | 269 | 269 |
| MSFT | 2025-10-21 | 2026-11-13 | 269 | 269 |
| RELIANCE.NS | 2025-10-20 | 2026-11-13 | 263 | 263 |

Validation ReturnRMSE selected ZeroReturn for AAPL and MSFT, and GRU for RELIANCE.NS.

| Ticker | Selected model | Strategy return | Strategy Sharpe | Strategy max drawdown | Buy-and-hold return | Buy-and-hold Sharpe | Buy-and-hold max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| AAPL | ZeroReturn | 0.0 | 0.0 | 0.0 | 0.3558245001682254 | 1.294024893053619 | -0.1382298375943214 |
| MSFT | ZeroReturn | 0.0 | 0.0 | 0.0 | 0.012015670635103337 | 0.19720580651605288 | -0.3491062375616114 |
| RELIANCE.NS | GRU | 0.31098549248317076 | 1.4046044061561367 | -0.11260289591914763 | -0.0866512772192759 | -0.2471657881826761 | -0.27275013220514344 |

MSFT never went long because ZeroReturn predicts exactly zero return and the strict `>` signal condition is always false.

## Cost sensitivity for validation-selected models

| Ticker | Transaction cost | Strategy return | Strategy Sharpe | Strategy max drawdown |
|---|---:|---:|---:|---:|
| AAPL | 0.0 | 0.0 | 0.0 | 0.0 |
| AAPL | 0.001 | 0.0 | 0.0 | 0.0 |
| AAPL | 0.003 | 0.0 | 0.0 | 0.0 |
| MSFT | 0.0 | 0.0 | 0.0 | 0.0 |
| MSFT | 0.001 | 0.0 | 0.0 | 0.0 |
| MSFT | 0.003 | 0.0 | 0.0 | 0.0 |
| RELIANCE.NS | 0.0 | 0.35218012603292803 | 1.5532319157042993 | -0.11080964424783146 |
| RELIANCE.NS | 0.001 | 0.31098549248317076 | 1.4046044061561367 | -0.11260289591914763 |
| RELIANCE.NS | 0.003 | 0.23220866890196712 | 1.104557622526941 | -0.1161839694640886 |

## Delayed-signal test

The delayed test moves the signal one additional trading day later.

| Ticker | Model | Delayed strategy return | Delayed Sharpe | Delayed max drawdown |
|---|---|---:|---:|---:|
| AAPL | ZeroReturn | 0.0 | 0.0 | 0.0 |
| MSFT | ZeroReturn | 0.0 | 0.0 | 0.0 |
| RELIANCE.NS | GRU | 0.09692710093785162 | 0.539932396519192 | -0.12642273124791503 |

The delayed result is lower for RELIANCE.NS because the signal is applied one day later than intended. It does not change AAPL or MSFT because their selected ZeroReturn strategy never enters the market.

## All-model backtests at 0.1% transaction cost

| Ticker | Model | Strategy return | Strategy Sharpe | Strategy max drawdown | Trades | Market exposure (%) |
|---|---|---:|---:|---:|---:|---:|
| AAPL | LSTM | 0.596473621417682 | 2.0494651988268098 | -0.14340081354363488 | 9 | 96.28252788104089 |
| AAPL | GRU | 0.00617569786012262 | 0.16955467728591742 | -0.023733396848946264 | 2 | 0.7434944237918215 |
| AAPL | Transformer | -0.054716400822909295 | -0.30486882936226944 | -0.15305316180775874 | 60 | 37.9182156133829 |
| AAPL | XGBoost | 0.040309859569960915 | 0.9814628343390527 | -0.019837726569386538 | 10 | 2.973977695167286 |
| AAPL | ZeroReturn | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| MSFT | LSTM | -0.022083315943565163 | 0.08585439501211363 | -0.341837799189868 | 15 | 84.01486988847584 |
| MSFT | GRU | 0.13948893375337867 | 0.5970736597840924 | -0.2624920687704344 | 69 | 53.53159851301115 |
| MSFT | Transformer | -0.04107708266539967 | -0.013865447522726311 | -0.33350827223423984 | 59 | 67.65799256505576 |
| MSFT | XGBoost | -0.17164887074778523 | -0.5083164648751355 | -0.38477515990564626 | 74 | 59.10780669144982 |
| MSFT | ZeroReturn | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| RELIANCE.NS | LSTM | 0.28336601851634025 | 1.457410816620193 | -0.06818135565295158 | 17 | 33.840304182509506 |
| RELIANCE.NS | GRU | 0.31098549248317076 | 1.4046044061561367 | -0.11260289591914763 | 31 | 55.893536121673 |
| RELIANCE.NS | Transformer | 0.02215014297908069 | 0.869757594742737 | -0.00611836110710781 | 8 | 1.520912547528517 |
| RELIANCE.NS | XGBoost | 0.04225953916233682 | 0.31562114318246215 | -0.1336305406227788 | 72 | 45.247148288973385 |
| RELIANCE.NS | ZeroReturn | 0.0 | 0.0 | 0.0 | 0 | 0.0 |

## Exposure- and trade-matched random baseline

Each random run matched the selected strategy’s market exposure and trade count. There were 1,000 runs per ticker.

| Ticker | Matched trades | Matched exposure (%) | Random mean return | Random return std | Random min return | Random max return | Strategy percentile |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| MSFT | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| RELIANCE.NS | 31 | 55.893536121673 | -0.06384876722791685 | 0.10393172484254543 | -0.3739874251206229 | 0.2647736524290323 | 100.0 |

## AAPL out-of-market periods

These periods use the validation-selected AAPL ZeroReturn strategy.

| Start date | End date | Actual market return avoided |
|---|---|---:|
| 2025-10-22 | 2026-11-13 | 0.3558245001682254 |

Total avoided return: `0.3558245001682254`.

## Prediction alignment samples

The window end is the day immediately before the prediction date in every row.

| Date | Window start | Window end | Predicted return | Actual return | Position |
|---|---|---|---:|---:|---:|
| 2025-10-22 | 2025-07-29 | 2025-10-21 | 0.0 | -0.01657678415969155 | 0.0 |
| 2025-10-23 | 2025-07-30 | 2025-10-22 | 0.0 | 0.004362589657176841 | 0.0 |
| 2026-05-06 | 2026-02-09 | 2026-05-05 | 0.0 | 0.011649861770647973 | 0.0 |
| 2026-11-12 | 2026-08-19 | 2026-11-11 | 0.0 | -0.0021111603446932747 | 0.0 |
| 2026-11-13 | 2026-08-20 | 2026-11-12 | 0.0 | 0.003909121599787291 | 0.0 |

## Neural model parameter counts

| Model | Parameters |
|---|---:|
| LSTM | 53634 |
| GRU | 40258 |
| Transformer | 71810 |

## Findings and limitations

- Validation-based selection chooses ZeroReturn for AAPL and MSFT, and GRU for RELIANCE.NS.
- The models do not clearly beat the ZeroReturn baseline overall.
- The AAPL and MSFT validation-selected backtests have no market exposure and therefore no strategy edge.
- The RELIANCE.NS strategy return falls from `0.35218012603292803` at zero cost to `0.23220866890196712` at 0.3% cost.
- The backtest covers one test period and is not evidence of a profitable strategy.
- The backtest does not model slippage, liquidity, taxes, corporate actions, or position sizing.
- The Transformer is intentionally small and uses learned positional encoding.
