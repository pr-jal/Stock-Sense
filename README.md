# StockSense — Machine-Learning Stock Direction Prediction

Predicts whether a stock's **next-day close will be higher or lower** than today's, using
machine learning on historical prices. Educational project — not financial advice.

## Features

- **4 ML models:** majority-class baseline, Logistic Regression, Random Forest, Gradient Boosting
- **15 engineered features:** returns, moving averages, RSI, MACD, volatility, volume, price range
- **Honest evaluation:** chronological train/validation/test split, plus 5-fold walk-forward testing
- **Explainable AI:** permutation feature importance and a per-prediction "why" chart
- **News sentiment (NLP):** recent headlines scored positive/negative/neutral
- **Dashboard:** overview, interactive charts, model comparison, backtest, stock comparison,
  prediction history, downloads, beginner tooltips, light/dark mode

## How to run (3 steps)

1. Install Python 3.10+, open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   ```
2. (Optional) Run the pipeline from the terminal and save results and charts to `results/`:
   ```
   python3 train.py --ticker RELIANCE.NS
   ```
3. Open the dashboard:
   ```
   streamlit run app.py
   ```
   Or double-click **Start StockSense.command** (Mac) or **Start StockSense (Windows).bat** (Windows).

Ticker examples: `TCS.NS`, `INFY.NS` (NSE India), `AAPL`, `MSFT` (US).
No internet? Use a CSV: `python3 train.py --csv data/mystock.csv --name MYSTOCK`.

## Files

| File | What it does |
|---|---|
| `data_loader.py` | Downloads, saves and cleans price data; company info; prediction history |
| `features.py` | Builds the 15 features and the target label (no look-ahead) |
| `model.py` | Models, chronological split, metrics, walk-forward test, explanations, backtest |
| `news.py` | Fetches recent headlines and scores their sentiment |
| `train.py` | Runs the pipeline from the terminal and saves results for the report |
| `app.py` | Streamlit dashboard |

## Method in short

- **Target:** 1 if tomorrow's close > today's close, else 0.
- **No look-ahead:** every feature uses only data up to that day.
- **Split by time:** 70% train → 15% validation (pick model) → 15% test (report once).
- **Walk-forward:** 5 expanding windows, each trained only on earlier data.
- **Reproducible:** downloaded prices are saved in `data/` and re-used; models use a fixed seed.

## Put it online (free, Streamlit Community Cloud)

1. Create a free account at **github.com** and make a new repository (e.g. `stocksense`).
2. Upload all the files in this folder (you can skip `data/` and `results/`).
3. Go to **share.streamlit.io**, sign in with GitHub, click **Create app**.
4. Choose your repository, set the main file to `app.py`, and click **Deploy**.
5. After a few minutes you get a public link (like `stocksense.streamlit.app`) to share.

## Expect honest results

Daily stock direction is very hard to predict. Test accuracy around 50–55% is normal.
Reporting that honestly (and comparing with the baseline) is part of the project.
