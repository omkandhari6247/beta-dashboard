# Nifty 50 — Beta & R² Dashboard

A Streamlit dashboard showing the **beta** and **R-squared** of every Nifty 50
stock versus the Nifty 50 index (`^NSEI`), computed on **daily returns** over a
selectable trailing horizon (**3 months → 1 year**).

## What it shows
- **Beta** — sensitivity of the stock to the index (1.0 = moves with Nifty).
- **R²** — fraction of the stock's daily moves explained by the Nifty (0–1).
- Plus correlation, annualised alpha, annualised volatility, and period return.
- Sortable table, colour gradients, CSV export, and a Beta-vs-R² scatter map.
- Horizon selector: 3M / 6M / 9M / 1Y.

## Method
For each stock, using daily simple returns on adjusted close over the window:

```
Beta  = cov(stock_ret, index_ret) / var(index_ret)
R²    = corr(stock_ret, index_ret) ** 2
Alpha = mean(stock_ret) - Beta * mean(index_ret)   (annualised ×252 in table)
```

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 (this project is configured for 8502 in
`.claude/launch.json` to avoid clashing with another local app).

## Data
Yahoo Finance via `yfinance` (free, end-of-day). ~14 months of history is
downloaded once and cached for 1 hour; the horizon slider slices it locally.
Use **🔄 Refresh data** to clear the cache.

## Maintenance
The constituent list is hardcoded at the top of `app.py` (`NIFTY_50`). Edit it
after an index rebalance. NSE tickers use the `.NS` suffix on Yahoo.
