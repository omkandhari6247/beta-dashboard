"""
Nifty 50 — Beta & R-squared Dashboard
=====================================
Computes each Nifty 50 constituent's beta and R-squared versus the
Nifty 50 index (^NSEI), using DAILY returns, over a selectable time
horizon from 3 months to 1 year.

Run:  streamlit run app.py
Data: Yahoo Finance via yfinance (free, ~15 min delayed / EOD).

Beta  = cov(stock_ret, index_ret) / var(index_ret)
R2    = corr(stock_ret, index_ret) ** 2
Alpha = mean(stock_ret) - beta * mean(index_ret)   (daily, annualised in table)
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# ---------------------------------------------------------------------------
# Nifty 50 constituents (Yahoo Finance NSE tickers use the ".NS" suffix).
# Edit this list whenever the index is rebalanced.
# ---------------------------------------------------------------------------
NIFTY_50 = {
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports & SEZ",
    "APOLLOHOSP": "Apollo Hospitals",
    "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",
    "BEL": "Bharat Electronics",
    "BHARTIARTL": "Bharti Airtel",
    "BPCL": "Bharat Petroleum",
    "BRITANNIA": "Britannia Industries",
    "CIPLA": "Cipla",
    "COALINDIA": "Coal India",
    "DRREDDY": "Dr Reddy's Labs",
    "EICHERMOT": "Eicher Motors",
    "GRASIM": "Grasim Industries",
    "HCLTECH": "HCL Technologies",
    "HDFCBANK": "HDFC Bank",
    "HDFCLIFE": "HDFC Life Insurance",
    "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco Industries",
    "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank",
    "INDUSINDBK": "IndusInd Bank",
    "INFY": "Infosys",
    "ITC": "ITC",
    "JIOFIN": "Jio Financial Services",
    "JSWSTEEL": "JSW Steel",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "M&M": "Mahindra & Mahindra",
    "MARUTI": "Maruti Suzuki",
    "NESTLEIND": "Nestle India",
    "NTPC": "NTPC",
    "ONGC": "Oil & Natural Gas Corp",
    "POWERGRID": "Power Grid Corp",
    "RELIANCE": "Reliance Industries",
    "SBILIFE": "SBI Life Insurance",
    "SBIN": "State Bank of India",
    "SHRIRAMFIN": "Shriram Finance",
    "SUNPHARMA": "Sun Pharma",
    "TATACONSUM": "Tata Consumer Products",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "TCS": "Tata Consultancy Services",
    "TECHM": "Tech Mahindra",
    "TITAN": "Titan Company",
    "TRENT": "Trent",
    "ULTRACEMCO": "UltraTech Cement",
    "WIPRO": "Wipro",
}

INDEX_TICKER = "^NSEI"  # Nifty 50 index
INDEX_NAME = "Nifty 50"

TRADING_DAYS_PER_MONTH = 21  # ~21 NSE trading days per month
MIN_MONTHS = 3
MAX_MONTHS = 12

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# How often the cached download is allowed to go stale before the next
# run re-pulls fresh end-of-day prices from Yahoo (seconds).
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_prices() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Download ~14 months of daily closes for all constituents + index.

    The ``period="14mo"`` window is relative to *now*, so each fresh download
    automatically includes the latest end-of-day bar — the dataset rolls
    forward on its own as new sessions settle.

    Returns (adjusted-close prices, fetch timestamp in UTC).
    Cached for CACHE_TTL_SECONDS so re-runs within that window don't re-hit
    Yahoo; after it expires the next run pulls the newest EOD prices.
    """
    tickers = [f"{s}.NS" for s in NIFTY_50] + [INDEX_TICKER]
    raw = yf.download(
        tickers,
        period="14mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # With multiple tickers yfinance returns a column MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:  # single ticker fallback
        prices = raw[["Close"]].copy()
    prices = prices.dropna(how="all")
    fetched_at = pd.Timestamp.now(tz="UTC")
    return prices, fetched_at


def compute_metrics(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute beta, R2, alpha, correlation and total return for each stock
    over the trailing `window` trading days vs the Nifty 50 index.
    """
    # Trailing window of prices, then daily simple returns.
    px = prices.tail(window + 1)
    rets = px.pct_change().dropna(how="all")

    idx = rets[INDEX_TICKER]
    rows = []
    for sym, name in NIFTY_50.items():
        col = f"{sym}.NS"
        if col not in rets.columns:
            continue
        pair = pd.concat([rets[col], idx], axis=1, keys=["stock", "index"]).dropna()
        if len(pair) < 20:  # not enough overlapping data
            continue
        y = pair["stock"].to_numpy()
        x = pair["index"].to_numpy()

        var_x = np.var(x, ddof=1)
        if var_x == 0 or np.isnan(var_x):
            continue
        cov_xy = np.cov(y, x, ddof=1)[0, 1]
        beta = cov_xy / var_x
        corr = np.corrcoef(y, x)[0, 1]
        r2 = corr ** 2
        alpha_daily = y.mean() - beta * x.mean()

        rows.append(
            {
                "Symbol": sym,
                "Company": name,
                "Beta": round(float(beta), 3),
                "R²": round(float(r2), 3),
                "Corr": round(float(corr), 3),
                "Alpha % (ann.)": round(float(alpha_daily) * 252 * 100, 2),
                "Volatility % (ann.)": round(float(y.std(ddof=1)) * np.sqrt(252) * 100, 1),
                "Return % (period)": round(float((1 + pd.Series(y)).prod() - 1) * 100, 1),
                "Obs (days)": int(len(pair)),
            }
        )

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Nifty 50 Beta & R²", layout="wide")

    st.title("📊 Nifty 50 — Beta & R² Dashboard")
    st.caption(
        "Beta and R-squared of each Nifty 50 constituent vs the Nifty 50 index "
        "(^NSEI), computed on **daily returns**. Source: Yahoo Finance (EOD)."
    )

    with st.sidebar:
        st.header("Settings")
        months = st.slider(
            "Time horizon (months, trailing)",
            min_value=MIN_MONTHS,
            max_value=MAX_MONTHS,
            value=MAX_MONTHS,
            step=1,
            help="Drag from 3 months up to 12 months (1 year).",
        )
        window = months * TRADING_DAYS_PER_MONTH
        horizon_label = "1 Year" if months == 12 else f"{months} Months"
        st.markdown(f"**Lookback:** {months} months (~{window} trading days)")

        st.divider()
        st.subheader("Beta reliability")
        r2_threshold = st.slider(
            "Flag stocks with R² below",
            min_value=0.05,
            max_value=0.60,
            value=0.30,
            step=0.05,
            help="When R² is low, the stock's moves are mostly NOT explained by "
            "the index, so its beta is statistically weak and shouldn't be read "
            "at face value (e.g. a stock that crashed on its own news). Flagged "
            "rows get a ⚠️ and their beta is greyed out.",
        )

        st.divider()
        st.subheader("Data refresh")
        auto_on = st.toggle(
            "Auto-update while open",
            value=True,
            help="Re-pull the latest end-of-day prices on a timer so a "
            "dashboard left open stays current.",
        )
        interval_label = st.selectbox(
            "Check every",
            ["15 min", "30 min", "60 min"],
            index=1,
            disabled=not auto_on,
        )
        if st.button("🔄 Update now"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown(
            "**Beta** — sensitivity to the index (1.0 = moves with Nifty).\n\n"
            "**R²** — fraction of the stock's daily moves explained by Nifty "
            "(0–1; high = closely index-driven)."
        )

    # Auto-update: reload the whole page on a browser-side timer. On reload the
    # script re-runs; load_prices() is cached for CACHE_TTL_SECONDS, so it only
    # re-downloads once that has elapsed — picking up the newest end-of-day bar
    # as soon as it's available. One reload per interval (no rerun loop).
    if auto_on:
        minutes = {"15 min": 15, "30 min": 30, "60 min": 60}[interval_label]
        components.html(
            f"<script>setTimeout(function() "
            f"{{ window.parent.location.reload(); }}, {minutes * 60 * 1000});</script>",
            height=0,
        )

    with st.spinner("Downloading prices from Yahoo Finance…"):
        prices, fetched_at = load_prices()

    if prices.empty or INDEX_TICKER not in prices.columns:
        st.error("Could not load price data. Check your internet connection and retry.")
        st.stop()

    # Display freshness in IST (UTC+5:30).
    fetched_ist = (fetched_at + pd.Timedelta(hours=5, minutes=30)).strftime("%d %b %Y, %H:%M IST")
    data_through = prices.index.max().date()
    df = compute_metrics(prices, window)

    if df.empty:
        st.warning("Not enough data to compute metrics for this horizon.")
        st.stop()

    # ---- Summary metrics ----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks covered", len(df))
    c2.metric("Median Beta", f"{df['Beta'].median():.2f}")
    c3.metric("Median R²", f"{df['R²'].median():.2f}")
    c4.metric("Latest session", data_through.strftime("%d %b %Y"))

    # Flag stocks whose beta is statistically weak (low R²).
    df["Beta OK?"] = np.where(df["R²"] < r2_threshold, "⚠️ weak", "✅ ok")
    n_weak = int((df["R²"] < r2_threshold).sum())

    refresh_note = (
        f"🟢 Auto-updating every {interval_label.lower()}" if auto_on else "⚪ Auto-update off"
    )
    st.caption(
        f"Prices fetched: **{fetched_ist}**  ·  latest EOD session: "
        f"**{data_through.strftime('%d %b %Y')}**  ·  {refresh_note}. "
        "The newest bar is provisional until the NSE session closes (~15:30 IST)."
    )

    if n_weak:
        st.warning(
            f"⚠️ **{n_weak} of {len(df)} stocks have R² < {r2_threshold:.2f}** — their beta is "
            "statistically weak (moves driven mostly by stock-specific factors, "
            "not the index). Beta is greyed out for these; treat it with caution."
        )

    # ---- Table --------------------------------------------------------------
    st.subheader(f"Metrics — trailing {horizon_label} (daily returns)")

    sort_col = st.selectbox(
        "Sort by",
        ["Beta", "R²", "Corr", "Alpha % (ann.)", "Volatility % (ann.)", "Return % (period)", "Symbol"],
        index=0,
    )
    ascending = sort_col == "Symbol"
    df_sorted = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    # Reorder so the reliability flag sits right after R².
    cols = list(df_sorted.columns)
    cols.insert(cols.index("R²") + 1, cols.pop(cols.index("Beta OK?")))
    df_sorted = df_sorted[cols]

    def _grey_weak_beta(row: pd.Series) -> list[str]:
        """De-emphasise the Beta cell when R² is below the threshold."""
        out = [""] * len(row)
        if row["R²"] < r2_threshold:
            out[row.index.get_loc("Beta")] = (
                "background-color:#eeeeee; color:#9aa0a6; font-style:italic;"
            )
        return out

    styled = (
        df_sorted.style.background_gradient(subset=["Beta"], cmap="RdYlGn_r")
        .background_gradient(subset=["R²"], cmap="Blues")
        .apply(_grey_weak_beta, axis=1)  # applied after gradient so it overrides
        .format(precision=3)
    )

    st.dataframe(styled, width="stretch", height=560, hide_index=True)

    csv = df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        csv,
        file_name=f"nifty50_beta_r2_{horizon_label.replace(' ', '')}_{data_through}.csv",
        mime="text/csv",
    )

    # ---- Scatter: Beta vs R² ------------------------------------------------
    st.subheader("Beta vs R² map")
    scatter_df = df_sorted.set_index("Symbol")[["Beta", "R²"]]
    st.scatter_chart(scatter_df.reset_index(), x="Beta", y="R²", color="Symbol", height=420)

    st.caption(
        f"**Beta OK?** flags whether beta is trustworthy: ⚠️ weak = R² below "
        f"{r2_threshold:.2f} (beta greyed out — the stock moves mostly on its own, "
        "not with the index, so a big price fall shows up as negative **Alpha**, "
        "not high beta). Adjust the threshold in the sidebar. "
        "Returns are simple daily pct-changes on adjusted close; beta uses sample "
        "covariance/variance; R² is the squared Pearson correlation; alpha and "
        "volatility are annualised (×252 / ×√252)."
    )


if __name__ == "__main__":
    main()
