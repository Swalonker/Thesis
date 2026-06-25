"""
Fetch daily S&P 500 (^GSPC) prices and build the returns CSV the pipeline reads.

    python fetch_data.py            # -> sp500_returns.csv

Output columns: Date, AdjClose, r   (r = 100 * log-return, percentage points)
This is the single, canonical input for run_all.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

START, END = "2000-01-01", "2025-01-01"   # END exclusive -> through 2024-12-31
OUT = Path(__file__).resolve().parent / "sp500_returns.csv"


def main():
    df = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit("yfinance returned no rows — check your connection.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    price = df["Close"]                       # auto_adjust=True -> Close is adjusted
    out = pd.DataFrame({
        "Date": pd.to_datetime(df.index).strftime("%Y-%m-%d"),
        "AdjClose": price.to_numpy(),
        "r": 100.0 * np.log(price / price.shift(1)),
    }).dropna()
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} rows to {OUT}  ({out['Date'].iloc[0]} -> {out['Date'].iloc[-1]})")


if __name__ == "__main__":
    main()
