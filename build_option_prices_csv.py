#!/usr/bin/env python3
"""
Télécharge les chaines d'options d'un ticker via yfinance et génère un CSV
contenant K, T (années), S0, type (Call/Put), lastPrice et impliedVolatility.

Sortie par défaut : ticker_prices.csv à la racine du projet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import pandas as pd
import yfinance as yf


def get_spot(ticker: str) -> float:
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if not hist.empty and "Close" in hist.columns:
        return float(hist["Close"].iloc[-1])
    info = tk.fast_info if hasattr(tk, "fast_info") else {}
    price = getattr(info, "last_price", None) or info.get("last_price") or info.get("lastPrice")
    if price is None:
        raise RuntimeError("Impossible de récupérer le spot.")
    return float(price)


def fetch_option_rows(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    expiries = tk.options or []
    if not expiries:
        raise RuntimeError("Aucune échéance d'option trouvée.")
    spot = get_spot(ticker)
    rows = []
    today = dt.datetime.utcnow().date()
    for expiry in expiries:
        exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
        T = (exp_date - today).days / 365.0
        chain = tk.option_chain(expiry)
        for opt_df, opt_type in [(chain.calls, "Call"), (chain.puts, "Put")]:
            for _, r in opt_df.iterrows():
                rows.append(
                    {
                        "expiry": expiry,
                        "option_type": opt_type,
                        "K": float(r["strike"]),
                        "T": float(T),
                        "S0": spot,
                        "lastPrice": float(r.get("lastPrice", float("nan"))),
                        "iv": float(r.get("impliedVolatility", float("nan"))),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit un CSV d'options pour un ticker.")
    parser.add_argument("ticker", help="Ticker Yahoo Finance (ex: AAPL)")
    parser.add_argument("--output", default="ticker_prices.csv", help="Chemin du CSV de sortie (défaut: ticker_prices.csv)")
    args = parser.parse_args()

    # Neutralise d’éventuelles options d’impersonation qui posent problème
    os.environ.pop("YF_IMPERSONATE", None)
    os.environ.pop("YF_SCRAPER_IMPERSONATE", None)
    try:
        yf.set_config(proxy=None)
    except Exception:
        pass

    df = fetch_option_rows(args.ticker)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"CSV écrit: {out_path} (shape={df.shape})")


if __name__ == "__main__":
    main()
