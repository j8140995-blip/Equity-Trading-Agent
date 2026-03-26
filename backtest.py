"""
backtest.py
-----------
Simplified backtester for the AI Trading Agent.

Uses:
  - Technical scoring: full module (EMA, RSI, MACD, VWAP, Volume, ATR)
  - Fundamental scoring: SEC EDGAR point-in-time (no lookahead)
  - Sentiment: fixed at 50.0 (no historical data available free)
  - Options: fixed at 50.0 (no historical data available free)

Signal generation: composite > 63.0
Entry: next day open price
Exit: stop (1x ATR below entry) or take-profit (5x ATR above entry)
Max hold: 3 trading days (time stop)

Outputs:
  - Per-trade log
  - Win rate, avg R/R, total return, Sharpe ratio
  - Saves results to /tmp/backtest_results.json
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import ta as ta_lib

sys.path.insert(0, ".")
from trading_agent.modules.fundamental import FundamentalAnalyzer
from trading_agent.modules.sector_map import get_sector, SECTOR_PE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Backtest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALPACA_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]

BACKTEST_START  = "2023-01-01"
BACKTEST_END    = "2024-12-31"
COMPOSITE_MIN   = 63.0
WEIGHT_TECH     = 0.35
WEIGHT_FUND     = 0.20
WEIGHT_SENT     = 0.20   # fixed at 50.0
WEIGHT_OPTS     = 0.25   # fixed at 50.0
MAX_POSITIONS   = 10
RISK_PCT        = 0.02   # 2% per trade
MAX_POS_PCT     = 0.05   # 5% concentration
RR_TARGET       = 2.5
TIME_STOP_DAYS  = 7
RESULTS_FILE    = "/tmp/backtest_results.json"

# Tickers to test — top scorers from recent live cycles
TEST_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN",
    "KSS", "MOS", "DRS", "RDW", "HRMY", "ADT", "CUBI",
    "REPX", "GDS", "ACHR", "UMAC", "BFH", "BKSY",
    "EC", "SBS", "RXO", "NU", "PSIX", "AMKR", "AVGO",
    "JPM", "BAC", "WFC", "GS", "MS",
    "XOM", "CVX", "COP", "SLB",
    "LMT", "RTX", "NOC", "GD",
    "UNH", "JNJ", "ABT", "PFE",
]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Alpaca for a date range."""
    import requests
    try:
        resp = requests.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            headers={
                "APCA-API-KEY-ID":     ALPACA_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET,
            },
            params={
                "symbols":    symbol,
                "timeframe":  "1Day",
                "start":      start,
                "end":        end,
                "limit":      1000,
                "feed":       "iex",
                "adjustment": "split",
            },
            timeout=15,
        )
        resp.raise_for_status()
        bars = resp.json().get("bars", {}).get(symbol, [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None)
        df = df.set_index("t").rename(columns={
            "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume",
        })
        return df[["open", "high", "low", "close", "volume"]].sort_index()
    except Exception as exc:
        logger.debug("OHLCV fetch failed %s: %s", symbol, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Technical scoring (same logic as live module)
# ---------------------------------------------------------------------------
def score_technical(df: pd.DataFrame, idx: int) -> float:
    """Score technical indicators at a specific bar index."""
    if idx < 50:
        return 0.0
    window = df.iloc[max(0, idx-119):idx+1].copy()
    if len(window) < 50:
        return 0.0

    window["ema20"]  = ta_lib.trend.ema_indicator(window["close"], window=20)
    window["ema50"]  = ta_lib.trend.ema_indicator(window["close"], window=50)
    window["ema200"] = ta_lib.trend.ema_indicator(window["close"], window=min(200, len(window)))
    window["rsi"]    = ta_lib.momentum.rsi(window["close"], window=14)
    window["macd"]   = ta_lib.trend.macd(window["close"])
    window["signal"] = ta_lib.trend.macd_signal(window["close"])
    window["hist"]   = ta_lib.trend.macd_diff(window["close"])
    window["atr"]    = ta_lib.volatility.average_true_range(
        window["high"], window["low"], window["close"], window=14
    )
    vwap_w = 20
    window["vwap"] = (
        (window["close"] * window["volume"]).rolling(vwap_w).sum()
        / window["volume"].rolling(vwap_w).sum()
    )

    row  = window.iloc[-1]
    prev = window.iloc[-2] if len(window) >= 2 else row
    total = 0.0

    # EMA trend (25 pts)
    if row["close"]  > row["ema20"]:  total += 8
    if row["ema20"]  > row["ema50"]:  total += 9
    if row["ema50"]  > row["ema200"]: total += 8

    # RSI (20 pts)
    rsi = row["rsi"]
    if pd.isna(rsi):   pass
    elif 50 <= rsi <= 65:   total += 20
    elif 40 <= rsi < 50:    total += 14
    elif 65 < rsi <= 70:    total += 12
    elif rsi > 70:          total += 5
    elif rsi < 30:          total += 3

    # MACD (20 pts)
    if not pd.isna(row["macd"]) and not pd.isna(row["signal"]):
        if (row["macd"] > row["signal"]
                and prev["macd"] <= prev["signal"]
                and row["hist"] > 0):
            total += 20
        elif row["macd"] > row["signal"]:
            total += 10

    # VWAP (15 pts)
    if not pd.isna(row["vwap"]) and row["vwap"] > 0:
        dev = (row["close"] - row["vwap"]) / row["vwap"]
        if dev >= 0.02:    total += 15
        elif dev >= 0.00:  total += 10
        elif dev >= -0.02: total += 5

    # Volume (12 pts)
    avg_vol = window["volume"].iloc[-21:-1].mean()
    if avg_vol > 0 and row["volume"] >= 1.5 * avg_vol: total += 12
    elif avg_vol > 0 and row["volume"] >= 1.0 * avg_vol: total += 6

    # ATR (8 pts)
    atr_pct = (row["atr"] / row["close"]) * 100 if row["close"] > 0 else 0
    if 2.5 <= atr_pct <= 8.0:   total += 8
    elif 2.0 <= atr_pct < 2.5:  total += 4

    return min(total, 100.0)


# ---------------------------------------------------------------------------
# Fundamental scoring (point-in-time)
# ---------------------------------------------------------------------------
async def fetch_fundamental_pit(
    symbol: str, as_of_date: str, fund: FundamentalAnalyzer,
    client: httpx.AsyncClient
) -> float:
    """
    Score fundamentals using only filings available as of as_of_date.
    Prevents lookahead bias.
    """
    await fund._load_cik_map(client)
    cik = fund._cik_map.get(symbol)
    if not cik:
        return 50.0

    # Get all concept data and filter to filings before as_of_date
    async def annual_pit(concept: str, unit: str) -> list[float]:
        entries = await fund._get_concept(cik, concept, unit, client)
        seen: dict[str, float] = {}
        for e in entries:
            if e.get("form") in ("10-K", "10-K/A"):
                filed = e.get("filed", "")
                if filed <= as_of_date:
                    seen[e["end"]] = float(e.get("val", 0))
        return [v for _, v in sorted(seen.items(), reverse=True)]

    async def quarterly_pit(concept: str, unit: str) -> list[float]:
        entries = await fund._get_concept(cik, concept, unit, client)
        seen: dict[str, float] = {}
        for e in entries:
            if e.get("form") in ("10-Q", "10-Q/A"):
                filed = e.get("filed", "")
                if filed <= as_of_date:
                    seen[e["end"]] = float(e.get("val", 0))
        return [v for _, v in sorted(seen.items(), reverse=True)]

    try:
        # EPS growth (30 pts)
        eps_vals = await annual_pit("EarningsPerShareBasic", "USD/shares")
        if len(eps_vals) >= 2 and eps_vals[1] != 0:
            growth = ((eps_vals[0] - eps_vals[1]) / abs(eps_vals[1])) * 100
        else:
            ni = await annual_pit("NetIncomeLoss", "USD")
            growth = ((ni[0] - ni[1]) / abs(ni[1])) * 100 if len(ni) >= 2 and ni[1] != 0 else 0

        if growth >= 30:    eps_score = 30.0
        elif growth >= 20:  eps_score = 25.0
        elif growth >= 10:  eps_score = 18.0
        elif growth >= 0:   eps_score = 10.0
        elif growth >= -10: eps_score = 4.0
        else:               eps_score = 0.0

        # D/E ratio (25 pts)
        debt   = await annual_pit("LongTermDebt", "USD")
        equity = await annual_pit("StockholdersEquity", "USD")
        if debt and equity and equity[0] > 0:
            de = debt[0] / equity[0]
            if de < 0.25:   de_score = 25.0
            elif de < 0.50: de_score = 20.0
            elif de < 1.00: de_score = 14.0
            elif de < 1.50: de_score = 8.0
            elif de < 2.00: de_score = 3.0
            else:           de_score = 0.0
        else:
            de_score = 12.5

        # Revenue acceleration (20 pts)
        revs = await quarterly_pit("RevenueFromContractWithCustomerExcludingAssessedTax", "USD")
        if len(revs) < 4:
            revs = await quarterly_pit("Revenues", "USD")
        if len(revs) >= 4:
            growths = [(revs[i]-revs[i+1])/abs(revs[i+1]) if revs[i+1] != 0 else 0 for i in range(3)]
            if all(g > 0 for g in growths) and all(growths[i] > growths[i+1] for i in range(2)):
                rev_score = 20.0
            elif all(g > 0 for g in growths):
                rev_score = 14.0
            elif growths[0] > 0:
                rev_score = 8.0
            else:
                rev_score = 2.0
        else:
            rev_score = 10.0

        return round(min(eps_score + 12.5 + de_score + rev_score, 100.0), 2)

    except Exception as exc:
        logger.debug("Fundamental PIT failed %s: %s", symbol, exc)
        return 50.0


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------
def simulate_trades(
    symbol: str,
    df: pd.DataFrame,
    tech_scores: list[float],
    fund_scores: list[float],
) -> list[dict]:
    """
    Simulate trades based on composite scores.
    Returns list of completed trade dicts.
    """
    trades = []
    in_position = False
    entry_price = 0.0
    stop_price  = 0.0
    target_price = 0.0
    entry_date  = None
    entry_idx   = 0
    entry_tech  = 0.0
    entry_fund  = 0.0
    entry_comp  = 0.0

    for i in range(len(df) - 1):
        row = df.iloc[i]

        if in_position:
            # Check exits on this bar
            next_open = df.iloc[i+1]["open"]
            hi = row["high"]
            lo = row["low"]

            days_held = i - entry_idx
            exited    = False
            exit_price = 0.0
            reason    = ""

            # Gap down through stop at open
            if next_open <= stop_price:
                exit_price = next_open
                reason = "STOP_LOSS"
                exited = True
            # Stop hit intraday
            elif lo <= stop_price:
                exit_price = stop_price
                reason = "STOP_LOSS"
                exited = True
            # Take profit hit intraday
            elif hi >= target_price:
                exit_price = target_price
                reason = "TAKE_PROFIT"
                exited = True
            # Time stop
            elif days_held >= TIME_STOP_DAYS:
                exit_price = df.iloc[i+1]["open"]
                reason = "TIME_STOP"
                exited = True

            if exited:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                risk_ps = entry_price - stop_price
                rr_realized = (exit_price - entry_price) / risk_ps if risk_ps > 0 else 0
                trades.append({
                    "symbol":       symbol,
                    "entry_date":   str(entry_date),
                    "exit_date":    str(df.index[i+1]),
                    "entry_price":  round(entry_price, 4),
                    "exit_price":   round(exit_price, 4),
                    "stop_price":   round(stop_price, 4),
                    "target_price": round(target_price, 4),
                    "pnl_pct":      round(pnl_pct, 4),
                    "rr_realized":  round(rr_realized, 4),
                    "win":          bool(exit_price > entry_price),
                    "reason":       reason,
                    "tech_score":   round(entry_tech, 1),
                    "fund_score":   round(entry_fund, 1),
                    "comp_score":   round(entry_comp, 1),
                })
                in_position = False

        if not in_position and i < len(tech_scores) and i < len(fund_scores):
            tech = tech_scores[i]
            fund = fund_scores[i]
            sent = 50.0
            opts = 50.0
            comp = (
                tech * WEIGHT_TECH
                + fund * WEIGHT_FUND
                + sent * WEIGHT_SENT
                + opts * WEIGHT_OPTS
            )

            if comp >= COMPOSITE_MIN and tech >= 30.0:
                # ATR for stop sizing
                window = df.iloc[max(0, i-13):i+1]
                try:
                    atr_series = ta_lib.volatility.average_true_range(
                        window["high"], window["low"], window["close"], window=min(14, len(window))
                    )
                    atr = float(atr_series.iloc[-1])
                except Exception:
                    atr = row["close"] * 0.02

                entry_price  = df.iloc[i+1]["open"]  # next day open
                stop_dist    = max(atr * 2.0, entry_price * 0.02, 0.05)
                stop_price   = round(entry_price - stop_dist, 4)
                # Target: min of 8% fixed or 3x ATR
                # Volatile stocks get the tighter 3x ATR target (reachable)
                # Stable stocks get the 8% fixed cap (still meaningful)
                atr_target   = entry_price + (3.0 * atr)
                pct_target   = entry_price * 1.08
                target_price = round(min(atr_target, pct_target), 4)
                entry_date   = df.index[i+1]
                entry_idx    = i + 1
                entry_tech   = tech
                entry_fund   = fund
                entry_comp   = comp
                in_position  = True

    return trades


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------
async def run_backtest():
    logger.info("=" * 55)
    logger.info("AI Trading Agent — Simplified Backtest")
    logger.info("Period : %s → %s", BACKTEST_START, BACKTEST_END)
    logger.info("Tickers: %d", len(TEST_UNIVERSE))
    logger.info("=" * 55)

    fund = FundamentalAnalyzer()
    all_trades: list[dict] = []

    async with httpx.AsyncClient() as client:
        await fund._load_cik_map(client)

        for sym in TEST_UNIVERSE:
            logger.info("Processing %s ...", sym)

            # Fetch OHLCV with extra history for indicator warmup
            df = fetch_ohlcv(sym, "2022-06-01", BACKTEST_END)
            if len(df) < 60:
                logger.info("  Skipping %s — insufficient data (%d bars)", sym, len(df))
                continue

            # Filter to backtest window
            bt_start = pd.Timestamp(BACKTEST_START)
            bt_end   = pd.Timestamp(BACKTEST_END)

            # Compute technical scores for each bar in backtest window
            tech_scores = []
            fund_scores = []
            bt_df = df[df.index >= bt_start].copy()

            # Fetch fundamental score ONCE per ticker using most recent filing
            # Fundamentals change quarterly — daily re-fetch is wasteful
            fund_score = await fetch_fundamental_pit(sym, BACKTEST_END, fund, client)
            logger.info("  %s fundamental score: %.1f", sym, fund_score)

            for i, (date, _) in enumerate(bt_df.iterrows()):
                # Technical: use all data up to and including this bar
                full_idx = df.index.get_loc(date)
                tech = score_technical(df, full_idx)
                tech_scores.append(tech)
                fund_scores.append(fund_score)

            if not tech_scores:
                continue

            trades = simulate_trades(sym, bt_df, tech_scores, fund_scores)
            all_trades.extend(trades)
            logger.info("  %s: %d trades", sym, len(trades))

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    if not all_trades:
        logger.info("No trades generated.")
        return

    wins   = [t for t in all_trades if t["win"]]
    losses = [t for t in all_trades if not t["win"]]
    total  = len(all_trades)
    win_rate = len(wins) / total * 100

    pnls     = [t["pnl_pct"] for t in all_trades]
    avg_pnl  = np.mean(pnls)
    avg_win  = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    avg_rr   = np.mean([t["rr_realized"] for t in all_trades])

    # Simple Sharpe (daily returns proxy)
    pnl_arr  = np.array(pnls)
    sharpe   = (np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)) if np.std(pnl_arr) > 0 else 0

    by_reason = {}
    for t in all_trades:
        r = t["reason"]
        by_reason.setdefault(r, {"count": 0, "wins": 0, "pnl": []})
        by_reason[r]["count"] += 1
        if t["win"]: by_reason[r]["wins"] += 1
        by_reason[r]["pnl"].append(t["pnl_pct"])

    logger.info("")
    logger.info("━" * 55)
    logger.info("BACKTEST RESULTS (%s → %s)", BACKTEST_START, BACKTEST_END)
    logger.info("━" * 55)
    logger.info("Total trades  : %d", total)
    logger.info("Win rate      : %.1f%%  (%d W / %d L)", win_rate, len(wins), len(losses))
    logger.info("Avg P&L/trade : %.2f%%", avg_pnl)
    logger.info("Avg win       : %.2f%%", avg_win)
    logger.info("Avg loss      : %.2f%%", avg_loss)
    logger.info("Avg R/R       : %.2fx", avg_rr)
    logger.info("Sharpe ratio  : %.2f", sharpe)
    logger.info("")
    logger.info("By exit reason:")
    for reason, data in by_reason.items():
        wr = data["wins"] / data["count"] * 100
        ap = np.mean(data["pnl"])
        logger.info("  %-15s  %3d trades  WR: %.0f%%  Avg: %.2f%%",
                    reason, data["count"], wr, ap)

    # Best and worst trades
    best  = max(all_trades, key=lambda t: t["pnl_pct"])
    worst = min(all_trades, key=lambda t: t["pnl_pct"])
    logger.info("")
    logger.info("Best trade  : %s  +%.2f%%  (%s)", best["symbol"], best["pnl_pct"], best["exit_date"])
    logger.info("Worst trade : %s  %.2f%%  (%s)", worst["symbol"], worst["pnl_pct"], worst["exit_date"])
    logger.info("━" * 55)

    # Save full results
    results = {
        "summary": {
            "period":     f"{BACKTEST_START} → {BACKTEST_END}",
            "total":      total,
            "wins":       len(wins),
            "losses":     len(losses),
            "win_rate":   round(win_rate, 2),
            "avg_pnl":    round(avg_pnl, 4),
            "avg_win":    round(avg_win, 4),
            "avg_loss":   round(avg_loss, 4),
            "avg_rr":     round(float(avg_rr), 4),
            "sharpe":     round(float(sharpe), 4),
            "by_reason":  {k: {"count": v["count"], "win_rate": round(v["wins"]/v["count"]*100,1),
                              "avg_pnl": round(float(np.mean(v["pnl"])),4)}
                          for k, v in by_reason.items()},
        },
        "trades": all_trades,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Full results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    asyncio.run(run_backtest())