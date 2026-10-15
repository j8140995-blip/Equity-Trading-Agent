# AI Trading Agent

An autonomous equities paper-trading agent that scans ~2,500 US tickers per cycle, scores them across four independent signal modules plus a macro regime filter, and executes bracketed orders through Alpaca's paper API. Runs 24/7 on a Replit Reserved VM with a live web dashboard.

**Status:** Paper trading only. No real capital. Currently paused pending signal validation (see [Findings](#findings)).

---

## Architecture

```
main.py                          Flask dashboard (:8080) + agent in background thread
performance_log.json             Trade ledger, persists across restarts
requirements.txt
backtest.py                      Standalone historical simulator
trading_agent/
├── scanner/
│   └── market_scanner.py        Config, scoring orchestration, risk, execution
└── modules/
    ├── fundamental.py           SEC EDGAR XBRL (async httpx, disk cache)
    ├── sentiment.py             VADER + custom financial lexicon
    ├── options_flow.py          CBOE chains with greeks, persistent IV history
    ├── macro_context.py         VIX, SPY/QQQ trend, sector rotation
    ├── performance_logger.py    Entry/exit tracking, P&L, realized R/R
    ├── dashboard.py             Flask UI
    └── sector_map.py            ~300 ticker→sector map + sector P/E baselines
```

**Entry point:** `python3 -u main.py` — Flask serves on port 8080 (Replit's exposed web port) while the trading agent runs in a daemon thread.

---

## Scan cycle

Each cycle runs every 5 minutes during market hours (pre-market 3:00 AM – 9:30 AM CT, regular 9:30 AM – 4:00 PM, after-hours 4:00 – 7:00 PM; sleeps overnight and weekends, with 2026 holidays hardcoded).

**Universe build** — pulls all tradable US equities from Alpaca, pre-filters on price ($5–$500), average dollar volume, and minimum volatility. Typically ~2,400 of ~4,900 tickers survive.

**Pass 1 — Technical** (~12 min, all candidates)
EMA alignment (20/50/200), RSI, MACD crossover, VWAP position, volume confirmation, ATR quality. Plus two trend-quality filters: penalties for sitting in the bottom of the 52-week range and for trading below the 200-day EMA. Tickers scoring below 30 are dropped.

**Pass 2 — Four modules concurrently**

| Module | Weight | Source | Signals |
|---|---|---|---|
| Technical | 35% | Alpaca OHLCV (2yr) | EMA/RSI/MACD/VWAP/volume/ATR, 52wk + 200d filters |
| Fundamental | 20% | SEC EDGAR XBRL | EPS growth YoY, P/E vs sector, debt/equity, revenue acceleration |
| Sentiment | 20% | Benzinga, Google News, SEC 8-K | VADER compound score with ~60-term financial lexicon |
| Options | 25% | CBOE delayed quotes | Put/call ratio, IV percentile, delta-weighted flow, OI concentration + unusual-activity detector |

**Macro overlay** — real VIX from Yahoo Finance, SPY/QQQ vs 200-day, sector-ETF relative strength, and news sentiment. Classifies the regime (`RISK_ON` / `NEUTRAL` / `RISK_OFF` / `CRISIS`), adjusts the composite threshold ±2–10 points, scales position size, applies ±3–5 point sector bonuses, and blocks all entries on scheduled macro event days (FOMC, CPI, NFP).

**Composite:** `tech×0.35 + fund×0.20 + sent×0.20 + opts×0.25`, entry at ≥68 (raised to 73 in risk-off).

---

## Risk and execution

| Parameter | Value |
|---|---|
| Max open positions | 10 |
| Max per sector | 3 |
| Portfolio risk per trade | 2% |
| Position concentration cap | 5% |
| Daily loss kill-switch | 3% |
| Stop | 2 × ATR below entry (stop-market, GTC) |
| Take-profit | min(3 × ATR, 8% above entry) |
| Time stop | 7 trading days |
| Minimum R/R | 1.0 |

Orders go in as GTC brackets — entry limit, stop-market child, take-profit limit child — so exits survive overnight and weekends. Stop-market rather than stop-limit, deliberately: a stop-limit won't fill on a gap-down, which is exactly when protection matters.

Other guards: long-only (shorts explicitly blocked), a 3% staleness check that skips orders when price has drifted since scoring, a 7-day cooldown on recently exited tickers to force fresh names, and position-limit accounting that counts pending buy orders but ignores sell-side bracket legs.

---

## Dashboard

Served at the deployment's root URL. Shows aggregate stats (total trades, win rate, total P&L, avg R/R, avg win/loss), open positions with entry/stop/target and the four module scores, and a closed-trade ledger with P&L, realized R/R, and exit reason (TP / SL / time stop). Auto-refreshes every 30s.

---

## Backtesting

`backtest.py` simulates the strategy on historical Alpaca bars with point-in-time SEC fundamentals (filings filtered by filing date to avoid lookahead). Sentiment and options are fixed at neutral 50 — no free historical source exists for either, which is a real limitation.

Parameter tuning across six runs moved Sharpe from −0.06 to ~2.6. **That number should be treated as inflated by curve-fitting**, since the parameters were adjusted repeatedly against the same sample.

---

## Findings

69 closed trades, late March – early August 2026:

```
Win rate        47.8%
Total P&L       +$1,729.93  (+1.73% on $100k)
Avg win         +$351.61
Avg loss        -$274.25
Payoff ratio    1.28
```

A rigorous post-mortem on the first 40 trades reached an uncomfortable conclusion:

- **No statistically demonstrable edge.** Expectancy +$11/trade, t ≈ 0.20, p ≈ 0.84. Profit factor 1.07. Indistinguishable from zero.
- **Composite score did not predict outcomes.** Bucketed by score, the 80–85 band — the highest-conviction trades — *lost* money, while 75–79 carried all the profit. No monotonic relationship.
- **Massive benchmark underperformance.** SPY returned ~+19% over the same window. Equal-weight ~+10%. T-bills ~+0.7%. The bot returned less than cash while taking equity risk.
- **Long-only risk-off logic backfired.** It de-risked into the March/April geopolitical bottom and under-participated in the recovery rally — the textbook momentum-crash failure mode.
- **Sector concentration hurt.** 11 energy trades netted roughly −$1,000 at a 27% win rate; the rest of the book made ~+$1,455.

Known bugs at time of pause: exit detection and entry logging both dropped trades, the log file was written in a container unreachable from the dev workspace, and buy volume decayed monotonically month over month (57 → 23 → 18 → 12 → 12 → 0) before stopping entirely.

---

## What's next

The project is paused on feature work and moved to signal validation. The literature is unambiguous that oscillator and pattern rules — which make up most of the technical module — largely don't survive data-snooping correction, while proximity to the 52-week high (George & Hwang, 2004) has independent predictive power that replicates internationally.

The next step is a single falsifiable test: *do stocks in the top quintile of price-to-52-week-high outperform the bottom quintile over 5 / 21 / 63 / 126 trading days?* Train on 2019–2023, hold out 2024–2026 untouched, benchmark against random baskets, SPY, and a size/volatility-matched basket. Pre-committed success criteria; accept the null if it fails.

Build nothing on a signal until it's been validated. That lesson is the main output of the project so far.

---

## Setup

```bash
pip install -r requirements.txt
python3 -u main.py
```

Environment variables: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`. All data sources are free and require no additional keys.

---

*Paper trading only. Not investment advice.*
