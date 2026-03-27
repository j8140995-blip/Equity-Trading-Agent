"""
macro_context.py
----------------
Macro Context Module — AI Trading Agent | Paper Trading Only

Provides market regime awareness that individual stock modules lack.
Runs once per scan cycle and returns regime flags used to:
  - Adjust composite threshold (tighter in risk-off)
  - Apply sector multipliers (boost leaders, penalize laggards)
  - Block entries on high-volatility macro event days
  - Reduce position sizing in elevated VIX environments

Data sources (all free):
  - Alpaca snapshots: SPY, QQQ, VIX proxy (VIXY), sector ETFs
  - FRED API: not needed — all signals derivable from price data

Regime classifications:
  RISK_ON   : SPY above 200d, VIX < 20, broad uptrend
  NEUTRAL   : Mixed signals
  RISK_OFF  : SPY below 200d OR VIX > 25
  CRISIS    : VIX > 35 — pause all new entries
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger("MacroContext")

# Sector ETFs mapped to sector names
SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financial",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrial",
    "XLY":  "Consumer_Discretionary",
    "XLP":  "Consumer_Staples",
    "XLU":  "Utilities",
    "XLRE": "Real_Estate",
    "XLB":  "Materials",
    "XLC":  "Communication",
}

# High-volatility macro event dates (FOMC, CPI, NFP) — update monthly
# Format: "YYYY-MM-DD"
MACRO_EVENT_DATES: set[str] = {
    "2026-03-19",  # FOMC
    "2026-04-01",  # NFP
    "2026-04-10",  # CPI
    "2026-04-29",  # FOMC
    "2026-05-07",  # NFP
    "2026-05-13",  # CPI
    "2026-06-10",  # CPI
    "2026-06-17",  # FOMC
}


@dataclass
class MacroRegime:
    """Current macro regime state."""
    regime: str = "NEUTRAL"          # RISK_ON, NEUTRAL, RISK_OFF, CRISIS
    spy_above_200d: bool = True
    qqq_above_200d: bool = True
    vix_level: float = 20.0
    is_macro_event_day: bool = False
    threshold_adjustment: float = 0.0   # added to COMPOSITE_MIN
    position_size_multiplier: float = 1.0
    sector_scores: dict[str, float] = field(default_factory=dict)  # sector -> relative strength
    leading_sectors: list[str] = field(default_factory=list)
    lagging_sectors: list[str] = field(default_factory=list)
    summary: str = ""


class MacroContextAnalyzer:
    """
    Analyzes macro regime once per cycle.
    Cached for 30 minutes — regime doesn't change tick by tick.
    """

    CACHE_TTL = 1_800  # 30 minutes

    def __init__(self, alpaca_key: str, alpaca_secret: str):
        self.key    = alpaca_key
        self.secret = alpaca_secret
        self._cache: Optional[tuple[float, MacroRegime]] = None

    def _get_bars(self, symbols: list[str], limit: int = 210) -> dict[str, list[dict]]:
        """Fetch daily bars for multiple symbols from Alpaca."""
        from datetime import date, timedelta
        end   = date.today().isoformat()
        start = (date.today() - timedelta(days=365)).isoformat()
        all_bars: dict[str, list[dict]] = {}
        # Fetch in batches of 5 to avoid response size limits
        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]
            try:
                resp = requests.get(
                    "https://data.alpaca.markets/v2/stocks/bars",
                    headers={
                        "APCA-API-KEY-ID":     self.key,
                        "APCA-API-SECRET-KEY": self.secret,
                    },
                    params={
                        "symbols":    ",".join(batch),
                        "timeframe":  "1Day",
                        "start":      start,
                        "end":        end,
                        "limit":      300,
                        "feed":       "iex",
                        "adjustment": "split",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                all_bars.update(resp.json().get("bars", {}))
            except Exception as exc:
                logger.warning("Macro bars fetch failed for %s: %s", batch, exc)
        return all_bars

    def _ema(self, prices: list[float], window: int) -> float:
        """Simple EMA calculation."""
        if len(prices) < window:
            return prices[-1] if prices else 0
        k = 2 / (window + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def _fetch_vix(self) -> float:
        """Fetch current VIX level from Yahoo Finance. Falls back to 20 on error."""
        try:
            resp = requests.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
                headers={"User-Agent": "trading-agent/1.0"},
                params={"interval": "1d", "range": "5d"},
                timeout=8,
            )
            resp.raise_for_status()
            data   = resp.json()
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            vix    = next((v for v in reversed(closes) if v is not None), 20.0)
            logger.info("VIX: %.1f (live)", vix)
            return round(float(vix), 1)
        except Exception as exc:
            logger.warning("VIX fetch failed: %s — using fallback", exc)
            return self._vix_from_spy_drawdown()

    def _vix_from_spy_drawdown(self) -> float:
        """Fallback: estimate VIX from SPY's drawdown from 20-day high."""
        try:
            resp = requests.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
                headers={"User-Agent": "trading-agent/1.0"},
                params={"interval": "1d", "range": "1mo"},
                timeout=8,
            )
            resp.raise_for_status()
            closes  = resp.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes  = [c for c in closes if c is not None]
            if len(closes) < 5:
                return 20.0
            high_20 = max(closes[-20:]) if len(closes) >= 20 else max(closes)
            current = closes[-1]
            drawdown = (high_20 - current) / high_20
            if drawdown >= 0.12:  return 35.0
            elif drawdown >= 0.08: return 28.0
            elif drawdown >= 0.05: return 24.0
            elif drawdown >= 0.03: return 21.0
            else:                  return 17.0
        except Exception:
            return 20.0

    def _fetch_market_news_sentiment(self) -> float:
        """
        Fetch recent market news headlines and score sentiment.
        Returns a score from -1 (very bearish) to +1 (very bullish).
        Used as an additional regime signal alongside price-based indicators.
        """
        BEARISH_TERMS = ["recession", "tariff", "crash", "crisis", "war", "inflation",
                         "selloff", "plunge", "collapse", "fear", "deficit", "downgrade"]
        BULLISH_TERMS = ["rally", "surge", "beat", "growth", "recovery", "record",
                         "upgrade", "boom", "breakout", "strong", "optimism"]
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={"q": "stock market S&P 500", "hl": "en-US", "gl": "US", "ceid": "US:en"},
                timeout=6,
            )
            resp.raise_for_status()
            import xml.etree.ElementTree as ET
            root  = ET.fromstring(resp.text)
            items = root.findall(".//item")[:10]
            bull  = 0
            bear  = 0
            for item in items:
                text = (item.findtext("title") or "").lower()
                bull += sum(1 for t in BULLISH_TERMS if t in text)
                bear += sum(1 for t in BEARISH_TERMS if t in text)
            total = bull + bear
            if total == 0:
                return 0.0
            return round((bull - bear) / total, 2)
        except Exception:
            return 0.0

    def _sector_relative_strength(self, bars: dict[str, list[dict]]) -> dict[str, float]:
        """
        Calculate 20-day relative strength for each sector ETF vs SPY.
        Returns dict of sector_name -> RS score (-1 to +1).
        """
        spy_bars = bars.get("SPY", [])
        if len(spy_bars) < 21:
            return {}

        spy_return = (spy_bars[-1]["c"] - spy_bars[-21]["c"]) / spy_bars[-21]["c"]
        rs_scores  = {}

        for etf, sector in SECTOR_ETFS.items():
            etf_bars = bars.get(etf, [])
            if len(etf_bars) < 21:
                continue
            etf_return = (etf_bars[-1]["c"] - etf_bars[-21]["c"]) / etf_bars[-21]["c"]
            rs_scores[sector] = round(etf_return - spy_return, 4)

        return rs_scores

    def analyze(self) -> MacroRegime:
        """
        Returns current macro regime. Cached 30 minutes.
        """
        if self._cache:
            ts, regime = self._cache
            if time.time() - ts < self.CACHE_TTL:
                return regime

        regime = self._compute_regime()
        self._cache = (time.time(), regime)
        return regime

    def _compute_regime(self) -> MacroRegime:
        regime = MacroRegime()

        # Check if today is a macro event day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        regime.is_macro_event_day = today in MACRO_EVENT_DATES

        # Fetch all needed symbols
        all_symbols = ["SPY", "QQQ", "VIXY"] + list(SECTOR_ETFS.keys())
        bars = self._get_bars(all_symbols, limit=210)

        if not bars:
            logger.warning("No macro data available — defaulting to NEUTRAL")
            regime.summary = "No data — NEUTRAL default"
            return regime

        # SPY 200-day EMA
        spy_bars = bars.get("SPY", [])
        if len(spy_bars) >= 200:
            spy_closes  = [b["c"] for b in spy_bars]
            spy_ema200  = self._ema(spy_closes, 200)
            spy_current = spy_closes[-1]
            regime.spy_above_200d = spy_current > spy_ema200
        else:
            regime.spy_above_200d = True  # assume bullish if insufficient data

        # QQQ 200-day EMA
        qqq_bars = bars.get("QQQ", [])
        if len(qqq_bars) >= 200:
            qqq_closes  = [b["c"] for b in qqq_bars]
            qqq_ema200  = self._ema(qqq_closes, 200)
            regime.qqq_above_200d = qqq_closes[-1] > qqq_ema200
        else:
            regime.qqq_above_200d = True

        # VIX: fetch real VIX level from Yahoo Finance (free, no auth)
        regime.vix_level = self._fetch_vix()

        # Sector relative strength
        rs = self._sector_relative_strength(bars)
        regime.sector_scores = rs

        # Leading = top 3 sectors outperforming SPY
        # Lagging = bottom 3 sectors underperforming SPY
        sorted_sectors = sorted(rs.items(), key=lambda x: x[1], reverse=True)
        regime.leading_sectors = [s for s, v in sorted_sectors[:3] if v > 0]
        regime.lagging_sectors = [s for s, v in sorted_sectors[-3:] if v < 0]

        # Market news sentiment
        news_sentiment = self._fetch_market_news_sentiment()
        logger.info("Market news sentiment: %.2f", news_sentiment)

        # Classify regime
        if regime.vix_level > 35:
            regime.regime = "CRISIS"
            regime.threshold_adjustment = +10.0   # raise threshold by 10
            regime.position_size_multiplier = 0.5  # half position sizes
        elif not regime.spy_above_200d or regime.vix_level > 25:
            regime.regime = "RISK_OFF"
            regime.threshold_adjustment = +5.0
            regime.position_size_multiplier = 0.75
        elif regime.spy_above_200d and regime.qqq_above_200d and regime.vix_level < 18 and news_sentiment >= -0.2:
            regime.regime = "RISK_ON"
            regime.threshold_adjustment = -2.0
            regime.position_size_multiplier = 1.0
        else:
            regime.regime = "NEUTRAL"
            # News sentiment nudges threshold: very bearish news adds 2pts
            regime.threshold_adjustment = -2.0 if news_sentiment > 0.3 else (2.0 if news_sentiment < -0.3 else 0.0)
            regime.position_size_multiplier = 1.0

        # Build summary
        event_note = " | ⚠ MACRO EVENT DAY" if regime.is_macro_event_day else ""
        news_note  = f" | News: {'+' if news_sentiment > 0 else ''}{news_sentiment:.2f}"
        regime.summary = (
            f"Regime: {regime.regime} | SPY>200d: {regime.spy_above_200d} | "
            f"VIX~{regime.vix_level:.0f} | "
            f"Leading: {','.join(regime.leading_sectors) or 'none'} | "
            f"Lagging: {','.join(regime.lagging_sectors) or 'none'}"
            f"{news_note}{event_note}"
        )

        logger.info("Macro: %s", regime.summary)
        return regime

    def get_sector_bonus(self, sector: str, regime: MacroRegime) -> float:
        """
        Returns composite score adjustment for a stock based on its sector.
        Leading sectors: +3 pts. Lagging sectors: -5 pts.
        """
        if sector in regime.leading_sectors:
            return 3.0
        if sector in regime.lagging_sectors:
            return -5.0
        return 0.0