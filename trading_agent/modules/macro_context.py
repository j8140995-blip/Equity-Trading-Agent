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
        try:
            resp = requests.get(
                "https://data.alpaca.markets/v2/stocks/bars",
                headers={
                    "APCA-API-KEY-ID":     self.key,
                    "APCA-API-SECRET-KEY": self.secret,
                },
                params={
                    "symbols":   ",".join(symbols),
                    "timeframe": "1Day",
                    "limit":     limit,
                    "feed":      "iex",
                    "adjustment": "split",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("bars", {})
        except Exception as exc:
            logger.warning("Macro bars fetch failed: %s", exc)
            return {}

    def _ema(self, prices: list[float], window: int) -> float:
        """Simple EMA calculation."""
        if len(prices) < window:
            return prices[-1] if prices else 0
        k = 2 / (window + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

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

        # VIX proxy via VIXY (ProShares VIX Short-Term Futures ETF)
        # VIXY moves with VIX — use it as a fear gauge
        vixy_bars = bars.get("VIXY", [])
        if len(vixy_bars) >= 20:
            vixy_closes   = [b["c"] for b in vixy_bars]
            vixy_current  = vixy_closes[-1]
            vixy_20d_avg  = sum(vixy_closes[-20:]) / 20
            # Normalize to approximate VIX level (VIXY ~= VIX/4 roughly)
            regime.vix_level = round(vixy_current * 4, 1)
        else:
            regime.vix_level = 20.0  # assume normal if no data

        # Sector relative strength
        rs = self._sector_relative_strength(bars)
        regime.sector_scores = rs

        # Leading = top 3 sectors outperforming SPY
        # Lagging = bottom 3 sectors underperforming SPY
        sorted_sectors = sorted(rs.items(), key=lambda x: x[1], reverse=True)
        regime.leading_sectors = [s for s, v in sorted_sectors[:3] if v > 0]
        regime.lagging_sectors = [s for s, v in sorted_sectors[-3:] if v < 0]

        # Classify regime
        if regime.vix_level > 35:
            regime.regime = "CRISIS"
            regime.threshold_adjustment = +10.0   # raise threshold by 10
            regime.position_size_multiplier = 0.5  # half position sizes
        elif not regime.spy_above_200d or regime.vix_level > 25:
            regime.regime = "RISK_OFF"
            regime.threshold_adjustment = +5.0
            regime.position_size_multiplier = 0.75
        elif regime.spy_above_200d and regime.qqq_above_200d and regime.vix_level < 18:
            regime.regime = "RISK_ON"
            regime.threshold_adjustment = -2.0    # slightly lower threshold
            regime.position_size_multiplier = 1.0
        else:
            regime.regime = "NEUTRAL"
            regime.threshold_adjustment = 0.0
            regime.position_size_multiplier = 1.0

        # Build summary
        event_note = " | ⚠ MACRO EVENT DAY" if regime.is_macro_event_day else ""
        regime.summary = (
            f"Regime: {regime.regime} | SPY>200d: {regime.spy_above_200d} | "
            f"VIX~{regime.vix_level:.0f} | "
            f"Leading: {','.join(regime.leading_sectors) or 'none'} | "
            f"Lagging: {','.join(regime.lagging_sectors) or 'none'}"
            f"{event_note}"
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