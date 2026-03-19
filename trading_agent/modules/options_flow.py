"""
options_flow.py
---------------
Phase 2 Module — Options Flow Analysis
AI Trading Agent | Paper Trading Only

Data source: CBOE delayed quotes (free, no auth, 3,376+ contracts per ticker)
  URL: https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json

  Fields available per contract:
    bid, ask, iv, open_interest, volume, delta, gamma, vega, theta,
    theo, change, percent_change, prev_day_close, last_trade_price,
    last_trade_time, tick

  Note: During market hours, volume/OI/IV are live delayed.
        After hours, volume=0 but bid/ask/greeks/OI remain valid.

Scoring rubric (0–100):
  Put/Call volume ratio   : 25 pts  (low P/C = bullish bias)
  IV percentile           : 25 pts  (expanding IV = directional move coming)
  Delta-weighted flow     : 25 pts  (smart money targets 0.3–0.5 delta calls)
  OI concentration (OTM)  : 25 pts  (rising OTM OI = accumulation)

Cache TTL: 20 minutes during market hours, 60 minutes after hours.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("OptionsAnalyzer")

CBOE_HEADERS = {"User-Agent": "trading-agent/1.0"}
CENTRAL       = ZoneInfo("America/Chicago")


def _is_market_hours() -> bool:
    now_ct  = datetime.now(CENTRAL)
    hour_ct = now_ct.hour + now_ct.minute / 60
    return 9.5 <= hour_ct <= 16.0 and now_ct.weekday() < 5


class OptionsAnalyzer:
    """
    Scores options flow for a ticker using CBOE's full chain data.

    Uses greeks directly from CBOE — no estimation needed:
      - delta  : probability proxy and moneyness indicator
      - iv     : implied volatility per contract
      - vega   : IV sensitivity (high vega = big move expected)
      - volume : fresh positioning vs existing OI
    """

    CACHE_TTL_LIVE  = 1_200   # 20 min during market hours
    CACHE_TTL_AFTER = 3_600   # 60 min after hours

    def __init__(self, alpaca_api_key: str = "", alpaca_secret_key: str = ""):
        # Alpaca keys retained for API compatibility but CBOE needs no auth
        self._cache: dict[str, tuple[float, float]]   = {}
        self._iv_history: dict[str, list[float]]       = {}   # rolling IV baseline

    def _cache_ttl(self) -> int:
        return self.CACHE_TTL_LIVE if _is_market_hours() else self.CACHE_TTL_AFTER

    def _cache_get(self, symbol: str) -> Optional[float]:
        if symbol in self._cache:
            ts, score = self._cache[symbol]
            if time.time() - ts < self._cache_ttl():
                return score
        return None

    def _cache_set(self, symbol: str, score: float) -> None:
        self._cache[symbol] = (time.time(), score)

    def _parse_chain(self, options: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Correctly splits options into calls and puts.

        Option symbol format: {TICKER}{YYMMDD}{C|P}{8-digit-strike}
        e.g. AAPL260318C00195000

        Simply checking 'C' or 'P' in the full string fails for tickers
        that contain those letters (AAPL has P, SPY has P, etc.).
        Instead we find C or P immediately before the 8-digit strike suffix.
        """
        import re
        call_pattern = re.compile(r'C\d{8}$')
        put_pattern  = re.compile(r'P\d{8}$')

        calls, puts = [], []
        for o in options:
            sym = str(o.get("option", ""))
            if call_pattern.search(sym):
                calls.append(o)
            elif put_pattern.search(sym):
                puts.append(o)
        return calls, puts

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------
    def _fetch_chain(self, symbol: str) -> tuple[list[dict], list[dict]]:
        """
        Fetches CBOE options chain and returns (calls, puts) as separate lists.
        Each contract dict has: option, bid, ask, iv, open_interest, volume,
        delta, gamma, vega, theta, theo, prev_day_close, last_trade_price.
        """
        try:
            url  = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
            resp = requests.get(url, headers=CBOE_HEADERS, timeout=15)
            resp.raise_for_status()
            data    = resp.json()
            options = data.get("data", {}).get("options", [])

            calls, puts = self._parse_chain(options)
            logger.debug(
                "%s CBOE chain: %d calls, %d puts (ts: %s)",
                symbol, len(calls), len(puts),
                data.get("timestamp", "?"),
            )
            return calls, puts

        except Exception as exc:
            logger.debug("CBOE fetch failed for %s: %s", symbol, exc)
            return [], []

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------
    def _score_put_call_ratio(
        self, calls: list[dict], puts: list[dict]
    ) -> float:
        """
        25 pts — Put/call volume ratio.
        After hours falls back to OI ratio since volume = 0.
        Low P/C = bullish sentiment.
        """
        call_vol = sum(float(c.get("volume", 0) or 0) for c in calls)
        put_vol  = sum(float(p.get("volume", 0) or 0) for p in puts)

        # After hours: use open interest as proxy
        if call_vol + put_vol == 0:
            call_vol = sum(float(c.get("open_interest", 0) or 0) for c in calls)
            put_vol  = sum(float(p.get("open_interest", 0) or 0) for p in puts)

        if call_vol + put_vol == 0:
            return 12.5   # no data — neutral

        pc = put_vol / call_vol if call_vol > 0 else 9.9

        if pc <= 0.35:   return 25.0
        elif pc <= 0.55: return 21.0
        elif pc <= 0.75: return 16.0
        elif pc <= 0.95: return 11.0
        elif pc <= 1.20: return 6.0
        elif pc <= 1.60: return 2.0
        else:            return 0.0

    def _score_iv_percentile(
        self, calls: list[dict], symbol: str
    ) -> float:
        """
        25 pts — IV expansion vs rolling baseline.
        Uses CBOE's per-contract IV field directly.
        Expanding IV signals that the market expects a significant move,
        which is when momentum trades have the best R/R.
        """
        ivs = [
            float(c.get("iv", 0))
            for c in calls
            if float(c.get("iv", 0) or 0) > 0.01
        ]
        if not ivs:
            return 12.5

        current_iv = sum(ivs) / len(ivs)

        # Maintain rolling IV history (last 48 observations)
        history = self._iv_history.setdefault(symbol, [])
        history.append(current_iv)
        if len(history) > 48:
            history.pop(0)

        if len(history) < 3:
            return 12.5   # not enough history yet

        baseline = sum(history[:-1]) / len(history[:-1])
        ratio    = current_iv / baseline if baseline > 0 else 1.0

        if ratio >= 1.60:   return 25.0   # IV 60%+ above baseline
        elif ratio >= 1.35: return 21.0
        elif ratio >= 1.15: return 16.0
        elif ratio >= 0.95: return 11.0
        elif ratio >= 0.80: return 6.0
        else:               return 2.0    # IV collapsed — low conviction

    def _score_delta_weighted_flow(self, calls: list[dict]) -> float:
        """
        25 pts — Delta-weighted call volume targeting smart money strikes.

        Smart money (institutions) typically buy 0.25–0.50 delta calls:
          - Far enough OTM to be cheap (high leverage)
          - Close enough to benefit from a moderate move
          - 0.5 delta = ATM = highest gamma, most sensitive to price moves

        Deep ITM (delta > 0.8) = stock replacement, not directional bet.
        Far OTM (delta < 0.15) = lottery tickets, low conviction.

        We weight volume by how 'smart' the delta target is.
        """
        smart_vol = 0.0
        total_vol = 0.0

        for c in calls:
            vol   = float(c.get("volume", 0) or 0)
            delta = abs(float(c.get("delta", 0) or 0))

            # After hours fallback: use OI as volume proxy
            if vol == 0:
                vol = float(c.get("open_interest", 0) or 0) * 0.1

            total_vol += vol

            # Weight by delta quality (peak weight at delta ~0.40)
            if 0.25 <= delta <= 0.55:
                smart_vol += vol * 1.5   # ideal sweet spot
            elif 0.15 <= delta < 0.25:
                smart_vol += vol * 0.8   # acceptable OTM
            elif 0.55 < delta <= 0.75:
                smart_vol += vol * 0.5   # getting expensive
            # < 0.15 (lottery) or > 0.75 (stock replacement) get 0 weight

        if total_vol == 0:
            return 12.5

        smart_ratio = smart_vol / total_vol

        if smart_ratio >= 0.60:   return 25.0
        elif smart_ratio >= 0.45: return 20.0
        elif smart_ratio >= 0.30: return 14.0
        elif smart_ratio >= 0.15: return 8.0
        else:                     return 3.0

    def _score_oi_concentration(self, calls: list[dict]) -> float:
        """
        25 pts — Open interest distribution across strikes.

        Checks two things:
          1. Is OTM call OI growing? (accumulation signal)
          2. Is OI concentrated at a specific strike? (target price signal)

        High OI concentration at OTM strikes = the market has a price target.
        """
        if len(calls) < 5:
            return 12.5

        try:
            # Sort by strike ascending
            sorted_calls = sorted(
                calls,
                key=lambda c: float(
                    str(c.get("option", "0")).split("C")[-1] or 0
                ),
            )
        except Exception:
            return 12.5

        n     = len(sorted_calls)
        lower = sorted_calls[:n // 3]    # ITM/ATM
        mid   = sorted_calls[n // 3: 2 * n // 3]  # ATM/slightly OTM
        upper = sorted_calls[2 * n // 3:]  # OTM

        lower_oi = sum(float(c.get("open_interest", 0) or 0) for c in lower)
        mid_oi   = sum(float(c.get("open_interest", 0) or 0) for c in mid)
        upper_oi = sum(float(c.get("open_interest", 0) or 0) for c in upper)
        total_oi = lower_oi + mid_oi + upper_oi

        if total_oi == 0:
            return 12.5

        # OTM + mid OTM concentration
        otm_ratio = (mid_oi + upper_oi) / total_oi

        # Peak OI strike concentration (is there a clear target?)
        all_ois   = [float(c.get("open_interest", 0) or 0) for c in sorted_calls]
        max_oi    = max(all_ois) if all_ois else 0
        peak_conc = max_oi / total_oi if total_oi > 0 else 0

        score = 0.0
        # OTM concentration component (15 pts)
        if otm_ratio >= 0.70:   score += 15.0
        elif otm_ratio >= 0.55: score += 11.0
        elif otm_ratio >= 0.40: score += 7.0
        else:                   score += 2.0

        # Peak concentration component (10 pts) — clear target = conviction
        if peak_conc >= 0.25:   score += 10.0
        elif peak_conc >= 0.15: score += 7.0
        elif peak_conc >= 0.08: score += 4.0
        else:                   score += 1.0

        return min(score, 25.0)

    # ------------------------------------------------------------------
    # Public score() method
    # ------------------------------------------------------------------
    def score(self, symbol: str) -> float:
        """
        Returns an options flow score 0–100 using CBOE full chain data.
        Combines P/C ratio, IV expansion, delta-weighted flow, and OI concentration.
        Cache TTL: 20 min market hours, 60 min after hours.
        """
        cached = self._cache_get(symbol)
        if cached is not None:
            return cached

        calls, puts = self._fetch_chain(symbol)

        if not calls and not puts:
            logger.debug("No options chain for %s — returning neutral.", symbol)
            return 50.0

        pc_score    = self._score_put_call_ratio(calls, puts)
        iv_score    = self._score_iv_percentile(calls, symbol)
        delta_score = self._score_delta_weighted_flow(calls)
        oi_score    = self._score_oi_concentration(calls)

        total = round(
            min(max(pc_score + iv_score + delta_score + oi_score, 0.0), 100.0),
            2,
        )

        logger.debug(
            "%s options — P/C: %.0f  IV: %.0f  Delta: %.0f  OI: %.0f  → %.1f",
            symbol, pc_score, iv_score, delta_score, oi_score, total,
        )

        self._cache_set(symbol, total)
        return total

    def cache_size(self) -> int:
        return len(self._cache)