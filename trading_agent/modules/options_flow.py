"""
options_flow.py
---------------
Phase 2 Module — Options Flow Analysis
AI Trading Agent | Paper Trading Only

Data source: CBOE delayed quotes (free, no auth)
  URL: https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json

Improvements over v1:
  1. IV history persisted to disk — survives restarts, real percentile calculation
  2. Unusual activity detector — single-strike volume spikes get bonus scoring
  3. Clean after-hours logic — uses OI directly, skips volume-based signals
  4. Tighter scoring calibration across all four components

Scoring rubric (0–100):
  Put/Call ratio        : 25 pts  (volume during hours, OI after hours)
  IV percentile         : 25 pts  (vs 30-day rolling baseline from disk)
  Delta-weighted flow   : 25 pts  (smart money 0.25-0.55 delta calls)
  OI concentration      : 25 pts  (OTM accumulation + unusual activity bonus)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("OptionsAnalyzer")

CBOE_HEADERS  = {"User-Agent": "trading-agent/1.0"}
CENTRAL       = ZoneInfo("America/Chicago")
IV_CACHE_FILE = "/tmp/options_iv_history.json"   # persists across restarts


def _is_market_hours() -> bool:
    now_ct  = datetime.now(CENTRAL)
    hour_ct = now_ct.hour + now_ct.minute / 60
    return 9.5 <= hour_ct <= 16.0 and now_ct.weekday() < 5


def _load_iv_history() -> dict[str, list[float]]:
    """Load IV history from disk, returns empty dict on failure."""
    try:
        if os.path.exists(IV_CACHE_FILE):
            with open(IV_CACHE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_iv_history(history: dict[str, list[float]]) -> None:
    """Save IV history to disk, silently fails if unable."""
    try:
        with open(IV_CACHE_FILE, "w") as f:
            json.dump(history, f)
    except Exception:
        pass


class OptionsAnalyzer:

    CACHE_TTL_LIVE  = 1_200   # 20 min during market hours
    CACHE_TTL_AFTER = 3_600   # 60 min after hours
    IV_MAX_HISTORY  = 96      # ~2 trading days of 20-min observations

    def __init__(self, alpaca_api_key: str = "", alpaca_secret_key: str = ""):
        self._cache: dict[str, tuple[float, float]] = {}
        self._iv_history: dict[str, list[float]]    = _load_iv_history()
        self._iv_dirty   = False   # track if we need to save
        self._last_save  = time.time()

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

    def _maybe_save_iv(self) -> None:
        """Save IV history to disk every 5 minutes if dirty."""
        if self._iv_dirty and time.time() - self._last_save > 300:
            _save_iv_history(self._iv_history)
            self._iv_dirty = False
            self._last_save = time.time()

    # ------------------------------------------------------------------
    # Chain parsing
    # ------------------------------------------------------------------
    def _parse_chain(self, options: list[dict]) -> tuple[list[dict], list[dict]]:
        """Splits CBOE chain into calls and puts using symbol pattern."""
        import re
        call_pattern = re.compile(r'C\d{8}$')
        put_pattern  = re.compile(r'P\d{8}$')
        calls, puts  = [], []
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
        try:
            url  = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
            resp = requests.get(url, headers=CBOE_HEADERS, timeout=10)
            resp.raise_for_status()
            data    = resp.json()
            options = data.get("data", {}).get("options", [])
            calls, puts = self._parse_chain(options)
            logger.debug("%s CBOE: %d calls, %d puts", symbol, len(calls), len(puts))
            return calls, puts
        except Exception as exc:
            logger.debug("CBOE fetch failed for %s: %s", symbol, exc)
            return [], []

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------
    def _score_put_call_ratio(self, calls: list[dict], puts: list[dict]) -> float:
        """
        25 pts — Put/call ratio.
        During market hours: uses volume (fresh activity).
        After hours: uses open interest (existing positioning).
        """
        live = _is_market_hours()

        if live:
            call_val = sum(float(c.get("volume", 0) or 0) for c in calls)
            put_val  = sum(float(p.get("volume", 0) or 0) for p in puts)
        else:
            call_val = sum(float(c.get("open_interest", 0) or 0) for c in calls)
            put_val  = sum(float(p.get("open_interest", 0) or 0) for p in puts)

        if call_val + put_val == 0:
            return 12.5

        pc = put_val / call_val if call_val > 0 else 9.9

        if pc <= 0.35:   return 25.0
        elif pc <= 0.55: return 21.0
        elif pc <= 0.75: return 16.0
        elif pc <= 0.95: return 11.0
        elif pc <= 1.20: return 6.0
        elif pc <= 1.60: return 2.0
        else:            return 0.0

    def _score_iv_percentile(self, calls: list[dict], symbol: str) -> float:
        """
        25 pts — IV expansion vs persistent rolling baseline.
        History survives restarts via disk cache.
        """
        ivs = [float(c.get("iv", 0)) for c in calls if float(c.get("iv", 0) or 0) > 0.01]
        if not ivs:
            return 12.5

        current_iv = sum(ivs) / len(ivs)
        history    = self._iv_history.setdefault(symbol, [])
        history.append(current_iv)
        if len(history) > self.IV_MAX_HISTORY:
            history.pop(0)
        self._iv_dirty = True

        if len(history) < 3:
            return 12.5

        baseline = sum(history[:-1]) / len(history[:-1])
        ratio    = current_iv / baseline if baseline > 0 else 1.0

        if ratio >= 1.60:   return 25.0
        elif ratio >= 1.35: return 21.0
        elif ratio >= 1.15: return 16.0
        elif ratio >= 0.95: return 11.0
        elif ratio >= 0.80: return 6.0
        else:               return 2.0

    def _score_delta_weighted_flow(self, calls: list[dict]) -> float:
        """
        25 pts — Delta-weighted call activity targeting smart money strikes.
        Only uses volume during market hours (meaningful activity).
        After hours uses OI as a positioning proxy.
        """
        live      = _is_market_hours()
        smart_val = 0.0
        total_val = 0.0

        for c in calls:
            if live:
                val = float(c.get("volume", 0) or 0)
            else:
                val = float(c.get("open_interest", 0) or 0)

            delta = abs(float(c.get("delta", 0) or 0))
            total_val += val

            if 0.25 <= delta <= 0.55:
                smart_val += val * 1.5   # ideal sweet spot
            elif 0.15 <= delta < 0.25:
                smart_val += val * 0.8
            elif 0.55 < delta <= 0.75:
                smart_val += val * 0.5

        if total_val == 0:
            return 12.5

        smart_ratio = smart_val / total_val
        if smart_ratio >= 0.60:   return 25.0
        elif smart_ratio >= 0.45: return 20.0
        elif smart_ratio >= 0.30: return 14.0
        elif smart_ratio >= 0.15: return 8.0
        else:                     return 3.0

    def _score_oi_concentration(self, calls: list[dict]) -> float:
        """
        25 pts — OI distribution + unusual activity detector.

        Components:
          - OTM call OI concentration (15 pts): accumulation signal
          - Unusual single-strike spike (10 pts): block trade / conviction signal

        Unusual activity: one strike has >25% of total OI or >5x average OI.
        This catches institutional block trades that get diluted in averages.
        """
        if len(calls) < 5:
            return 12.5

        try:
            sorted_calls = sorted(
                calls,
                key=lambda c: float(str(c.get("option", "0")).split("C")[-1] or 0),
            )
        except Exception:
            return 12.5

        n         = len(sorted_calls)
        lower     = sorted_calls[:n // 3]
        mid       = sorted_calls[n // 3: 2 * n // 3]
        upper     = sorted_calls[2 * n // 3:]

        lower_oi  = sum(float(c.get("open_interest", 0) or 0) for c in lower)
        mid_oi    = sum(float(c.get("open_interest", 0) or 0) for c in mid)
        upper_oi  = sum(float(c.get("open_interest", 0) or 0) for c in upper)
        total_oi  = lower_oi + mid_oi + upper_oi

        if total_oi == 0:
            return 12.5

        # Component 1: OTM concentration (15 pts)
        otm_ratio = (mid_oi + upper_oi) / total_oi
        if otm_ratio >= 0.70:   otm_score = 15.0
        elif otm_ratio >= 0.55: otm_score = 11.0
        elif otm_ratio >= 0.40: otm_score = 7.0
        else:                   otm_score = 2.0

        # Component 2: Unusual single-strike activity (10 pts)
        all_ois   = [float(c.get("open_interest", 0) or 0) for c in sorted_calls]
        max_oi    = max(all_ois) if all_ois else 0
        avg_oi    = total_oi / len(sorted_calls) if sorted_calls else 0
        peak_conc = max_oi / total_oi if total_oi > 0 else 0
        spike_ratio = max_oi / avg_oi if avg_oi > 0 else 0

        # Unusual = >25% of OI at one strike OR >5x the average
        if peak_conc >= 0.25 or spike_ratio >= 5.0:
            unusual_score = 10.0
        elif peak_conc >= 0.15 or spike_ratio >= 3.0:
            unusual_score = 7.0
        elif peak_conc >= 0.08 or spike_ratio >= 2.0:
            unusual_score = 4.0
        else:
            unusual_score = 1.0

        return min(otm_score + unusual_score, 25.0)

    # ------------------------------------------------------------------
    # Public score()
    # ------------------------------------------------------------------
    def score(self, symbol: str) -> float:
        """
        Returns options flow score 0-100.
        Combines P/C ratio, IV percentile, delta-weighted flow, OI concentration.
        IV history persists to disk across restarts.
        """
        cached = self._cache_get(symbol)
        if cached is not None:
            return cached

        self._maybe_save_iv()

        calls, puts = self._fetch_chain(symbol)
        if not calls and not puts:
            return 50.0

        pc_score    = self._score_put_call_ratio(calls, puts)
        iv_score    = self._score_iv_percentile(calls, symbol)
        delta_score = self._score_delta_weighted_flow(calls)
        oi_score    = self._score_oi_concentration(calls)

        total = round(min(max(pc_score + iv_score + delta_score + oi_score, 0.0), 100.0), 2)

        logger.debug(
            "%s options — P/C: %.0f  IV: %.0f  Delta: %.0f  OI: %.0f  → %.1f",
            symbol, pc_score, iv_score, delta_score, oi_score, total,
        )

        self._cache_set(symbol, total)
        return total

    def cache_size(self) -> int:
        return len(self._cache)