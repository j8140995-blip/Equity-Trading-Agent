"""
fundamental.py
--------------
Phase 2 Module — Fundamental Analysis
AI Trading Agent | Paper Trading Only

Uses httpx async client — genuinely cancellable by asyncio timeouts,
unlike requests which blocks OS threads that asyncio cannot cancel.

Data: SEC EDGAR targeted concept endpoints (50-100KB each).
Cache TTL: 24 hours.

Scoring rubric (0-100):
  EPS growth YoY       : 30 pts
  P/E vs sector avg    : 25 pts
  Debt-to-equity       : 25 pts
  Revenue acceleration : 20 pts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("FundamentalAnalyzer")

HEADERS        = {"User-Agent": "ai-trading-agent contact@trading-agent.local"}
FUND_CACHE_FILE = os.path.join(os.getcwd(), "fundamental_cache.json")
SEC_BASE = "https://data.sec.gov"
WWW_SEC  = "https://www.sec.gov"

from trading_agent.modules.sector_map import get_sector, SECTOR_PE


class FundamentalAnalyzer:

    CACHE_TTL = 86_400   # 24 hours

    def __init__(self, fmp_api_key: str = ""):
        self._cik_map: dict[str, str]                  = {}
        self._score_cache: dict[str, tuple[float, float]] = {}
        self._price_map: dict[str, float]              = {}
        self._load_cache_from_disk()

    def _load_cache_from_disk(self) -> None:
        """Load persisted fundamental scores from disk on startup."""
        try:
            if os.path.exists(FUND_CACHE_FILE):
                with open(FUND_CACHE_FILE) as f:
                    raw = json.load(f)
                now = time.time()
                loaded = 0
                for sym, (ts, score) in raw.items():
                    if now - ts < self.CACHE_TTL:   # only load non-expired
                        self._score_cache[sym] = (ts, score)
                        loaded += 1
                logger.info("Fundamental disk cache loaded: %d valid entries.", loaded)
        except Exception as exc:
            logger.debug("Fundamental cache load failed: %s", exc)

    def _save_cache_to_disk(self) -> None:
        """Persist current score cache to disk."""
        try:
            with open(FUND_CACHE_FILE, "w") as f:
                json.dump(self._score_cache, f)
        except Exception as exc:
            logger.debug("Fundamental cache save failed: %s", exc)

    # ------------------------------------------------------------------
    # Async HTTP — genuinely cancellable
    # ------------------------------------------------------------------
    async def _aget(self, url: str, client: httpx.AsyncClient) -> Optional[dict]:
        try:
            resp = await client.get(url, headers=HEADERS, timeout=8.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("SEC request failed (%s): %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # CIK map — async load
    # ------------------------------------------------------------------
    async def _load_cik_map(self, client: httpx.AsyncClient) -> None:
        if self._cik_map:
            return
        logger.info("Loading SEC ticker→CIK map …")
        data = await self._aget(f"{WWW_SEC}/files/company_tickers.json", client)
        if not data:
            return
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik    = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                self._cik_map[ticker] = cik
        logger.info("CIK map loaded: %d tickers.", len(self._cik_map))

    def preload_cik_map(self) -> None:
        """Sync wrapper for compatibility — runs async load in new event loop."""
        if not self._cik_map:
            asyncio.run(self._async_preload())

    async def _async_preload(self) -> None:
        async with httpx.AsyncClient() as client:
            await self._load_cik_map(client)

    # ------------------------------------------------------------------
    # Concept fetcher
    # ------------------------------------------------------------------
    async def _get_concept(
        self, cik: str, concept: str, unit: str, client: httpx.AsyncClient
    ) -> list[dict]:
        url  = f"{SEC_BASE}/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        data = await self._aget(url, client)
        if not data:
            return []
        try:
            entries = data.get("units", {}).get(unit, [])
            return sorted(entries, key=lambda x: x.get("end", ""), reverse=True)
        except Exception:
            return []

    async def _annual(self, cik: str, concept: str, unit: str, client: httpx.AsyncClient) -> list[float]:
        entries = await self._get_concept(cik, concept, unit, client)
        seen: dict[str, float] = {}
        for e in entries:
            if e.get("form") in ("10-K", "10-K/A") and e.get("end"):
                seen[e["end"]] = float(e.get("val", 0))
        return [v for _, v in sorted(seen.items(), reverse=True)]

    async def _quarterly(self, cik: str, concept: str, unit: str, client: httpx.AsyncClient) -> list[float]:
        entries = await self._get_concept(cik, concept, unit, client)
        seen: dict[str, float] = {}
        for e in entries:
            if e.get("form") in ("10-Q", "10-Q/A") and e.get("end"):
                seen[e["end"]] = float(e.get("val", 0))
        return [v for _, v in sorted(seen.items(), reverse=True)]

    # ------------------------------------------------------------------
    # Sub-scorers — all async
    # ------------------------------------------------------------------
    async def _score_eps(self, cik: str, client: httpx.AsyncClient) -> float:
        eps = await self._annual(cik, "EarningsPerShareBasic", "USD/shares", client)
        if len(eps) < 2:
            ni = await self._annual(cik, "NetIncomeLoss", "USD", client)
            if len(ni) < 2 or ni[1] == 0:
                return 15.0
            growth = ((ni[0] - ni[1]) / abs(ni[1])) * 100
        else:
            if eps[1] == 0:
                return 15.0
            growth = ((eps[0] - eps[1]) / abs(eps[1])) * 100

        if growth >= 30:    return 30.0
        elif growth >= 20:  return 25.0
        elif growth >= 10:  return 18.0
        elif growth >= 0:   return 10.0
        elif growth >= -10: return 4.0
        else:               return 0.0

    async def _score_pe(self, symbol: str, cik: str, client: httpx.AsyncClient) -> float:
        price = self._price_map.get(symbol, 0.0)
        if price <= 0:
            return 12.5

        eps_vals = await self._annual(cik, "EarningsPerShareBasic", "USD/shares", client)
        if eps_vals and eps_vals[0] > 0:
            eps = eps_vals[0]
        else:
            ni     = await self._annual(cik, "NetIncomeLoss", "USD", client)
            shares = await self._annual(cik, "CommonStockSharesOutstanding", "shares", client)
            if not ni or not shares or shares[0] == 0:
                return 12.5
            eps = ni[0] / shares[0]

        if eps <= 0:
            return 12.5

        pe        = price / eps
        sector    = get_sector(symbol)
        sector_pe = SECTOR_PE.get(sector, SECTOR_PE["Unknown"])
        ratio     = pe / sector_pe

        if ratio <= 0.70:   return 25.0
        elif ratio <= 0.85: return 20.0
        elif ratio <= 1.00: return 15.0
        elif ratio <= 1.20: return 8.0
        elif ratio <= 1.50: return 3.0
        else:               return 0.0

    async def _score_de(self, cik: str, client: httpx.AsyncClient) -> float:
        debt = await self._annual(cik, "LongTermDebt", "USD", client)
        if not debt:
            debt = await self._annual(cik, "LongTermDebtNoncurrent", "USD", client)
        equity = await self._annual(cik, "StockholdersEquity", "USD", client)
        if not equity:
            equity = await self._annual(cik, "RetainedEarningsAccumulatedDeficit", "USD", client)

        if not debt or not equity or equity[0] <= 0:
            return 12.5

        de = debt[0] / equity[0]
        if de < 0.25:   return 25.0
        elif de < 0.50: return 20.0
        elif de < 1.00: return 14.0
        elif de < 1.50: return 8.0
        elif de < 2.00: return 3.0
        else:           return 0.0

    async def _score_rev(self, cik: str, client: httpx.AsyncClient) -> float:
        revs = await self._quarterly(
            cik, "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", client
        )
        if len(revs) < 4:
            revs = await self._quarterly(cik, "Revenues", "USD", client)
        if len(revs) < 4:
            return 10.0

        growths = [
            (revs[i] - revs[i+1]) / abs(revs[i+1]) if revs[i+1] != 0 else 0
            for i in range(3)
        ]
        acc = all(growths[i] > growths[i+1] for i in range(2))
        pos = all(g > 0 for g in growths)

        if acc and pos: return 20.0
        elif pos:       return 14.0
        elif growths[0] > 0: return 8.0
        else:           return 2.0

    # ------------------------------------------------------------------
    # Public async score
    # ------------------------------------------------------------------
    async def ascore(self, symbol: str) -> float:
        """
        Native async score — genuinely cancellable by asyncio.timeout.
        All HTTP calls use httpx.AsyncClient, no blocking threads.
        """
        if symbol in self._score_cache:
            ts, sc = self._score_cache[symbol]
            if time.time() - ts < self.CACHE_TTL:
                return sc

        async with httpx.AsyncClient() as client:
            await self._load_cik_map(client)
            cik = self._cik_map.get(symbol)
            if not cik:
                return 50.0

            eps = await self._score_eps(cik, client)
            pe  = await self._score_pe(symbol, cik, client)
            de  = await self._score_de(cik, client)
            rev = await self._score_rev(cik, client)

        total = round(min(max(eps + pe + de + rev, 0.0), 100.0), 2)
        logger.debug(
            "%s fundamentals — EPS:%.0f P/E:%.0f D/E:%.0f Rev:%.0f → %.1f",
            symbol, eps, pe, de, rev, total,
        )
        self._score_cache[symbol] = (time.time(), total)
        self._save_cache_to_disk()
        return total

    def score(self, symbol: str) -> float:
        """Sync wrapper for backward compatibility."""
        if symbol in self._score_cache:
            ts, sc = self._score_cache[symbol]
            if time.time() - ts < self.CACHE_TTL:
                return sc
        return asyncio.run(self.ascore(symbol))

    def set_price(self, symbol: str, price: float) -> None:
        self._price_map[symbol] = price

    def quota_remaining(self) -> int:
        return 999

    def cache_size(self) -> int:
        return len(self._score_cache)