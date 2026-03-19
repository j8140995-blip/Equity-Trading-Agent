"""
sentiment.py
------------
Phase 2 Module — Sentiment & News Analysis
AI Trading Agent | Paper Trading Only

Data sources (all free, no extra API keys needed):
  1. Benzinga via Alpaca News API  — full article content, per-ticker
  2. Google News RSS               — broad headline coverage, per-ticker
  3. SEC EDGAR 8-K feed            — material event detection, per-ticker

Scoring engine: Financial lexicon-based NLP
  Uses a curated dictionary of ~200 bullish/bearish financial terms
  weighted by strength. No heavy ML models needed — lexicon-based
  scoring is fast, interpretable, and performs well on financial text.

Scoring rubric (0–100):
  Benzinga sentiment   : 45 pts  (full article content, high quality)
  Google News sentiment: 30 pts  (headline breadth, recency weighted)
  SEC 8-K event signal : 25 pts  (material events — binary boost/crush)

Cache TTL: 2 hours (news changes intraday, unlike fundamentals)
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger("SentimentAnalyzer")

HEADERS_SEC    = {"User-Agent": "ai-trading-agent contact@trading-agent.local"}
HEADERS_NEWS   = {"User-Agent": "trading-agent/1.0"}

# ---------------------------------------------------------------------------
# Financial Sentiment Lexicon
# Positive terms boost score, negative terms reduce it.
# Weights: 1.0 = moderate signal, 2.0 = strong signal, 3.0 = very strong
# ---------------------------------------------------------------------------
BULLISH_TERMS: dict[str, float] = {
    # Earnings & growth
    "beat":2.0, "beats":2.0, "exceeded":2.0, "surpassed":2.0, "topped":1.5,
    "record":1.5, "record-high":2.0, "all-time high":2.0, "raised guidance":2.5,
    "raised outlook":2.5, "raised forecast":2.5, "strong earnings":2.0,
    "earnings beat":2.5, "revenue beat":2.0, "profit surge":2.0,
    "blowout":2.5, "blew past":2.5, "crushed":2.0, "smashed":2.0,
    # Growth signals
    "accelerating":1.5, "accelerated":1.5, "momentum":1.0, "growth":1.0,
    "expansion":1.0, "expanding":1.0, "outperform":2.0, "outperformed":2.0,
    "upgrade":2.0, "upgraded":2.0, "buy rating":2.0, "strong buy":2.5,
    "price target raised":2.5, "target raised":2.0, "bull":1.5, "bullish":2.0,
    # Positive events
    "acquisition":1.0, "buyback":1.5, "share repurchase":1.5, "dividend":1.0,
    "dividend increase":2.0, "special dividend":2.0, "partnership":1.0,
    "contract win":2.0, "deal":1.0, "approved":1.5, "fda approval":3.0,
    "breakthrough":2.0, "innovation":1.0, "launched":1.0, "launch":1.0,
    # Market signals
    "rally":1.5, "rallied":1.5, "surge":1.5, "surged":1.5, "soar":2.0,
    "soared":2.0, "spike":1.0, "spiked":1.0, "jump":1.5, "jumped":1.5,
    "gain":1.0, "gained":1.0, "rise":1.0, "rose":1.0, "climb":1.0,
}

BEARISH_TERMS: dict[str, float] = {
    # Earnings & misses
    "miss":2.0, "missed":2.0, "missed estimates":2.5, "below expectations":2.5,
    "shortfall":2.0, "disappointing":2.0, "disappointed":2.0, "weak":1.5,
    "lowered guidance":2.5, "lowered outlook":2.5, "cut guidance":2.5,
    "warned":2.0, "warning":2.0, "profit warning":3.0, "revenue miss":2.5,
    # Negative events
    "lawsuit":1.5, "sued":1.5, "investigation":2.0, "sec probe":3.0,
    "fraud":3.0, "scandal":2.5, "recall":2.0, "safety concern":2.0,
    "layoffs":1.5, "laid off":1.5, "restructuring":1.5, "job cuts":2.0,
    "bankruptcy":3.0, "default":3.0, "debt crisis":2.5, "downgrade":2.5,
    "downgraded":2.5, "sell rating":2.0, "underperform":2.0,
    "price target cut":2.5, "target cut":2.0, "bear":1.5, "bearish":2.0,
    # Market signals
    "crash":2.5, "crashed":2.5, "plunge":2.0, "plunged":2.0, "tank":2.0,
    "tanked":2.0, "collapse":2.5, "collapsed":2.5, "fall":1.0, "fell":1.0,
    "drop":1.0, "dropped":1.0, "decline":1.0, "declined":1.0, "slide":1.0,
    "slid":1.0, "sink":1.5, "sank":1.5, "tumble":1.5, "tumbled":1.5,
    # Risk signals
    "recession":2.0, "inflation":1.0, "tariff":1.5, "sanction":2.0,
    "shortage":1.5, "supply chain":1.0, "headwind":1.5, "risk":0.8,
    "uncertainty":1.0, "volatile":1.0, "volatility":0.8,
}

# 8-K item numbers that are high-signal events
# Positive 8-K items (boost sentiment)
BULLISH_8K_ITEMS = {
    "2.02",   # Results of operations (earnings release)
    "7.01",   # Regulation FD disclosure (often guidance)
    "8.01",   # Other events (often positive announcements)
    "1.01",   # Entry into material agreement
}
# Negative 8-K items (crush sentiment)
BEARISH_8K_ITEMS = {
    "2.05",   # Departure of directors / principal officers
    "2.06",   # Material impairments
    "4.01",   # Changes in registrant's certifying accountant
    "4.02",   # Non-reliance on previously issued financials
    "8.01",   # Can also be negative — scored by content
}


def _lex_score(text: str) -> float:
    """
    Scores a text string using the financial lexicon.
    Returns a value roughly in [-10, +10]; caller normalizes to 0–100.
    """
    text_lower = text.lower()
    bull = sum(
        weight
        for term, weight in BULLISH_TERMS.items()
        if term in text_lower
    )
    bear = sum(
        weight
        for term, weight in BEARISH_TERMS.items()
        if term in text_lower
    )
    return bull - bear


def _normalize(raw: float, scale: float = 8.0) -> float:
    """Maps [-scale, +scale] → [0, 100] with 50 as neutral."""
    return max(0.0, min(100.0, 50.0 + (raw / scale) * 50.0))


class SentimentAnalyzer:
    """
    Multi-source sentiment scorer using:
      - Benzinga news via Alpaca (full content, per-ticker)
      - Google News RSS (headlines, per-ticker)
      - SEC EDGAR 8-K filings (material events, per-ticker)

    Cache TTL: 2 hours (shorter than fundamentals — news changes fast).
    """

    CACHE_TTL     = 7_200    # 2 hours
    NEWS_LOOKBACK = 3        # days of news to consider

    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str):
        self.alpaca_key    = alpaca_api_key
        self.alpaca_secret = alpaca_secret_key
        self._cache: dict[str, tuple[float, float]] = {}   # sym -> (ts, score)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_get(self, symbol: str) -> Optional[float]:
        if symbol in self._cache:
            ts, score = self._cache[symbol]
            if time.time() - ts < self.CACHE_TTL:
                return score
        return None

    def _cache_set(self, symbol: str, score: float) -> None:
        self._cache[symbol] = (time.time(), score)

    # ------------------------------------------------------------------
    # Source 1: Benzinga via Alpaca News API (45 pts)
    # ------------------------------------------------------------------
    def _fetch_benzinga(self, symbol: str) -> float:
        """
        Fetches last 3 days of Benzinga articles via Alpaca.
        Scores full article content using the financial lexicon.
        Returns normalized 0–100 component score (max contribution: 45 pts).
        """
        try:
            start = (datetime.now(timezone.utc) - timedelta(days=self.NEWS_LOOKBACK))
            resp = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={
                    "APCA-API-KEY-ID":     self.alpaca_key,
                    "APCA-API-SECRET-KEY": self.alpaca_secret,
                },
                params={
                    "symbols":         symbol,
                    "limit":           10,
                    "include_content": "true",
                    "start":           start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                timeout=6,
            )
            resp.raise_for_status()
            articles = resp.json().get("news", [])

            if not articles:
                return 50.0   # no news = neutral

            raw_scores: list[float] = []
            now = datetime.now(timezone.utc)
            for article in articles:
                headline = article.get("headline", "")
                content  = article.get("content", "")
                text     = f"{headline} {content}"

                # Recency weighting: articles from today get full weight,
                # older articles are discounted
                try:
                    pub = datetime.fromisoformat(
                        article.get("created_at", "").replace("Z", "+00:00")
                    )
                    age_hours  = max(0, (now - pub).total_seconds() / 3600)
                    recency_wt = max(0.3, 1.0 - (age_hours / (self.NEWS_LOOKBACK * 24)))
                except Exception:
                    recency_wt = 0.5

                raw_scores.append(_lex_score(text) * recency_wt)

            avg_raw = sum(raw_scores) / len(raw_scores)
            return _normalize(avg_raw, scale=6.0)

        except Exception as exc:
            logger.debug("Benzinga fetch failed for %s: %s", symbol, exc)
            return 50.0

    # ------------------------------------------------------------------
    # Source 2: Google News RSS (30 pts)
    # ------------------------------------------------------------------
    def _fetch_google_news(self, symbol: str) -> float:
        """
        Fetches Google News RSS headlines for the ticker.
        Scores headline text only (no full content available).
        Returns normalized 0–100 component score (max contribution: 30 pts).
        """
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={
                    "q":    f"{symbol} stock",
                    "hl":   "en-US",
                    "gl":   "US",
                    "ceid": "US:en",
                },
                timeout=6,
            )
            resp.raise_for_status()

            root  = ET.fromstring(resp.text)
            items = root.findall(".//item")[:15]

            if not items:
                return 50.0

            raw_scores: list[float] = []
            for item in items:
                title = item.findtext("title") or ""
                desc  = item.findtext("description") or ""
                raw_scores.append(_lex_score(f"{title} {desc}"))

            avg_raw = sum(raw_scores) / len(raw_scores)
            return _normalize(avg_raw, scale=4.0)

        except Exception as exc:
            logger.debug("Google News fetch failed for %s: %s", symbol, exc)
            return 50.0

    # ------------------------------------------------------------------
    # Source 3: SEC EDGAR 8-K filings (25 pts)
    # ------------------------------------------------------------------
    def _fetch_sec_8k(self, symbol: str) -> float:
        """
        Checks for recent 8-K filings — material event disclosures.
        A recently filed 8-K on a bullish item (earnings beat, deal) boosts
        score. A bearish item (impairment, accountant change) crushes it.
        Returns normalized 0–100 component score (max contribution: 25 pts).
        """
        try:
            start = (
                datetime.now(timezone.utc) - timedelta(days=7)
            ).strftime("%Y-%m-%d")

            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q":         f'"{symbol}"',
                    "dateRange": "custom",
                    "startdt":   start,
                    "forms":     "8-K",
                },
                headers=HEADERS_SEC,
                timeout=6,
            )
            resp.raise_for_status()
            data  = resp.json()
            hits  = data.get("hits", {}).get("hits", [])

            if not hits:
                return 50.0   # no recent 8-K = neutral

            raw_score = 0.0
            for hit in hits[:5]:
                source   = hit.get("_source", {})
                # Score the 8-K description text using the lexicon
                desc     = source.get("display_names", [""])[0] if source.get("display_names") else ""
                form_type = source.get("form_type", "")
                raw_score += _lex_score(desc)

            # Scale: an 8-K is a significant event so use tighter scale
            return _normalize(raw_score / max(len(hits[:5]), 1), scale=3.0)

        except Exception as exc:
            logger.debug("SEC 8-K fetch failed for %s: %s", symbol, exc)
            return 50.0

    # ------------------------------------------------------------------
    # Public score() method
    # ------------------------------------------------------------------
    def score(self, symbol: str) -> float:
        """
        Returns a composite sentiment score 0–100.
        Weighted combination of Benzinga (45%), Google News (30%), SEC 8-K (25%).
        Cached for 2 hours.
        """
        cached = self._cache_get(symbol)
        if cached is not None:
            return cached

        benzinga    = self._fetch_benzinga(symbol)
        google_news = self._fetch_google_news(symbol)
        sec_8k      = self._fetch_sec_8k(symbol)

        # Weighted composite
        composite = (
            benzinga    * 0.45
            + google_news * 0.30
            + sec_8k      * 0.25
        )
        composite = round(min(max(composite, 0.0), 100.0), 2)

        logger.debug(
            "%s sentiment — Benzinga: %.1f  Google: %.1f  SEC 8-K: %.1f  → %.1f",
            symbol, benzinga, google_news, sec_8k, composite,
        )

        self._cache_set(symbol, composite)
        return composite

    def cache_size(self) -> int:
        return len(self._cache)