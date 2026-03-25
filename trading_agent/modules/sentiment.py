"""
sentiment.py
------------
Phase 2 Module — Sentiment & News Analysis
AI Trading Agent | Paper Trading Only

Scoring engine: VADER + Financial Lexicon Hybrid
  - VADER (Valence Aware Dictionary and sEntiment Reasoner) is purpose-built
    for short news/social text and handles negation, intensifiers, punctuation
  - Custom financial term boosts tune VADER specifically for market language
  - Applied to full article content from Benzinga + Google News + SEC 8-K

Why VADER over raw lexicon:
  - Handles negation: "not a good quarter" scores negative (lexicon misses this)
  - Intensifiers: "CRUSHED earnings" scores higher than "beat earnings"
  - Punctuation: "Earnings BEAT!!!" scores higher than "earnings beat"
  - Produces compound score in [-1, +1], maps cleanly to 0-100

Scoring rubric (0-100):
  Benzinga articles   : 50% weight (full content, per-ticker)
  Google News         : 30% weight (headlines, breadth)
  SEC 8-K             : 20% weight (material events)

Cache TTL: 2 hours
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger("SentimentAnalyzer")

HEADERS_SEC  = {"User-Agent": "ai-trading-agent contact@trading-agent.local"}

# ---------------------------------------------------------------------------
# Financial lexicon boosts for VADER
# These are added to VADER's internal dictionary to tune it for finance.
# Positive = bullish signal strength, Negative = bearish signal strength
# VADER uses [-4, +4] scale internally
# ---------------------------------------------------------------------------
FINANCIAL_LEXICON = {
    # Strong bullish
    "beat":          2.5,
    "beats":         2.5,
    "blowout":       3.5,
    "crush":         3.0,
    "crushed":       3.0,
    "smashed":       2.5,
    "raised guidance": 3.0,
    "raised outlook":  3.0,
    "upgrade":       2.5,
    "upgraded":      2.5,
    "strong buy":    3.0,
    "outperform":    2.5,
    "fda approval":  3.5,
    "buyback":       2.0,
    "dividend increase": 2.0,
    "contract win":  2.5,
    "record revenue": 3.0,
    "record earnings": 3.0,
    "accelerating":  1.5,
    "momentum":      1.0,
    "breakout":      2.0,
    "bullish":       2.0,

    # Strong bearish
    "miss":         -2.5,
    "missed":       -2.5,
    "lowered guidance": -3.0,
    "cut guidance": -3.0,
    "downgrade":    -2.5,
    "downgraded":   -2.5,
    "underperform": -2.5,
    "sell rating":  -2.0,
    "profit warning": -3.5,
    "bankruptcy":   -4.0,
    "fraud":        -4.0,
    "sec probe":    -3.5,
    "investigation": -2.5,
    "recall":       -2.5,
    "layoffs":      -2.0,
    "job cuts":     -2.5,
    "debt crisis":  -3.0,
    "default":      -3.5,
    "impairment":   -2.5,
    "shortfall":    -2.5,
    "disappointing": -2.0,
    "headwinds":    -1.5,
    "bearish":      -2.0,
    "tariff":       -1.5,
    "sanction":     -2.0,
}


def _build_analyzer() -> SentimentIntensityAnalyzer:
    """Build VADER analyzer with financial lexicon injected."""
    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(FINANCIAL_LEXICON)
    return vader


# Module-level singleton — one analyzer for the whole process
_VADER = _build_analyzer()


def _vader_score(text: str) -> float:
    """
    Score text with VADER. Returns compound score in [-1, +1].
    Compound is VADER's normalized aggregate — best single metric.
    """
    if not text or not text.strip():
        return 0.0
    scores = _VADER.polarity_scores(text)
    return scores["compound"]


def _compound_to_100(compound: float, scale: float = 0.35) -> float:
    """
    Map VADER compound to [0, 100] with 50 as neutral.
    Real financial news rarely exceeds ±0.35 when averaged across articles,
    so we scale to that realistic range rather than the full [-1, +1].
    This gives full spread: compound=-0.35 → ~0, compound=+0.35 → ~100.
    """
    return round(max(0.0, min(100.0, 50.0 + (compound / scale) * 50.0)), 2)


class SentimentAnalyzer:
    """
    Multi-source sentiment scorer using VADER + financial lexicon.
    Sources: Benzinga (50%), Google News (30%), SEC 8-K (20%).
    Cache TTL: 2 hours.
    """

    CACHE_TTL     = 7_200   # 2 hours
    NEWS_LOOKBACK = 3       # days of news to consider

    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str):
        self.alpaca_key    = alpaca_api_key
        self.alpaca_secret = alpaca_secret_key
        self._cache: dict[str, tuple[float, float]] = {}

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
    # Source 1: Benzinga via Alpaca (50%)
    # ------------------------------------------------------------------
    def _score_benzinga(self, symbol: str) -> float:
        try:
            start = datetime.now(timezone.utc) - timedelta(days=self.NEWS_LOOKBACK)
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
                return 50.0

            now = datetime.now(timezone.utc)
            weighted_scores = []
            for article in articles:
                headline = article.get("headline", "")
                content  = article.get("content", "")[:800]   # cap content length
                text     = f"{headline}. {content}"

                # Recency weighting — today = 1.0, 3 days ago = 0.3
                try:
                    pub = datetime.fromisoformat(
                        article.get("created_at", "").replace("Z", "+00:00")
                    )
                    age_hours  = max(0, (now - pub).total_seconds() / 3600)
                    recency_wt = max(0.3, 1.0 - age_hours / (self.NEWS_LOOKBACK * 24))
                except Exception:
                    recency_wt = 0.5

                compound = _vader_score(text)
                weighted_scores.append(compound * recency_wt)

            avg = sum(weighted_scores) / len(weighted_scores)
            return _compound_to_100(avg)

        except Exception as exc:
            logger.debug("Benzinga fetch failed for %s: %s", symbol, exc)
            return 50.0

    # ------------------------------------------------------------------
    # Source 2: Google News RSS (30%)
    # ------------------------------------------------------------------
    def _score_google_news(self, symbol: str) -> float:
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={"q": f"{symbol} stock", "hl": "en-US",
                        "gl": "US", "ceid": "US:en"},
                timeout=6,
            )
            resp.raise_for_status()
            root  = ET.fromstring(resp.text)
            items = root.findall(".//item")[:15]
            if not items:
                return 50.0

            scores = []
            for item in items:
                title = item.findtext("title") or ""
                desc  = item.findtext("description") or ""
                scores.append(_vader_score(f"{title}. {desc}"))

            avg = sum(scores) / len(scores)
            return _compound_to_100(avg)

        except Exception as exc:
            logger.debug("Google News failed for %s: %s", symbol, exc)
            return 50.0

    # ------------------------------------------------------------------
    # Source 3: SEC EDGAR 8-K (20%)
    # ------------------------------------------------------------------
    def _score_sec_8k(self, symbol: str) -> float:
        try:
            start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": f'"{symbol}"', "dateRange": "custom",
                        "startdt": start, "forms": "8-K"},
                headers=HEADERS_SEC,
                timeout=6,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                return 50.0

            scores = []
            for hit in hits[:5]:
                src  = hit.get("_source", {})
                desc = " ".join(src.get("display_names", []))
                if desc:
                    scores.append(_vader_score(desc))

            if not scores:
                return 50.0

            avg = sum(scores) / len(scores)
            return _compound_to_100(avg)

        except Exception as exc:
            logger.debug("SEC 8-K failed for %s: %s", symbol, exc)
            return 50.0

    # ------------------------------------------------------------------
    # Public score()
    # ------------------------------------------------------------------
    def score(self, symbol: str) -> float:
        """
        Returns sentiment score 0-100 using VADER + financial lexicon.
        Benzinga 50% + Google News 30% + SEC 8-K 20%.
        Cached 2 hours.
        """
        cached = self._cache_get(symbol)
        if cached is not None:
            return cached

        benzinga = self._score_benzinga(symbol)
        google   = self._score_google_news(symbol)
        sec_8k   = self._score_sec_8k(symbol)

        composite = round(
            benzinga * 0.50
            + google  * 0.30
            + sec_8k  * 0.20,
            2,
        )
        composite = max(0.0, min(100.0, composite))

        logger.debug(
            "%s sentiment — Benzinga: %.1f  Google: %.1f  SEC 8-K: %.1f  → %.1f",
            symbol, benzinga, google, sec_8k, composite,
        )

        self._cache_set(symbol, composite)
        return composite

    def cache_size(self) -> int:
        return len(self._cache)