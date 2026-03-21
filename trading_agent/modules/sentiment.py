"""
sentiment.py
------------
Phase 2 Module — Sentiment & News Analysis
AI Trading Agent | Paper Trading Only

Data sources:
  1. Benzinga via Alpaca News API  — full article content, per-ticker
  2. Google News RSS               — headline coverage, per-ticker
  3. SEC EDGAR 8-K feed            — material event detection, per-ticker

Scoring engine: Claude API (claude-haiku-4-5-20251001)
  Replaces lexicon-based NLP. Claude reads actual article content and
  returns a calibrated 0-100 sentiment score with reasoning.
  Only called when articles exist — tickers with no news return 50.0.
  Cached 2 hours to minimize API usage.

Scoring rubric (0–100):
  0-20  : Strongly bearish (fraud, bankruptcy, severe miss)
  21-40 : Bearish (guidance cut, downgrade, negative catalyst)
  41-59 : Neutral (no meaningful signal)
  60-75 : Bullish (beat, upgrade, positive catalyst)
  76-100: Strongly bullish (blowout, major deal, FDA approval)
"""

from __future__ import annotations

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger("SentimentAnalyzer")

HEADERS_SEC  = {"User-Agent": "ai-trading-agent contact@trading-agent.local"}
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_URL   = "https://api.anthropic.com/v1/messages"


def _call_claude(prompt: str) -> Optional[str]:
    """Calls Claude API and returns the text response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — falling back to neutral sentiment.")
        return None
    try:
        resp = requests.post(
            CLAUDE_URL,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      CLAUDE_MODEL,
                "max_tokens": 150,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()
    except Exception as exc:
        logger.debug("Claude API call failed: %s", exc)
        return None


def _score_with_claude(symbol: str, articles: list[dict]) -> float:
    """
    Sends article headlines + summaries to Claude and gets a 0-100 score.
    Falls back to 50.0 on any failure.
    """
    if not articles:
        return 50.0

    # Build article summary (keep it compact to save tokens)
    article_text = ""
    for i, a in enumerate(articles[:5], 1):
        headline = a.get("headline", "")
        summary  = a.get("summary", "") or a.get("content", "")[:300]
        date     = a.get("created_at", "")[:10]
        article_text += f"{i}. [{date}] {headline}\n{summary}\n\n"

    prompt = f"""You are a financial sentiment analyst. Score the news sentiment for stock ticker {symbol} based on these recent articles.

ARTICLES:
{article_text.strip()}

Return ONLY a JSON object with this exact format:
{{"score": <integer 0-100>, "reason": "<10 words max>"}}

Scoring guide:
- 0-20: Severely bearish (fraud, bankruptcy, massive miss)
- 21-40: Bearish (guidance cut, downgrade, bad earnings)
- 41-59: Neutral (mixed news or no clear signal)
- 60-75: Bullish (earnings beat, upgrade, positive catalyst)
- 76-100: Strongly bullish (blowout quarter, major deal, FDA approval)

JSON only, no other text:"""

    response = _call_claude(prompt)
    if not response:
        return 50.0

    try:
        # Strip any markdown fences if present
        clean = response.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        score = float(data.get("score", 50))
        score = max(0.0, min(100.0, score))
        reason = data.get("reason", "")
        logger.debug("%s sentiment: %.0f — %s", symbol, score, reason)
        return score
    except Exception as exc:
        logger.debug("Claude response parse failed for %s: %s — raw: %s", symbol, exc, response[:100])
        return 50.0


class SentimentAnalyzer:
    """
    Multi-source sentiment scorer using Claude API for scoring.
    Data: Benzinga (primary), Google News (headlines), SEC 8-K (events).
    Cache TTL: 2 hours.
    """

    CACHE_TTL     = 7_200   # 2 hours
    NEWS_LOOKBACK = 3       # days of news

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
    # Source 1: Benzinga via Alpaca News API
    # ------------------------------------------------------------------
    def _fetch_benzinga_articles(self, symbol: str) -> list[dict]:
        """Fetches last 3 days of Benzinga articles. Returns raw article dicts."""
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
                    "limit":           5,
                    "include_content": "true",
                    "start":           start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                timeout=6,
            )
            resp.raise_for_status()
            return resp.json().get("news", [])
        except Exception as exc:
            logger.debug("Benzinga fetch failed for %s: %s", symbol, exc)
            return []

    # ------------------------------------------------------------------
    # Source 2: Google News RSS headlines
    # ------------------------------------------------------------------
    def _fetch_google_headlines(self, symbol: str) -> list[dict]:
        """Fetches Google News RSS headlines as article dicts."""
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={"q": f"{symbol} stock", "hl": "en-US", "gl": "US", "ceid": "US:en"},
                timeout=6,
            )
            resp.raise_for_status()
            root    = ET.fromstring(resp.text)
            items   = root.findall(".//item")[:5]
            articles = []
            for item in items:
                articles.append({
                    "headline": item.findtext("title") or "",
                    "summary":  item.findtext("description") or "",
                    "created_at": item.findtext("pubDate") or "",
                })
            return articles
        except Exception as exc:
            logger.debug("Google News fetch failed for %s: %s", symbol, exc)
            return []

    # ------------------------------------------------------------------
    # Source 3: SEC EDGAR 8-K
    # ------------------------------------------------------------------
    def _fetch_sec_8k_articles(self, symbol: str) -> list[dict]:
        """Fetches recent 8-K filings as article dicts."""
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
            articles = []
            for hit in hits[:3]:
                src = hit.get("_source", {})
                articles.append({
                    "headline": f"SEC 8-K Filing: {src.get('form_type', '8-K')}",
                    "summary":  src.get("display_names", [""])[0] if src.get("display_names") else "",
                    "created_at": src.get("file_date", ""),
                })
            return articles
        except Exception as exc:
            logger.debug("SEC 8-K fetch failed for %s: %s", symbol, exc)
            return []

    # ------------------------------------------------------------------
    # Public score()
    # ------------------------------------------------------------------
    def score(self, symbol: str) -> float:
        """
        Returns sentiment score 0-100 using Claude API.
        Combines Benzinga, Google News, and SEC 8-K articles.
        Cached 2 hours. Returns 50.0 if no articles found.
        """
        cached = self._cache_get(symbol)
        if cached is not None:
            return cached

        # Gather articles from all three sources
        articles = []
        articles += self._fetch_benzinga_articles(symbol)
        if not articles:
            articles += self._fetch_google_headlines(symbol)
        articles += self._fetch_sec_8k_articles(symbol)

        if not articles:
            # No news = neutral, don't waste an API call
            self._cache_set(symbol, 50.0)
            return 50.0

        score = _score_with_claude(symbol, articles)
        self._cache_set(symbol, score)
        return score

    def cache_size(self) -> int:
        return len(self._cache)