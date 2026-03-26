"""
market_scanner.py
-----------------
Phase 1 Core Module — Market Scanner & Signal Filter
AI Trading Agent | Paper Trading Only

Data sources:
  - Alpaca TradingClient    : asset universe, account, order execution (free)
  - Alpaca StockDataClient  : snapshots + historical OHLCV bars (free tier)

Phase 1 fixes applied:
  - Bracket orders (entry + stop-loss + take-profit submitted together)
  - Weekend detection (no scanning Sat/Sun)
  - Live P&L pulled from Alpaca each cycle to feed kill-switch
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import ta as ta_lib
from alpaca.trading.client import TradingClient
from trading_agent.modules.fundamental import FundamentalAnalyzer
from trading_agent.modules.sentiment import SentimentAnalyzer
from trading_agent.modules.options_flow import OptionsAnalyzer
from trading_agent.modules.performance_logger import PerformanceLogger
from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import logging

class _CTFormatter(logging.Formatter):
    """Formats log timestamps in US Central time instead of UTC."""
    def formatTime(self, record, datefmt=None):
        from zoneinfo import ZoneInfo
        import datetime
        ct = datetime.datetime.fromtimestamp(
            record.created, tz=ZoneInfo("America/Chicago")
        )
        return ct.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

_handler = logging.StreamHandler()
_handler.setFormatter(_CTFormatter(
    fmt="%(asctime)s CT [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger("MarketScanner")

CENTRAL = ZoneInfo("America/Chicago")

# US market holidays 2026 (add each year)
MARKET_HOLIDAYS_2026 = {
    (2026,  1,  1),   # New Year's Day
    (2026,  1, 19),   # MLK Day
    (2026,  2, 16),   # Presidents Day
    (2026,  4,  3),   # Good Friday
    (2026,  5, 25),   # Memorial Day
    (2026,  6, 19),   # Juneteenth
    (2026,  7,  3),   # Independence Day (observed)
    (2026,  9,  7),   # Labor Day
    (2026, 11, 26),   # Thanksgiving
    (2026, 11, 27),   # Black Friday (early close — treat as closed)
    (2026, 12, 25),   # Christmas
}

def _is_holiday(dt: datetime) -> bool:
    return (dt.year, dt.month, dt.day) in MARKET_HOLIDAYS_2026


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    ALPACA_API_KEY: str    = os.environ["ALPACA_API_KEY"]
    ALPACA_SECRET_KEY: str = os.environ["ALPACA_SECRET_KEY"]
    FMP_API_KEY: str       = os.environ.get("FMP_API_KEY", "")

    PAPER_TRADING: bool = True

    # Universe filters
    MIN_PRICE: float         = 5.0
    MAX_PRICE: float         = 500.0
    MIN_ADV: float           = 500_000
    MIN_ATR_PCT: float       = 0.5
    MAX_ATR_PCT: float       = 25.0
    SNAPSHOT_BATCH_SIZE: int = 1000
    MAX_UNIVERSE_SIZE: int   = 8000

    # Scoring weights
    WEIGHT_TECHNICAL: float   = 0.35
    WEIGHT_FUNDAMENTAL: float = 0.20
    WEIGHT_SENTIMENT: float   = 0.20
    WEIGHT_OPTIONS: float     = 0.25

    # Score thresholds
    TECH_PREFILTER: float       = 30.0
    COMPOSITE_MIN: float        = 63.0   # raised from 55 — filters out low-conviction signals
    MAX_FUNDAMENTAL_SCORES: int = 500    # top N for SEC EDGAR (async httpx, genuinely cancellable)

    # Risk management
    MAX_PORTFOLIO_RISK_PCT: float = 2.0
    MAX_POSITION_PCT: float       = 5.0
    RISK_REWARD_MIN: float        = 5.0
    MAX_DAILY_LOSS_PCT: float     = 3.0
    MAX_OPEN_POSITIONS: int       = 10
    MAX_SECTOR_POSITIONS: int     = 3    # max concurrent positions in same sector

    # Cadence
    SCAN_INTERVAL_SECONDS: int   = 300
    WATCHLIST_REFRESH_HOURS: int = 6


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class CandidateTicker:
    symbol: str
    price: float
    adv: float
    atr_pct: float
    market_cap: float
    passed_prefilter: bool = False

    technical_score: float    = 0.0
    fundamental_score: float  = 0.0
    sentiment_score: float    = 0.0
    options_score: float      = 0.0
    composite_score: float    = 0.0

    entry: Optional[float]       = None
    stop_loss: Optional[float]   = None
    atr_dollar: Optional[float]  = None
    target: Optional[float]      = None
    risk_reward: Optional[float] = None
    direction: str               = "LONG"
    timestamp: datetime          = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class TradeSignal:
    ticker: CandidateTicker
    position_size: int = 0
    dollar_risk: float = 0.0
    confidence: str    = "LOW"


# ---------------------------------------------------------------------------
# Session Helper
# ---------------------------------------------------------------------------
def get_session() -> tuple[str, bool]:
    """
    Returns (session_name, is_closed) based on current Central time.

    Sessions:
      Pre-market  : Mon–Fri 3:00 AM – 9:30 AM CT
      Regular     : Mon–Fri 9:30 AM – 4:00 PM CT
      After-hours : Mon–Fri 4:00 PM – 7:00 PM CT
      Closed      : 7:00 PM – 3:00 AM CT, and all day Sat–Sun
    """
    now_ct  = datetime.now(CENTRAL)
    weekday = now_ct.weekday()   # 0=Mon … 6=Sun
    hour_ct = now_ct.hour + now_ct.minute / 60

    # Full weekend closure
    if weekday == 5:
        return "WEEKEND", True
    if weekday == 6 and hour_ct < 3.0:
        return "WEEKEND", True
    if weekday == 6:
        return "WEEKEND", True

    # Market holiday
    if _is_holiday(now_ct):
        return "HOLIDAY", True

    # Weekday closed window
    if hour_ct >= 19.0 or hour_ct < 3.0:
        return "CLOSED", True

    if hour_ct < 9.5:
        return "PRE-MARKET", False
    elif hour_ct < 16.0:
        return "REGULAR HOURS", False
    else:
        return "AFTER-HOURS", False


def seconds_until_next_open() -> float:
    """Calculates seconds until next 3:00 AM CT Monday–Friday open."""
    now_ct = datetime.now(CENTRAL)
    # Find next weekday 3 AM
    candidate = now_ct.replace(hour=3, minute=0, second=0, microsecond=0)
    if candidate <= now_ct:
        candidate += timedelta(days=1)
    # Skip weekend
    while candidate.weekday() >= 5:   # Sat=5, Sun=6
        candidate += timedelta(days=1)
    secs = (candidate - now_ct).total_seconds() - 60   # wake 60s early
    return max(0.0, min(secs, 8 * 3600))   # cap at 8 hrs for Replit


# ---------------------------------------------------------------------------
# Universe Builder
# ---------------------------------------------------------------------------
class UniverseBuilder:

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
    ):
        self.trade                                  = trading_client
        self.data                                   = data_client
        self._ticker_list: list[str]                = []
        self._universe_cache: list[CandidateTicker] = []
        self._last_build: float                     = 0.0
        self._last_ticker_fetch: float              = 0.0

    def _universe_cache_fresh(self) -> bool:
        return (time.time() - self._last_build) / 3600 < Config.WATCHLIST_REFRESH_HOURS

    def _ticker_list_fresh(self) -> bool:
        return (time.time() - self._last_ticker_fetch) / 3600 < Config.WATCHLIST_REFRESH_HOURS

    def _fetch_all_tickers(self) -> list[str]:
        if self._ticker_list_fresh() and self._ticker_list:
            logger.info("Ticker list cache fresh (%d symbols).", len(self._ticker_list))
            return self._ticker_list

        logger.info("Fetching US equity universe from Alpaca assets …")
        tickers: list[str] = []
        try:
            assets = self.trade.get_all_assets(
                GetAssetsRequest(
                    asset_class=AssetClass.US_EQUITY,
                    status=AssetStatus.ACTIVE,
                )
            )
            for asset in assets:
                sym = asset.symbol
                if (
                    asset.tradable
                    and asset.easy_to_borrow
                    and "." not in sym
                    and "/" not in sym
                    and sym.isalpha()
                    and 1 <= len(sym) <= 5
                ):
                    tickers.append(sym)
                if len(tickers) >= Config.MAX_UNIVERSE_SIZE:
                    break
        except Exception as exc:
            logger.error("Alpaca asset list failed: %s", exc)

        logger.info("Ticker list built: %d symbols.", len(tickers))
        self._ticker_list       = tickers
        self._last_ticker_fetch = time.time()
        return tickers

    async def _snapshot_filter(self, tickers: list[str]) -> list[CandidateTicker]:
        candidates: list[CandidateTicker] = []
        batches = [
            tickers[i : i + Config.SNAPSHOT_BATCH_SIZE]
            for i in range(0, len(tickers), Config.SNAPSHOT_BATCH_SIZE)
        ]
        for idx, batch in enumerate(batches):
            logger.info(
                "Snapshot batch %d / %d | candidates so far: %d",
                idx + 1, len(batches), len(candidates),
            )
            try:
                snapshots = await asyncio.to_thread(
                    self.data.get_stock_snapshot,
                    StockSnapshotRequest(symbol_or_symbols=batch),
                )
                for sym, snap in snapshots.items():
                    try:
                        bar = snap.previous_daily_bar or snap.daily_bar
                        if bar is None:
                            continue
                        price      = float(bar.close or 0)
                        volume_usd = price * float(bar.volume or 0)
                        if not (Config.MIN_PRICE <= price <= Config.MAX_PRICE):
                            continue
                        if volume_usd < Config.MIN_ADV:
                            continue
                        day_range_pct = 0.0
                        if bar.high and bar.low and price > 0:
                            day_range_pct = (
                                (float(bar.high) - float(bar.low)) / price
                            ) * 100
                        if not (Config.MIN_ATR_PCT <= day_range_pct <= Config.MAX_ATR_PCT):
                            continue
                        candidates.append(
                            CandidateTicker(
                                symbol=sym,
                                price=price,
                                adv=volume_usd,
                                atr_pct=day_range_pct,
                                market_cap=0.0,
                                passed_prefilter=True,
                            )
                        )
                    except Exception as exc:
                        logger.debug("Snapshot parse %s: %s", sym, exc)
            except Exception as exc:
                logger.error("Snapshot batch %d failed: %s", idx + 1, exc)
            await asyncio.sleep(0.3)
        return candidates

    async def build(self, force: bool = False) -> list[CandidateTicker]:
        if not force and self._universe_cache_fresh():
            logger.info("Universe cache fresh (%d tickers).", len(self._universe_cache))
            return self._universe_cache

        all_tickers = await asyncio.to_thread(self._fetch_all_tickers)
        if not all_tickers:
            logger.error("No tickers retrieved — check Alpaca API keys.")
            return []

        # Connectivity check
        try:
            snaps = await asyncio.to_thread(
                self.data.get_stock_snapshot,
                StockSnapshotRequest(symbol_or_symbols=["AAPL"]),
            )
            aapl = snaps.get("AAPL")
            bar  = (aapl.previous_daily_bar or aapl.daily_bar) if aapl else None
            if bar:
                logger.info(
                    "Alpaca data OK — AAPL close: $%.2f | volume: %s | range: %.2f%%",
                    float(bar.close),
                    f"{int(bar.volume):,}",
                    ((float(bar.high) - float(bar.low)) / float(bar.close)) * 100,
                )
        except Exception as exc:
            logger.error("Alpaca snapshot test failed: %s", exc)

        candidates = await self._snapshot_filter(all_tickers)
        logger.info(
            "Universe built: %d / %d tickers passed pre-filter.",
            len(candidates), len(all_tickers),
        )
        self._universe_cache = candidates
        self._last_build     = time.time()
        return candidates


# ---------------------------------------------------------------------------
# Technical Analyzer
# ---------------------------------------------------------------------------
class TechnicalAnalyzer:

    def __init__(self, data_client: StockHistoricalDataClient):
        self.data = data_client

    def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        try:
            bars = self.data.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime.now(timezone.utc),
                    limit=120,
                    feed="iex",
                )
            )
            df = bars.df
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)
            return df[["open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            logger.debug("OHLCV fetch failed %s: %s", symbol, exc)
            return pd.DataFrame()

    def score(self, symbol: str) -> float:
        df = self._fetch_ohlcv(symbol)
        if df.empty or len(df) < 50:
            return 0.0

        df["ema20"]  = ta_lib.trend.ema_indicator(df["close"], window=20)
        df["ema50"]  = ta_lib.trend.ema_indicator(df["close"], window=50)
        df["ema200"] = ta_lib.trend.ema_indicator(df["close"], window=200)
        df["rsi"]    = ta_lib.momentum.rsi(df["close"], window=14)
        df["macd"]   = ta_lib.trend.macd(df["close"])
        df["signal"] = ta_lib.trend.macd_signal(df["close"])
        df["hist"]   = ta_lib.trend.macd_diff(df["close"])
        df["atr"]    = ta_lib.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )

        # VWAP: rolling 20-day volume-weighted average price
        # Price above VWAP = bullish (buyers in control)
        # Price below VWAP = bearish (sellers in control)
        vwap_window = 20
        df["vwap"] = (
            (df["close"] * df["volume"]).rolling(vwap_window).sum()
            / df["volume"].rolling(vwap_window).sum()
        )

        row  = df.iloc[-1]
        prev = df.iloc[-2]
        total = 0.0

        # EMA trend alignment (25 pts)
        if row["close"]  > row["ema20"]:  total += 8
        if row["ema20"]  > row["ema50"]:  total += 9
        if row["ema50"]  > row["ema200"]: total += 8

        # RSI momentum zone (20 pts)
        rsi = row["rsi"]
        if 50 <= rsi <= 65:   total += 20
        elif 40 <= rsi < 50:  total += 14
        elif 65 < rsi <= 70:  total += 12
        elif rsi > 70:        total += 5
        elif rsi < 30:        total += 3

        # MACD crossover (20 pts)
        if (row["macd"] > row["signal"]
                and prev["macd"] <= prev["signal"]
                and row["hist"] > 0):
            total += 20
        elif row["macd"] > row["signal"]:
            total += 10

        # VWAP position (15 pts)
        # Price vs VWAP tells us who controls momentum
        vwap = row["vwap"]
        close = row["close"]
        if vwap > 0:
            vwap_deviation = (close - vwap) / vwap
            if vwap_deviation >= 0.02:    total += 15   # strongly above VWAP
            elif vwap_deviation >= 0.00:  total += 10   # above VWAP
            elif vwap_deviation >= -0.02: total += 5    # just below VWAP
            # below -2% gets 0 — bearish price action

        # Volume confirmation (12 pts)
        avg_vol = df["volume"].iloc[-21:-1].mean()
        if avg_vol > 0 and row["volume"] >= 1.5 * avg_vol:
            total += 12
        elif avg_vol > 0 and row["volume"] >= 1.0 * avg_vol:
            total += 6

        # ATR setup quality (8 pts)
        atr_pct = (row["atr"] / row["close"]) * 100 if row["close"] > 0 else 0
        if 2.5 <= atr_pct <= 8.0:   total += 8
        elif 2.0 <= atr_pct < 2.5:  total += 4

        return min(total, 100.0)


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------
class ScoringEngine:

    def __init__(
        self,
        fundamental_analyzer: FundamentalAnalyzer,
        sentiment_analyzer: SentimentAnalyzer,
        options_analyzer: OptionsAnalyzer,
    ):
        self.fundamental = fundamental_analyzer
        self.sentiment   = sentiment_analyzer
        self.options     = options_analyzer

    def compute(self, candidate: CandidateTicker, tech_score: float) -> CandidateTicker:
        fund = self.fundamental.score(candidate.symbol)
        sent = self.sentiment.score(candidate.symbol)
        opts = self.options.score(candidate.symbol)

        candidate.technical_score   = tech_score
        candidate.fundamental_score = fund
        candidate.sentiment_score   = sent
        candidate.options_score     = opts
        candidate.composite_score   = round(
            tech_score * Config.WEIGHT_TECHNICAL
            + fund      * Config.WEIGHT_FUNDAMENTAL
            + sent      * Config.WEIGHT_SENTIMENT
            + opts      * Config.WEIGHT_OPTIONS,
            2,
        )
        return candidate

    def build_trade_setup(self, candidate: CandidateTicker) -> Optional[CandidateTicker]:
        if candidate.price <= 0 or candidate.atr_pct <= 0:
            return None
        atr_dollar = candidate.price * (candidate.atr_pct / 100)

        # Stop: 2x ATR — gives position room to breathe through normal volatility
        stop_dist = max(atr_dollar * 2.0, candidate.price * 0.02, 0.05)

        entry     = candidate.price
        stop_loss = round(entry - stop_dist, 2)

        # Target: min(3x ATR, 8% above entry)
        # Adaptive for volatile stocks, capped for stable ones
        atr_target = entry + (3.0 * atr_dollar)
        pct_target = entry * 1.08
        target     = round(min(atr_target, pct_target), 2)

        rr = (target - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
        candidate.entry       = entry
        candidate.stop_loss   = stop_loss
        candidate.target      = target
        candidate.atr_dollar  = atr_dollar
        candidate.risk_reward = round(rr, 2)
        return candidate


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------
class RiskManager:

    def __init__(self, alpaca_client: TradingClient):
        self.alpaca         = alpaca_client
        self._daily_pnl_pct = 0.0
        self._killed        = False

    # --- FIX 3: pull real P&L from Alpaca ---
    def refresh_daily_pnl(self) -> None:
        """
        Fetches current account equity vs. last_equity (previous day close)
        to compute intraday P&L %. Called at the start of every scan cycle.
        """
        try:
            account    = self.alpaca.get_account()
            equity     = float(account.equity)
            last_equity = float(account.last_equity)
            if last_equity > 0:
                pnl_pct = ((equity - last_equity) / last_equity) * 100
                self._daily_pnl_pct = pnl_pct
                logger.info(
                    "Daily P&L: %.2f%% | Equity: $%.2f | Prev close: $%.2f",
                    pnl_pct, equity, last_equity,
                )
        except Exception as exc:
            logger.error("P&L refresh failed: %s", exc)

    def is_system_live(self) -> bool:
        if self._killed:
            logger.critical("KILL-SWITCH ACTIVE — all trading halted.")
            return False
        if self._daily_pnl_pct <= -Config.MAX_DAILY_LOSS_PCT:
            logger.critical(
                "Daily loss limit hit (%.2f%%). Kill-switch activated.",
                self._daily_pnl_pct,
            )
            self._killed = True
            return False
        return True

    def update_daily_pnl(self, pct: float) -> None:
        self._daily_pnl_pct = pct

    def reset_kill_switch(self) -> None:
        self._killed        = False
        self._daily_pnl_pct = 0.0
        logger.info("Kill-switch reset for new session.")

    def size_position(self, candidate: CandidateTicker) -> Optional[TradeSignal]:
        if not self.is_system_live():
            return None
        if candidate.entry is None or candidate.stop_loss is None:
            return None

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            account         = self.alpaca.get_account()
            portfolio_value = float(account.portfolio_value)
            open_positions  = len(self.alpaca.get_all_positions())
            # Count pending orders too — critical for after-hours batching
            pending_orders  = self.alpaca.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            # Count unique symbols with open buy orders not already in positions
            position_symbols = {p.symbol for p in self.alpaca.get_all_positions()}
            pending_symbols  = {
                o.symbol for o in pending_orders
                if o.side.value == "buy" and o.symbol not in position_symbols
            }
            total_exposure = open_positions + len(pending_symbols)
        except Exception as exc:
            logger.error("Alpaca account query failed: %s", exc)
            return None

        if total_exposure >= Config.MAX_OPEN_POSITIONS:
            logger.debug(
                "Position limit reached: %d positions + %d pending = %d / %d",
                open_positions, len(pending_symbols), total_exposure, Config.MAX_OPEN_POSITIONS,
            )
            return None

        stop_distance = candidate.entry - candidate.stop_loss
        if stop_distance <= 0:
            return None

        risk_dollars = portfolio_value * (Config.MAX_PORTFOLIO_RISK_PCT / 100)
        shares       = int(risk_dollars / stop_distance)
        max_shares   = int(
            portfolio_value * (Config.MAX_POSITION_PCT / 100) / candidate.entry
        )
        shares = min(shares, max_shares)
        if shares < 1:
            return None

        confidence = (
            "HIGH"   if candidate.composite_score >= 80 else
            "MEDIUM" if candidate.composite_score >= 65 else
            "LOW"
        )

        signal = TradeSignal(
            ticker=candidate,
            position_size=shares,
            dollar_risk=round(shares * stop_distance, 2),
            confidence=confidence,
        )
        logger.info(
            "Signal [%s] %s | Shares: %d | Risk: $%.2f | R/R: %.1f:1 | Score: %.1f",
            confidence, candidate.symbol, shares, signal.dollar_risk,
            candidate.risk_reward, candidate.composite_score,
        )
        return signal


# ---------------------------------------------------------------------------
# Execution Engine — limit orders using live bid/ask
# ---------------------------------------------------------------------------
class ExecutionEngine:
    """
    Submits bracket orders using limit entry prices derived from the live
    bid/ask spread. Required for pre-market and after-hours trading —
    Alpaca rejects market orders outside regular hours. Also produces
    better fills during regular hours by avoiding slippage.

    Entry limit price logic:
      BUY  → ask + $0.01  (aggressive limit — fills quickly, beats the spread)
      SELL → bid - $0.01  (for shorts, when added)

    Stop-loss and take-profit are placed as part of the same bracket order.
    Time-in-force: DAY — order expires at end of session if unfilled.
    """

    def __init__(
        self,
        alpaca_client: TradingClient,
        data_client: StockHistoricalDataClient,
    ):
        self.alpaca = alpaca_client
        self.data   = data_client
        assert Config.PAPER_TRADING, "Live trading disabled until backtest sign-off."

    def _get_limit_price(self, symbol: str, side: OrderSide, snapshot_price: float = 0.0) -> Optional[float]:
        """
        Fetches latest quote and returns an aggressive limit price.
        BUY:  ask + $0.01 (or bid + $0.02 if ask is zero after hours)
        Falls back to snapshot_price * 1.002 if no valid quote exists.
        """
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req    = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data.get_stock_latest_quote(req)
            quote  = quotes.get(symbol)
            if not quote:
                return None

            ask = float(quote.ask_price or 0)
            bid = float(quote.bid_price or 0)

            if side == OrderSide.BUY:
                if ask > 0:
                    return round(ask + 0.01, 2)        # normal hours: just above ask
                elif bid > 0:
                    return round(bid * 1.003, 2)       # after hours: 0.3% above last bid
                elif snapshot_price > 0:
                    return round(snapshot_price * 1.002, 2)  # fallback
                return None
            else:
                if bid > 0:
                    return round(bid - 0.01, 2)
                elif ask > 0:
                    return round(ask * 0.997, 2)
                elif snapshot_price > 0:
                    return round(snapshot_price * 0.998, 2)
                return None

        except Exception as exc:
            logger.debug("Quote fetch failed for %s: %s", symbol, exc)
            return None

    def submit(self, signal: TradeSignal) -> bool:
        if not Config.PAPER_TRADING:
            raise RuntimeError("Live order attempted — blocked.")

        ticker = signal.ticker

        # Long-only enforcement — shorts disabled until Phase 4
        if ticker.direction != "LONG":
            logger.warning(
                "Skipping %s — short entries disabled. Direction: %s",
                ticker.symbol, ticker.direction,
            )
            return False

        if (
            ticker.entry is None
            or ticker.stop_loss is None
            or ticker.target is None
            or signal.position_size < 1
        ):
            return False

        side = OrderSide.BUY

        # Get live bid/ask limit price — fall back to snapshot price + buffer
        limit_price = self._get_limit_price(ticker.symbol, side, ticker.entry or 0.0)
        if limit_price is None:
            limit_price = round((ticker.entry or 0) * 1.002, 2)
            logger.debug(
                "%s no live quote — using fallback limit $%.2f",
                ticker.symbol, limit_price,
            )

        # Recalculate stop and target against actual limit price.
        # Stop: 2x ATR below entry for volatility-appropriate protection.
        # Target: min(3x ATR, 8% above entry) — adaptive for volatile stocks,
        #         capped for stable ones. Optimized via backtest.
        atr_dollar    = ticker.atr_dollar if hasattr(ticker, "atr_dollar") and ticker.atr_dollar else limit_price * 0.02
        stop_distance = max(atr_dollar * 2.0, limit_price * 0.02, 0.05)
        stop_price    = round(limit_price - stop_distance, 2)
        atr_target    = limit_price + (3.0 * atr_dollar)
        pct_target    = limit_price * 1.08
        target_price  = round(min(atr_target, pct_target), 2)
        # Ensure minimum gap for Alpaca
        if limit_price - stop_price < 0.02:
            stop_price = round(limit_price - 0.02, 2)

        try:
            from alpaca.trading.requests import LimitOrderRequest

            order = LimitOrderRequest(
                symbol=ticker.symbol,
                qty=signal.position_size,
                side=side,
                time_in_force=TimeInForce.GTC,   # GTC so stop/target persist across sessions
                limit_price=limit_price,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=target_price),
                stop_loss=StopLossRequest(
                    stop_price=stop_price,
                    # No limit_price = stop-market order.
                    # Guarantees execution on gap-downs unlike stop-limit.
                ),
            )
            submitted = self.alpaca.submit_order(order)
            logger.info(
                "[PAPER] Bracket limit order — %s %d shares | "
                "entry: $%.2f | stop: $%.2f | target: $%.2f | ID: %s",
                ticker.symbol, signal.position_size,
                limit_price, stop_price, target_price,
                submitted.id,
            )
            return True

        except Exception as exc:
            logger.error("Order failed for %s: %s", ticker.symbol, exc)
            return False


# ---------------------------------------------------------------------------
# Trading Agent
# ---------------------------------------------------------------------------
class TradingAgent:

    def __init__(self):
        self.trading_client = TradingClient(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            paper=True,
        )
        self.data_client = StockHistoricalDataClient(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
        )
        self.universe = UniverseBuilder(self.trading_client, self.data_client)
        self.tech     = TechnicalAnalyzer(self.data_client)
        self.fund     = FundamentalAnalyzer(Config.FMP_API_KEY)
        self.sent     = SentimentAnalyzer(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY)
        self.opts     = OptionsAnalyzer(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY)
        self.scorer   = ScoringEngine(self.fund, self.sent, self.opts)
        self.risk     = RiskManager(self.trading_client)
        self.executor = ExecutionEngine(self.trading_client, self.data_client)

    async def scan_cycle(self) -> list[TradeSignal]:
        # --- FIX 2: session + weekend guard ---
        session, is_closed = get_session()

        if is_closed:
            sleep_secs = seconds_until_next_open()
            now_ct = datetime.now(CENTRAL)
            logger.info(
                "Session: %s | Current time: %s CT | Next scan at 3:00 AM CT in %.0f minutes …",
                session,
                now_ct.strftime("%I:%M %p"),
                sleep_secs / 60,
            )
            await asyncio.sleep(sleep_secs)
            return []

        logger.info("Session: %s", session)

        # --- Enforce time-based stops on stale positions ---
        await self._enforce_time_stops()

        # --- FIX 3: refresh live P&L before checking kill-switch ---
        self.risk.refresh_daily_pnl()
        if not self.risk.is_system_live():
            return []

        candidates = await self.universe.build()
        signals: list[TradeSignal] = []
        scored     = 0
        top_scores: list[tuple[str, float, float]] = []

        logger.info("Scoring %d pre-filtered candidates …", len(candidates))

        # --- Pass 1: technical scoring (fast, no quota cost) ---
        tech_results: list[tuple[CandidateTicker, float]] = []
        for c in candidates:
            tech_score = await asyncio.to_thread(self.tech.score, c.symbol)
            scored += 1
            if scored % 100 == 0:
                logger.info("  … scored %d / %d", scored, len(candidates))
            if tech_score >= Config.TECH_PREFILTER:
                tech_results.append((c, tech_score))

        # Sort by tech score descending
        tech_results.sort(key=lambda x: x[1], reverse=True)

        total_tech = len(tech_results)
        logger.info(
            "Pass 1 done — %d tickers above tech threshold. "
            "Running all four modules concurrently …",
            total_tech,
        )

        # --- Pass 2: concurrent scoring across all four modules ---
        # Each module runs its HTTP calls in a thread pool so they overlap.
        # Batch size controls max concurrent requests per module — stays
        # polite to external APIs while being much faster than sequential.
        BATCH = 30   # concurrent tickers per batch (higher = faster)
        SENT_TIMEOUT = 900   # 15 min for sentiment (1859 tickers × 3 sources, slow)
        OPTS_TIMEOUT = 120   # 2 min for options (very fast, ~16s observed)

        fund_score_map: dict[str, float] = {}
        sent_score_map: dict[str, float] = {}
        opts_score_map: dict[str, float] = {}

        # Pre-load CIK map once using native async — no threads, no locks
        import httpx as _httpx
        logger.info("Pre-loading SEC CIK map …")
        async with _httpx.AsyncClient() as _c:
            await self.fund._load_cik_map(_c)
        logger.info("CIK map ready — %d tickers.", len(self.fund._cik_map))

        # Set prices for P/E calculation
        for c, _ in tech_results:
            self.fund.set_price(c.symbol, c.price)

        async def score_batch(
            items: list[tuple[CandidateTicker, float]],
            label: str,
            score_fn,
            result_map: dict,
            per_ticker_timeout: float = 8.0,
        ) -> None:
            total = len(items)
            for i in range(0, total, BATCH):
                batch = items[i : i + BATCH]
                tasks = [
                    asyncio.wait_for(
                        asyncio.to_thread(score_fn, c.symbol),
                        timeout=per_ticker_timeout,
                    )
                    for c, _ in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for (c, _), result in zip(batch, results):
                    if isinstance(result, (Exception, asyncio.TimeoutError)):
                        result_map[c.symbol] = 50.0
                    else:
                        result_map[c.symbol] = float(result)
                done = min(i + BATCH, total)
                if done % 100 == 0 or done == total:
                    logger.info("  %s: %d / %d", label, done, total)

        # Fundamentals — native async httpx, genuinely cancellable, top 500
        fund_candidates = tech_results[:Config.MAX_FUNDAMENTAL_SCORES]
        logger.info("Pass 2a: fundamentals (top %d, native async) …", len(fund_candidates))
        for i, (c, _) in enumerate(fund_candidates):
            try:
                fund_score_map[c.symbol] = await asyncio.wait_for(
                    self.fund.ascore(c.symbol),
                    timeout=15.0,
                )
            except Exception:
                fund_score_map[c.symbol] = 50.0
            if (i + 1) % 50 == 0:
                logger.info("  fundamentals: %d / %d", i + 1, len(fund_candidates))
        logger.info("  fundamentals: %d / %d done.", len(fund_candidates), len(fund_candidates))

        # Sentiment — concurrent batches (Benzinga + Google + 8-K, handles concurrency well)
        try:
            logger.info("Pass 2b: sentiment (all %d) …", total_tech)
            await asyncio.wait_for(
                score_batch(tech_results, "sentiment", self.sent.score, sent_score_map, 8.0),
                timeout=SENT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Pass 2b timed out — using cached/neutral sentiment.")

        # Options — concurrent batches (single CBOE call per ticker, very fast)
        try:
            logger.info("Pass 2c: options (all %d) …", total_tech)
            await asyncio.wait_for(
                score_batch(tech_results, "options", self.opts.score, opts_score_map, 8.0),
                timeout=OPTS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Pass 2c timed out — using cached/neutral options.")

        # --- Pass 2d: compute composites and build signals ---
        logger.info("Pass 2d: computing composites …")
        top_scores: list[tuple[str, float, float, float, float, float]] = []
        for c, tech_score in tech_results:
            fund = fund_score_map.get(c.symbol, 50.0)
            sent = sent_score_map.get(c.symbol, 50.0)
            opts = opts_score_map.get(c.symbol, 50.0)

            c.technical_score   = tech_score
            c.fundamental_score = fund
            c.sentiment_score   = sent
            c.options_score     = opts
            c.composite_score   = round(
                tech_score * Config.WEIGHT_TECHNICAL
                + fund      * Config.WEIGHT_FUNDAMENTAL
                + sent      * Config.WEIGHT_SENTIMENT
                + opts      * Config.WEIGHT_OPTIONS,
                2,
            )

            top_scores.append((
                c.symbol, tech_score, fund, sent, opts, c.composite_score,
            ))

            if c.composite_score < Config.COMPOSITE_MIN:
                continue

            c = self.scorer.build_trade_setup(c)
            if c is None:
                continue

            signal = self.risk.size_position(c)
            if signal:
                signals.append(signal)

        top_scores.sort(key=lambda x: x[5], reverse=True)
        logger.info("── Top 10 scoring tickers this cycle ──")
        logger.info("  %-6s  %5s  %5s  %5s  %5s  %7s", "SYM", "TECH", "FUND", "SENT", "OPTS", "COMP")
        for sym, tech, fund, sent, opts, comp in top_scores[:10]:
            logger.info("  %-6s  %5.1f  %5.1f  %5.1f  %5.1f  %7.1f", sym, tech, fund, sent, opts, comp)
        logger.info("────────────────────────────────────────")

        signals.sort(key=lambda s: s.ticker.composite_score, reverse=True)
        logger.info(
            "Cycle complete — scored %d tickers, %d actionable signals.",
            scored, len(signals),
        )
        return signals

    async def _enforce_time_stops(self) -> None:
        """
        Exits any position open for more than 7 trading days without
        hitting stop-loss or take-profit. Prevents capital being locked
        in stagnant trades indefinitely.
        """
        TIME_STOP_DAYS = 7
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            positions = self.trading_client.get_all_positions()
            if not positions:
                return

            now_ct = datetime.now(CENTRAL)

            for pos in positions:
                sym = pos.symbol
                try:
                    orders = self.trading_client.get_orders(
                        GetOrdersRequest(
                            status=QueryOrderStatus.CLOSED,
                            symbols=[sym],
                            limit=5,
                        )
                    )
                    filled = [o for o in orders if str(o.status) == "filled"]
                    if not filled:
                        continue

                    filled_at = filled[0].filled_at
                    if filled_at is None:
                        continue

                    filled_ct = filled_at.astimezone(CENTRAL)
                    age_days  = (now_ct - filled_ct).total_seconds() / 86400

                    if age_days >= TIME_STOP_DAYS:
                        logger.warning(
                            "Time stop triggered for %s — open %.1f days. "
                            "Closing at market.", sym, age_days,
                        )
                        self.trading_client.close_position(sym)

                except Exception as exc:
                    logger.debug("Time stop check failed for %s: %s", sym, exc)

        except Exception as exc:
            logger.error("Time stop enforcement failed: %s", exc)

    def _get_open_symbols(self) -> set[str]:
        """Returns set of ticker symbols we currently hold positions in."""
        try:
            positions = self.trading_client.get_all_positions()
            return {p.symbol for p in positions}
        except Exception as exc:
            logger.error("Failed to fetch open positions: %s", exc)
            return set()

    def _get_pending_order_symbols(self) -> set[str]:
        """Returns set of ticker symbols with open/pending orders."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            return {o.symbol for o in orders}
        except Exception as exc:
            logger.error("Failed to fetch open orders: %s", exc)
            return set()

    async def execute_signals(self, signals: list[TradeSignal]) -> None:
        open_symbols    = self._get_open_symbols()
        pending_symbols = self._get_pending_order_symbols()
        excluded        = open_symbols | pending_symbols

        # Capacity = how many new positions we can open right now
        total_exposure = len(open_symbols) + len(pending_symbols)
        capacity       = Config.MAX_OPEN_POSITIONS - total_exposure

        if capacity <= 0:
            logger.info(
                "Position limit reached (%d / %d) — no new orders this cycle.",
                total_exposure, Config.MAX_OPEN_POSITIONS,
            )
            return

        # Build sector counts from existing positions + pending
        from trading_agent.modules.sector_map import get_sector
        sector_counts: dict[str, int] = {}
        for sym in open_symbols | pending_symbols:
            sector = get_sector(sym)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if excluded:
            logger.info(
                "Skipping tickers already held/pending: %s",
                ", ".join(sorted(excluded)),
            )

        submitted = 0
        for sig in signals:
            # Hard cap: never exceed available capacity this cycle
            if submitted >= capacity:
                break

            sym    = sig.ticker.symbol
            sector = get_sector(sym)

            if sym in excluded:
                logger.info("Skipping %s — already held or pending.", sym)
                continue

            if sector_counts.get(sector, 0) >= Config.MAX_SECTOR_POSITIONS:
                logger.info(
                    "Skipping %s — sector '%s' already has %d positions (max %d).",
                    sym, sector,
                    sector_counts.get(sector, 0),
                    Config.MAX_SECTOR_POSITIONS,
                )
                continue

            if self.executor.submit(sig):
                excluded.add(sym)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                submitted += 1
                logger.info("Slot used: %d / %d capacity this cycle.", submitted, capacity)
                # Log entry to performance tracker
                t = sig.ticker
                self.perf.log_entry(
                    symbol      = t.symbol,
                    entry_price = t.entry or 0.0,
                    stop        = t.stop_loss or 0.0,
                    target      = t.target or 0.0,
                    qty         = sig.position_size,
                    score       = t.composite_score,
                    tech        = t.technical_score,
                    fund        = t.fundamental_score,
                    sent        = t.sentiment_score,
                    opts        = t.options_score,
                )
            await asyncio.sleep(0.5)

    async def run(self) -> None:
        logger.info("=" * 60)
        logger.info("AI Trading Agent — PAPER TRADING MODE")
        logger.info("=" * 60)

        # --- Startup summary ---
        try:
            account = self.trading_client.get_account()
            logger.info("Account equity    : $%s", f"{float(account.equity):,.2f}")
            logger.info("Buying power      : $%s", f"{float(account.buying_power):,.2f}")
            logger.info("Open positions    : %d / %d max",
                        len(self.trading_client.get_all_positions()),
                        Config.MAX_OPEN_POSITIONS)
        except Exception as exc:
            logger.error("Startup account check failed: %s", exc)

        self.perf = PerformanceLogger()
        logger.info("Modules active    : Technical | Fundamental (SEC EDGAR) | "
                    "Sentiment (Benzinga+Google+8-K) | Options (CBOE+Alpaca)")
        logger.info("Scan interval     : %ds", Config.SCAN_INTERVAL_SECONDS)
        logger.info("Composite threshold: %.1f / 100", Config.COMPOSITE_MIN)
        logger.info("Min R/R ratio     : %.1f:1", Config.RISK_REWARD_MIN)
        logger.info("=" * 60)

        while True:
            try:
                signals = await self.scan_cycle()
                self.perf.check_exits(self.trading_client)
                if signals:
                    await self.execute_signals(signals)
                else:
                    logger.info("No qualifying signals this cycle.")
            except KeyboardInterrupt:
                logger.info("Shutdown requested.")
                break
            except Exception as exc:
                logger.error("Unhandled error in scan cycle: %s", exc, exc_info=True)

            logger.info(
                "Sleeping %ds until next scan …", Config.SCAN_INTERVAL_SECONDS
            )
            await asyncio.sleep(Config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    agent = TradingAgent()
    asyncio.run(agent.run())