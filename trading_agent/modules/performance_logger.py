"""
performance_logger.py
---------------------
Tracks every trade from entry to exit, persisting to disk.

Records:
  - Entry: symbol, entry price, stop, target, qty, composite score, timestamp
  - Exit: exit price, exit reason (stop/target/time/manual), P&L $, P&L %, R/R realized
  - Aggregates: win rate, avg P&L, avg R/R, total return

Data persists to /tmp/performance_log.json across restarts.
Summary printed to logs every 10 cycles.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("PerformanceLogger")

PERF_LOG_FILE = "/tmp/performance_log.json"


def _load_log() -> dict:
    try:
        if os.path.exists(PERF_LOG_FILE):
            with open(PERF_LOG_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {"open_trades": {}, "closed_trades": [], "meta": {"created": _now_str()}}


def _save_log(data: dict) -> None:
    try:
        with open(PERF_LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save performance log: %s", exc)


def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


class PerformanceLogger:
    """
    Tracks trade performance across the full lifecycle.

    Usage:
      perf = PerformanceLogger()
      perf.log_entry(signal)                    # on order submission
      perf.check_exits(trading_client)          # each scan cycle
      perf.print_summary()                      # periodic reporting
    """

    def __init__(self):
        self._data       = _load_log()
        self._cycle_count = 0
        logger.info(
            "PerformanceLogger loaded: %d open, %d closed trades.",
            len(self._data["open_trades"]),
            len(self._data["closed_trades"]),
        )

    # ------------------------------------------------------------------
    # Entry logging
    # ------------------------------------------------------------------
    def log_entry(self, symbol: str, entry_price: float, stop: float,
                  target: float, qty: int, score: float,
                  tech: float, fund: float, sent: float, opts: float) -> None:
        """Record a new trade entry."""
        self._data["open_trades"][symbol] = {
            "symbol":       symbol,
            "entry_price":  entry_price,
            "stop":         stop,
            "target":       target,
            "qty":          qty,
            "score":        score,
            "tech":         tech,
            "fund":         fund,
            "sent":         sent,
            "opts":         opts,
            "entry_time":   _now_str(),
            "risk_per_share": round(entry_price - stop, 4),
            "reward_per_share": round(target - entry_price, 4),
        }
        _save_log(self._data)
        logger.info(
            "PERF | Entry logged: %s @ $%.2f | stop $%.2f | target $%.2f | score %.1f",
            symbol, entry_price, stop, target, score,
        )

    # ------------------------------------------------------------------
    # Exit detection
    # ------------------------------------------------------------------
    def check_exits(self, trading_client) -> None:
        """
        Compare open trades against current Alpaca positions.
        Any open trade no longer in positions has been closed — log the exit.
        """
        self._cycle_count += 1

        if not self._data["open_trades"]:
            return

        try:
            positions     = trading_client.get_all_positions()
            open_symbols  = {p.symbol for p in positions}
            position_map  = {p.symbol: p for p in positions}
        except Exception as exc:
            logger.warning("PERF | Failed to fetch positions: %s", exc)
            return

        closed_this_cycle = []
        for symbol, trade in list(self._data["open_trades"].items()):
            if symbol not in open_symbols:
                # Position closed — determine exit price from Alpaca activities
                exit_price = self._get_exit_price(trading_client, symbol,
                                                   trade["entry_time"])
                if exit_price is None:
                    # Can't determine exit yet — skip this cycle
                    continue

                self._record_exit(symbol, trade, exit_price)
                closed_this_cycle.append(symbol)

        for sym in closed_this_cycle:
            del self._data["open_trades"][sym]

        if closed_this_cycle:
            _save_log(self._data)

        # Print summary every 10 cycles
        if self._cycle_count % 10 == 0:
            self.print_summary()

    def _get_exit_price(self, trading_client, symbol: str,
                        entry_time: str) -> Optional[float]:
        """Fetch the fill price of the most recent sell order for this symbol."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            orders = trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=50)
            )
            sell_orders = [
                o for o in orders
                if o.symbol == symbol
                and str(o.side) in ("OrderSide.SELL", "sell")
                and o.filled_avg_price is not None
            ]
            if not sell_orders:
                return None

            # Most recent sell
            sell_orders.sort(
                key=lambda o: o.filled_at or o.submitted_at,
                reverse=True,
            )
            return float(sell_orders[0].filled_avg_price)
        except Exception as exc:
            logger.debug("PERF | Exit price fetch failed for %s: %s", symbol, exc)
            return None

    def _record_exit(self, symbol: str, trade: dict, exit_price: float) -> None:
        """Compute P&L and log the closed trade."""
        entry    = trade["entry_price"]
        stop     = trade["stop"]
        target   = trade["target"]
        qty      = trade["qty"]
        risk_ps  = trade["risk_per_share"]
        reward_ps = trade["reward_per_share"]

        pnl_per_share = exit_price - entry
        pnl_dollars   = round(pnl_per_share * qty, 2)
        pnl_pct       = round((pnl_per_share / entry) * 100, 2) if entry > 0 else 0

        # Determine exit reason
        if exit_price >= target * 0.98:
            reason = "TAKE_PROFIT"
        elif exit_price <= stop * 1.02:
            reason = "STOP_LOSS"
        else:
            reason = "MANUAL_OR_TIME"

        # Realized R/R: how many R multiples did we capture?
        rr_realized = round(pnl_per_share / risk_ps, 2) if risk_ps > 0 else 0

        closed = {
            **trade,
            "exit_price":   exit_price,
            "exit_time":    _now_str(),
            "exit_reason":  reason,
            "pnl_dollars":  pnl_dollars,
            "pnl_pct":      pnl_pct,
            "rr_realized":  rr_realized,
            "win":          pnl_dollars > 0,
        }
        self._data["closed_trades"].append(closed)

        emoji = "✓" if pnl_dollars > 0 else "✗"
        logger.info(
            "PERF | %s Exit: %s @ $%.2f | P&L: $%.2f (%.2f%%) | R/R: %.2fx | %s",
            emoji, symbol, exit_price, pnl_dollars, pnl_pct, rr_realized, reason,
        )

    # ------------------------------------------------------------------
    # Summary reporting
    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print aggregate performance metrics to logs."""
        closed = self._data["closed_trades"]
        open_t = self._data["open_trades"]

        if not closed and not open_t:
            logger.info("PERF | No trades recorded yet.")
            return

        logger.info("━" * 50)
        logger.info("PERF SUMMARY")
        logger.info("━" * 50)

        if closed:
            total      = len(closed)
            wins       = sum(1 for t in closed if t["win"])
            losses     = total - wins
            win_rate   = round(wins / total * 100, 1)
            total_pnl  = round(sum(t["pnl_dollars"] for t in closed), 2)
            avg_pnl    = round(total_pnl / total, 2)
            avg_rr     = round(sum(t["rr_realized"] for t in closed) / total, 2)
            avg_win    = round(sum(t["pnl_dollars"] for t in closed if t["win"]) / max(wins, 1), 2)
            avg_loss   = round(sum(t["pnl_dollars"] for t in closed if not t["win"]) / max(losses, 1), 2)

            logger.info("Closed trades : %d  (W: %d / L: %d)", total, wins, losses)
            logger.info("Win rate      : %.1f%%", win_rate)
            logger.info("Total P&L     : $%.2f", total_pnl)
            logger.info("Avg P&L/trade : $%.2f", avg_pnl)
            logger.info("Avg R/R       : %.2fx", avg_rr)
            logger.info("Avg win       : $%.2f", avg_win)
            logger.info("Avg loss      : $%.2f", avg_loss)

            # Best and worst
            best  = max(closed, key=lambda t: t["pnl_dollars"])
            worst = min(closed, key=lambda t: t["pnl_dollars"])
            logger.info("Best trade    : %s $%.2f (%.2f%%)", best["symbol"], best["pnl_dollars"], best["pnl_pct"])
            logger.info("Worst trade   : %s $%.2f (%.2f%%)", worst["symbol"], worst["pnl_dollars"], worst["pnl_pct"])

        if open_t:
            logger.info("Open trades   : %d — %s", len(open_t), ", ".join(open_t.keys()))

        logger.info("━" * 50)

    def get_stats(self) -> dict:
        """Return summary stats as a dict for programmatic use."""
        closed = self._data["closed_trades"]
        if not closed:
            return {}
        total  = len(closed)
        wins   = sum(1 for t in closed if t["win"])
        return {
            "total_trades": total,
            "wins":         wins,
            "losses":       total - wins,
            "win_rate":     round(wins / total * 100, 1),
            "total_pnl":    round(sum(t["pnl_dollars"] for t in closed), 2),
            "avg_rr":       round(sum(t["rr_realized"] for t in closed) / total, 2),
        }