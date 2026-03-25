"""
main.py
-------
Entry point for the AI Trading Agent.
Flask dashboard runs as the main process on port 8080 (Replit web URL).
Trading agent runs in a background thread.
"""

import asyncio
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s CT [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from trading_agent.modules.dashboard import app
from trading_agent.scanner.market_scanner import TradingAgent


def _run_agent():
    """Run the async trading agent in a dedicated thread."""
    agent = TradingAgent()
    asyncio.run(agent.run())


if __name__ == "__main__":
    # Start trading agent in background thread
    agent_thread = threading.Thread(target=_run_agent, daemon=True)
    agent_thread.start()

    # Flask serves on port 8080 — Replit routes this to the web URL
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)