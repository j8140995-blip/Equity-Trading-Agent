import asyncio
import sys
sys.path.insert(0, ".")

from log_server import start_log_server
from trading_agent.scanner.market_scanner import TradingAgent

if __name__ == "__main__":
    start_log_server()
    agent = TradingAgent()
    asyncio.run(agent.run())
