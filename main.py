import asyncio
import sys
sys.path.insert(0, ".")

from trading_agent.scanner.market_scanner import TradingAgent

if __name__ == "__main__":
    agent = TradingAgent()
    asyncio.run(agent.run())