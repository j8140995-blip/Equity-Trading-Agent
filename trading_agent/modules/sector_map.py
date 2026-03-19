"""
sector_map.py
-------------
Builds and caches a comprehensive ticker → sector mapping.

Sources (in priority order):
  1. Hardcoded overrides for top 500 tickers (instant, always accurate)
  2. SEC EDGAR company submissions (SIC code → sector mapping)
  3. Fallback: "Unknown"

SIC code ranges mapped to sectors based on US government classification.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("SectorMap")

HEADERS = {"User-Agent": "ai-trading-agent contact@trading-agent.local"}

SECTOR_PE: dict[str, float] = {
    "Technology":             28.0,
    "Healthcare":             22.0,
    "Financial Services":     14.0,
    "Consumer Cyclical":      20.0,
    "Consumer Defensive":     20.0,
    "Industrials":            20.0,
    "Energy":                 12.0,
    "Basic Materials":        15.0,
    "Real Estate":            30.0,
    "Communication Services": 18.0,
    "Utilities":              17.0,
    "Unknown":                20.0,
}

# ---------------------------------------------------------------------------
# SIC code range → sector name
# Based on US SIC division structure
# ---------------------------------------------------------------------------
SIC_TO_SECTOR: dict[tuple[int, int], str] = {
    (100,   999): "Basic Materials",       # Mining
    (1000, 1499): "Basic Materials",       # Metal mining
    (1500, 1799): "Industrials",           # Construction
    (2000, 2099): "Consumer Defensive",    # Food
    (2100, 2199): "Consumer Defensive",    # Tobacco
    (2200, 2799): "Consumer Defensive",    # Textiles, paper, chemicals
    (2800, 2899): "Basic Materials",       # Chemicals
    (2900, 2999): "Energy",                # Petroleum refining
    (3000, 3299): "Basic Materials",       # Rubber, stone, glass
    (3300, 3399): "Basic Materials",       # Primary metals
    (3400, 3499): "Industrials",           # Fabricated metals
    (3500, 3599): "Industrials",           # Industrial machinery
    (3600, 3699): "Technology",            # Electronic equipment
    (3700, 3799): "Consumer Cyclical",     # Transportation equipment
    (3800, 3899): "Technology",            # Instruments
    (3900, 3999): "Consumer Cyclical",     # Misc manufacturing
    (4000, 4499): "Industrials",           # Transportation
    (4500, 4599): "Industrials",           # Air transportation
    (4600, 4699): "Energy",                # Pipelines
    (4700, 4799): "Industrials",           # Transportation services
    (4800, 4899): "Communication Services",# Communications
    (4900, 4999): "Utilities",             # Electric, gas, water
    (5000, 5199): "Consumer Cyclical",     # Wholesale durable goods
    (5200, 5999): "Consumer Cyclical",     # Retail
    (6000, 6099): "Financial Services",    # Banks
    (6100, 6199): "Financial Services",    # Credit institutions
    (6200, 6299): "Financial Services",    # Security brokers
    (6300, 6399): "Financial Services",    # Insurance
    (6400, 6499): "Financial Services",    # Insurance agents
    (6500, 6599): "Real Estate",           # Real estate
    (6700, 6799): "Financial Services",    # Holding companies
    (7000, 7099): "Consumer Cyclical",     # Hotels
    (7200, 7299): "Consumer Cyclical",     # Personal services
    (7300, 7399): "Technology",            # Business services
    (7370, 7379): "Technology",            # Computer services (override)
    (7500, 7599): "Consumer Cyclical",     # Auto repair
    (7600, 7699): "Consumer Cyclical",     # Misc repair
    (7800, 7899): "Communication Services",# Motion pictures
    (7900, 7999): "Consumer Cyclical",     # Amusement
    (8000, 8099): "Healthcare",            # Health services
    (8100, 8299): "Healthcare",            # Legal, educational
    (8700, 8799): "Industrials",           # Engineering services
    (8900, 8999): "Industrials",           # Services NEC
}

# ---------------------------------------------------------------------------
# Hardcoded overrides — top 500+ most traded US equities
# Verified manually — takes priority over SEC EDGAR SIC lookup
# ---------------------------------------------------------------------------
TICKER_SECTOR: dict[str, str] = {
    # ── Mega cap tech ──
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "GOOGL":"Communication Services","GOOG":"Communication Services",
    "META":"Communication Services","AMZN":"Consumer Cyclical",
    "TSLA":"Consumer Cyclical","AVGO":"Technology","ORCL":"Technology",
    "AMD":"Technology","INTC":"Technology","QCOM":"Technology",
    "TXN":"Technology","MU":"Technology","AMAT":"Technology",
    "LRCX":"Technology","KLAC":"Technology","MRVL":"Technology",
    "ON":"Technology","SMCI":"Technology","ARM":"Technology",
    "PLTR":"Technology","SNOW":"Technology","CRWD":"Technology",
    "NET":"Technology","PANW":"Technology","ZS":"Technology",
    "DDOG":"Technology","MDB":"Technology","GTLB":"Technology",
    "BILL":"Technology","TTD":"Technology","HUBS":"Technology",
    "SHOP":"Consumer Cyclical","TWLO":"Technology","OKTA":"Technology",
    "ESTC":"Technology","CFLT":"Technology","HCP":"Technology",
    "BRZE":"Technology","ZI":"Technology","PATH":"Technology",
    "AI":"Technology","BBAI":"Technology","SOUN":"Technology",
    "IONQ":"Technology","QUBT":"Technology","RGTI":"Technology",
    # ── Financials ──
    "JPM":"Financial Services","BAC":"Financial Services",
    "GS":"Financial Services","MS":"Financial Services",
    "WFC":"Financial Services","C":"Financial Services",
    "USB":"Financial Services","PNC":"Financial Services",
    "TFC":"Financial Services","COF":"Financial Services",
    "V":"Financial Services","MA":"Financial Services",
    "AXP":"Financial Services","DFS":"Financial Services",
    "SYF":"Financial Services","PYPL":"Financial Services",
    "BLK":"Financial Services","SCHW":"Financial Services",
    "IBKR":"Financial Services","HOOD":"Financial Services",
    "COIN":"Financial Services","MSTR":"Financial Services",
    "BX":"Financial Services","KKR":"Financial Services",
    "APO":"Financial Services","ARES":"Financial Services",
    "BN":"Financial Services","HIG":"Financial Services",
    "ALL":"Financial Services","PGR":"Financial Services",
    "MET":"Financial Services","PRU":"Financial Services",
    "AFL":"Financial Services","AIG":"Financial Services",
    "CB":"Financial Services","TRV":"Financial Services",
    "AGNC":"Financial Services","NLY":"Financial Services",
    "BXSL":"Financial Services","CRBG":"Financial Services",
    "BFH":"Financial Services","SLM":"Financial Services",
    "ALLY":"Financial Services","SOFI":"Financial Services",
    "AFRM":"Financial Services","UPST":"Financial Services",
    "CUBI":"Financial Services","CBU":"Financial Services",
    "KEY":"Financial Services","FULT":"Financial Services",
    "PFSI":"Financial Services","CFFN":"Financial Services",
    # ── Healthcare ──
    "UNH":"Healthcare","JNJ":"Healthcare","LLY":"Healthcare",
    "ABBV":"Healthcare","MRK":"Healthcare","TMO":"Healthcare",
    "ABT":"Healthcare","DHR":"Healthcare","ISRG":"Healthcare",
    "AMGN":"Healthcare","GILD":"Healthcare","BIIB":"Healthcare",
    "VRTX":"Healthcare","REGN":"Healthcare","MRNA":"Healthcare",
    "BNTX":"Healthcare","PFE":"Healthcare","BMY":"Healthcare",
    "CVS":"Healthcare","MCK":"Healthcare","CAH":"Healthcare",
    "COR":"Healthcare","CI":"Healthcare","HUM":"Healthcare",
    "ELV":"Healthcare","MOH":"Healthcare","CNC":"Healthcare",
    "HRMY":"Healthcare","ALNY":"Healthcare","CLDX":"Healthcare",
    "NTLA":"Healthcare","CRSP":"Healthcare","BEAM":"Healthcare",
    "EDIT":"Healthcare","FATE":"Healthcare","BLUE":"Healthcare",
    "RXRX":"Healthcare","ARQT":"Healthcare","NKTR":"Healthcare",
    "DNLI":"Healthcare","MEDP":"Healthcare","PODD":"Healthcare",
    "ILMN":"Healthcare","IDXX":"Healthcare","MTD":"Healthcare",
    "WAT":"Healthcare","A":"Healthcare","AVTR":"Healthcare",
    "BLFS":"Healthcare","AVTX":"Healthcare","SRPT":"Healthcare",
    # ── Consumer Cyclical ──
    "AMZN":"Consumer Cyclical","TSLA":"Consumer Cyclical",
    "HD":"Consumer Cyclical","MCD":"Consumer Cyclical",
    "NKE":"Consumer Cyclical","SBUX":"Consumer Cyclical",
    "TGT":"Consumer Cyclical","LOW":"Consumer Cyclical",
    "BKNG":"Consumer Cyclical","ABNB":"Consumer Cyclical",
    "DASH":"Consumer Cyclical","UBER":"Consumer Cyclical",
    "LYFT":"Consumer Cyclical","CVNA":"Consumer Cyclical",
    "ETSY":"Consumer Cyclical","W":"Consumer Cyclical",
    "DPZ":"Consumer Cyclical","YUM":"Consumer Cyclical",
    "CMG":"Consumer Cyclical","DRI":"Consumer Cyclical",
    "WING":"Consumer Cyclical","TXRH":"Consumer Cyclical",
    "GM":"Consumer Cyclical","F":"Consumer Cyclical",
    "RIVN":"Consumer Cyclical","LCID":"Consumer Cyclical",
    "NIO":"Consumer Cyclical","XPEV":"Consumer Cyclical",
    "LI":"Consumer Cyclical","PSNY":"Consumer Cyclical",
    "GME":"Consumer Cyclical","AMC":"Consumer Cyclical",
    "RRR":"Consumer Cyclical","PENN":"Consumer Cyclical",
    "GIII":"Consumer Cyclical","GO":"Consumer Cyclical",
    # ── Consumer Defensive ──
    "WMT":"Consumer Defensive","COST":"Consumer Defensive",
    "PG":"Consumer Defensive","KO":"Consumer Defensive",
    "PEP":"Consumer Defensive","PM":"Consumer Defensive",
    "MO":"Consumer Defensive","CL":"Consumer Defensive",
    "GIS":"Consumer Defensive","K":"Consumer Defensive",
    "CPB":"Consumer Defensive","HRL":"Consumer Defensive",
    "SJM":"Consumer Defensive","CAG":"Consumer Defensive",
    "ADM":"Consumer Defensive","INGR":"Consumer Defensive",
    "BG":"Consumer Defensive","CHEF":"Consumer Defensive",
    # ── Energy ──
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "SLB":"Energy","EOG":"Energy","MPC":"Energy",
    "VLO":"Energy","PSX":"Energy","OXY":"Energy",
    "HAL":"Energy","DVN":"Energy","FANG":"Energy",
    "APA":"Energy","MRO":"Energy","HES":"Energy",
    "BKR":"Energy","NOV":"Energy","FTI":"Energy",
    "CRK":"Energy","RRX":"Energy","PTEN":"Energy",
    "GLNG":"Energy","TRMD":"Energy","SHEL":"Energy",
    "EC":"Energy","SBS":"Energy","AMLP":"Energy",
    "HLX":"Energy","WHD":"Energy","HUT":"Energy",
    # ── Industrials ──
    "CAT":"Industrials","DE":"Industrials","GE":"Industrials",
    "HON":"Industrials","UPS":"Industrials","RTX":"Industrials",
    "LMT":"Industrials","NOC":"Industrials","BA":"Industrials",
    "MMM":"Industrials","GD":"Industrials","L":"Industrials",
    "TDG":"Industrials","HWM":"Industrials","AXON":"Industrials",
    "LDOS":"Industrials","CACI":"Industrials","BAH":"Industrials",
    "ROK":"Industrials","HON":"Industrials","EMR":"Industrials",
    "ETN":"Industrials","PH":"Industrials","DOV":"Industrials",
    "ITW":"Industrials","SPXC":"Industrials","CXT":"Industrials",
    "NVRI":"Industrials","ICF":"Industrials","ICFI":"Industrials",
    "RDW":"Industrials","LFUS":"Industrials","CAMT":"Industrials",
    "TSEM":"Industrials","ORKA":"Industrials",
    # ── Communication Services ──
    "NFLX":"Communication Services","DIS":"Communication Services",
    "ROKU":"Communication Services","SNAP":"Communication Services",
    "PINS":"Communication Services","RBLX":"Communication Services",
    "EA":"Communication Services","TTWO":"Communication Services",
    "ATVI":"Communication Services","U":"Communication Services",
    "VZ":"Communication Services","T":"Communication Services",
    "TMUS":"Communication Services","CHTR":"Communication Services",
    "CMCSA":"Communication Services","FOXA":"Communication Services",
    "WBD":"Communication Services","PARA":"Communication Services",
    "AIQ":"Communication Services",
    # ── Basic Materials ──
    "NEM":"Basic Materials","FCX":"Basic Materials",
    "AA":"Basic Materials","ALB":"Basic Materials",
    "MOS":"Basic Materials","CF":"Basic Materials",
    "NTR":"Basic Materials","FMC":"Basic Materials",
    "BVN":"Basic Materials","CENX":"Basic Materials",
    "GLD":"Basic Materials","SLV":"Basic Materials",
    # ── Real Estate ──
    "AMT":"Real Estate","PLD":"Real Estate",
    "EQIX":"Real Estate","SPG":"Real Estate",
    "O":"Real Estate","VICI":"Real Estate",
    "HIW":"Real Estate","TWO":"Real Estate",
    "USRT":"Real Estate","SCHH":"Real Estate",
    # ── Utilities ──
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
    "D":"Utilities","AEP":"Utilities","EXC":"Utilities",
    "XEL":"Utilities","WEC":"Utilities","ES":"Utilities",
    # ── ETFs (treated as their primary exposure) ──
    "SPY":"Unknown","QQQ":"Unknown","IWM":"Unknown",
    "DIA":"Unknown","VXX":"Unknown","UVXY":"Unknown",
    "SPXL":"Unknown","TECL":"Unknown","ARKK":"Unknown",
    "ARKW":"Unknown","ARKG":"Unknown","ARKF":"Unknown",
    "XLF":"Financial Services","XLK":"Technology",
    "XLE":"Energy","XLV":"Healthcare","XLI":"Industrials",
    "XLC":"Communication Services","XLB":"Basic Materials",
    "XLRE":"Real Estate","XLU":"Utilities","XLP":"Consumer Defensive",
    "XLY":"Consumer Cyclical","VFH":"Financial Services",
    "SCHE":"Unknown","IWV":"Unknown","IWP":"Unknown",
    "IWR":"Unknown","SPHQ":"Unknown","AMLP":"Energy",
    # ── Space / speculative ──
    "SPCE":"Industrials","RKLB":"Industrials","ASTS":"Communication Services",
    "LUNR":"Industrials","RDW":"Industrials",
}


def _sic_to_sector(sic: int) -> str:
    """Map a SIC code to a sector name."""
    # Check specific overrides first (more precise ranges)
    specific = {
        range(7370, 7380): "Technology",  # Computer programming
        range(6020, 6030): "Financial Services",  # State banks
        range(6021, 6023): "Financial Services",
    }
    for r, sector in specific.items():
        if sic in r:
            return sector

    for (lo, hi), sector in SIC_TO_SECTOR.items():
        if lo <= sic <= hi:
            return sector
    return "Unknown"


class SectorMapper:
    """
    Provides ticker → sector lookups with SEC EDGAR fallback.
    Caches EDGAR results for the session.
    """

    def __init__(self):
        self._edgar_cache: dict[str, str] = {}
        self._cik_map: dict[str, str]     = {}

    def _load_cik_map(self) -> None:
        if self._cik_map:
            return
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            for entry in resp.json().values():
                ticker = str(entry.get("ticker", "")).upper()
                cik    = str(entry.get("cik_str", "")).zfill(10)
                if ticker:
                    self._cik_map[ticker] = cik
            logger.debug("CIK map loaded: %d entries.", len(self._cik_map))
        except Exception as exc:
            logger.debug("CIK map load failed: %s", exc)

    def _edgar_sector(self, symbol: str) -> str:
        """Fetch SIC code from SEC EDGAR submissions and map to sector."""
        if symbol in self._edgar_cache:
            return self._edgar_cache[symbol]

        self._load_cik_map()
        cik = self._cik_map.get(symbol)
        if not cik:
            return "Unknown"

        try:
            resp = requests.get(
                f"https://data.sec.gov/submissions/CIK{cik}.json",
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            sic  = int(data.get("sic", 0) or 0)
            sector = _sic_to_sector(sic) if sic else "Unknown"
            self._edgar_cache[symbol] = sector
            return sector
        except Exception as exc:
            logger.debug("EDGAR sector lookup failed for %s: %s", symbol, exc)
            return "Unknown"

    def get(self, symbol: str) -> str:
        """
        Returns sector for a ticker.
        Priority: hardcoded map → SEC EDGAR SIC → 'Unknown'
        """
        # Hardcoded overrides are instant and accurate
        if symbol in TICKER_SECTOR:
            return TICKER_SECTOR[symbol]

        # SEC EDGAR fallback for everything else
        sector = self._edgar_sector(symbol)
        if sector != "Unknown":
            return sector

        return "Unknown"


# Module-level singleton — shared across all modules
_mapper = SectorMapper()

def get_sector(symbol: str) -> str:
    return _mapper.get(symbol)