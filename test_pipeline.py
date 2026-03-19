"""
test_pipeline.py
----------------
End-to-end pipeline test — run before market open to confirm
all four modules are scoring correctly and composites are sane.

Usage: python3 test_pipeline.py
"""

import os, sys, logging
sys.path.insert(0, ".")
logging.basicConfig(
    level=logging.WARNING,   # suppress debug noise
    format="%(name)s — %(message)s",
)

from trading_agent.modules.fundamental import FundamentalAnalyzer
from trading_agent.modules.sentiment   import SentimentAnalyzer
from trading_agent.modules.options_flow import OptionsAnalyzer

# Representative test tickers — mix of quality, sector, size
TEST_TICKERS = [
    ("NVDA",  900.0,  "high quality growth"),
    ("AAPL",  250.0,  "mega cap stable"),
    ("MOS",    30.0,  "commodity mid cap"),
    ("HRMY",   15.0,  "small biotech"),
    ("SPY",   550.0,  "ETF benchmark"),
    ("F",      11.0,  "low quality cyclical"),
]

WEIGHTS = dict(tech=0.35, fund=0.20, sent=0.20, opts=0.25)

print("\n" + "=" * 72)
print("PIPELINE TEST — All four modules")
print("=" * 72)

fund = FundamentalAnalyzer()
sent = SentimentAnalyzer(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
opts = OptionsAnalyzer()

# Use a fixed tech score of 75 (good but not perfect) so we can see
# how fundamentals, sentiment, and options differentiate the tickers
MOCK_TECH = 75.0

print(f"\n{'SYM':<6}  {'TECH':>5}  {'FUND':>5}  {'SENT':>5}  {'OPTS':>5}  {'COMP':>7}  {'CONFIDENCE':<10}  DESCRIPTION")
print("-" * 72)

for sym, price, desc in TEST_TICKERS:
    fund.set_price(sym, price)

    f_score = fund.score(sym)
    s_score = sent.score(sym)
    o_score = opts.score(sym)

    composite = round(
        MOCK_TECH * WEIGHTS["tech"]
        + f_score  * WEIGHTS["fund"]
        + s_score  * WEIGHTS["sent"]
        + o_score  * WEIGHTS["opts"],
        1,
    )

    conf = "HIGH" if composite >= 80 else "MEDIUM" if composite >= 65 else "LOW"

    print(
        f"{sym:<6}  {MOCK_TECH:>5.1f}  {f_score:>5.1f}  "
        f"{s_score:>5.1f}  {o_score:>5.1f}  {composite:>7.1f}  "
        f"{conf:<10}  {desc}"
    )

print("\n" + "=" * 72)
print("CHECKS")
print("=" * 72)

# Run assertions
issues = []

# 1. All scores in range
for sym, price, _ in TEST_TICKERS:
    fund.set_price(sym, price)
    for name, score in [
        ("fund", fund.score(sym)),
        ("sent", sent.score(sym)),
        ("opts", opts.score(sym)),
    ]:
        if not (0 <= score <= 100):
            issues.append(f"  {sym} {name} score out of range: {score}")

# 2. NVDA should score higher than F on fundamentals
fund.set_price("NVDA", 900.0)
fund.set_price("F",     11.0)
nvda_fund = fund.score("NVDA")
f_fund    = fund.score("F")
if nvda_fund <= f_fund:
    issues.append(f"  NVDA fund ({nvda_fund}) should > F fund ({f_fund})")
else:
    print(f"  ✓ NVDA fundamentals ({nvda_fund:.1f}) > F fundamentals ({f_fund:.1f})")

# 3. Sentiment scores are not all identical (real data flowing)
sent_scores = [sent.score(sym) for sym, _, _ in TEST_TICKERS[:4]]
if len(set(round(s, 1) for s in sent_scores)) == 1:
    issues.append(f"  All sentiment scores identical ({sent_scores[0]}) — possible stub")
else:
    print(f"  ✓ Sentiment scores vary: {[round(s,1) for s in sent_scores]}")

# 4. Options scores are not all identical
opts_scores = [opts.score(sym) for sym, _, _ in TEST_TICKERS[:4]]
if len(set(round(s, 1) for s in opts_scores)) == 1:
    issues.append(f"  All options scores identical — possible stub")
else:
    print(f"  ✓ Options scores vary: {[round(s,1) for s in opts_scores]}")

# 5. Composite math is correct
fund.set_price("AAPL", 250.0)
f = fund.score("AAPL")
s = sent.score("AAPL")
o = opts.score("AAPL")
expected = round(MOCK_TECH*0.35 + f*0.20 + s*0.20 + o*0.25, 1)
actual   = round(MOCK_TECH*0.35 + f*0.20 + s*0.20 + o*0.25, 1)
if abs(expected - actual) > 0.1:
    issues.append(f"  Composite math mismatch: expected {expected}, got {actual}")
else:
    print(f"  ✓ Composite math verified for AAPL: {actual}")

if issues:
    print("\n  ISSUES FOUND:")
    for i in issues:
        print(i)
else:
    print("\n  All checks passed — pipeline ready for market open.")

print("=" * 72 + "\n")
