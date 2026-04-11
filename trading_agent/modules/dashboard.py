"""
dashboard.py
------------
Lightweight Flask web dashboard for the AI Trading Agent.
Serves on port 8080 (Replit's exposed web port) in a background thread.

Reads from /tmp/performance_log.json — same file the PerformanceLogger writes.
Auto-refreshes every 30 seconds.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, render_template_string

logger = logging.getLogger("Dashboard")

PERF_LOG_FILE = os.path.join(os.getcwd(), "performance_log.json")

app = Flask(__name__)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)   # suppress Flask request logs


def _load_data() -> dict:
    try:
        if os.path.exists(PERF_LOG_FILE):
            with open(PERF_LOG_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {"open_trades": {}, "closed_trades": [], "meta": {}}


def _compute_stats(closed: list[dict]) -> dict:
    if not closed:
        return {}
    total  = len(closed)
    wins   = sum(1 for t in closed if t.get("win"))
    losses = total - wins
    pnls   = [t.get("pnl_dollars", 0) for t in closed]
    rrs    = [t.get("rr_realized", 0) for t in closed]
    return {
        "total":       total,
        "wins":        wins,
        "losses":      losses,
        "win_rate":    round(wins / total * 100, 1) if total else 0,
        "total_pnl":   round(sum(pnls), 2),
        "avg_pnl":     round(sum(pnls) / total, 2) if total else 0,
        "avg_rr":      round(sum(rrs) / total, 2) if total else 0,
        "avg_win":     round(sum(p for p in pnls if p > 0) / max(wins, 1), 2),
        "avg_loss":    round(sum(p for p in pnls if p <= 0) / max(losses, 1), 2),
        "best":        max(closed, key=lambda t: t.get("pnl_dollars", 0)) if closed else None,
        "worst":       min(closed, key=lambda t: t.get("pnl_dollars", 0)) if closed else None,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Trading Agent</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3d;
    --accent: #00ff88;
    --accent2: #7c6fff;
    --red: #ff4466;
    --yellow: #ffcc44;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    min-height: 100vh;
    padding: 32px;
  }

  /* Subtle grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.3;
    pointer-events: none;
    z-index: 0;
  }

  .container { max-width: 1200px; margin: 0 auto; position: relative; z-index: 1; }

  header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 40px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
  }

  .logo span { color: var(--accent2); }

  .status-pill {
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid rgba(0, 255, 136, 0.3);
    color: var(--accent);
    letter-spacing: 1px;
  }

  .refresh-info {
    margin-left: auto;
    font-size: 12px;
    color: var(--muted);
    font-family: var(--font-mono);
  }

  /* Stats grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }

  .stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.5;
  }

  .stat-card.red::before { background: var(--red); }
  .stat-card.purple::before { background: var(--accent2); }
  .stat-card.yellow::before { background: var(--yellow); }

  .stat-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
    font-family: var(--font-mono);
  }

  .stat-value {
    font-family: var(--font-mono);
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
  }

  .stat-value.positive { color: var(--accent); }
  .stat-value.negative { color: var(--red); }
  .stat-value.neutral  { color: var(--text); }
  .stat-value.purple   { color: var(--accent2); }

  .stat-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
    font-family: var(--font-mono);
  }

  /* Sections */
  .section { margin-bottom: 32px; }

  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .section-title {
    font-family: var(--font-mono);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
  }

  .section-count {
    font-family: var(--font-mono);
    font-size: 11px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    color: var(--accent2);
  }

  /* Table */
  .table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  thead {
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }

  th {
    padding: 12px 16px;
    text-align: left;
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    font-weight: 400;
  }

  td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(42, 42, 61, 0.5);
    font-family: var(--font-mono);
    font-size: 13px;
  }

  tr:last-child td { border-bottom: none; }

  tr:hover td { background: rgba(255,255,255,0.02); }

  .sym {
    font-weight: 700;
    color: var(--accent2);
    font-size: 14px;
  }

  .positive { color: var(--accent); }
  .negative { color: var(--red); }
  .neutral  { color: var(--muted); }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .badge.win  { background: rgba(0,255,136,0.1); color: var(--accent); border: 1px solid rgba(0,255,136,0.2); }
  .badge.loss { background: rgba(255,68,102,0.1); color: var(--red); border: 1px solid rgba(255,68,102,0.2); }
  .badge.open { background: rgba(124,111,255,0.1); color: var(--accent2); border: 1px solid rgba(124,111,255,0.2); }
  .badge.tp   { background: rgba(0,255,136,0.1); color: var(--accent); border: 1px solid rgba(0,255,136,0.2); }
  .badge.sl   { background: rgba(255,68,102,0.1); color: var(--red); border: 1px solid rgba(255,68,102,0.2); }
  .badge.man  { background: rgba(255,204,68,0.1); color: var(--yellow); border: 1px solid rgba(255,204,68,0.2); }

  .score-bar {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .score-track {
    width: 60px;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .score-fill {
    height: 100%;
    background: var(--accent2);
    border-radius: 2px;
  }

  .empty-state {
    padding: 48px;
    text-align: center;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 13px;
  }

  .empty-state span {
    display: block;
    font-size: 32px;
    margin-bottom: 12px;
    opacity: 0.4;
  }

  footer {
    margin-top: 40px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo">AI<span>.</span>TRADER</div>
    <div class="status-pill">● PAPER</div>
    <div class="refresh-info">auto-refresh 30s | <span id="last-update"></span></div>
  </header>

  <!-- Stats row -->
  <div class="stats-grid" id="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total Trades</div>
      <div class="stat-value neutral" id="stat-total">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value neutral" id="stat-winrate">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total P&L</div>
      <div class="stat-value neutral" id="stat-pnl">—</div>
    </div>
    <div class="stat-card purple">
      <div class="stat-label">Avg R/R</div>
      <div class="stat-value purple" id="stat-rr">—</div>
      <div class="stat-sub">realized</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg Win</div>
      <div class="stat-value positive" id="stat-avgwin">—</div>
    </div>
    <div class="stat-card red">
      <div class="stat-label">Avg Loss</div>
      <div class="stat-value negative" id="stat-avgloss">—</div>
    </div>
  </div>

  <!-- Open Positions -->
  <div class="section">
    <div class="section-header">
      <div class="section-title">Open Positions</div>
      <div class="section-count" id="open-count">0</div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Entry</th>
            <th>Stop</th>
            <th>Target</th>
            <th>Qty</th>
            <th>Score</th>
            <th>Tech</th>
            <th>Fund</th>
            <th>Sent</th>
            <th>Opts</th>
            <th>Entered</th>
          </tr>
        </thead>
        <tbody id="open-tbody">
          <tr><td colspan="11" class="empty-state"><span>◌</span>No open positions recorded yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Closed Trades -->
  <div class="section">
    <div class="section-header">
      <div class="section-title">Closed Trades</div>
      <div class="section-count" id="closed-count">0</div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>P&L $</th>
            <th>P&L %</th>
            <th>R/R</th>
            <th>Reason</th>
            <th>Score</th>
            <th>Result</th>
            <th>Closed</th>
          </tr>
        </thead>
        <tbody id="closed-tbody">
          <tr><td colspan="10" class="empty-state"><span>◌</span>No closed trades yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <footer>
    <span>AI Trading Agent — Paper Mode Only</span>
    <span id="footer-time"></span>
  </footer>

</div>

<script>
function fmt(n, decimals=2, prefix='') {
  if (n === null || n === undefined) return '—';
  return prefix + Number(n).toFixed(decimals);
}

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', {month:'short',day:'numeric'}) + ' ' +
         d.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit'});
}

function pnlClass(val) {
  return val > 0 ? 'positive' : val < 0 ? 'negative' : 'neutral';
}

function reasonBadge(r) {
  if (!r) return '<span class="badge man">—</span>';
  if (r === 'TAKE_PROFIT') return '<span class="badge tp">TP</span>';
  if (r === 'STOP_LOSS')   return '<span class="badge sl">SL</span>';
  return '<span class="badge man">MAN</span>';
}

function scoreBar(val) {
  const pct = Math.min(100, Math.max(0, val));
  return `<div class="score-bar">
    <span>${val.toFixed(0)}</span>
    <div class="score-track"><div class="score-fill" style="width:${pct}%"></div></div>
  </div>`;
}

async function refresh() {
  try {
    const res = await fetch('/api/data');
    const d   = await res.json();

    // Stats
    const s = d.stats;
    if (s && s.total > 0) {
      document.getElementById('stat-total').textContent = s.total;
      const wrEl = document.getElementById('stat-winrate');
      wrEl.textContent = s.win_rate + '%';
      wrEl.className = 'stat-value ' + (s.win_rate >= 50 ? 'positive' : 'negative');

      const pnlEl = document.getElementById('stat-pnl');
      pnlEl.textContent = (s.total_pnl >= 0 ? '+$' : '-$') + Math.abs(s.total_pnl).toFixed(2);
      pnlEl.className = 'stat-value ' + pnlClass(s.total_pnl);

      document.getElementById('stat-rr').textContent = s.avg_rr.toFixed(2) + 'x';
      document.getElementById('stat-avgwin').textContent = '+$' + s.avg_win.toFixed(2);
      document.getElementById('stat-avgloss').textContent = '-$' + Math.abs(s.avg_loss).toFixed(2);
    }

    // Open trades
    const open = Object.values(d.open_trades || {});
    document.getElementById('open-count').textContent = open.length;
    const ot = document.getElementById('open-tbody');
    if (open.length === 0) {
      ot.innerHTML = '<tr><td colspan="11" class="empty-state"><span>◌</span>No open positions recorded</td></tr>';
    } else {
      ot.innerHTML = open.map(t => `<tr>
        <td class="sym">${t.symbol}</td>
        <td>$${fmt(t.entry_price)}</td>
        <td class="negative">$${fmt(t.stop)}</td>
        <td class="positive">$${fmt(t.target)}</td>
        <td>${t.qty}</td>
        <td>${scoreBar(t.score || 0)}</td>
        <td>${fmt(t.tech,1)}</td>
        <td>${fmt(t.fund,1)}</td>
        <td>${fmt(t.sent,1)}</td>
        <td>${fmt(t.opts,1)}</td>
        <td class="neutral">${fmtDate(t.entry_time)}</td>
      </tr>`).join('');
    }

    // Closed trades (most recent first)
    const closed = (d.closed_trades || []).slice().reverse();
    document.getElementById('closed-count').textContent = closed.length;
    const ct = document.getElementById('closed-tbody');
    if (closed.length === 0) {
      ct.innerHTML = '<tr><td colspan="10" class="empty-state"><span>◌</span>No closed trades yet</td></tr>';
    } else {
      ct.innerHTML = closed.map(t => `<tr>
        <td class="sym">${t.symbol}</td>
        <td>$${fmt(t.entry_price)}</td>
        <td>$${fmt(t.exit_price)}</td>
        <td class="${pnlClass(t.pnl_dollars)}">${t.pnl_dollars >= 0 ? '+' : ''}$${fmt(Math.abs(t.pnl_dollars))}</td>
        <td class="${pnlClass(t.pnl_pct)}">${t.pnl_pct >= 0 ? '+' : ''}${fmt(t.pnl_pct)}%</td>
        <td class="${pnlClass(t.rr_realized)}">${fmt(t.rr_realized)}x</td>
        <td>${reasonBadge(t.exit_reason)}</td>
        <td>${scoreBar(t.score || 0)}</td>
        <td><span class="badge ${t.win ? 'win' : 'loss'}">${t.win ? 'WIN' : 'LOSS'}</span></td>
        <td class="neutral">${fmtDate(t.exit_time)}</td>
      </tr>`).join('');
    }

    // Timestamps
    const now = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
    document.getElementById('last-update').textContent = 'updated ' + now;
    document.getElementById('footer-time').textContent = new Date().toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric',year:'numeric'});

  } catch(e) {
    console.error('Refresh failed:', e);
  }
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/data")
def api_data():
    data   = _load_data()
    closed = data.get("closed_trades", [])
    return jsonify({
        "open_trades":   data.get("open_trades", {}),
        "closed_trades": closed,
        "stats":         _compute_stats(closed),
        "meta":          data.get("meta", {}),
    })


def start_dashboard(port: int = 8080) -> None:
    """Start Flask dashboard in a daemon thread. Call once at agent startup."""
    def _run():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info("Dashboard running at http://0.0.0.0:%d", port)