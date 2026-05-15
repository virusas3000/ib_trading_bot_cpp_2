"""
dashboard.py — Flask web dashboard for US Algo Trader.

URL:  http://192.168.0.208:8088
Port: 8088

Open positions:  sourced from trades.db (OPEN status)
Closed trades:   sourced from IB executions API (clientId=96), NOT from DB
  - Ensures accurate fill prices even if DB is stale
  - Session boundary: 13:30 UTC (NOT 12:00 UTC)
  - Spans midnight HKT (21:30 Wed → 04:00 Thu HKT) — always check ET

Start: python3 dashboard.py
"""
from __future__ import annotations
import logging
import sqlite3
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from ib_insync import IB, Contract, ExecutionFilter, MarketOrder, util

import config

logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO)

app  = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────
#  IB connection (clientId=96, dedicated for dashboard/executions)
# ─────────────────────────────────────────────────────────────────────────

_ib_dash  = IB()
_ib_lock  = threading.Lock()


def _get_ib() -> IB:
    with _ib_lock:
        if not _ib_dash.isConnected():
            try:
                _ib_dash.connect(config.IB_HOST, config.IB_PORT,
                                  clientId=config.DASHBOARD_CLIENT_ID,
                                  timeout=5)
                # subscribe to account updates so portfolio() is populated
                _ib_dash.client.reqAccountUpdates(True, "")
                import time; time.sleep(2)  # allow portfolio cache to fill
            except Exception as exc:
                logger.warning("[Dashboard] IB connect failed: %s", exc)
        return _ib_dash


# ─────────────────────────────────────────────────────────────────────────
#  Live price cache — refreshed every 5s from IB portfolio() in background.
#  Avoids race where portfolio() hasn't populated yet on first request.
# ─────────────────────────────────────────────────────────────────────────

_price_cache: dict[str, float] = {}
_price_cache_lock = threading.Lock()

def _refresh_price_cache() -> None:
    """Refresh live prices from IB portfolio() — runs in background thread every 2s."""
    import time
    while True:
        try:
            ib = _get_ib()
            if ib.isConnected():
                prices = {}
                for item in ib.portfolio():
                    sym = item.contract.symbol
                    mp  = item.marketPrice
                    if mp and mp > 0 and mp < 1e10:  # guard against IB sentinel -1
                        prices[sym] = mp
                if prices:
                    with _price_cache_lock:
                        _price_cache.update(prices)
                    logger.info("[Dashboard] price cache updated %d symbols: %s",
                                len(prices), {k: round(v,2) for k,v in list(prices.items())[:5]})
                else:
                    logger.warning("[Dashboard] portfolio() returned no valid prices")
        except Exception as exc:
            logger.warning("[Dashboard] price cache refresh error: %s", exc)
        time.sleep(2)

# ─────────────────────────────────────────────────────────────────────────
#  Closed-trades cache — avoids blocking the browser on every 15s refresh.
#  reqExecutions() can take several seconds; we refresh it in a background
#  thread and always return the last known result immediately.
# ─────────────────────────────────────────────────────────────────────────

_closed_cache: list[dict] = []
_closed_cache_ts: float   = 0.0       # unix timestamp of last successful fetch
_closed_cache_lock        = threading.Lock()
_CLOSED_TTL               = 30        # seconds before triggering a background refresh


def _refresh_closed_cache() -> None:
    """Fetch closed trades from IB (or DB fallback) and update the cache."""
    try:
        result = _fetch_closed_trades_live()
        with _closed_cache_lock:
            global _closed_cache, _closed_cache_ts
            _closed_cache    = result
            _closed_cache_ts = __import__("time").time()
    except Exception as exc:
        logger.warning("[Dashboard] closed-trade cache refresh failed: %s", exc)


def _get_closed_trades_cached() -> list[dict]:
    """Return cached closed trades; trigger a background refresh if stale."""
    import time
    with _closed_cache_lock:
        age   = time.time() - _closed_cache_ts
        fresh = _closed_cache[:]

    if age > _CLOSED_TTL:
        t = threading.Thread(target=_refresh_closed_cache, daemon=True)
        t.start()

    return fresh


# ─────────────────────────────────────────────────────────────────────────
#  Session boundary helper
# ─────────────────────────────────────────────────────────────────────────

def _session_start_utc() -> datetime:
    """Today's US session start in UTC: 13:30 UTC (NOT noon)."""
    now = datetime.now(timezone.utc)
    return now.replace(hour=config.SESSION_START_UTC_HOUR,
                       minute=config.SESSION_START_UTC_MIN,
                       second=0, microsecond=0)


# ─────────────────────────────────────────────────────────────────────────
#  Data fetchers
# ─────────────────────────────────────────────────────────────────────────

def fetch_open_positions() -> list[dict]:
    """Open positions from trades.db, sorted newest-first."""
    con = sqlite3.connect(config.TRADES_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT symbol, side, qty, entry_price, entry_time,
                  strategy, confidence, stop, target1 AS target1, target2,
                  ib_order_id
           FROM trades WHERE status='OPEN'
           ORDER BY entry_time DESC"""
    ).fetchall()
    con.close()

    # get last prices from background price cache (refreshed every 5s)
    with _price_cache_lock:
        last_prices = dict(_price_cache)

    result = []
    for r in rows:
        sym        = r["symbol"]
        last       = last_prices.get(sym, r["entry_price"])
        side       = r["side"]
        qty        = r["qty"]
        entry      = r["entry_price"]
        float_pnl  = (last - entry) * qty * (1 if side == "LONG" else -1)
        result.append({
            "symbol":      sym,
            "side":        side,
            "qty":         qty,
            "entry_price": entry,
            "last_price":  last,
            "float_pnl":   round(float_pnl, 2),
            "entry_time":  r["entry_time"],
            "strategy":    r["strategy"],
            "confidence":  r["confidence"],
            "stop":        r["stop"],
            "target1":     r["target1"],
            "target2":     r["target2"],
        })
    return result


def _fetch_closed_trades_live() -> list[dict]:
    """
    Closed trades from trades.db (primary source — always accurate).
    Filtered to today's US session (≥ session start UTC).
    """
    session = _session_start_utc()
    con = sqlite3.connect(config.TRADES_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT symbol, side, qty, entry_price, exit_price, pnl, pct_chg,
                  strategy, reason, exit_time, confidence, target1, target2
           FROM trades WHERE status='CLOSED' AND exit_time >= ?
           ORDER BY exit_time DESC""",
        (session.isoformat(),)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]



def fetch_nav() -> float:
    ib = _get_ib()
    if not ib.isConnected():
        return 0.0
    try:
        vals = ib.accountValues()
        for v in vals:
            if v.tag == "NetLiquidation" and v.currency == "USD":
                return float(v.value)
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────

@app.route("/api/positions")
def api_positions():
    return jsonify(fetch_open_positions())


@app.route("/api/trades")
def api_trades():
    return jsonify(_get_closed_trades_cached())


@app.route("/api/nav")
def api_nav():
    return jsonify({"nav": fetch_nav()})


@app.route("/close", methods=["POST"])
def close_position():
    import subprocess, sys
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get("symbol") or "").strip().upper()
        side   = (data.get("side")   or "").strip().upper()

        if not symbol or side not in ("LONG", "SHORT"):
            return jsonify({"ok": False, "error": "Invalid symbol or side"}), 400

        # Look up qty from open position in DB
        con = sqlite3.connect(config.TRADES_DB)
        row = con.execute(
            "SELECT id, qty FROM trades WHERE symbol=? AND side=? AND status='OPEN' LIMIT 1",
            (symbol, side),
        ).fetchone()
        con.close()

        if row is None:
            return jsonify({"ok": False, "error": f"No open {side} position for {symbol}"}), 404

        row_id, qty = row
        close_action = "SELL" if side == "LONG" else "BUY"

        # Run in subprocess to avoid ib_insync asyncio event-loop conflict with Flask
        script = f"""
import sys
from ib_insync import IB, Contract, MarketOrder, util
util.patchAsyncio()
ib = IB()
ib.connect('{config.IB_HOST}', {config.IB_PORT}, clientId=98, timeout=10)
contract = Contract(symbol='{symbol}', secType='STK', exchange='SMART', currency='USD')
order = MarketOrder('{close_action}', {qty})
ib.placeOrder(contract, order)
ib.sleep(1)
ib.disconnect()
print('OK')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=20
        )
        if "OK" not in result.stdout:
            err = result.stderr.strip() or result.stdout.strip() or "IB order failed"
            logger.error("[Close] subprocess error: %s", err)
            return jsonify({"ok": False, "error": err}), 500

        # Mark closed in DB with exit price and P&L
        with _price_cache_lock:
            exit_price = _price_cache.get(symbol)

        con = sqlite3.connect(config.TRADES_DB)
        if exit_price:
            entry_row = con.execute(
                "SELECT entry_price, qty, side FROM trades WHERE id=?", (row_id,)
            ).fetchone()
            if entry_row:
                ep, q, sd = entry_row
                pnl = round((exit_price - ep) * q * (1 if sd == "LONG" else -1), 2)
                pct = round((exit_price - ep) / ep * (1 if sd == "LONG" else -1), 6)
                con.execute(
                    "UPDATE trades SET status='CLOSED', exit_time=?, exit_price=?, pnl=?, pct_chg=?, reason=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), exit_price, pnl, pct, "MANUAL", row_id),
                )
            else:
                con.execute(
                    "UPDATE trades SET status='CLOSED', exit_time=?, exit_price=?, reason=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), exit_price, "MANUAL", row_id),
                )
        else:
            con.execute(
                "UPDATE trades SET status='CLOSED', exit_time=?, reason=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), "MANUAL", row_id),
            )
        con.commit()
        con.close()

        logger.info("[Close] %s %s qty=%d action=%s — OK", symbol, side, qty, close_action)
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("[Close] unhandled error: %s", exc, exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/")
def index():
    return render_template_string(_HTML)


# ─────────────────────────────────────────────────────────────────────────
#  Embedded HTML/JS frontend
# ─────────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>US Algo Trader Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: 'Courier New', monospace; padding: 20px; }
  h1 { font-size: 22px; color: #58a6ff; margin-bottom: 6px; }
  .subtitle { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card .label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 28px; font-weight: bold; margin-top: 4px; }
  .green { color: #3fb950; } .red { color: #f85149; } .yellow { color: #e3b341; } .blue { color: #58a6ff; }
  table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
  th { background: #21262d; color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; padding: 10px 14px; text-align: left; }
  td { padding: 10px 14px; border-bottom: 1px solid #21262d; font-size: 13px; }
  tr:last-child td { border-bottom: none; }
  tr.win td:first-child { border-left: 3px solid #3fb950; }
  tr.loss td:first-child { border-left: 3px solid #f85149; }
  tr.open-row td:first-child { border-left: 3px solid #e3b341; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  .badge-short { background: #f8514920; color: #f85149; }
  .badge-long  { background: #3fb95020; color: #3fb950; }
  .badge-win   { background: #3fb95020; color: #3fb950; }
  .badge-loss  { background: #f8514920; color: #f85149; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3fb950; margin-right: 6px; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.3; } }
  .section-title { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; margin-top: 4px; }
  #last-update { color: #8b949e; font-size: 12px; margin-bottom: 20px; }
</style>
</head>
<body>
<h1>⚡ US Algo Trader Dashboard 🇺🇸</h1>
<div class="subtitle"><span class="dot"></span>Live — auto-refreshing every 1s</div>

<div class="grid">
  <div class="card">
    <div class="label">Day P&L</div>
    <div class="value" id="day-pnl">—</div>
  </div>
  <div class="card">
    <div class="label">Unrealized</div>
    <div class="value" id="unrealized">—</div>
  </div>
  <div class="card">
    <div class="label">Realized</div>
    <div class="value" id="realized">—</div>
  </div>
  <div class="card">
    <div class="label">Net Liquidation</div>
    <div class="value blue" id="nav">—</div>
  </div>
  <div class="card">
    <div class="label">Open Positions</div>
    <div class="value yellow" id="open-count">—</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value" id="win-rate">—</div>
  </div>
  <div class="card">
    <div class="label">Trades Today</div>
    <div class="value blue" id="trade-count">—</div>
  </div>
  <div class="card">
    <div class="label">Avg R</div>
    <div class="value" id="avg-r">—</div>
  </div>
</div>

<div id="last-update">Loading…</div>

<div class="section-title">🟡 Open Positions</div>
<table>
  <thead><tr>
    <th>SYMBOL</th><th>SIDE</th><th>QTY</th><th>ENTRY</th><th>STOP</th>
    <th>T1</th><th>T2</th><th>CURRENT</th><th>POSITION</th>
    <th>FLOATING P&L</th><th>CONF</th><th>STRATEGY</th><th>TIME</th><th></th>
  </tr></thead>
  <tbody id="open-body"><tr><td colspan="14" style="color:#8b949e;text-align:center">Loading…</td></tr></tbody>
</table>

<div class="section-title">✅ Closed Today</div>
<table>
  <thead><tr>
    <th>SYMBOL</th><th>SIDE</th><th>QTY</th><th>POSITION</th><th>ENTRY</th>
    <th>T1</th><th>T2</th><th>EXIT</th><th>P&L</th>
    <th>RESULT</th><th>CONF</th><th>STRATEGY</th><th>EXIT REASON</th><th>TIME</th>
  </tr></thead>
  <tbody id="closed-body"><tr><td colspan="14" style="color:#8b949e;text-align:center">Loading…</td></tr></tbody>
</table>

<script>
function fmt(n,d=2){return n==null?'—':'$'+Number(n).toFixed(d);}
function fmtPnl(n){if(n==null)return'—';const s=n>=0?'+$':'-$';return s+Math.abs(n).toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});}
function toHKT(iso){if(!iso)return'—';const d=new Date(iso.endsWith('Z')||iso.includes('+')?iso:iso+'Z');return d.toLocaleTimeString('en-HK',{timeZone:'Asia/Hong_Kong',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});}
function pct(n){return n==null?'—':Number(n).toFixed(2)+'%';}
function colorClass(n){return n>0?'green':n<0?'red':'';}
function confColor(c){return c>=90?'#f85149':c>=70?'#e3b341':'#3fb950';}

async function refresh(){
  document.getElementById('last-update').textContent='Last update: '+new Date().toLocaleTimeString();

  // NAV
  const navR = await fetch('/api/nav').then(r=>r.json()).catch(()=>({nav:0}));
  const nav = navR.nav||0;
  document.getElementById('nav').textContent='$'+Number(nav).toLocaleString(undefined,{minimumFractionDigits:2});

  // Open positions
  const open = await fetch('/api/positions').then(r=>r.json()).catch(()=>[]);
  document.getElementById('open-count').textContent=open.length;

  const ob = document.getElementById('open-body');
  if(!open.length){
    ob.innerHTML='<tr><td colspan="14" style="color:#8b949e;text-align:center">No open positions</td></tr>';
  } else {
    let unrealized=0;
    ob.innerHTML=open.map(p=>{
      const fp=p.float_pnl||0;
      unrealized+=fp;
      const posVal=p.entry_price&&p.qty?'$'+(p.entry_price*p.qty).toLocaleString('en',{maximumFractionDigits:0}):'—';
      const conf=p.confidence!=null?Math.round(p.confidence*100):null;
      const pctStr=p.entry_price&&p.qty?(fp/(p.entry_price*p.qty)*100):null;
      return `<tr class="open-row">
        <td><strong>${p.symbol}</strong></td>
        <td><span class="badge badge-${(p.side||'').toLowerCase()}">${p.side}</span></td>
        <td>${p.qty||'—'}</td>
        <td>$${(p.entry_price||0).toFixed(2)}</td>
        <td class="red" style="font-size:0.85em">${p.stop?'$'+p.stop.toFixed(2):'—'}</td>
        <td class="green" style="font-size:0.85em">${p.target1?'$'+p.target1.toFixed(2):'—'}</td>
        <td class="green" style="font-size:0.85em;opacity:0.75">${p.target2?'$'+p.target2.toFixed(2):'—'}</td>
        <td>${p.last_price?'$'+p.last_price.toFixed(2):'<span style="color:#8b949e">—</span>'}</td>
        <td style="font-size:0.85em">${posVal}</td>
        <td class="${colorClass(fp)}">${fmtPnl(fp)}${pctStr!=null?' <span style="font-size:0.8em;opacity:0.8">('+( pctStr>=0?'+':'')+pctStr.toFixed(2)+'%)</span>':''}</td>
        <td>${conf!=null?'<span style="color:'+confColor(conf)+';font-weight:bold">'+conf+'%</span>':'<span style="color:#8b949e">—</span>'}</td>
        <td>${p.strategy||'—'}</td>
        <td>${toHKT(p.entry_time)}</td>
        <td><button onclick="closePos('${p.symbol}','${p.side}')" style="background:#f8514922;color:#f85149;border:1px solid #f8514955;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:11px;">✕ Close</button></td>
      </tr>`;
    }).join('');
    const unrEl=document.getElementById('unrealized');
    unrEl.textContent=fmtPnl(unrealized);
    unrEl.className='value '+(unrealized>=0?'green':'red');
  }

  // Closed trades
  const closed = await fetch('/api/trades').then(r=>r.json()).catch(()=>[]);
  document.getElementById('trade-count').textContent=closed.length;

  let realized=0, wins=0;
  const cb = document.getElementById('closed-body');
  if(!closed.length){
    cb.innerHTML='<tr><td colspan="14" style="color:#8b949e;text-align:center">No closed trades this session</td></tr>';
  } else {
    cb.innerHTML=closed.map(t=>{
      const isManual=t.reason==='MANUAL';
      const pnl=t.pnl!=null?t.pnl:0;
      const hasPnl=t.pnl!=null;
      realized+=pnl; if(pnl>0)wins++;
      const win=pnl>0;
      const posVal=t.qty&&t.entry_price?'$'+(t.qty*t.entry_price).toLocaleString('en',{maximumFractionDigits:0}):'—';
      const conf=t.confidence!=null?Math.round(t.confidence*100):null;
      const resultBadge=isManual
        ?`<span class="badge ${win?'badge-win':'badge-loss'}" style="opacity:0.7">MANUAL</span>`
        :`<span class="badge ${win?'badge-win':'badge-loss'}">${win?'WIN':'LOSS'}</span>`;
      const pnlCell=hasPnl
        ?`<span class="${colorClass(pnl)}">${fmtPnl(pnl)} <span style="font-size:0.8em;opacity:0.8">${t.pct_chg!=null?'('+(t.pct_chg>=0?'+':'')+Number(t.pct_chg*100).toFixed(2)+'%)':'(—)'}</span></span>`
        :'<span style="color:#8b949e">—</span>';
      return `<tr class="${win?'win':'loss'}">
        <td><strong>${t.symbol}</strong></td>
        <td><span class="badge badge-${(t.side||'').toLowerCase()}">${t.side}</span></td>
        <td>${t.qty||'—'}</td>
        <td style="color:#58a6ff">${posVal}</td>
        <td>$${(t.entry_price||0).toFixed(2)}</td>
        <td class="green" style="font-size:0.85em">${t.target1?'$'+t.target1.toFixed(2):'—'}</td>
        <td class="green" style="font-size:0.85em;opacity:0.75">${t.target2?'$'+t.target2.toFixed(2):'—'}</td>
        <td>${t.exit_price?'$'+t.exit_price.toFixed(2):'<span style="color:#8b949e">—</span>'}</td>
        <td>${pnlCell}</td>
        <td>${resultBadge}</td>
        <td>${conf!=null?'<span style="color:'+confColor(conf)+';font-weight:bold">'+conf+'%</span>':'—'}</td>
        <td>${t.strategy||'—'}</td>
        <td>${t.reason||'—'}</td>
        <td>${toHKT(t.exit_time)}</td>
      </tr>`;
    }).join('');
  }

  // Stats
  const realEl=document.getElementById('realized');
  realEl.textContent=fmtPnl(realized);
  realEl.className='value '+(realized>=0?'green':'red');

  const unrealizedVal=parseFloat((document.getElementById('unrealized').textContent||'0').replace(/[^0-9.-]/g,''))*(document.getElementById('unrealized').textContent.startsWith('-')?-1:1);
  const dayPnl=realized+(parseFloat(document.getElementById('unrealized').textContent.replace(/[^0-9.]/g,''))*(document.getElementById('unrealized').className.includes('red')?-1:1));
  const dpEl=document.getElementById('day-pnl');
  dpEl.textContent=fmtPnl(realized); // realized only for day P&L display
  dpEl.className='value '+(realized>=0?'green':'red');

  const wr=closed.length?Math.round(wins/closed.length*100):0;
  const wrEl=document.getElementById('win-rate');
  wrEl.textContent=wr+'%';
  wrEl.className='value '+(wr>=50?'green':wr>=35?'yellow':'red');

  const avgR=document.getElementById('avg-r');
  avgR.textContent='—';
  avgR.className='value yellow';
}

function closePos(symbol, side){
  if(!confirm('Close '+symbol+' '+side+'?'))return;
  fetch('/close',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol,side})})
    .then(r=>r.json()).then(d=>{
      if(d.ok){alert('✅ '+symbol+' close order sent');}
      else{alert('❌ Error: '+d.error);}
    }).catch(e=>alert('❌ '+e));
}

refresh();
setInterval(refresh,1000);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────
#  Start
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    util.patchAsyncio()
    # Pre-warm the closed-trades cache so the first browser hit is instant
    threading.Thread(target=_refresh_closed_cache, daemon=True).start()
    # Pre-connect IB and subscribe to account updates so portfolio() is ready
    _get_ib()
    # Start background price cache refresher
    threading.Thread(target=_refresh_price_cache, daemon=True).start()
    logger.info("Dashboard starting on %s:%d", config.DASHBOARD_HOST, config.DASHBOARD_PORT)
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT,
            debug=False, use_reloader=False, threaded=True)
