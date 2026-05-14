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

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
from ib_insync import IB, ExecutionFilter, util

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
            except Exception as exc:
                logger.warning("[Dashboard] IB connect failed: %s", exc)
        return _ib_dash


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

    # get last prices from IB
    ib = _get_ib()
    last_prices: dict[str, float] = {}
    if ib.isConnected():
        try:
            syms = list({r["symbol"] for r in rows})
            for sym in syms:
                from ib_insync import Contract
                c = Contract(symbol=sym, secType="STK",
                             exchange="SMART", currency="USD")
                ib.qualifyContracts(c)
                ticker = ib.reqMktData(c, "", False, False)
                ib.sleep(0.3)
                if ticker.last and ticker.last > 0:
                    last_prices[sym] = ticker.last
                ib.cancelMktData(c)
        except Exception as exc:
            logger.warning("[Dashboard] last price fetch: %s", exc)

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


def fetch_closed_trades() -> list[dict]:
    """
    Closed trades from IB executions API (clientId 96).
    Falls back to trades.db if IB unavailable.
    Filtered to today's session (≥ 13:30 UTC).
    """
    ib       = _get_ib()
    session  = _session_start_utc()

    if ib.isConnected():
        try:
            filt  = ExecutionFilter()
            execs = ib.reqExecutions(filt)
            # pair up buys and sells for same symbol
            buys:  dict[str, list] = {}
            sells: dict[str, list] = {}
            for ex in execs:
                ts = datetime.fromisoformat(
                    ex.execution.time.replace("  ", " ").split(" ")[0] + "T" +
                    ex.execution.time.split(" ")[-1]
                ).replace(tzinfo=timezone.utc) if len(ex.execution.time) > 8 else session
                if ts < session:
                    continue
                sym    = ex.contract.symbol
                action = ex.execution.side.upper()  # BOT / SLD
                side   = "LONG" if action == "BOT" else "SHORT"
                entry  = {"sym": sym, "price": ex.execution.price,
                          "qty": ex.execution.shares, "side": side, "time": ts}
                if action == "BOT":
                    buys.setdefault(sym, []).append(entry)
                else:
                    sells.setdefault(sym, []).append(entry)

            trades = []
            for sym, buy_list in buys.items():
                sell_list = sells.get(sym, [])
                for b, s in zip(buy_list, sell_list):
                    entry_p = b["price"]
                    exit_p  = s["price"]
                    qty     = min(b["qty"], s["qty"])
                    pnl     = (exit_p - entry_p) * qty
                    pct_chg = (exit_p - entry_p) / entry_p * 100
                    trades.append({
                        "symbol":      sym,
                        "side":        "LONG",
                        "qty":         qty,
                        "entry_price": entry_p,
                        "exit_price":  exit_p,
                        "pnl":         round(pnl, 2),
                        "pct_chg":     round(pct_chg, 2),
                        "strategy":    "—",
                        "reason":      "—",
                        "exit_time":   str(s["time"]),
                        "confidence":  None,
                    })
            # enrich strategy/reason from trades.db where available
            _enrich_from_db(trades, session)
            return sorted(trades, key=lambda t: t["exit_time"], reverse=True)
        except Exception as exc:
            logger.warning("[Dashboard] IB executions failed, using DB: %s", exc)

    # DB fallback
    con = sqlite3.connect(config.TRADES_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT symbol, side, qty, entry_price, exit_price, pnl, pct_chg,
                  strategy, reason, exit_time, confidence
           FROM trades WHERE status='CLOSED' AND exit_time>=?
           ORDER BY exit_time DESC""",
        (session.isoformat(),)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _enrich_from_db(trades: list[dict], since: datetime) -> None:
    """Add strategy/reason from trades.db to IB-sourced executions."""
    try:
        con = sqlite3.connect(config.TRADES_DB)
        rows = con.execute(
            "SELECT symbol, strategy, reason, confidence FROM trades "
            "WHERE status='CLOSED' AND exit_time>=?", (since.isoformat(),)
        ).fetchall()
        con.close()
        db_map = {r[0]: (r[1], r[2], r[3]) for r in rows}
        for t in trades:
            if t["symbol"] in db_map:
                t["strategy"], t["reason"], t["confidence"] = db_map[t["symbol"]]
    except Exception:
        pass


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
    return jsonify(fetch_closed_trades())


@app.route("/api/nav")
def api_nav():
    return jsonify({"nav": fetch_nav()})


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
<meta http-equiv="refresh" content="15">
<style>
  body{font-family:monospace;background:#111;color:#eee;margin:1rem 2rem}
  h2{color:#7af}
  table{border-collapse:collapse;width:100%;margin-bottom:2rem}
  th{background:#222;color:#7af;padding:6px 10px;text-align:left;border-bottom:2px solid #444}
  td{padding:5px 10px;border-bottom:1px solid #222}
  .pos{color:#2f2}
  .neg{color:#f44}
  .nav-box{font-size:1.4rem;margin-bottom:1rem;color:#ff9}
  #ts{font-size:.8rem;color:#777}
</style>
</head>
<body>
<div class="nav-box" id="nav-box">NAV: loading…</div>
<span id="ts"></span>
<h2>Open Positions</h2>
<table id="open-tbl">
<thead><tr>
  <th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Last</th>
  <th>Float P&L</th><th>Entry Time</th><th>Strategy</th>
  <th>Conf</th><th>Stop</th><th>T1</th><th>T2</th>
</tr></thead>
<tbody id="open-body"><tr><td colspan="12">Loading…</td></tr></tbody>
</table>

<h2>Closed Trades (today's session)</h2>
<table id="closed-tbl">
<thead><tr>
  <th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th>
  <th>P&L</th><th>%</th><th>Strategy</th><th>Reason</th>
  <th>Exit Time</th><th>Conf</th>
</tr></thead>
<tbody id="closed-body"><tr><td colspan="11">Loading…</td></tr></tbody>
</table>

<script>
function fmt(n,d=2){return n==null?'—':'$'+Number(n).toFixed(d);}
function pct(n){return n==null?'—':Number(n).toFixed(2)+'%';}
function cls(n){return n>0?'pos':n<0?'neg':'';}

async function refresh(){
  document.getElementById('ts').textContent='Last refresh: '+new Date().toLocaleTimeString();

  // NAV
  const navR = await fetch('/api/nav').then(r=>r.json()).catch(()=>({nav:0}));
  document.getElementById('nav-box').textContent='NAV: $'+Number(navR.nav).toLocaleString(undefined,{minimumFractionDigits:2});

  // Open positions
  const open = await fetch('/api/positions').then(r=>r.json()).catch(()=>[]);
  const ob = document.getElementById('open-body');
  if(!open.length){ob.innerHTML='<tr><td colspan="12">No open positions</td></tr>';return;}
  ob.innerHTML=open.map(p=>`<tr>
    <td>${p.symbol}</td>
    <td>${p.side}</td>
    <td>${p.qty}</td>
    <td>${fmt(p.entry_price)}</td>
    <td>${fmt(p.last_price)}</td>
    <td class="${cls(p.float_pnl)}">${fmt(p.float_pnl)}</td>
    <td>${p.entry_time?p.entry_time.slice(0,16):''}</td>
    <td>${p.strategy||''}</td>
    <td>${p.confidence?(p.confidence*100).toFixed(0)+'%':'—'}</td>
    <td>${fmt(p.stop)}</td>
    <td>${fmt(p.target1)}</td>
    <td>${fmt(p.target2)}</td>
  </tr>`).join('');

  // Closed trades
  const closed = await fetch('/api/trades').then(r=>r.json()).catch(()=>[]);
  const cb = document.getElementById('closed-body');
  if(!closed.length){cb.innerHTML='<tr><td colspan="11">No closed trades this session</td></tr>';return;}
  cb.innerHTML=closed.map(t=>`<tr>
    <td>${t.symbol}</td>
    <td>${t.side}</td>
    <td>${t.qty}</td>
    <td>${fmt(t.entry_price)}</td>
    <td>${fmt(t.exit_price)}</td>
    <td class="${cls(t.pnl)}">${fmt(t.pnl)}</td>
    <td class="${cls(t.pct_chg)}">${pct(t.pct_chg)}</td>
    <td>${t.strategy||'—'}</td>
    <td>${t.reason||'—'}</td>
    <td>${t.exit_time?t.exit_time.slice(0,16):''}</td>
    <td>${t.confidence?(t.confidence*100).toFixed(0)+'%':'—'}</td>
  </tr>`).join('');
}
refresh();
setInterval(refresh,15000);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────
#  Start
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    util.patchAsyncio()
    logger.info("Dashboard starting on %s:%d", config.DASHBOARD_HOST, config.DASHBOARD_PORT)
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT,
            debug=False, use_reloader=False)
