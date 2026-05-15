"""
trader.py — US Algo Trader main bot.

Bot:     ~/Desktop/ib_algo_trader/trader.py
Account: TWS Paper DU4127455 (port 7497)
Client:  CLIENT_ID = 32

Architecture:
  - ib_insync event loop for IB connectivity
  - IB streaming 1-min bars (reqHistoricalData keepUpToDate=True) for all watchlist symbols
  - On each bar: run strategy engine → quality gate → ML/LLM confidence → size → order
  - Positions restored on restart via reqPositions (avgCost fallback)
  - EOD force-close at 15:45 ET
  - Telegram alerts throughout
"""
from __future__ import annotations
import asyncio
import logging
import os
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from ib_insync import (
    IB, Contract, MarketOrder, StopOrder,
    BarData, util,
)
from zoneinfo import ZoneInfo

import config
from llm_confidence    import get_llm_confidence
from ml_integration    import MLTradingIntegration
from strategy          import StrategyEngine
from telegram_notifier import TelegramNotifier
from trade_quality_gate import run_all as quality_gate

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trader.log", mode="a"),
    ],
)
logger = logging.getLogger("trader")

ET = ZoneInfo("America/New_York")


@dataclass
class Bar:
    date:   object
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


# ─────────────────────────────────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────────────────────────────────

class BotState:
    def __init__(self) -> None:
        self.nav:          float = 0.0
        self.starting_nav: float = 0.0
        self.positions:    dict[str, dict] = {}   # symbol → position dict
        self.bar_data:     dict[str, list[Bar]] = {}
        self.prev_close:   dict[str, float] = {}
        self.daily_commission: float = 0.0
        self.halted:       bool  = False
        self.ib_order_map: dict[int, str] = {}    # orderId → symbol

    @property
    def open_count(self) -> int:
        return len(self.positions)


state  = BotState()
ib     = IB()
notifier   = TelegramNotifier()
ml_engine  = MLTradingIntegration()
strategy   = StrategyEngine()


# ─────────────────────────────────────────────────────────────────────────
#  Database helpers
# ─────────────────────────────────────────────────────────────────────────

def db_open_position(symbol: str, side: str, qty: int,
                      entry: float, stop: float, t1: float, t2: float,
                      strategy_name: str, confidence: float,
                      ib_order_id: int) -> int:
    con = sqlite3.connect(config.TRADES_DB)
    cur = con.execute(
        """INSERT INTO trades
           (symbol, side, qty, entry_price, entry_time, strategy, confidence,
            stop, target1, target2, status, ib_order_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,'OPEN',?)""",
        (symbol, side, qty, entry,
         datetime.now(timezone.utc).isoformat(),
         strategy_name, confidence, stop, t1, t2, ib_order_id),
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def db_close_position(row_id: int, exit_price: float, pnl: float,
                       pct_chg: float, reason: str,
                       commission: float = 0.0) -> None:
    con = sqlite3.connect(config.TRADES_DB)
    con.execute(
        """UPDATE trades SET exit_price=?, exit_time=?, pnl=?, pct_chg=?,
           reason=?, status='CLOSED', commission=?
           WHERE id=?""",
        (exit_price, datetime.now(timezone.utc).isoformat(),
         pnl, pct_chg, reason, commission, row_id),
    )
    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────
#  IB helpers
# ─────────────────────────────────────────────────────────────────────────

def make_contract(symbol: str) -> Contract:
    return Contract(symbol=symbol, secType="STK",
                    exchange="SMART", currency="USD")


def get_nav() -> float:
    # Try accountValues first (fast if already populated)
    for v in ib.accountValues():
        if v.tag == "NetLiquidation" and v.currency == "USD":
            return float(v.value)
    # Fallback: accountSummary (works even before reqAccountUpdates completes)
    try:
        summary = ib.accountSummary()
        for s in summary:
            if s.tag == "NetLiquidation" and s.currency == "USD":
                return float(s.value)
    except Exception:
        pass
    return 0.0


def get_avg_volume_20(symbol: str) -> float:
    """Return estimated daily volume from 1-min bars (20-bar avg scaled × 390 min/day)."""
    bars = state.bar_data.get(symbol, [])
    if len(bars) < 2:
        return 0.0
    vols = [b.volume for b in bars[-21:-1]]
    avg_per_min = sum(vols) / len(vols) if vols else 0.0
    return avg_per_min * 390


# ─────────────────────────────────────────────────────────────────────────
#  Position restore on startup
# ─────────────────────────────────────────────────────────────────────────

def restore_positions() -> int:
    """
    Restore open positions from IB on bot restart.
    Uses avgCost as fallback entry price when trades.db record missing.
    Does NOT auto-close existing positions.
    """
    ib_positions = ib.positions()
    restored = 0

    con = sqlite3.connect(config.TRADES_DB)
    open_rows = {r[0]: r for r in con.execute(
        "SELECT symbol, id, side, qty, entry_price, stop, target1, target2, "
        "strategy, confidence FROM trades WHERE status='OPEN'"
    ).fetchall()}
    con.close()

    for pos in ib_positions:
        sym = pos.contract.symbol
        if sym not in config.WATCHLIST:
            continue

        qty_ib = int(abs(pos.position))
        side   = "LONG" if pos.position > 0 else "SHORT"

        if sym in open_rows:
            r = open_rows[sym]
            state.positions[sym] = {
                "row_id":    r[1],
                "side":      r[2],
                "qty":       r[3],
                "entry":     r[4],
                "stop":      r[5],
                "target1":   r[6],
                "target2":   r[7],
                "strategy":  r[8],
                "confidence":r[9],
            }
        else:
            # avgCost fallback — no DB record
            avg_cost = float(pos.avgCost)
            stop_pct = (config.VOLATILE_STOP_PCT
                        if sym in config.VOLATILE_SYMBOLS
                        else config.DEFAULT_STOP_PCT)
            stop = avg_cost * (1 - stop_pct) if side == "LONG" else avg_cost * (1 + stop_pct)
            row_id = db_open_position(
                sym, side, qty_ib, avg_cost, stop,
                avg_cost * (1 + config.TARGET1_PCT) if side == "LONG" else avg_cost * (1 - config.TARGET1_PCT),
                avg_cost * (1 + config.TARGET2_PCT) if side == "LONG" else avg_cost * (1 - config.TARGET2_PCT),
                "RESTORED", 0.5, 0,
            )
            t1_restored = avg_cost * (1 + config.TARGET1_PCT) if side == "LONG" else avg_cost * (1 - config.TARGET1_PCT)
            t2_restored = avg_cost * (1 + config.TARGET2_PCT) if side == "LONG" else avg_cost * (1 - config.TARGET2_PCT)
            state.positions[sym] = {
                "row_id": row_id, "side": side, "qty": qty_ib,
                "entry": avg_cost, "stop": stop,
                "target1": t1_restored, "target2": t2_restored,
                "strategy": "RESTORED", "confidence": 0.5,
            }
            logger.warning("[RESTORE] %s avgCost=%.2f (no DB record)", sym, avg_cost)

        restored += 1
        logger.info("[RESTORE] %s %s qty=%d entry=%.2f",
                    sym, side, qty_ib, state.positions[sym]["entry"])

    return restored


# ─────────────────────────────────────────────────────────────────────────
#  Order management
# ─────────────────────────────────────────────────────────────────────────

async def place_entry(symbol: str, signal: dict, nav: float) -> None:
    try:
        await _place_entry(symbol, signal, nav)
    except Exception:
        logger.exception("[PLACE_ENTRY_ERROR] %s", symbol)


async def _place_entry(symbol: str, signal: dict, nav: float) -> None:
    if state.halted:
        logger.info("[SKIP] %s: bot is halted", symbol)
        return

    direction = signal["signal"]
    entry     = signal.get("entry_price", 0.0)
    stop_p    = signal["stop"]
    t1        = signal["target1"]
    t2        = signal["target2"]
    strat     = signal["strategy"]
    conf      = signal["confidence"]

    logger.info("[SIGNAL] %s %s entry=%.2f stop=%.2f t1=%.2f conf=%.2f strat=%s",
                symbol, direction, entry, stop_p, t1, conf, strat)

    if entry <= 0:
        logger.info("[SKIP] %s: entry price is zero", symbol)
        return

    if nav <= 0:
        logger.info("[SKIP] %s: NAV is zero (nav=%.2f)", symbol, nav)
        return

    # ── Position sizing via C++ ────────────────────────────────────────
    try:
        import trading_engine as te
        sz = te.calc_size(nav, config.MAX_POSITION_PCT, config.MAX_RISK_PCT,
                          entry, stop_p, t1,
                          config.MIN_T1_PROFIT_USD, config.MIN_POSITION_VALUE)
        qty = sz.qty
        if qty == 0:
            logger.info("[SKIP_MIN_PROFIT] %s %s entry=%.2f t1=%.2f reason=%s",
                        symbol, direction, entry, t1, sz.reason)
            return
    except ImportError:
        qty = max(1, int(nav * config.MAX_POSITION_PCT / entry))

    logger.info("[SIZING] %s qty=%d nav=%.2f", symbol, qty, nav)

    # ── Quality gate ──────────────────────────────────────────────────
    avg_vol = get_avg_volume_20(symbol)
    logger.info("[QUALITY] %s avg_vol=%.0f entry=%.2f dollar_vol=$%.1fM",
                symbol, avg_vol, entry, avg_vol * entry / 1e6)
    ok, reason = quality_gate(
        symbol, nav, state.starting_nav,
        state.open_count, avg_vol, entry,
    )
    if not ok:
        logger.info("[BLOCKED] %s: %s", symbol, reason)
        return

    # ── ML confidence ─────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    features = MLTradingIntegration.build_features(
        rsi14        = signal.get("rsi14",  50.0),
        rvol         = signal.get("rvol",    1.0),
        atr_pct      = signal.get("atr_pct", 0.005),
        macd_hist    = signal.get("macd_hist", 0.0),
        bb_pct       = signal.get("bb_pct", 0.5),
        vwap_dev_pct = signal.get("vwap_dev_pct", 0.0),
        entry_time   = now_utc,
    )
    ml_conf  = ml_engine.predict_proba(features)
    llm_conf = await get_llm_confidence(symbol, direction, strat,
                                         {"price": entry, "rsi": signal.get("rsi14", 0)})
    combined_conf = (conf + ml_conf + llm_conf) / 3.0

    logger.info("[CONF] %s signal=%.2f ml=%.2f llm=%.2f combined=%.2f threshold=%.2f",
                symbol, conf, ml_conf, llm_conf, combined_conf, config.MIN_SIGNAL_CONFIDENCE)

    if combined_conf < config.MIN_SIGNAL_CONFIDENCE:
        logger.info("[LOW_CONF] %s %s combined=%.2f < %.2f",
                    symbol, direction, combined_conf, config.MIN_SIGNAL_CONFIDENCE)
        return

    # ── Place order ───────────────────────────────────────────────────
    contract    = make_contract(symbol)
    action      = "BUY" if direction == "LONG" else "SELL"
    entry_order = MarketOrder(action, qty)
    trade = ib.placeOrder(contract, entry_order)
    await asyncio.sleep(0.5)
    fill_price = entry

    stop_action = "SELL" if direction == "LONG" else "BUY"
    stop_order  = StopOrder(stop_action, qty, stop_p)
    ib.placeOrder(contract, stop_order)

    row_id = db_open_position(
        symbol, direction, qty, fill_price,
        stop_p, t1, t2, strat, combined_conf,
        trade.order.orderId,
    )
    state.positions[symbol] = {
        "row_id":    row_id,
        "side":      direction,
        "qty":       qty,
        "entry":     fill_price,
        "stop":      stop_p,
        "target1":   t1,
        "target2":   t2,
        "strategy":  strat,
        "confidence": combined_conf,
    }

    logger.info("ORDER PLACED %s %s %d @ %.2f stop=%.2f conf=%.2f",
                direction, symbol, qty, fill_price, stop_p, combined_conf)
    await notifier.send_entry(symbol, direction, qty, fill_price,
                               stop_p, t1, t2, strat, combined_conf)


async def close_position(symbol: str, reason: str) -> None:
    if symbol not in state.positions:
        return

    pos        = state.positions[symbol]
    side       = pos["side"]
    qty        = pos["qty"]
    entry      = pos["entry"]
    close_action = "SELL" if side == "LONG" else "BUY"

    contract = make_contract(symbol)
    ib.placeOrder(contract, MarketOrder(close_action, qty))
    await asyncio.sleep(0.2)

    # use last known price as proxy exit
    bars = state.bar_data.get(symbol, [])
    exit_price = bars[-1].close if bars else entry

    pnl     = (exit_price - entry) * qty * (1 if side == "LONG" else -1)
    pct_chg = (exit_price - entry) / entry * (1 if side == "LONG" else -1)

    db_close_position(pos["row_id"], exit_price, pnl, pct_chg, reason)
    await notifier.send_exit(symbol, side, qty, entry, exit_price, pnl, reason, pos["strategy"])

    del state.positions[symbol]
    logger.info("CLOSED %s %s @ %.2f pnl=$%.2f reason=%s",
                side, symbol, exit_price, pnl, reason)


# ─────────────────────────────────────────────────────────────────────────
#  Bar processing
# ─────────────────────────────────────────────────────────────────────────

def build_df(symbol: str) -> Optional[pd.DataFrame]:
    bars = state.bar_data.get(symbol, [])
    if len(bars) < 30:
        return None
    data = {
        "open":   [b.open   for b in bars],
        "high":   [b.high   for b in bars],
        "low":    [b.low    for b in bars],
        "close":  [b.close  for b in bars],
        "volume": [b.volume for b in bars],
    }
    return pd.DataFrame(data)


async def _start_ib_bars(contracts: dict) -> None:
    """Subscribe to IB 1-min streaming bars (keepUpToDate=True) for all watchlist symbols."""
    today = datetime.now(ET).date()

    for sym, contract in contracts.items():
        try:
            bar_list = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                keepUpToDate=True,
            )

            # Seed prev_close from last bar before today
            for b in reversed(list(bar_list)[:-1]):
                bar_date = b.date.date() if hasattr(b.date, "date") else b.date
                if bar_date < today:
                    state.prev_close[sym] = b.close
                    break

            def _make_handler(s: str):
                def _handler(bars, has_new: bool) -> None:
                    ib_bars = [
                        Bar(date=b.date, open=b.open, high=b.high,
                            low=b.low, close=b.close, volume=b.volume)
                        for b in bars
                    ]
                    asyncio.ensure_future(on_bar(ib_bars, has_new, s))
                return _handler

            bar_list.updateEvent += _make_handler(sym)
            logger.info("[IB_BARS] Subscribed to %s (%d bars loaded)", sym, len(bar_list))
        except Exception as exc:
            logger.error("[IB_BARS] Failed to subscribe %s: %s", sym, exc)
        await asyncio.sleep(0.1)


async def on_bar(bars, has_new_bar: bool, symbol: str) -> None:
    if not has_new_bar:
        return

    state.bar_data[symbol] = list(bars)

    # Check stop / target1 for open positions
    if symbol in state.positions:
        pos   = state.positions[symbol]
        price = bars[-1].close
        side  = pos["side"]

        if side == "LONG":
            if price <= pos["stop"]:
                await close_position(symbol, "STOP")
                return
            if price >= pos["target1"]:
                await close_position(symbol, "TARGET1")
                return
        else:
            if price >= pos["stop"]:
                await close_position(symbol, "STOP")
                return
            if price <= pos["target1"]:
                await close_position(symbol, "TARGET1")
                return

    # Hard-stop failsafe (0.2%)
    if symbol in state.positions:
        pos   = state.positions[symbol]
        price = bars[-1].close
        hard_stop_pct = config.HARD_STOP_PCT
        hard_stop = (pos["entry"] * (1 - hard_stop_pct) if pos["side"] == "LONG"
                     else pos["entry"] * (1 + hard_stop_pct))
        if (pos["side"] == "LONG"  and price <= hard_stop) or \
           (pos["side"] == "SHORT" and price >= hard_stop):
            await close_position(symbol, "HARD_STOP")
            return

    if state.halted or symbol in state.positions:
        return

    # ── EOD window check ──────────────────────────────────────────────
    now_et    = datetime.now(ET)
    cutoff_min = config.NO_NEW_POSITIONS_HOUR * 60 + config.NO_NEW_POSITIONS_MIN
    if now_et.hour * 60 + now_et.minute >= cutoff_min:
        return

    # ── Signal generation ─────────────────────────────────────────────
    df = build_df(symbol)
    if df is None:
        return

    prev_close = state.prev_close.get(symbol, 0.0)
    sig = strategy.run_all(df, symbol, prev_close)
    if sig is None:
        return

    sig["entry_price"] = df["close"].iloc[-1]

    state.nav = get_nav()
    await place_entry(symbol, sig, state.nav)


# ─────────────────────────────────────────────────────────────────────────
#  EOD force-close
# ─────────────────────────────────────────────────────────────────────────

async def eod_force_close() -> None:
    logger.info("[EOD] Force-closing all %d open positions", len(state.positions))
    for sym in list(state.positions.keys()):
        await close_position(sym, "FORCE_CLOSE")

    # EOD summary
    today = date.today().isoformat()
    con   = sqlite3.connect(config.TRADES_DB)
    rows  = con.execute(
        "SELECT pnl, commission FROM trades WHERE status='CLOSED' AND date(exit_time)=?",
        (today,)
    ).fetchall()
    con.close()

    total_pnl  = sum(r[0] for r in rows if r[0] is not None)
    total_comm = sum(r[1] for r in rows if r[1] is not None)
    wins       = sum(1 for r in rows if r[0] is not None and r[0] > 0)
    losses     = sum(1 for r in rows if r[0] is not None and r[0] < 0)

    await notifier.send_eod_summary(today, state.nav, total_pnl,
                                     len(rows), wins, losses, total_comm)


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

async def main() -> None:
    # ── Connect ───────────────────────────────────────────────────────
    logger.info("Connecting to IB on %s:%d (clientId=%d)...",
                config.IB_HOST, config.IB_PORT, config.CLIENT_ID)
    await ib.connectAsync(config.IB_HOST, config.IB_PORT,
                          clientId=config.CLIENT_ID)
    logger.info("Connected to IB")

    # ── Account info ──────────────────────────────────────────────────
    # Subscribe to account updates (non-blocking fire-and-forget)
    ib.client.reqAccountUpdates(True, "")
    # Give IB 3s to stream account data, then fall back to accountSummary
    await asyncio.sleep(3)
    state.nav = get_nav()
    state.starting_nav = state.nav
    logger.info("Account equity: $%.2f", state.nav)
    if state.nav <= 0:
        logger.warning("NAV still zero after wait — proceeding with $100k placeholder")
        state.nav = 100_000.0
        state.starting_nav = state.nav

    # ── ML model ─────────────────────────────────────────────────────
    logger.info("ML model loaded: %s", ml_engine._model is not None)

    # ── Restore positions ─────────────────────────────────────────────
    n_restored = restore_positions()
    logger.info("Existing IB positions restored: %d", n_restored)

    # ── Watchlist — qualify contracts for IB order execution ─────────────
    logger.info("Qualifying %d contracts for order execution...", len(config.WATCHLIST))
    contracts = {}
    for sym in config.WATCHLIST:
        contract = make_contract(sym)
        ib.qualifyContracts(contract)
        contracts[sym] = contract

    await _start_ib_bars(contracts)
    logger.info("IB streaming bars started (%d symbols)", len(config.WATCHLIST))
    await notifier.send_startup(config.IB_PORT, state.nav, n_restored)

    # ── EOD scheduler ─────────────────────────────────────────────────
    force_close_done = False

    async def _scheduler() -> None:
        nonlocal force_close_done
        while True:
            now_et = datetime.now(ET)
            fc_min = config.FORCE_CLOSE_HOUR * 60 + config.FORCE_CLOSE_MIN
            if now_et.hour * 60 + now_et.minute >= fc_min and not force_close_done:
                await eod_force_close()
                force_close_done = True
            await asyncio.sleep(30)

    asyncio.ensure_future(_scheduler())

    # ── Run until interrupted ─────────────────────────────────────────
    logger.info("Bot running. Press Ctrl-C to stop.")
    # Keep event loop alive indefinitely
    while True:
        await asyncio.sleep(60)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def _shutdown(signum, frame):
        logger.info("Shutdown signal received")
        ib.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    util.startLoop()
    asyncio.run(main())
