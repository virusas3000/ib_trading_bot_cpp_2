"""
trade_quality_gate.py — pre-trade quality checks before order submission.

Called by trader.py for every candidate signal before sizing / placement.
Returns (passed: bool, reason: str).
"""
from __future__ import annotations
import sqlite3
import logging
from datetime import date, timezone, datetime

import config

logger = logging.getLogger(__name__)


def check_daily_loss(nav: float, starting_nav: float) -> tuple[bool, str]:
    """Block trading if daily drawdown exceeds MAX_DAILY_LOSS_PCT."""
    if starting_nav <= 0:
        return True, ""
    loss_pct = (starting_nav - nav) / starting_nav
    if loss_pct >= config.MAX_DAILY_LOSS_PCT:
        return False, f"MAX_DAILY_LOSS hit ({loss_pct*100:.2f}% >= {config.MAX_DAILY_LOSS_PCT*100:.0f}%)"
    return True, ""


def check_consecutive_losses(db_path: str = config.TRADES_DB) -> tuple[bool, str]:
    """Block if the last N closed trades are all losses."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT pnl FROM trades WHERE status='CLOSED' ORDER BY exit_time DESC LIMIT ?",
        (config.MAX_CONSECUTIVE_LOSSES,)
    ).fetchall()
    con.close()
    if len(rows) < config.MAX_CONSECUTIVE_LOSSES:
        return True, ""
    if all(r[0] is not None and r[0] < 0 for r in rows):
        return False, f"MAX_CONSECUTIVE_LOSSES ({config.MAX_CONSECUTIVE_LOSSES}) hit"
    return True, ""


def check_daily_commission(db_path: str = config.TRADES_DB) -> tuple[bool, str]:
    """Block if today's commission already exceeds MAX_DAILY_COMMISSION."""
    today = date.today().isoformat()
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT COALESCE(SUM(commission),0) FROM trades WHERE date(entry_time)=?",
        (today,)
    ).fetchone()
    con.close()
    total_comm = row[0] if row else 0.0
    if total_comm >= config.MAX_DAILY_COMMISSION:
        return False, f"MAX_DAILY_COMMISSION ${config.MAX_DAILY_COMMISSION:.0f} hit (${total_comm:.2f})"
    return True, ""


def check_max_positions(open_count: int) -> tuple[bool, str]:
    """Block if already at max open positions."""
    if open_count >= config.MAX_POSITIONS:
        return False, f"MAX_POSITIONS ({config.MAX_POSITIONS}) reached"
    return True, ""


def check_dollar_volume(avg_volume: float, price: float) -> tuple[bool, str]:
    """Block thinly-traded symbols."""
    dv = avg_volume * price
    if dv < config.MIN_DOLLAR_VOLUME:
        return False, f"[LOW_VOLUME] ${dv/1e6:.1f}M < ${config.MIN_DOLLAR_VOLUME/1e6:.0f}M daily"
    return True, ""


def check_duplicate_position(symbol: str,
                              db_path: str = config.TRADES_DB) -> tuple[bool, str]:
    """Block opening a second position in the same symbol."""
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT COUNT(*) FROM trades WHERE symbol=? AND status='OPEN'", (symbol,)
    ).fetchone()
    con.close()
    if row and row[0] > 0:
        return False, f"Already have open position in {symbol}"
    return True, ""


def check_no_new_positions_window() -> tuple[bool, str]:
    """Block new entries after NO_NEW_POSITIONS time (ET)."""
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    cutoff_min = config.NO_NEW_POSITIONS_HOUR * 60 + config.NO_NEW_POSITIONS_MIN
    now_min    = now_et.hour * 60 + now_et.minute
    if now_min >= cutoff_min:
        return False, f"No new positions after {config.NO_NEW_POSITIONS_HOUR}:{config.NO_NEW_POSITIONS_MIN:02d} ET"
    return True, ""


def run_all(
    symbol: str,
    nav: float,
    starting_nav: float,
    open_count: int,
    avg_volume: float,
    price: float,
    db_path: str = config.TRADES_DB,
) -> tuple[bool, str]:
    """Run every gate in sequence; return first failure or (True, '')."""
    checks = [
        lambda: check_no_new_positions_window(),
        lambda: check_daily_loss(nav, starting_nav),
        lambda: check_consecutive_losses(db_path),
        lambda: check_daily_commission(db_path),
        lambda: check_max_positions(open_count),
        lambda: check_dollar_volume(avg_volume, price),
        lambda: check_duplicate_position(symbol, db_path),
    ]
    for fn in checks:
        ok, reason = fn()
        if not ok:
            logger.warning("[QUALITY_GATE] %s blocked: %s", symbol, reason)
            return False, reason
    return True, ""
