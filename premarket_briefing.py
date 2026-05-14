"""
premarket_briefing.py — morning briefing sent via Telegram at 9:00 AM ET.

Cron: runs at 9:00 PM HKT = 9:00 AM ET (UTC 13:00) Mon–Fri
  0 13 * * 1-5  python3 ~/Desktop/ib_algo_trader/premarket_briefing.py

Covers:
  - Today's expected market session (ET open/close times)
  - Active strategies
  - Open positions carried from previous session (if any)
  - Recent P&L summary (last 5 trading days)
  - ML model age
  - Any critical config changes to review
"""
from __future__ import annotations
import asyncio
import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timezone

import config
from telegram_notifier import TelegramNotifier

logger = logging.getLogger("premarket")
logging.basicConfig(level=logging.INFO)

ET_OPEN  = "9:30 AM ET"
ET_CLOSE = "4:00 PM ET"


def open_position_count() -> int:
    if not os.path.exists(config.TRADES_DB):
        return 0
    con = sqlite3.connect(config.TRADES_DB)
    row = con.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()
    con.close()
    return row[0] if row else 0


def recent_pnl_summary(days: int = 5) -> str:
    if not os.path.exists(config.TRADES_DB):
        return "No trade history."
    con = sqlite3.connect(config.TRADES_DB)
    rows = con.execute(
        """SELECT date(exit_time) AS d, SUM(pnl) AS daily_pnl, COUNT(*) AS n
           FROM trades WHERE status='CLOSED' AND exit_time IS NOT NULL
           GROUP BY d ORDER BY d DESC LIMIT ?""",
        (days,),
    ).fetchall()
    con.close()
    if not rows:
        return "No recent closed trades."
    lines = []
    for d, pnl, n in reversed(rows):
        emoji = "📈" if (pnl or 0) >= 0 else "📉"
        lines.append(f"  {emoji} {d}: ${(pnl or 0):+,.2f} ({n} trades)")
    return "\n".join(lines)


def ml_model_age() -> str:
    if not os.path.exists(config.ML_MODEL_PATH):
        return "Model not found"
    age_h = (time.time() - os.path.getmtime(config.ML_MODEL_PATH)) / 3600
    return f"{age_h:.1f}h old"


async def send_briefing() -> None:
    n        = TelegramNotifier()
    today    = date.today().strftime("%A %d %b %Y")
    open_pos = open_position_count()
    active   = [s for s in [
        "SUPPORT_RESISTANCE",
        "MOMENTUM", "SMC_ORDER_BLOCK", "GAP_AND_GO",
        "MACD_CROSS", "RSI_DIVERGENCE", "ROUND_NUMBER_MAGNET",
        "VWAP_REVERSION", "HOD_BREAKOUT",
    ] if s not in config.DISABLED_STRATEGIES]

    text = (
        f"☀️ <b>Premarket Briefing — {today}</b>\n\n"
        f"<b>Session:</b> {ET_OPEN} → {ET_CLOSE}\n"
        f"<b>Active strategies:</b> {', '.join(active) or 'None'}\n"
        f"<b>Open positions:</b> {open_pos}\n"
        f"<b>ML model:</b> {ml_model_age()}\n\n"
        f"<b>Recent P&L (last 5 days):</b>\n"
        f"{recent_pnl_summary(5)}\n\n"
        f"<i>Remember: US session = 13:30–20:00 UTC (21:30–04:00 HKT)</i>"
    )
    await n.send_message(text)
    logger.info("Premarket briefing sent")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    asyncio.run(send_briefing())
