"""
telegram_notifier.py — send alerts to the Hermes Swarm thread.

All sends are fire-and-forget async; errors are logged but never raise.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone

import aiohttp

import config

logger = logging.getLogger(__name__)

_BASE = "https://api.telegram.org/bot{token}/{method}"


async def _post(method: str, payload: dict) -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.debug("[Telegram] No credentials — skipping %s", method)
        return
    url = _BASE.format(token=config.TELEGRAM_BOT_TOKEN, method=method)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("[Telegram] %s returned %d: %s", method, resp.status, body[:200])
    except Exception as exc:
        logger.warning("[Telegram] send failed: %s", exc)


class TelegramNotifier:
    """Async Telegram notifier — sends to TELEGRAM_CHAT_ID / thread TELEGRAM_THREAD_ID."""

    async def send_message(self, text: str, parse_mode: str = "HTML") -> None:
        payload: dict = {
            "chat_id":                  config.TELEGRAM_CHAT_ID,
            "text":                     text,
            "parse_mode":               parse_mode,
            "disable_web_page_preview": True,
        }
        if config.TELEGRAM_THREAD_ID:
            payload["message_thread_id"] = config.TELEGRAM_THREAD_ID
        await _post("sendMessage", payload)

    # ── Trade alerts ──────────────────────────────────────────────────────

    async def send_entry(self, symbol: str, side: str, qty: int,
                          entry: float, stop: float, t1: float, t2: float,
                          strategy: str, confidence: float) -> None:
        emoji = "🟢" if side == "LONG" else "🔴"
        ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        text  = (
            f"{emoji} <b>ENTRY {side} {symbol}</b> [{strategy}]\n"
            f"Price: ${entry:.2f}  Qty: {qty}\n"
            f"Stop: ${stop:.2f}  T1: ${t1:.2f}  T2: ${t2:.2f}\n"
            f"Conf: {confidence:.0%}  {ts}"
        )
        await self.send_message(text)

    async def send_exit(self, symbol: str, side: str, qty: int,
                         entry: float, exit_price: float, pnl: float,
                         reason: str, strategy: str) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        pct   = ((exit_price - entry) / entry) * (1 if side == "LONG" else -1) * 100
        ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        text  = (
            f"{emoji} <b>EXIT {side} {symbol}</b> [{reason}]\n"
            f"Entry: ${entry:.2f}  Exit: ${exit_price:.2f}  ({pct:+.2f}%)\n"
            f"P&L: ${pnl:+.2f}  Strategy: {strategy}  {ts}"
        )
        await self.send_message(text)

    async def send_eod_summary(self, date_str: str, nav: float,
                                pnl: float, num_trades: int,
                                wins: int, losses: int,
                                commission: float) -> None:
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0.0
        emoji    = "📈" if pnl >= 0 else "📉"
        text     = (
            f"{emoji} <b>EOD Summary {date_str}</b>\n"
            f"NAV: ${nav:,.2f}  P&L: ${pnl:+,.2f}\n"
            f"Trades: {num_trades}  W/L: {wins}/{losses} ({win_rate:.0f}%)\n"
            f"Commission: ${commission:.2f}"
        )
        await self.send_message(text)

    async def send_alert(self, title: str, body: str) -> None:
        text = f"⚠️ <b>{title}</b>\n{body}"
        await self.send_message(text)

    async def send_startup(self, port: int, nav: float, positions: int) -> None:
        text = (
            f"🚀 <b>US Bot started</b> (port {port})\n"
            f"NAV: ${nav:,.2f}  Existing positions: {positions}"
        )
        await self.send_message(text)
