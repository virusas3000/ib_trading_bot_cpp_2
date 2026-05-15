"""
fetch_premarket_test.py — fetch live pre-market bars from IB and run all
strategy signal tests. Works outside RTH (useRTH=False).

Usage: python3 fetch_premarket_test.py
"""
from __future__ import annotations
import logging
import math
import sqlite3
import sys
from datetime import datetime, timezone

import pandas as pd
from ib_insync import IB, Contract, util

import config
from strategy import (
    check_support_resistance,
    check_vwap_reversion,
    check_vwap_reclaim,
    check_gap_and_go,
    check_hod_breakout,
    check_abcd_pattern,
    check_fallen_angel,
    check_breakout,
    check_compression_breakout,
    check_round_number_magnet,
)

logging.basicConfig(level=logging.WARNING)  # suppress ib_insync noise

PASS = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"
WARN = "\033[93m⚠️ \033[0m"
INFO = "\033[94mℹ️ \033[0m"

# Fetch a manageable subset for speed; covers diverse sectors
TEST_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD",
                "META", "AMZN", "GOOGL"]


def fetch_bars(ib: IB, symbol: str, n_bars: int = 90) -> list:
    """Request n_bars of 1-min data including pre-market (useRTH=False)."""
    contract = Contract(symbol=symbol, secType="STK", exchange="SMART", currency="USD")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=f"{max(1, n_bars // 60 + 1)} D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        keepUpToDate=False,
    )
    return bars[-n_bars:] if bars else []


def fetch_prev_close(ib: IB, symbol: str) -> float:
    """Return previous session's closing price via daily bar."""
    contract = Contract(symbol=symbol, secType="STK", exchange="SMART", currency="USD")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="3 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        keepUpToDate=False,
    )
    # Last completed day is second-to-last bar (today's may be incomplete)
    if len(bars) >= 2:
        return float(bars[-2].close)
    if bars:
        return float(bars[-1].close)
    return 0.0


def bars_to_df(bars: list) -> pd.DataFrame:
    rows = []
    for b in bars:
        rows.append({
            "timestamp": b.date.isoformat() if hasattr(b.date, "isoformat") else str(b.date),
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": int(b.volume),
        })
    df = pd.DataFrame(rows)
    return df


def store_in_db(symbol: str, df: pd.DataFrame) -> int:
    con = sqlite3.connect(config.MARKET_DB)
    inserted = 0
    for _, row in df.iterrows():
        try:
            con.execute(
                "INSERT OR IGNORE INTO market_data "
                "(symbol, timestamp, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?)",
                (symbol, row["timestamp"], row["open"], row["high"],
                 row["low"], row["close"], int(row["volume"])),
            )
            inserted += con.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass
    con.commit()
    con.close()
    return inserted


def run_all_strategies(df: pd.DataFrame, symbol: str, prev_close: float = 0.0) -> list[dict]:
    signals = []
    checkers_simple = [
        check_support_resistance,
        check_vwap_reversion,
        check_vwap_reclaim,
        check_hod_breakout,
        check_abcd_pattern,
        check_breakout,
        check_compression_breakout,
        check_round_number_magnet,
    ]
    for fn in checkers_simple:
        try:
            sig = fn(df, symbol)
            if sig:
                signals.append(sig)
        except Exception as exc:
            signals.append({"strategy": fn.__name__, "error": str(exc)})
    # strategies that need prev_close
    for fn in (check_gap_and_go, check_fallen_angel):
        try:
            sig = fn(df, symbol, prev_close)
            if sig:
                signals.append(sig)
        except Exception as exc:
            signals.append({"strategy": fn.__name__, "error": str(exc)})
    return signals


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def main() -> int:
    print("\n" + "=" * 55)
    print("  Pre-Market Data Fetch + Strategy Test")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 55)

    # ── Connect to IB ────────────────────────────────────────────────
    section("Connecting to IB (clientId=98)")
    ib = IB()
    try:
        ib.connect(config.IB_HOST, config.IB_PORT, clientId=98, timeout=10)
    except Exception as exc:
        print(f"  {FAIL} IB connect failed: {exc}")
        return 1

    if not ib.isConnected():
        print(f"  {FAIL} Not connected")
        return 1
    print(f"  {PASS} Connected to IB port {config.IB_PORT}")

    # ── Fetch + store bars ───────────────────────────────────────────
    section("Fetching pre-market 1-min bars (useRTH=False)")
    symbol_dfs: dict[str, pd.DataFrame] = {}

    prev_closes: dict[str, float] = {}
    for sym in TEST_SYMBOLS:
        try:
            bars = fetch_bars(ib, sym, n_bars=90)
            if not bars:
                print(f"  {WARN} {sym}: no bars returned")
                continue
            df = bars_to_df(bars)
            inserted = store_in_db(sym, df)
            prev_close = fetch_prev_close(ib, sym)
            prev_closes[sym] = prev_close
            latest = df["close"].iloc[-1]
            ts     = df["timestamp"].iloc[-1]
            print(f"  {PASS} {sym:6s}  {len(bars):3d} bars  last={latest:.2f}  "
                  f"prev_close={prev_close:.2f}  ts={ts}  +{inserted} new rows → market.db")
            symbol_dfs[sym] = df
        except Exception as exc:
            print(f"  {FAIL} {sym}: {exc}")

    ib.disconnect()

    if not symbol_dfs:
        print(f"\n  {FAIL} No data fetched — cannot run strategy tests")
        return 1

    # ── Run strategy signals ─────────────────────────────────────────
    section("Strategy signal scan")
    any_signal = False

    for sym, df in symbol_dfs.items():
        prev_close = prev_closes.get(sym, 0.0)
        signals = run_all_strategies(df, sym, prev_close)
        price   = df["close"].iloc[-1]
        for s in signals:
            any_signal = True
            if "error" in s:
                print(f"  {WARN} {sym:6s} {s['strategy']:30s} ERROR: {s['error']}")
            else:
                direction = s.get("signal", "?")
                strategy  = s.get("strategy", "?")
                conf      = s.get("confidence", 0.0)
                stop      = s.get("stop", 0.0)
                t1        = s.get("target1", 0.0)
                conf_flag = PASS if conf >= config.MIN_SIGNAL_CONFIDENCE else WARN
                print(f"  {conf_flag} {sym:6s} {strategy:30s} "
                      f"{direction:5s} price={price:.2f} "
                      f"conf={conf:.2f} stop={stop:.2f} t1={t1:.2f}")

    if not any_signal:
        print(f"  {INFO} No signals fired on pre-market data "
              f"(normal outside RTH — low volume/rvol)")

    # ── DB row count ─────────────────────────────────────────────────
    section("market.db summary")
    con = sqlite3.connect(config.MARKET_DB)
    total = con.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
    syms  = con.execute(
        "SELECT symbol, COUNT(*) as n, MAX(timestamp) as latest "
        "FROM market_data GROUP BY symbol ORDER BY latest DESC"
    ).fetchall()
    con.close()
    print(f"  {PASS} Total rows: {total:,}")
    for sym, n, latest in syms:
        print(f"       {sym:6s}  {n:5,} bars  latest={latest}")

    print(f"\n{'='*55}")
    print(f"  Pre-market test complete.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
