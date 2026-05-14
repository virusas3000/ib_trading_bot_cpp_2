"""
health_check.py — pre-flight health check for US Algo Trader.

Run before every trading session:
  python3 health_check.py

Exits 0 if all critical checks pass (warnings are OK).
Exits 1 if any critical check fails.

Covers: environment, config sanity, strategy rules, IB connectivity,
        field integrity, NAV, $20 min profit, position restore, ML load.
"""
from __future__ import annotations
import ast
import importlib
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ── pretty printer ─────────────────────────────────────────────────────────
PASS  = "\033[92m✅\033[0m"
FAIL  = "\033[91m❌\033[0m"
WARN  = "\033[93m⚠️ \033[0m"

_results: list[tuple[str, str, str]] = []  # (status, category, message)


def ok(msg: str, cat: str = "") -> None:
    _results.append(("PASS", cat, msg))
    print(f"  {PASS} {msg}")


def fail(msg: str, cat: str = "") -> None:
    _results.append(("FAIL", cat, msg))
    print(f"  {FAIL} {msg}")


def warn(msg: str, cat: str = "") -> None:
    _results.append(("WARN", cat, msg))
    print(f"  {WARN} {msg}")


def section(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ─────────────────────────────────────────────────────────────────────────

def check_env() -> None:
    section("1. Environment")
    import config

    # .env keys
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "OPENROUTER_API_KEY"):
        val = getattr(config, key, "")
        if val:
            ok(f"{key} set")
        else:
            warn(f"{key} missing from .env")

    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        val = getattr(config, key, "")
        if val:
            ok(f"{key} set")
        else:
            warn(f"{key} missing — Telegram alerts disabled")

    # trades.db
    if os.path.exists(config.TRADES_DB):
        try:
            con = sqlite3.connect(config.TRADES_DB)
            con.execute("SELECT 1 FROM trades LIMIT 1")
            con.close()
            ok("trades.db accessible and not locked")
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                warn("trades.db exists but schema missing — run setup_db.py")
            else:
                fail(f"trades.db locked or corrupt: {e}")
    else:
        fail("trades.db missing — run setup_db.py")

    # market.db
    if os.path.exists(config.MARKET_DB):
        size_gb = os.path.getsize(config.MARKET_DB) / 1e9
        ok(f"market.db exists ({size_gb:.1f} GB)")
    else:
        warn("market.db missing — ML feature lookups will fail")

    # ML models
    for p in (config.ML_MODEL_PATH, config.ML_SCALER_PATH, config.ML_LABELS_PATH):
        if os.path.exists(p):
            age_h = (time.time() - os.path.getmtime(p)) / 3600
            if age_h > 48:
                warn(f"{p} is {age_h:.0f}h old — consider retraining")
            else:
                ok(f"{p} exists ({age_h:.1f}h old)")
        else:
            warn(f"{p} missing — ML will return neutral 0.5")

    # trade_quality_gate.py
    if os.path.exists("trade_quality_gate.py"):
        ok("trade_quality_gate.py exists")
    else:
        fail("trade_quality_gate.py missing")

    # orphan check
    try:
        out = subprocess.check_output(["pgrep", "-f", "trader.py"], text=True)
        pids = out.strip().split()
        my_pid = str(os.getpid())
        others = [p for p in pids if p != my_pid]
        if others:
            warn(f"Possible orphan trader.py processes: PIDs {others}")
        else:
            ok("No orphan trader.py processes")
    except subprocess.CalledProcessError:
        ok("No orphan trader.py processes")


def check_config() -> None:
    section("2. Config Sanity")
    import config

    checks = [
        ("CLIENT_ID",              32,    True),
        ("IB_PORT",                7497,  True),
        ("LONG_ONLY",              False, True),
        ("MAX_POSITIONS",          10,    False),
        ("DEFAULT_STOP_PCT",       0.005, True),
        ("VOLATILE_STOP_PCT",      0.010, True),
        ("DAY_TRADE_ONLY",         True,  True),
        ("FORCE_CLOSE_HOUR",       15,    True),
        ("FORCE_CLOSE_MIN",        45,    True),
        ("NO_NEW_POSITIONS_HOUR",  15,    True),
        ("NO_NEW_POSITIONS_MIN",   30,    True),
        ("MIN_SIGNAL_CONFIDENCE",  0.40,  True),
        ("MIN_T1_PROFIT_USD",      20.0,  True),
        ("MAX_CONSECUTIVE_LOSSES", 4,     False),
        ("HARD_STOP_PCT",          0.002, True),
        ("ML_RETRAIN_INTERVAL",    24,    False),
    ]
    for attr, expected, critical in checks:
        val = getattr(config, attr, None)
        if val == expected:
            ok(f"{attr} = {val}")
        else:
            msg = f"{attr} = {val} (expected {expected})"
            fail(msg) if critical else warn(msg)

    # session start UTC
    if config.SESSION_START_UTC_HOUR == 13 and config.SESSION_START_UTC_MIN == 30:
        ok("SESSION_START_UTC = 13:30 (correct US market open)")
    else:
        fail(f"SESSION_START_UTC = {config.SESSION_START_UTC_HOUR}:{config.SESSION_START_UTC_MIN:02d} "
             f"— MUST be 13:30. NEVER use 12:00.")

    # watchlist length
    n = len(config.WATCHLIST)
    if n == 50:
        ok(f"WATCHLIST has {n} symbols")
    else:
        warn(f"WATCHLIST has {n} symbols (expected 50)")

    ok(f"DISABLED_STRATEGIES: {config.DISABLED_STRATEGIES}")
    ok(f"SR_EXCLUDED_SYMBOLS:  {config.SR_EXCLUDED_SYMBOLS}")


def check_strategy_rules() -> None:
    section("3. Strategy Code Rules")
    trader_src = Path("trader.py").read_text() if Path("trader.py").exists() else ""
    strategy_src = Path("strategy.py").read_text() if Path("strategy.py").exists() else ""

    # All calc_size() calls pass t1=t1
    combined = trader_src + strategy_src
    import re
    calls = re.findall(r'calc_size\([^)]+\)', combined)
    for call in calls:
        if "t1" in call or "t1_price" in call:
            ok(f"calc_size() call includes t1: {call[:60]}…")
        else:
            fail(f"calc_size() call missing t1: {call[:60]}…")

    if "SKIP_MIN_PROFIT" in strategy_src or "SKIP_MIN_PROFIT" in trader_src:
        ok("SKIP_MIN_PROFIT log tag present")
    else:
        warn("SKIP_MIN_PROFIT log tag not found in strategy/trader")

    if "RESTORE" in trader_src or "avgCost" in trader_src:
        ok("Position restore (avgCost fallback) present in trader.py")
    else:
        fail("Position restore code missing from trader.py")

    if "HARD_STOP" in trader_src:
        ok("Hard stop failsafe present in trader.py")
    else:
        fail("HARD_STOP code missing from trader.py")

    # Session boundary
    if "SESSION_START_UTC" in trader_src or "13" in trader_src:
        ok("Session boundary references found in trader.py")
    else:
        warn("Session boundary check may be missing in trader.py")


def check_ib_connectivity() -> None:
    section("4. IB Connectivity")
    import config
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(config.IB_HOST, config.IB_PORT, clientId=99, timeout=5)
        if ib.isConnected():
            ok(f"IB connection OK (port {config.IB_PORT})")
            vals = ib.accountValues()
            nav  = next((float(v.value) for v in vals
                         if v.tag == "NetLiquidation" and v.currency == "USD"), 0.0)
            if nav > 0:
                ok(f"Account equity: ${nav:,.2f}")
            else:
                fail("Account NAV is $0 — check TWS paper account")
            ib.disconnect()
        else:
            fail(f"IB connection FAILED on port {config.IB_PORT}")
    except Exception as exc:
        fail(f"IB connection error: {exc}")


def check_modules() -> None:
    section("5. Python Modules")
    required = [
        "ib_insync", "pandas", "numpy", "sklearn",
        "joblib", "aiohttp", "flask",
    ]
    for mod in required:
        try:
            importlib.import_module(mod)
            ok(f"{mod} importable")
        except ImportError:
            fail(f"{mod} not installed — run: pip3 install -r requirements.txt")

    # C++ engine
    try:
        import trading_engine  # type: ignore
        ok("trading_engine C++ module loaded")
    except ImportError:
        warn("trading_engine C++ module not found — run ./build.sh")

    # ML classes
    try:
        from ml_integration import MLTradingIntegration
        ok("MLTradingIntegration importable")
    except Exception as exc:
        fail(f"MLTradingIntegration import failed: {exc}")


def check_min_profit_logic() -> None:
    section("6. Min T1 Profit Enforcement")
    try:
        import trading_engine as te
        import config

        # Should PASS: 500 shares × $0.05 = $25 > $20
        r = te.calc_size(100_000, 0.10, 0.005, 100.0, 99.5, 100.05,
                         config.MIN_T1_PROFIT_USD, config.MIN_POSITION_VALUE)
        if r.qty > 0:
            ok(f"Wide T1 ($0.05/share): qty={r.qty} profit=${r.max_profit_usd:.2f} ✓ PASS")
        else:
            fail(f"Wide T1 should pass but returned qty=0 reason={r.reason}")

        # Should SKIP: entry=100, T1=100.02 → $0.02/share × ~500 = $10 < $20
        r2 = te.calc_size(100_000, 0.10, 0.005, 100.0, 99.5, 100.02,
                          config.MIN_T1_PROFIT_USD, config.MIN_POSITION_VALUE)
        if r2.qty == 0 and r2.skip_min_profit:
            ok(f"Tight T1 ($0.02/share): qty=0 reason={r2.reason} ✓ SKIP")
        else:
            fail(f"Tight T1 should be skipped but qty={r2.qty}")

    except ImportError:
        warn("trading_engine not built — skipping min-profit test")


def main() -> int:
    print("\n" + "=" * 50)
    print("  US Algo Trader Health Check")
    print("=" * 50)

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

    check_env()
    check_config()
    check_strategy_rules()
    check_ib_connectivity()
    check_modules()
    check_min_profit_logic()

    # Summary
    fails = [r for r in _results if r[0] == "FAIL"]
    warns = [r for r in _results if r[0] == "WARN"]
    print(f"\n{'='*50}")
    print(f"  RESULT: {len(_results)} checks — "
          f"{len(_results)-len(fails)-len(warns)} passed, "
          f"{len(warns)} warnings, {len(fails)} failures")
    if fails:
        print(f"\n  CRITICAL FAILURES:")
        for _, cat, msg in fails:
            print(f"    {FAIL} {msg}")
        print(f"\n  ❌ Health check FAILED — fix issues before trading.\n")
        return 1
    else:
        print(f"\n  ✅ All critical checks passed. Bot is ready.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
