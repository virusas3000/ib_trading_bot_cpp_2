"""
backtest_full.py — full historical backtest across all strategies.

Usage:
  python3 backtest_full.py --start 2024-01-01 --end 2024-02-01 --symbols SPY
  python3 backtest_full.py --start 2015-01-01 --end 2024-01-01   # 9-year run

Data source: market.db (8.7 GB, 28 M rows, 1-minute bars)
Output:      backtest_full_results.csv + strategy ranking table to stdout
Baseline:    SPY Jan–Feb 2024 → +$1,506
"""
from __future__ import annotations
import argparse
import csv
import logging
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
from tabulate import tabulate

import config
from strategy import StrategyEngine

logger = logging.getLogger("backtest")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")

SLIPPAGE_PCT  = 0.0005   # 0.05% each way (conservative paper assumption)
COMMISSION    = 0.005    # $0.005 per share (IB tiered)
LOOKBACK_BARS = 60       # bars of history before first signal eligible


# ─────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────

def load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    con = sqlite3.connect(config.MARKET_DB)
    df  = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM market_data "
        "WHERE symbol=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
        con, params=(symbol, start, end),
    )
    con.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    return df


# ─────────────────────────────────────────────────────────────────────────
#  Trade simulation
# ─────────────────────────────────────────────────────────────────────────

def simulate_symbol(symbol: str, df: pd.DataFrame,
                    nav: float = 100_000.0) -> list[dict]:
    """Run all strategies bar-by-bar and simulate fills."""
    engine  = StrategyEngine()
    trades  = []
    open_pos: dict | None = None

    for i in range(LOOKBACK_BARS, len(df)):
        window    = df.iloc[i - LOOKBACK_BARS: i + 1].copy()
        price     = float(window["close"].iloc[-1])
        prev_close = float(df["close"].iloc[i - 1]) if i > 0 else 0.0

        # ── Manage open position ───────────────────────────────────────
        if open_pos is not None:
            side  = open_pos["side"]
            entry = open_pos["entry"]
            stop  = open_pos["stop"]
            t1    = open_pos["target1"]
            bar_h = float(window["high"].iloc[-1])
            bar_l = float(window["low"].iloc[-1])

            exit_price = None
            reason     = None

            if side == "LONG":
                if bar_l <= stop:
                    exit_price, reason = stop,  "STOP"
                elif bar_h >= t1:
                    exit_price, reason = t1,    "TARGET1"
            else:
                if bar_h >= stop:
                    exit_price, reason = stop,  "STOP"
                elif bar_l <= t1:
                    exit_price, reason = t1,    "TARGET1"

            # EOD close at last bar of session
            if exit_price is None:
                ts = window.index[-1]
                if ts.hour >= config.FORCE_CLOSE_HOUR and ts.minute >= config.FORCE_CLOSE_MIN:
                    exit_price, reason = price, "FORCE_CLOSE"

            if exit_price is not None:
                # slippage
                if reason == "STOP":
                    exit_price *= (1 - SLIPPAGE_PCT) if side == "LONG" else (1 + SLIPPAGE_PCT)
                qty   = open_pos["qty"]
                pnl   = (exit_price - entry) * qty * (1 if side == "LONG" else -1)
                comm  = qty * COMMISSION * 2  # entry + exit
                pnl  -= comm
                nav  += pnl

                trades.append({
                    "symbol":     symbol,
                    "side":       side,
                    "strategy":   open_pos["strategy"],
                    "entry":      entry,
                    "exit":       exit_price,
                    "qty":        qty,
                    "pnl":        round(pnl, 2),
                    "commission": round(comm, 2),
                    "reason":     reason,
                    "entry_time": open_pos["entry_time"],
                    "exit_time":  str(window.index[-1]),
                })
                open_pos = None
            continue  # don't look for new signals while in trade

        # ── Signal scan ────────────────────────────────────────────────
        now_ts = window.index[-1]
        if now_ts.hour >= config.NO_NEW_POSITIONS_HOUR and \
           now_ts.minute >= config.NO_NEW_POSITIONS_MIN:
            continue

        sig = engine.run_all(window, symbol, prev_close)
        if sig is None:
            continue

        entry  = price * (1 + SLIPPAGE_PCT if sig["signal"] == "LONG" else 1 - SLIPPAGE_PCT)
        stop_p = sig["stop"]
        t1     = sig["target1"]

        # T1 profit gate
        t1_profit = abs(t1 - entry)
        max_pos_val = nav * config.MAX_POSITION_PCT
        qty = max(1, int(max_pos_val / entry))

        try:
            import trading_engine as te
            sz  = te.calc_size(nav, config.MAX_POSITION_PCT, config.MAX_RISK_PCT,
                               entry, stop_p, t1,
                               config.MIN_T1_PROFIT_USD, config.MIN_POSITION_VALUE)
            qty = sz.qty
            if qty == 0:
                continue
        except ImportError:
            if qty * t1_profit < config.MIN_T1_PROFIT_USD:
                continue

        open_pos = {
            "side":       sig["signal"],
            "entry":      entry,
            "stop":       stop_p,
            "target1":    t1,
            "target2":    sig["target2"],
            "qty":        qty,
            "strategy":   sig["strategy"],
            "entry_time": str(now_ts),
        }

    return trades


# ─────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {}

    by_strat: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_strat[t["strategy"]].append(t)

    rows = []
    for strat, ts in sorted(by_strat.items(), key=lambda x: -sum(t["pnl"] for t in x[1])):
        pnls  = [t["pnl"] for t in ts]
        wins  = [p for p in pnls if p > 0]
        loses = [p for p in pnls if p <= 0]
        pf    = (sum(wins) / abs(sum(loses))) if loses and sum(loses) != 0 else float("inf")
        avg_w = sum(wins) / len(wins)   if wins  else 0.0
        avg_l = sum(loses) / len(loses) if loses else 0.0
        rows.append({
            "Strategy":       strat,
            "Trades":         len(ts),
            "Win%":           f"{len(wins)/len(ts)*100:.1f}",
            "PF":             f"{pf:.2f}" if math.isfinite(pf) else "∞",
            "AvgW":           f"${avg_w:.2f}",
            "AvgL":           f"${avg_l:.2f}",
            "Net P&L":        f"${sum(pnls):+,.2f}",
            "Commission":     f"${sum(t['commission'] for t in ts):.2f}",
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full backtest")
    parser.add_argument("--start",   default="2024-01-01")
    parser.add_argument("--end",     default="2024-02-01")
    parser.add_argument("--symbols", default=",".join(config.WATCHLIST))
    parser.add_argument("--nav",     type=float, default=100_000.0)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    all_trades: list[dict] = []
    for sym in symbols:
        logger.info("Backtesting %s %s → %s", sym, args.start, args.end)
        df = load_bars(sym, args.start, args.end)
        if df.empty:
            logger.warning("No data for %s — check market.db", sym)
            continue
        trades = simulate_symbol(sym, df, args.nav)
        all_trades.extend(trades)
        logger.info("  %s: %d trades, P&L=$%.2f",
                    sym, len(trades), sum(t["pnl"] for t in trades))

    if not all_trades:
        logger.error("No trades generated. Check market.db has data for the requested range.")
        sys.exit(1)

    # ── Save CSV ──────────────────────────────────────────────────────
    out_file = "backtest_full_results.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_trades[0].keys())
        writer.writeheader()
        writer.writerows(all_trades)
    logger.info("Saved %d trades to %s", len(all_trades), out_file)

    # ── Strategy ranking table ────────────────────────────────────────
    rows = compute_metrics(all_trades)
    print("\n" + "=" * 70)
    print(f"  BACKTEST {args.start} → {args.end}  |  {len(symbols)} symbols  |  {len(all_trades)} trades")
    print("=" * 70)
    print(tabulate(rows, headers="keys", tablefmt="simple"))
    total_pnl = sum(t["pnl"] for t in all_trades)
    print(f"\n  TOTAL NET P&L: ${total_pnl:+,.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
