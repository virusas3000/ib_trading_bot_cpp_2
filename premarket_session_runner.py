"""
premarket_session_runner.py — pre-market paper execution window + dashboard
restart scheduler.

Behavior:
  1. Starts trader.py in paper mode during pre-market and stops it at
     9:25 AM ET so strategies can be tested on live pre-market data.
  2. Can still run the older signal-only scan mode when requested.
  3. Restarts the dashboard listener on the configured dashboard port at
     9:25 PM local time so it reconnects cleanly.

Usage:
  python3 premarket_session_runner.py
  python3 premarket_session_runner.py --signal-only --scan-once --skip-dashboard-restart
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from ib_insync import IB

from fetch_premarket_test import (
    TEST_SYMBOLS,
    bars_to_df,
    fetch_bars,
    fetch_prev_close,
    run_all_strategies,
    store_in_db,
)

logger = logging.getLogger("premarket_runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-18s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def _format_td(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def premarket_cutoff_et(now: datetime | None = None) -> datetime:
    now_et = (now or datetime.now(ET)).astimezone(ET)
    return now_et.replace(hour=9, minute=25, second=0, microsecond=0)


def next_dashboard_restart_local(now: datetime | None = None) -> datetime:
    local_now = (now or datetime.now().astimezone()).astimezone()
    target = local_now.replace(hour=21, minute=25, second=0, microsecond=0)
    if target <= local_now:
        target += timedelta(days=1)
    return target


def _run_ps() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, command = parts
        try:
            rows.append((int(pid_s), command))
        except ValueError:
            continue
    return rows


def run_scan_once(client_id: int = 98) -> tuple[int, int]:
    ib = IB()
    fetched = 0
    signals_found = 0

    logger.info("Connecting to IB on %s:%d (clientId=%d)", config.IB_HOST, config.IB_PORT, client_id)
    try:
        ib.connect(config.IB_HOST, config.IB_PORT, clientId=client_id, timeout=10)
    except Exception as exc:
        raise RuntimeError(f"IB connect failed: {exc}") from exc

    if not ib.isConnected():
        raise RuntimeError("IB connect returned without an active connection")

    try:
        for sym in TEST_SYMBOLS:
            bars = fetch_bars(ib, sym, n_bars=90)
            if not bars:
                logger.warning("%s: no pre-market bars returned", sym)
                continue

            df = bars_to_df(bars)
            inserted = store_in_db(sym, df)
            prev_close = fetch_prev_close(ib, sym)
            signals = run_all_strategies(df, sym, prev_close)
            fetched += 1

            latest_price = float(df["close"].iloc[-1])
            latest_ts = str(df["timestamp"].iloc[-1])
            logger.info(
                "%s bars=%d last=%.2f prev_close=%.2f inserted=%d latest=%s",
                sym,
                len(df),
                latest_price,
                prev_close,
                inserted,
                latest_ts,
            )

            if not signals:
                continue

            for signal_data in signals:
                if "error" in signal_data:
                    logger.warning("%s %s error=%s", sym, signal_data.get("strategy", "?"), signal_data["error"])
                    continue

                signals_found += 1
                logger.info(
                    "SIGNAL %s %s side=%s conf=%.2f stop=%.2f t1=%.2f",
                    sym,
                    signal_data.get("strategy", "?"),
                    signal_data.get("signal", "?"),
                    float(signal_data.get("confidence", 0.0)),
                    float(signal_data.get("stop", 0.0)),
                    float(signal_data.get("target1", 0.0)),
                )
    finally:
        ib.disconnect()

    logger.info("Pre-market scan complete: fetched=%d symbols, signals=%d", fetched, signals_found)
    return fetched, signals_found


def run_signal_scan_window(interval_seconds: int, scan_once: bool) -> None:
    cutoff = premarket_cutoff_et()
    now_et = datetime.now(ET)
    if now_et >= cutoff:
        logger.info("Skipping pre-market scan window; cutoff already passed (%s ET)", cutoff.strftime("%H:%M"))
        return

    while True:
        now_et = datetime.now(ET)
        if now_et >= cutoff:
            logger.info("Reached pre-market cutoff at %s ET", cutoff.strftime("%H:%M"))
            return

        remaining = (cutoff - now_et).total_seconds()
        logger.info("Running paper pre-market scan; %s remaining until 09:25 ET", _format_td(remaining))
        run_scan_once()

        if scan_once:
            return

        sleep_for = min(interval_seconds, max(0, int((cutoff - datetime.now(ET)).total_seconds())))
        if sleep_for <= 0:
            return
        logger.info("Sleeping %ss before next pre-market scan", sleep_for)
        time.sleep(sleep_for)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _pid_listening_on_port(port: int) -> list[int]:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        check=False,
        capture_output=True,
        text=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def stop_dashboard_processes(timeout_seconds: int = 10) -> None:
    pids = _pid_listening_on_port(config.DASHBOARD_PORT)
    if not pids:
        logger.info("No process is listening on dashboard port %d", config.DASHBOARD_PORT)
        return

    logger.info("Stopping dashboard process(es): %s", ", ".join(str(pid) for pid in pids))
    for pid in pids:
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_exists(pid)]
        if not alive:
            logger.info("Dashboard process(es) stopped cleanly")
            return
        time.sleep(0.5)

    alive = [pid for pid in pids if _pid_exists(pid)]
    for pid in alive:
        logger.warning("Dashboard PID %d did not exit after SIGTERM; sending SIGKILL", pid)
        os.kill(pid, signal.SIGKILL)


def start_dashboard(log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "dashboard.py"],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    logger.info("Started dashboard.py with PID %d", proc.pid)
    return proc.pid


def wait_for_dashboard(timeout_seconds: int = 20) -> None:
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{config.DASHBOARD_PORT}/api/nav"
    last_error: str | None = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    logger.info("Dashboard is responding on port %d", config.DASHBOARD_PORT)
                    return
                last_error = f"unexpected HTTP status {resp.status}"
        except urllib.error.URLError as exc:
            last_error = str(exc)
        time.sleep(1)

    raise RuntimeError(f"dashboard did not become ready: {last_error or 'timeout'}")


def restart_dashboard(log_path: Path) -> None:
    stop_dashboard_processes()
    start_dashboard(log_path)
    wait_for_dashboard()


def wait_until(target: datetime, label: str) -> None:
    while True:
        now = datetime.now(target.tzinfo or datetime.now().astimezone().tzinfo)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        sleep_for = min(remaining, 60)
        logger.info(
            "Waiting for %s at %s (%s remaining)",
            label,
            target.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
            _format_td(remaining),
        )
        time.sleep(sleep_for)


def find_trader_pids() -> list[int]:
    return [pid for pid, command in _run_ps() if "trader.py" in command]


def stop_trader_processes(timeout_seconds: int = 10) -> None:
    pids = find_trader_pids()
    if not pids:
        logger.info("No running trader.py process found")
        return

    logger.info("Stopping trader process(es): %s", ", ".join(str(pid) for pid in pids))
    for pid in pids:
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_exists(pid)]
        if not alive:
            logger.info("Trader process(es) stopped cleanly")
            return
        time.sleep(0.5)

    alive = [pid for pid in pids if _pid_exists(pid)]
    for pid in alive:
        logger.warning("Trader PID %d did not exit after SIGTERM; sending SIGKILL", pid)
        os.kill(pid, signal.SIGKILL)


def start_trader(log_path: Path) -> int:
    if find_trader_pids():
        raise RuntimeError("trader.py is already running; refusing to start a duplicate paper trader")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "trader.py"],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    logger.info("Started trader.py with PID %d", proc.pid)
    return proc.pid


def wait_for_trader_ready(log_path: Path, pid: int, timeout_seconds: int = 25) -> None:
    deadline = time.time() + timeout_seconds
    success_markers = ("Connected to IB", "Bot running. Press Ctrl-C to stop.")

    while time.time() < deadline:
        if not _pid_exists(pid):
            raise RuntimeError(f"trader.py exited before startup completed; check {log_path}")
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in content for marker in success_markers):
                logger.info("Trader startup confirmed")
                return
        time.sleep(1)

    raise RuntimeError(f"trader.py did not confirm startup within {timeout_seconds}s; check {log_path}")


def run_premarket_execution_window(trader_log: Path) -> None:
    cutoff = premarket_cutoff_et()
    now_et = datetime.now(ET)
    if now_et >= cutoff:
        logger.info("Skipping pre-market execution window; cutoff already passed (%s ET)", cutoff.strftime("%H:%M"))
        return

    remaining = (cutoff - now_et).total_seconds()
    logger.info("Starting paper execution window; %s remaining until 09:25 ET", _format_td(remaining))
    trader_pid = start_trader(trader_log)
    wait_for_trader_ready(trader_log, trader_pid)
    wait_until(cutoff, "pre-market cutoff")
    stop_trader_processes()
    logger.info("Pre-market paper execution window finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper pre-market execution until 9:25 ET, then restart the dashboard at 9:25 PM local time.",
    )
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Use the old signal-only scan mode instead of starting trader.py.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Seconds between repeated scans in signal-only mode (default: 60).",
    )
    parser.add_argument(
        "--scan-once",
        action="store_true",
        help="Run one pre-market scan immediately and exit the signal-only scan loop.",
    )
    parser.add_argument(
        "--skip-dashboard-restart",
        action="store_true",
        help="Exit after the pre-market scan window without restarting dashboard.py.",
    )
    parser.add_argument(
        "--dashboard-log",
        default=str(PROJECT_ROOT / "dashboard_runtime.log"),
        help="Log file used when dashboard.py is restarted.",
    )
    parser.add_argument(
        "--trader-log",
        default=str(PROJECT_ROOT / "premarket_trader_runtime.log"),
        help="Log file used when trader.py is started for paper execution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    try:
        if args.signal_only:
            run_signal_scan_window(args.interval_seconds, args.scan_once)
        else:
            run_premarket_execution_window(Path(args.trader_log))
        if args.skip_dashboard_restart:
            logger.info("Skipping dashboard restart by request")
            return 0

        target = next_dashboard_restart_local()
        logger.info("Dashboard restart scheduled for %s", target.strftime("%Y-%m-%d %H:%M:%S %Z"))
        wait_until(target, "dashboard restart")
        restart_dashboard(Path(args.dashboard_log))
        logger.info("Dashboard restart completed")
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
