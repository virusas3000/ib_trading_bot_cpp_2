"""
strategy.py — Signal generation for all US trading strategies.

Each check_*() function receives a pandas DataFrame (OHLCV + indicators)
and returns a signal dict or None.

Signal dict format:
  {
    "signal":     "LONG" | "SHORT",
    "strategy":   str,
    "confidence": float,          # raw strategy confidence (combined later)
    "stop":       float,
    "target1":    float,
    "target2":    float,
  }

The C++ trading_engine module is used for all numeric-heavy computations.
"""
from __future__ import annotations
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

import config

# ── C++ engine ────────────────────────────────────────────────────────────
try:
    import trading_engine as te
    _CPP = True
except ImportError:
    _CPP = False
    logging.getLogger(__name__).warning(
        "trading_engine C++ module not found. Run ./build.sh first. "
        "Falling back to pure-Python indicators (slower)."
    )

logger = logging.getLogger(__name__)

Signal = Optional[dict]


# ─────────────────────────────────────────────────────────────────────────
#  Pure-Python indicator fallbacks (used only when C++ module absent)
# ─────────────────────────────────────────────────────────────────────────

def _rsi_py(closes: list[float], period: int = 14) -> list[float]:
    """Wilder RSI — pure Python fallback."""
    n   = len(closes)
    out = [float("nan")] * n
    if n <= period:
        return out
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = sum(gains) / period, sum(losses) / period
    out[period] = 100 - 100 / (1 + ag / al) if al > 1e-10 else 100.0
    for i in range(period + 1, n):
        d  = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(d, 0.0))  / period
        al = (al * (period - 1) + max(-d, 0.0)) / period
        out[i] = 100 - 100 / (1 + ag / al) if al > 1e-10 else 100.0
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

def _lists(df: pd.DataFrame):
    """Return (opens, highs, lows, closes, volumes) as Python float lists."""
    return (
        df["open"].tolist(),
        df["high"].tolist(),
        df["low"].tolist(),
        df["close"].tolist(),
        df["volume"].tolist(),
    )


def _get_rsi(closes: list[float], period: int = 14) -> list[float]:
    return te.rsi(closes, period) if _CPP else _rsi_py(closes, period)


def _get_rvol(volumes: list[float], period: int = 20) -> float:
    return te.calc_rvol(volumes, period) if _CPP else (
        volumes[-1] / (sum(volumes[-period - 1:-1]) / period)
        if len(volumes) > period else 1.0
    )


def _stop_target(entry: float, direction: str, symbol: str,
                 atr: float) -> tuple[float, float, float]:
    """Return (stop, t1, t2) for an entry."""
    stop_pct = (config.VOLATILE_STOP_PCT
                if symbol in config.VOLATILE_SYMBOLS
                else config.DEFAULT_STOP_PCT)
    if direction == "LONG":
        stop = entry * (1 - stop_pct)
        t1   = entry + atr * 1.5
        t2   = entry + atr * 3.0
    else:
        stop = entry * (1 + stop_pct)
        t1   = entry - atr * 1.5
        t2   = entry - atr * 3.0
    return stop, t1, t2


# ─────────────────────────────────────────────────────────────────────────
#  1. SUPPORT_RESISTANCE (always active, excludes NVDA/TSLA/AMD)
# ─────────────────────────────────────────────────────────────────────────

def check_support_resistance(df: pd.DataFrame, symbol: str) -> Signal:
    if symbol in config.SR_EXCLUDED_SYMBOLS:
        return None
    if len(df) < 30:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    rsi_vals = _get_rsi(closes)
    rsi_now  = rsi_vals[-1]
    if math.isnan(rsi_now):
        return None

    rvol = _get_rvol(volumes)
    if rvol < 1.2:
        return None

    if _CPP:
        levels = te.find_sr_levels(highs, lows, closes, 60, 0.003)
        atr_v  = te.atr(highs, lows, closes, 14)
    else:
        return None  # S/R detection requires C++ for accuracy

    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005

    for lvl in levels[:5]:  # check top-5 levels by touch count
        dist_pct = (price - lvl.price) / lvl.price

        # LONG: price just bounced off support (within 0.3%)
        if lvl.is_support and -0.003 <= dist_pct <= 0.003 and rsi_now < 60:
            stop, t1, t2 = _stop_target(price, "LONG", symbol, atr_last)
            conf = min(0.5 + lvl.touches * 0.05 + (rvol - 1) * 0.05, 0.85)
            logger.info("SUPPORT_RESISTANCE signal LONG %s conf=%.2f", symbol, conf)
            return {"signal": "LONG", "strategy": "SUPPORT_RESISTANCE",
                    "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

        # SHORT: price just hit resistance (within 0.3%)
        if not lvl.is_support and -0.003 <= dist_pct <= 0.003 and rsi_now > 40:
            if config.LONG_ONLY:
                continue
            stop, t1, t2 = _stop_target(price, "SHORT", symbol, atr_last)
            conf = min(0.5 + lvl.touches * 0.05 + (rvol - 1) * 0.05, 0.85)
            logger.info("SUPPORT_RESISTANCE signal SHORT %s conf=%.2f", symbol, conf)
            return {"signal": "SHORT", "strategy": "SUPPORT_RESISTANCE",
                    "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  2. MOMENTUM
# ─────────────────────────────────────────────────────────────────────────

def check_momentum(df: pd.DataFrame, symbol: str) -> Signal:
    if "MOMENTUM" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 20:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    rsi_vals = _get_rsi(closes)
    rsi_now  = rsi_vals[-1]
    if math.isnan(rsi_now):
        return None

    rvol = _get_rvol(volumes)

    if _CPP:
        ema9  = te.ema(closes, 9)
        ema21 = te.ema(closes, 21)
        atr_v = te.atr(highs, lows, closes, 14)
    else:
        return None

    if math.isnan(ema9[-1]) or math.isnan(ema21[-1]) or math.isnan(atr_v[-1]):
        return None

    atr_last = atr_v[-1]

    # LONG momentum: ema9 > ema21, RSI 50-75, rvol ≥ 1.5
    if ema9[-1] > ema21[-1] and 50 <= rsi_now <= 75 and rvol >= 1.5:
        # confirm prior bar's ema9 < ema21 → crossover
        if ema9[-2] <= ema21[-2]:
            stop, t1, t2 = _stop_target(price, "LONG", symbol, atr_last)
            conf = min(0.55 + (rvol - 1.5) * 0.05, 0.80)
            return {"signal": "LONG", "strategy": "MOMENTUM",
                    "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    # SHORT momentum: ema9 < ema21, RSI 25-50, rvol ≥ 1.5
    if not config.LONG_ONLY and ema9[-1] < ema21[-1] and 25 <= rsi_now <= 50 and rvol >= 1.5:
        if ema9[-2] >= ema21[-2]:
            stop, t1, t2 = _stop_target(price, "SHORT", symbol, atr_last)
            conf = min(0.55 + (rvol - 1.5) * 0.05, 0.80)
            return {"signal": "SHORT", "strategy": "MOMENTUM",
                    "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  3. MACD_CROSS
# ─────────────────────────────────────────────────────────────────────────

def check_macd_cross(df: pd.DataFrame, symbol: str) -> Signal:
    if "MACD_CROSS" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 40:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    if not _CPP:
        return None

    macd_line, signal_line, histogram = te.macd(closes)
    atr_v = te.atr(highs, lows, closes, 14)

    if any(math.isnan(v) for v in [macd_line[-1], signal_line[-1],
                                    macd_line[-2], signal_line[-2]]):
        return None
    if math.isnan(atr_v[-1]):
        return None

    atr_last = atr_v[-1]
    rvol = _get_rvol(volumes)

    # Bullish cross: macd crossed above signal
    if macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]:
        if macd_line[-1] < 0:  # cross below zero line = stronger signal
            stop, t1, t2 = _stop_target(price, "LONG", symbol, atr_last)
            conf = 0.55 + min(rvol - 1.0, 0.5) * 0.1
            return {"signal": "LONG", "strategy": "MACD_CROSS",
                    "confidence": min(conf, 0.80), "stop": stop,
                    "target1": t1, "target2": t2}

    # Bearish cross
    if not config.LONG_ONLY:
        if macd_line[-2] >= signal_line[-2] and macd_line[-1] < signal_line[-1]:
            if macd_line[-1] > 0:
                stop, t1, t2 = _stop_target(price, "SHORT", symbol, atr_last)
                conf = 0.55 + min(rvol - 1.0, 0.5) * 0.1
                return {"signal": "SHORT", "strategy": "MACD_CROSS",
                        "confidence": min(conf, 0.80), "stop": stop,
                        "target1": t1, "target2": t2}
    return None


# ─────────────────────────────────────────────────────────────────────────
#  4. GAP_AND_GO
# ─────────────────────────────────────────────────────────────────────────

def check_gap_and_go(df: pd.DataFrame, symbol: str,
                      prev_close: float) -> Signal:
    if "GAP_AND_GO" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 5 or prev_close <= 0:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    open_price = opens[0]   # today's open
    price      = closes[-1]
    gap_pct    = (open_price - prev_close) / prev_close

    if not _CPP:
        return None

    rvol     = _get_rvol(volumes)
    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005

    rsi_vals = _get_rsi(closes)
    rsi_now  = rsi_vals[-1]

    # Gap up ≥ 1%: price holding above open, rvol ≥ 2, RSI > 55
    if gap_pct >= 0.01 and price > open_price and rvol >= 2.0 and not math.isnan(rsi_now) and rsi_now > 55:
        stop, t1, t2 = _stop_target(price, "LONG", symbol, atr_last)
        conf = min(0.5 + gap_pct * 10 + (rvol - 2) * 0.05, 0.80)
        return {"signal": "LONG", "strategy": "GAP_AND_GO",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    # Gap down ≥ 1% (short)
    if not config.LONG_ONLY and gap_pct <= -0.01 and price < open_price and rvol >= 2.0 and not math.isnan(rsi_now) and rsi_now < 45:
        stop, t1, t2 = _stop_target(price, "SHORT", symbol, atr_last)
        conf = min(0.5 + abs(gap_pct) * 10 + (rvol - 2) * 0.05, 0.80)
        return {"signal": "SHORT", "strategy": "GAP_AND_GO",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  5. RSI_DIVERGENCE
# ─────────────────────────────────────────────────────────────────────────

def check_rsi_divergence(df: pd.DataFrame, symbol: str) -> Signal:
    if "RSI_DIVERGENCE" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 30 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price    = closes[-1]
    rsi_vals = te.rsi(closes, 14)
    atr_v    = te.atr(highs, lows, closes, 14)

    if math.isnan(atr_v[-1]):
        return None
    atr_last = atr_v[-1]

    if te.detect_bullish_divergence(closes, rsi_vals, 20):
        stop, t1, t2 = _stop_target(price, "LONG", symbol, atr_last)
        return {"signal": "LONG", "strategy": "RSI_DIVERGENCE",
                "confidence": 0.60, "stop": stop, "target1": t1, "target2": t2}

    if not config.LONG_ONLY and te.detect_bearish_divergence(highs, rsi_vals, 20):
        stop, t1, t2 = _stop_target(price, "SHORT", symbol, atr_last)
        return {"signal": "SHORT", "strategy": "RSI_DIVERGENCE",
                "confidence": 0.60, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  6. ROUND_NUMBER_MAGNET (2026-05-12 rewrite)
# ─────────────────────────────────────────────────────────────────────────

def check_round_number_magnet(df: pd.DataFrame, symbol: str) -> Signal:
    """
    Entry only when price is 0.3%–2.0% BELOW a meaningful round level.
    Reject if price just crossed a round (within 0.3% above any round).
    rvol ≥ 2.0, RSI rising 3 bars and RSI < 75.
    """
    if "ROUND_NUMBER_MAGNET" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 10 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price    = closes[-1]
    rsi_vals = te.rsi(closes, 14)
    atr_v    = te.atr(highs, lows, closes, 14)

    rsi_now = rsi_vals[-1]
    if math.isnan(rsi_now) or math.isnan(atr_v[-1]):
        return None

    if not te.rsi_rising(rsi_vals, config.RNM_RSI_RISING_BARS):
        return None
    if rsi_now >= config.RNM_RSI_MAX:
        return None

    rvol = te.calc_rvol(volumes)
    if rvol < config.RNM_RVOL_MIN:
        return None

    atr_last = atr_v[-1]
    round_levels = te.find_round_levels(price, 0.05)

    for rl in round_levels:
        dist = rl.distance_pct  # positive = price ABOVE round

        # reject: price just crossed above (within 0.3% above)
        if 0.0 <= dist <= config.RNM_REJECT_ABOVE_PCT:
            return None

        # entry: price 0.3%–2.0% BELOW round (magnet pull upward)
        if -(config.RNM_ENTRY_MAX_BELOW_PCT) <= dist <= -(config.RNM_ENTRY_MIN_BELOW_PCT):
            stop  = price * (1 - 0.0015)   # 0.15% below entry
            t1    = rl.price                # target = the round number
            t2    = rl.price * 1.002        # tiny extension past round
            conf  = 0.58 + min(rvol - 2.0, 1.0) * 0.05
            logger.info("[RNM] LONG %s @ %.2f targeting round %.2f", symbol, price, rl.price)
            return {"signal": "LONG", "strategy": "ROUND_NUMBER_MAGNET",
                    "confidence": min(conf, 0.80), "stop": stop,
                    "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  7. VWAP_REVERSION (2026-05-12 update — trend-day + momentum-fading filters)
# ─────────────────────────────────────────────────────────────────────────

def check_vwap_reversion(df: pd.DataFrame, symbol: str) -> Signal:
    """
    Trend-day filter: ≥26/30 bars same VWAP side → skip.
    Momentum fading: deviation must be shrinking vs prior bar.
    Entry: price 2.5%+ from VWAP, RSI oversold/overbought, deviation shrinking.
    """
    if "VWAP_REVERSION" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 10 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price     = closes[-1]
    vwap_vals = te.vwap(highs, lows, closes, volumes)
    vwap_now  = vwap_vals[-1]

    # Trend day filter
    if te.is_trend_day(closes, vwap_vals, config.VWAP_TREND_DAY_BARS,
                       config.VWAP_TREND_DAY_THRESH / config.VWAP_TREND_DAY_BARS):
        return None

    # Momentum fading: must be shrinking
    if te.vwap_deviation_expanding(closes, vwap_vals):
        return None

    dev_pct = (price - vwap_now) / vwap_now
    rsi_vals = te.rsi(closes, 14)
    rsi_now  = rsi_vals[-1]
    if math.isnan(rsi_now):
        return None

    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005

    rvol = te.calc_rvol(volumes)

    # LONG: price 2.5%+ BELOW vwap, RSI oversold, pulling back toward VWAP
    if dev_pct <= -(config.VWAP_DEVIATION_PCT) and rsi_now <= config.VWAP_RSI_OVERSOLD:
        stop = price * (1 - config.DEFAULT_STOP_PCT)
        t1   = vwap_now
        t2   = vwap_now * 1.005
        conf = min(0.55 + abs(dev_pct) * 5 + (rvol - 1) * 0.03, 0.80)
        return {"signal": "LONG", "strategy": "VWAP_REVERSION",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    # SHORT: price 2.5%+ ABOVE vwap, RSI overbought
    if not config.LONG_ONLY and dev_pct >= config.VWAP_DEVIATION_PCT and rsi_now >= config.VWAP_RSI_OVERBOUGHT:
        stop = price * (1 + config.DEFAULT_STOP_PCT)
        t1   = vwap_now
        t2   = vwap_now * 0.995
        conf = min(0.55 + dev_pct * 5 + (rvol - 1) * 0.03, 0.80)
        return {"signal": "SHORT", "strategy": "VWAP_REVERSION",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  8. HOD_BREAKOUT (2026-05-12 update — stricter RSI + volume)
# ─────────────────────────────────────────────────────────────────────────

def check_hod_breakout(df: pd.DataFrame, symbol: str) -> Signal:
    """
    RSI ≥ 65, rising 3 consecutive bars, rvol ≥ 2.5.
    New HOD = entry trigger.
    """
    if "HOD_BREAKOUT" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 10 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    rsi_vals = te.rsi(closes, 14)
    rsi_now  = rsi_vals[-1]
    if math.isnan(rsi_now):
        return None

    if rsi_now < config.HOD_RSI_MIN:
        return None
    if not te.rsi_rising(rsi_vals, config.HOD_RSI_RISING_BARS):
        return None

    rvol = te.calc_rvol(volumes)
    if rvol < config.HOD_RVOL_MIN:
        return None

    hod = te.get_hod(highs[:-1])  # HOD excluding current bar
    if price <= hod:  # current price must break above prior HOD
        return None

    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005

    stop, t1, t2 = _stop_target(price, "LONG", symbol, atr_last)
    conf = min(0.60 + (rvol - 2.5) * 0.05 + (rsi_now - 65) * 0.005, 0.82)
    logger.info("[HOD_BREAKOUT] LONG %s RSI=%.1f rvol=%.1f", symbol, rsi_now, rvol)
    return {"signal": "LONG", "strategy": "HOD_BREAKOUT",
            "confidence": conf, "stop": stop, "target1": t1, "target2": t2}


# ─────────────────────────────────────────────────────────────────────────
#  9. SMC_ORDER_BLOCK
# ─────────────────────────────────────────────────────────────────────────

def check_smc_order_block(df: pd.DataFrame, symbol: str) -> Signal:
    if "SMC_ORDER_BLOCK" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 15 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    obs      = te.find_order_blocks(opens, highs, lows, closes, 30)
    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005
    rvol     = te.calc_rvol(volumes)
    rsi_vals = te.rsi(closes, 14)
    rsi_now  = rsi_vals[-1]

    for ob in obs:
        in_zone = ob.low <= price <= ob.high
        if not in_zone:
            continue

        # Bullish OB: price re-enters demand zone from above → LONG
        if ob.bullish and not math.isnan(rsi_now) and rsi_now < 55 and rvol >= 1.5:
            stop  = ob.low  * 0.999
            t1    = price + atr_last * 1.5
            t2    = price + atr_last * 3.0
            conf  = min(0.58 + (rvol - 1.5) * 0.05, 0.82)
            return {"signal": "LONG", "strategy": "SMC_ORDER_BLOCK",
                    "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

        # Bearish OB: price re-enters supply zone from below → SHORT
        if not ob.bullish and not config.LONG_ONLY and not math.isnan(rsi_now) and rsi_now > 45 and rvol >= 1.5:
            stop  = ob.high * 1.001
            t1    = price - atr_last * 1.5
            t2    = price - atr_last * 3.0
            conf  = min(0.58 + (rvol - 1.5) * 0.05, 0.82)
            return {"signal": "SHORT", "strategy": "SMC_ORDER_BLOCK",
                    "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  StrategyEngine — runs all strategies and returns first valid signal
# ─────────────────────────────────────────────────────────────────────────

class StrategyEngine:
    """Runs all enabled strategies; returns the highest-confidence signal."""

    def run_all(self, df: pd.DataFrame, symbol: str,
                prev_close: float = 0.0) -> Signal:
        """
        Run every enabled strategy and return the highest-confidence signal,
        or None if nothing fires.
        """
        candidates: list[dict] = []

        for fn, kwargs in [
            (check_support_resistance, {"df": df, "symbol": symbol}),
            (check_momentum,           {"df": df, "symbol": symbol}),
            (check_macd_cross,         {"df": df, "symbol": symbol}),
            (check_gap_and_go,         {"df": df, "symbol": symbol, "prev_close": prev_close}),
            (check_rsi_divergence,     {"df": df, "symbol": symbol}),
            (check_round_number_magnet,{"df": df, "symbol": symbol}),
            (check_vwap_reversion,     {"df": df, "symbol": symbol}),
            (check_hod_breakout,       {"df": df, "symbol": symbol}),
            (check_smc_order_block,    {"df": df, "symbol": symbol}),
        ]:
            try:
                sig = fn(**kwargs)
                if sig is not None:
                    candidates.append(sig)
            except Exception as exc:
                logger.error("[Strategy] %s raised: %s", fn.__name__, exc, exc_info=True)

        if not candidates:
            return None
        return max(candidates, key=lambda s: s["confidence"])
