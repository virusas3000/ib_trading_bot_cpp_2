"""
strategy.py — Signal generation for US trading strategies.

Active strategies:
  1. SUPPORT_RESISTANCE   — multi-touch S/R bounce + breakout
  2. VWAP_REVERSION       — extended deviation + trend-day filter
  3. VWAP_RECLAIM         — price dips below VWAP then reclaims it
  4. GAP_AND_GO           — news-driven gap continuation
  5. HOD_BREAKOUT         — new high of day with momentum
  6. ABCD_PATTERN         — A spike → B pullback → C push → D breakout (Aziz)
  7. FALLEN_ANGEL         — gap-down recovery / gap-fill long
  8. BREAKOUT             — tight consolidation range breakout
  9. COMPRESSION_BREAKOUT — ATR compression squeeze then explosion
 10. ROUND_NUMBER_MAGNET  — price approaching psychological round number

Each check_*() returns None or a signal dict:
  {"signal": "LONG"|"SHORT", "strategy": str, "confidence": float,
   "stop": float, "target1": float, "target2": float}

C++ trading_engine used for all indicator-heavy work.
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
        t1   = entry * (1 + config.TARGET1_PCT)
        t2   = entry * (1 + config.TARGET2_PCT)
    else:
        stop = entry * (1 + stop_pct)
        t1   = entry * (1 - config.TARGET1_PCT)
        t2   = entry * (1 - config.TARGET2_PCT)
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

    if not _CPP:
        return None  # S/R detection requires C++ for accuracy

    levels = te.find_sr_levels(highs, lows, closes, 60, 0.003)
    atr_v  = te.atr(highs, lows, closes, 14)
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
#  2. VWAP_REVERSION (trend-day + momentum-fading filters)
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
    rvol     = te.calc_rvol(volumes)

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
#  3. VWAP_RECLAIM — price dips below VWAP then closes back above (ported HK)
# ─────────────────────────────────────────────────────────────────────────

def check_vwap_reclaim(df: pd.DataFrame, symbol: str) -> Signal:
    """
    Price dips below VWAP for 2–8 bars, then reclaims it with momentum.
    High win-rate setup: 62-68% per HK bot history.
    Stop: session low. T1: +0.8%, T2: +1.5%
    """
    if "VWAP_RECLAIM" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 20 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    vwap_vals = te.vwap(highs, lows, closes, volumes)
    vwap_now  = vwap_vals[-1]
    if math.isnan(vwap_now) or vwap_now <= 0:
        return None

    # Must have reclaimed VWAP
    if price <= vwap_now:
        return None

    # Not too far above (stale signal)
    if (price - vwap_now) / vwap_now > 0.015:
        return None

    # Count how many of the last 2–8 bars were below VWAP
    n = len(closes)
    bars_below = sum(
        1 for i in range(-9, -1)
        if n + i >= 0 and closes[i] < vwap_vals[i]
    )
    if bars_below < 2:
        return None  # never dipped below VWAP

    # Volume confirmation
    rvol = _get_rvol(volumes)
    if rvol < 1.2:
        return None

    # RSI: not extreme, and rising
    rsi_vals = _get_rsi(closes)
    rsi_now  = rsi_vals[-1]
    if math.isnan(rsi_now) or not (40 <= rsi_now <= 75):
        return None
    rsi_3ago = rsi_vals[-4] if len(rsi_vals) >= 4 and not math.isnan(rsi_vals[-4]) else rsi_now
    if rsi_now < rsi_3ago:
        return None  # RSI falling — no momentum

    sess_low = min(lows)
    stop = max(sess_low * 0.998, price * 0.993)
    t1   = price * 1.008
    t2   = price * 1.015
    conf = min(0.50 + (bars_below / 20.0) + (rvol - 1.0) * 0.1, 0.80)
    logger.info("[VWAP_RECLAIM] LONG %s close=%.2f vwap=%.2f bars_below=%d rsi=%.1f",
                symbol, price, vwap_now, bars_below, rsi_now)
    return {"signal": "LONG", "strategy": "VWAP_RECLAIM",
            "confidence": conf, "stop": stop, "target1": t1, "target2": t2}


# ─────────────────────────────────────────────────────────────────────────
#  4. GAP_AND_GO — news-driven gap continuation
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
#  5. HOD_BREAKOUT — new high of day with momentum (stricter RSI + volume)
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
    if price <= hod:               # current price must break above prior HOD
        return None

    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005

    stop, t1, t2 = _stop_target(price, "LONG", symbol, atr_last)
    conf = min(0.60 + (rvol - 2.5) * 0.05 + (rsi_now - 65) * 0.005, 0.82)
    logger.info("[HOD_BREAKOUT] LONG %s RSI=%.1f rvol=%.1f", symbol, rsi_now, rvol)
    return {"signal": "LONG", "strategy": "HOD_BREAKOUT",
            "confidence": conf, "stop": stop, "target1": t1, "target2": t2}


# ─────────────────────────────────────────────────────────────────────────
#  6. ABCD_PATTERN — Andrew Aziz pattern (ported from HK bot)
#     A=spike, B=pullback on low vol, C=second push, D=pullback → breakout
# ─────────────────────────────────────────────────────────────────────────

def check_abcd_pattern(df: pd.DataFrame, symbol: str) -> Signal:
    """
    ~65% win rate on strong momentum stocks per Aziz back-testing.
    Entry: current close breaks above C with returning volume and above VWAP.
    Stop: below D low (or B low if D not established).
    Target: measured move (A-B distance projected from C).
    """
    if "ABCD_PATTERN" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 10 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]
    n     = len(closes)

    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005

    vwap_vals = te.vwap(highs, lows, closes, volumes)
    vwap_now  = vwap_vals[-1]
    rvol      = _get_rvol(volumes)

    # A: highest point in last 10 bars (by high)
    window = min(10, n)
    a_rel  = int(np.argmax(highs[n - window:]))
    a_idx  = n - window + a_rel
    a_price = highs[a_idx]

    if a_idx >= n - 2:
        return None  # too recent — no pattern formed yet

    # B: lowest low after A (pullback)
    after_a_lows = lows[a_idx:]
    b_rel    = int(np.argmin(after_a_lows))
    b_idx    = a_idx + b_rel
    b_price  = lows[b_idx]

    # Pullback must be 20–70% of A's move from its base
    base   = min(lows[max(0, a_idx - 5): a_idx + 1])
    a_move = a_price - base
    pullback = a_price - b_price
    if a_move <= 0 or not (0.20 <= pullback / a_move <= 0.70):
        return None

    # Volume drying up during pullback (confirms A-B is corrective not impulsive)
    if b_idx > a_idx:
        pb_vol   = sum(volumes[a_idx: b_idx + 1]) / max(1, b_idx - a_idx + 1)
        pre_a_vol = sum(volumes[max(0, a_idx - 5): a_idx]) / max(1, 5)
        if pre_a_vol > 0 and pb_vol > pre_a_vol * 0.8:
            return None

    # C: highest high after B (second push toward A)
    if b_idx >= n - 1:
        return None
    after_b_highs = highs[b_idx:]
    c_rel    = int(np.argmax(after_b_highs))
    c_idx    = b_idx + c_rel
    c_price  = highs[c_idx]

    # C within 2 ATR of A (forms double-top / equal-highs area)
    if abs(c_price - a_price) > atr_last * 2:
        return None

    # D: current pullback — must be a higher low than B (bullish structure)
    current_low = lows[-1]
    if current_low <= b_price:
        return None  # structure broken

    # Entry: close above C with volume and above VWAP
    if price > c_price and rvol >= 1.5 and not math.isnan(vwap_now) and price > vwap_now:
        stop         = current_low if current_low > b_price else b_price
        measured     = a_price - b_price
        t1           = c_price + measured * 0.6
        t2           = c_price + measured * 1.0
        conf         = min(0.58 + (rvol - 1.5) * 0.05, 0.82)
        logger.info("[ABCD_PATTERN] LONG %s A=%.2f B=%.2f C=%.2f rvol=%.1f",
                    symbol, a_price, b_price, c_price, rvol)
        return {"signal": "LONG", "strategy": "ABCD_PATTERN",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  7. FALLEN_ANGEL — gap-down recovery / gap-fill long (ported from HK bot)
#     Stock gaps down on weak/no news, sector strong → gap fill long
# ─────────────────────────────────────────────────────────────────────────

def check_fallen_angel(df: pd.DataFrame, symbol: str,
                        prev_close: float) -> Signal:
    """
    Gap down 2–8% with no fundamental news catalyst.
    Entry: stock recovering — close above VWAP and above open.
    Target: 99%–100% gap fill (near prev close).
    """
    if "FALLEN_ANGEL" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 3 or prev_close <= 0 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price      = closes[-1]
    open_price = opens[0]

    gap_pct = (open_price - prev_close) / prev_close
    # Must be a gap DOWN of 2–8% (not catastrophic)
    if not (-0.08 <= gap_pct <= -0.02):
        return None

    vwap_vals = te.vwap(highs, lows, closes, volumes)
    vwap_now  = vwap_vals[-1]
    rvol      = _get_rvol(volumes)

    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005

    # Stock recovering: close above VWAP and above open
    if math.isnan(vwap_now) or price <= vwap_now or price <= open_price:
        return None
    if rvol < 1.5:
        return None

    stop = vwap_now - atr_last * 0.5
    t1   = prev_close * 0.99   # 99% gap fill
    t2   = prev_close           # full gap fill
    if t1 <= price:
        return None
    rr = (t1 - price) / (price - stop) if price != stop else 0
    if rr < 1.2:
        return None

    conf = min(0.55 + abs(gap_pct) * 5 + (rvol - 1.5) * 0.05, 0.80)
    logger.info("[FALLEN_ANGEL] LONG %s gap=%.1f%% rvol=%.1f", symbol, gap_pct * 100, rvol)
    return {"signal": "LONG", "strategy": "FALLEN_ANGEL",
            "confidence": conf, "stop": stop, "target1": t1, "target2": t2}


# ─────────────────────────────────────────────────────────────────────────
#  8. BREAKOUT — tight consolidation range breakout (ported from HK bot)
# ─────────────────────────────────────────────────────────────────────────

def check_breakout(df: pd.DataFrame, symbol: str) -> Signal:
    """
    Price consolidates in a tight range for 15 bars (< 2x ATR), then breaks
    out with 1.3x+ volume.
    Long: close > range high + 0.1%
    Short: close < range low − 0.1%
    T1: range × 1.5, T2: range × 3.0, Stop: opposite side of range.
    """
    if "BREAKOUT" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 30 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    atr_v    = te.atr(highs, lows, closes, 14)
    atr_last = atr_v[-1] if not math.isnan(atr_v[-1]) else price * 0.005
    rvol     = _get_rvol(volumes)

    # Consolidation range: last 15 bars (excluding current)
    consol_high = max(highs[-16:-1])
    consol_low  = min(lows[-16:-1])
    range_size  = consol_high - consol_low

    # Tight squeeze: range < 2x ATR = consolidation, not wide chop
    if range_size <= 0 or range_size > atr_last * 2.0:
        return None

    if rvol < 1.3:
        return None

    # Long breakout
    if price > consol_high * 1.001:
        stop = consol_low
        t1   = price + range_size * 1.5
        t2   = price + range_size * 3.0
        conf = min(0.58 + (rvol - 1.3) * 0.05, 0.82)
        logger.info("[BREAKOUT] LONG %s above %.2f range=%.2f rvol=%.1f",
                    symbol, consol_high, range_size, rvol)
        return {"signal": "LONG", "strategy": "BREAKOUT",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    # Short breakout
    if not config.LONG_ONLY and price < consol_low * 0.999:
        stop = consol_high
        t1   = price - range_size * 1.5
        t2   = price - range_size * 3.0
        conf = min(0.58 + (rvol - 1.3) * 0.05, 0.82)
        return {"signal": "SHORT", "strategy": "BREAKOUT",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  9. COMPRESSION_BREAKOUT — ATR squeeze then explosion
# ─────────────────────────────────────────────────────────────────────────

def check_compression_breakout(df: pd.DataFrame, symbol: str) -> Signal:
    """
    Volatility contraction breakout (Larry Williams / Bollinger squeeze).
    Conditions:
      • Current ATR < 70% of its 20-bar rolling mean (significant compression)
      • Price breaks 10-bar consolidation high/low
      • Volume surge ≥ 2.0x avg (confirms genuine breakout, not noise)
      • RSI confirms direction (>50 for long, <50 for short)
    Wider targets than regular BREAKOUT because compressed moves tend to extend.
    """
    if "COMPRESSION_BREAKOUT" in config.DISABLED_STRATEGIES:
        return None
    if len(df) < 30 or not _CPP:
        return None

    opens, highs, lows, closes, volumes = _lists(df)
    price = closes[-1]

    atr_v = te.atr(highs, lows, closes, 14)
    if math.isnan(atr_v[-1]):
        return None
    atr_current = atr_v[-1]

    # 20-bar ATR mean (rolling mean of recent values, excluding latest bar)
    atr_recent = [v for v in atr_v[-21:-1] if not math.isnan(v)]
    if len(atr_recent) < 10:
        return None
    atr_mean = sum(atr_recent) / len(atr_recent)

    # Compression: current ATR < 70% of recent mean
    if atr_mean <= 0 or atr_current >= atr_mean * 0.70:
        return None

    rvol = _get_rvol(volumes)
    if rvol < 2.0:
        return None  # require strong volume to confirm breakout from squeeze

    # 10-bar consolidation range (excluding current bar)
    consol_high = max(highs[-11:-1])
    consol_low  = min(lows[-11:-1])
    range_size  = consol_high - consol_low

    rsi_vals = _get_rsi(closes)
    rsi_now  = rsi_vals[-1]
    if math.isnan(rsi_now):
        return None

    atr_ratio = atr_current / atr_mean  # < 0.70

    # Long compression breakout
    if price > consol_high * 1.001 and rsi_now > 50:
        stop = consol_low
        t1   = price + range_size * 2.0  # wider targets — compressed moves extend
        t2   = price + range_size * 4.0
        conf = min(0.60 + (1.0 - atr_ratio) * 0.30 + (rvol - 2.0) * 0.05, 0.85)
        logger.info("[COMPRESSION_BREAKOUT] LONG %s atr_ratio=%.2f rvol=%.1f",
                    symbol, atr_ratio, rvol)
        return {"signal": "LONG", "strategy": "COMPRESSION_BREAKOUT",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    # Short compression breakout
    if not config.LONG_ONLY and price < consol_low * 0.999 and rsi_now < 50:
        stop = consol_high
        t1   = price - range_size * 2.0
        t2   = price - range_size * 4.0
        conf = min(0.60 + (1.0 - atr_ratio) * 0.30 + (rvol - 2.0) * 0.05, 0.85)
        return {"signal": "SHORT", "strategy": "COMPRESSION_BREAKOUT",
                "confidence": conf, "stop": stop, "target1": t1, "target2": t2}

    return None


# ─────────────────────────────────────────────────────────────────────────
#  10. ROUND_NUMBER_MAGNET (0.3%–2.0% below round level, RSI rising, rvol ≥ 2)
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

    atr_last    = atr_v[-1]
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
#  StrategyEngine — runs all 10 strategies, returns highest-confidence signal
# ─────────────────────────────────────────────────────────────────────────

class StrategyEngine:
    """Runs all enabled strategies; returns the highest-confidence signal."""

    def run_all(self, df: pd.DataFrame, symbol: str,
                prev_close: float = 0.0) -> Signal:
        """
        Run every enabled strategy and return the highest-confidence signal,
        or None if nothing fires.
        prev_close required for GAP_AND_GO and FALLEN_ANGEL.
        """
        candidates: list[dict] = []

        for fn, kwargs in [
            (check_support_resistance,   {"df": df, "symbol": symbol}),
            (check_vwap_reversion,       {"df": df, "symbol": symbol}),
            (check_vwap_reclaim,         {"df": df, "symbol": symbol}),
            (check_gap_and_go,           {"df": df, "symbol": symbol, "prev_close": prev_close}),
            (check_hod_breakout,         {"df": df, "symbol": symbol}),
            (check_abcd_pattern,         {"df": df, "symbol": symbol}),
            (check_fallen_angel,         {"df": df, "symbol": symbol, "prev_close": prev_close}),
            (check_breakout,             {"df": df, "symbol": symbol}),
            (check_compression_breakout, {"df": df, "symbol": symbol}),
            (check_round_number_magnet,  {"df": df, "symbol": symbol}),
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
