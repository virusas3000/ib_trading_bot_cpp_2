"""
config.py — US Algo Trader configuration (CLIENT_ID=32)
HK Bot uses CLIENT_ID=31. Never run both simultaneously on same TWS instance.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── IB Connection ──────────────────────────────────────────────────────────
CLIENT_ID = 32          # US bot. HK bot = 31. IB rejects duplicate client IDs.
IB_HOST   = "127.0.0.1"
IB_PORT   = 7497        # Paper TWS. Live = 7496. Never mix.

# ── API Keys ───────────────────────────────────────────────────────────────
ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY  = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL    = "https://paper-api.alpaca.markets"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = "anthropic/claude-3-haiku"  # fast + cheap for confidence scoring

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_THREAD_ID = 279  # Hermes Swarm thread

# ── Risk Parameters ────────────────────────────────────────────────────────
LONG_ONLY          = False           # both directions enabled
MAX_POSITIONS      = 10
MAX_POSITION_PCT   = 0.10            # max 10% NAV per position
MAX_RISK_PCT       = 0.005           # risk at most 0.5% NAV on any single trade

MIN_POSITION_VALUE = 1_500.0         # always at least $1,500 per position (ATR override)
MIN_DOLLAR_VOLUME  = 25_000_000      # skip symbol if 20-bar avg_vol × price < $25M

DEFAULT_STOP_PCT   = 0.005           # 0.5% stop for most symbols
VOLATILE_STOP_PCT  = 0.010           # 1.0% stop for NVDA/TSLA/AMD
HARD_STOP_PCT      = 0.002           # 0.2% hard-stop failsafe (circuit breaker)
VOLATILE_SYMBOLS   = ["NVDA", "TSLA", "AMD"]

MIN_SIGNAL_CONFIDENCE  = 0.40        # combined ML+LLM confidence threshold
MIN_T1_PROFIT_USD      = 20.0        # skip trade if max possible profit < $20
MAX_CONSECUTIVE_LOSSES = 4           # halt after 4 consecutive losses
MAX_DAILY_LOSS_PCT     = 0.02        # halt trading at 2% daily drawdown
MAX_DAILY_COMMISSION   = 50.0        # halt if commission > $50/day

# ── Session Timing (US Eastern / ET) ──────────────────────────────────────
DAY_TRADE_ONLY        = True
FORCE_CLOSE_HOUR      = 15           # force-close all at 15:45 ET
FORCE_CLOSE_MIN       = 45
NO_NEW_POSITIONS_HOUR = 15           # no new entries after 15:30 ET
NO_NEW_POSITIONS_MIN  = 30

# US session in UTC: 13:30–20:00. NEVER use 12:00 UTC as cutoff.
# In HKT this is 21:30–04:00 (spans midnight). Always verify ET time.
SESSION_START_UTC_HOUR = 13
SESSION_START_UTC_MIN  = 30
SESSION_END_UTC_HOUR   = 20
SESSION_END_UTC_MIN    = 0

# ── ML / AI ────────────────────────────────────────────────────────────────
ML_RETRAIN_INTERVAL = 24   # hours between automatic retrains from trades.db
ML_MIN_SAMPLES      = 50   # minimum trade records needed to train
ML_MODEL_PATH       = "ml_model.pkl"
ML_SCALER_PATH      = "ml_scaler.pkl"
ML_LABELS_PATH      = "ml_labels.pkl"

# ── Database ───────────────────────────────────────────────────────────────
TRADES_DB = "trades.db"
MARKET_DB = "market.db"  # 8.7 GB, 28 M rows — ML feature store

# ── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_HOST      = "0.0.0.0"
DASHBOARD_PORT      = 8088
DASHBOARD_CLIENT_ID = 96   # dedicated IB client ID for executions API

# ── Strategies ─────────────────────────────────────────────────────────────
# Remove from this list one at a time after 5-day paper validation.
# SUPPORT_RESISTANCE is always active (not listed here).
DISABLED_STRATEGIES: list[str] = [
    "MOMENTUM",
    "SMC_ORDER_BLOCK",
    "GAP_AND_GO",
    "MACD_CROSS",
    "RSI_DIVERGENCE",
    "ROUND_NUMBER_MAGNET",
    "VWAP_REVERSION",
    "HOD_BREAKOUT",
]

# SUPPORT_RESISTANCE excluded symbols (net negative P&L confirmed in backtest)
SR_EXCLUDED_SYMBOLS = ["NVDA", "TSLA", "AMD"]

# ROUND_NUMBER_MAGNET thresholds (2026-05-12 rewrite)
RNM_ENTRY_MIN_BELOW_PCT = 0.003   # must be ≥0.3% below round level
RNM_ENTRY_MAX_BELOW_PCT = 0.020   # must be ≤2.0% below round level
RNM_REJECT_ABOVE_PCT    = 0.003   # reject if price just crossed round (within 0.3% above)
RNM_RVOL_MIN            = 2.0     # rvol ≥ 2.0×
RNM_RSI_RISING_BARS     = 3
RNM_RSI_MAX             = 75

# VWAP_REVERSION thresholds (2026-05-12 update)
VWAP_DEVIATION_PCT      = 0.025   # 2.5% min deviation from VWAP
VWAP_RSI_OVERSOLD       = 35
VWAP_RSI_OVERBOUGHT     = 65
VWAP_TREND_DAY_BARS     = 30      # look back 30 bars
VWAP_TREND_DAY_THRESH   = 26      # ≥26/30 bars same VWAP side = trend day, skip

# HOD_BREAKOUT thresholds (2026-05-12 update)
HOD_RSI_MIN             = 65      # was any RSI
HOD_RSI_RISING_BARS     = 3
HOD_RVOL_MIN            = 2.5     # was 1.5×

# ── Watchlist (50 symbols) ─────────────────────────────────────────────────
WATCHLIST: list[str] = [
    "SPY",  "QQQ",  "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "UBER", "LYFT", "SNAP", "PINS",  "SQ",   "PYPL", "SOFI", "HOOD", "RIVN",
    "LCID", "NIO",  "XPEV", "LI",   "PLTR",  "RBLX", "COIN", "MSTR", "GME",  "AMC",
    "BB",   "NOK",  "F",    "GM",   "BAC",   "JPM",  "WFC",  "C",    "GS",   "MS",
    "SCHW", "V",    "MA",   "AXP",  "DIS",   "INTC", "MU",   "QCOM", "ARM",  "SMCI",
]
