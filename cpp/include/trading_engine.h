#pragma once
/**
 * trading_engine.h — C++ core for US Algo Trader
 *
 * Exposed to Python via pybind11 as the 'trading_engine' module.
 * Build: ./build.sh  (requires CMake ≥ 3.18, C++17, pybind11)
 *
 * Architecture:
 *   - All numeric-heavy work lives here (indicators, S/R, sizing)
 *   - Python handles IB API, DB, ML, LLM, and orchestration
 */
#include <vector>
#include <tuple>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>

namespace trading
{

    // ─────────────────────────────────────────────────────────────────────────
    //  Basic helpers
    // ─────────────────────────────────────────────────────────────────────────

    inline double nanval() { return std::numeric_limits<double>::quiet_NaN(); }
    inline bool isnan_(double v) { return std::isnan(v); }

    // ─────────────────────────────────────────────────────────────────────────
    //  Indicators
    // ─────────────────────────────────────────────────────────────────────────

    /** EMA — same-length output; first `period-1` values are NaN. */
    std::vector<double> ema(const std::vector<double> &prices, int period);

    /** SMA — same-length output; first `period-1` values are NaN. */
    std::vector<double> sma(const std::vector<double> &prices, int period);

    /**
     * RSI using Wilder's smoothing.
     * Same-length output; first `period` values are NaN.
     */
    std::vector<double> rsi(const std::vector<double> &closes, int period = 14);

    /**
     * MACD — returns {macd_line, signal_line, histogram}.
     * All vectors have the same length as `closes`; short-history bars are NaN.
     */
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    macd(const std::vector<double> &closes,
         int fast = 12,
         int slow = 26,
         int signal_period = 9);

    /**
     * Wilder ATR.
     * Same-length output; first `period` values are NaN.
     */
    std::vector<double> atr(const std::vector<double> &highs,
                            const std::vector<double> &lows,
                            const std::vector<double> &closes,
                            int period = 14);

    /**
     * Intraday VWAP (provide current-day bars only; resets each call).
     * Returns VWAP series; all bars have valid values.
     */
    std::vector<double> vwap(const std::vector<double> &highs,
                             const std::vector<double> &lows,
                             const std::vector<double> &closes,
                             const std::vector<double> &volumes);

    /**
     * Bollinger Bands — returns {upper, middle, lower}.
     */
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    bollinger(const std::vector<double> &closes,
              int period = 20,
              double std_devs = 2.0);

    /** Relative Volume vs. `period`-bar average. Returns scalar for latest bar. */
    double calc_rvol(const std::vector<double> &volumes, int period = 20);

    /** Returns true if RSI has been rising for the last `n` bars. */
    bool rsi_rising(const std::vector<double> &rsi_vals, int n = 3);

    /** High-of-day from a vector of high prices. */
    double get_hod(const std::vector<double> &highs);

    /** Low-of-day from a vector of low prices. */
    double get_lod(const std::vector<double> &lows);

    /**
     * Trend-day filter for VWAP_REVERSION.
     * Returns true if ≥ `threshold` fraction of the last `bars` closes
     * are on the same side of their corresponding VWAP value (no counter-trend).
     */
    bool is_trend_day(const std::vector<double> &closes,
                      const std::vector<double> &vwap_vals,
                      int bars = 30,
                      double threshold = 26.0 / 30.0);

    /**
     * Returns true if VWAP deviation is still expanding (not fading).
     * Used in VWAP_REVERSION to reject entries where price is still diverging.
     */
    bool vwap_deviation_expanding(const std::vector<double> &closes,
                                  const std::vector<double> &vwap_vals);

    // ─────────────────────────────────────────────────────────────────────────
    //  Support / Resistance
    // ─────────────────────────────────────────────────────────────────────────

    struct Level
    {
        double price;
        int touches;
        bool is_support;
    };

    /**
     * Identify S/R levels by clustering pivot highs/lows within `tolerance_pct`.
     * Returns levels sorted by touch count descending.
     */
    std::vector<Level> find_sr_levels(const std::vector<double> &highs,
                                      const std::vector<double> &lows,
                                      const std::vector<double> &closes,
                                      int lookback = 60,
                                      double tolerance_pct = 0.003);

    // ─────────────────────────────────────────────────────────────────────────
    //  Round Number Levels
    // ─────────────────────────────────────────────────────────────────────────

    struct RoundLevel
    {
        double price;
        double magnitude;    ///< 0.5 / 1 / 5 / 10 / 25 / 50 / 100
        double distance_pct; ///< positive = price above round; negative = below
    };

    /**
     * Return all meaningful round levels within `search_range_pct` of `price`.
     * Magnitudes checked: 0.5, 1, 5, 10, 25, 50, 100.
     */
    std::vector<RoundLevel> find_round_levels(double price,
                                              double search_range_pct = 0.05);

    // ─────────────────────────────────────────────────────────────────────────
    //  SMC Order Blocks
    // ─────────────────────────────────────────────────────────────────────────

    struct OrderBlock
    {
        double high;
        double low;
        double mid;
        int bar_index; ///< bars from end; 0 = most recent
        bool bullish;  ///< true = demand zone; false = supply zone
    };

    /**
     * Detect order blocks using Smart Money Concepts methodology.
     * Bullish OB = last bearish candle before an impulsive move up.
     * Bearish OB = last bullish candle before an impulsive move down.
     * Mitigated OBs (price has already revisited) are excluded.
     */
    std::vector<OrderBlock> find_order_blocks(const std::vector<double> &opens,
                                              const std::vector<double> &highs,
                                              const std::vector<double> &lows,
                                              const std::vector<double> &closes,
                                              int lookback = 30);

    // ─────────────────────────────────────────────────────────────────────────
    //  Divergence
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Bullish divergence: price making lower lows, RSI making higher lows.
     */
    bool detect_bullish_divergence(const std::vector<double> &closes,
                                   const std::vector<double> &rsi_vals,
                                   int lookback = 20);

    /**
     * Bearish divergence: price making higher highs, RSI making lower highs.
     */
    bool detect_bearish_divergence(const std::vector<double> &highs,
                                   const std::vector<double> &rsi_vals,
                                   int lookback = 20);

    // ─────────────────────────────────────────────────────────────────────────
    //  Position Sizing
    // ─────────────────────────────────────────────────────────────────────────

    struct SizeResult
    {
        int qty; ///< 0 = skip trade
        double risk_usd;
        double max_profit_usd;
        bool skip_min_profit; ///< T1 profit < MIN_T1_PROFIT_USD
        bool skip_min_value;  ///< position value < MIN_POSITION_VALUE
        std::string reason;   ///< human-readable skip reason
    };

    /**
     * Calculate position size.
     *
     * Algorithm:
     *   risk_qty  = floor(nav * max_risk_pct  / |entry - stop|)
     *   nav_qty   = floor(nav * max_pos_pct   / entry)
     *   qty       = min(risk_qty, nav_qty)
     *   → bump up if qty * entry < min_position_value
     *   → return qty=0 if qty * |t1 - entry| < min_t1_profit_usd   [SKIP_MIN_PROFIT]
     */
    SizeResult calc_size(double nav,
                         double max_position_pct,
                         double max_risk_pct,
                         double entry_price,
                         double stop_price,
                         double t1_price,
                         double min_t1_profit_usd,
                         double min_position_value);

} // namespace trading
