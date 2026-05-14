/**
 * signals.cpp — S/R detection, round levels, order blocks, divergence, sizing
 * Part of trading_engine pybind11 module.
 */
#include "../include/trading_engine.h"
#include <map>
#include <cmath>

namespace trading
{

    // ─────────────────────────────────────────────────────────────────────────
    //  Support / Resistance — pivot clustering
    // ─────────────────────────────────────────────────────────────────────────

    static bool is_pivot_high(const std::vector<double> &highs, int i, int wing = 3)
    {
        if (i < wing || i >= static_cast<int>(highs.size()) - wing)
            return false;
        for (int j = i - wing; j <= i + wing; ++j)
            if (j != i && highs[j] >= highs[i])
                return false;
        return true;
    }

    static bool is_pivot_low(const std::vector<double> &lows, int i, int wing = 3)
    {
        if (i < wing || i >= static_cast<int>(lows.size()) - wing)
            return false;
        for (int j = i - wing; j <= i + wing; ++j)
            if (j != i && lows[j] <= lows[i])
                return false;
        return true;
    }

    std::vector<Level> find_sr_levels(const std::vector<double> &highs,
                                      const std::vector<double> &lows,
                                      const std::vector<double> &closes,
                                      int lookback, double tolerance_pct)
    {
        const int n = static_cast<int>(closes.size());
        const int start = std::max(0, n - lookback);

        // collect raw pivot prices
        std::vector<std::pair<double, bool>> pivots; // {price, is_support}
        for (int i = start; i < n; ++i)
        {
            if (is_pivot_high(highs, i))
                pivots.push_back({highs[i], false});
            if (is_pivot_low(lows, i))
                pivots.push_back({lows[i], true});
        }

        // cluster nearby pivots
        std::vector<Level> levels;
        std::sort(pivots.begin(), pivots.end());

        for (auto &[price, is_sup] : pivots)
        {
            bool merged = false;
            for (auto &lv : levels)
            {
                if (std::abs(lv.price - price) / lv.price <= tolerance_pct)
                {
                    lv.price = (lv.price * lv.touches + price) / (lv.touches + 1);
                    lv.touches++;
                    merged = true;
                    break;
                }
            }
            if (!merged)
                levels.push_back({price, 1, is_sup});
        }

        // sort by touch count descending
        std::sort(levels.begin(), levels.end(),
                  [](const Level &a, const Level &b)
                  { return a.touches > b.touches; });
        return levels;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Round Number Levels
    // ─────────────────────────────────────────────────────────────────────────

    std::vector<RoundLevel> find_round_levels(double price, double search_range_pct)
    {
        static const std::vector<double> magnitudes = {0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0};
        std::vector<RoundLevel> result;

        for (double mag : magnitudes)
        {
            // find the nearest round levels above and below
            double below = std::floor(price / mag) * mag;
            double above = std::ceil(price / mag) * mag;

            for (double lvl : {below, above})
            {
                if (lvl <= 0.0)
                    continue;
                double dist_pct = (price - lvl) / lvl; // positive = price above round
                if (std::abs(dist_pct) <= search_range_pct)
                    result.push_back({lvl, mag, dist_pct});
            }
        }

        // sort by distance ascending
        std::sort(result.begin(), result.end(),
                  [](const RoundLevel &a, const RoundLevel &b)
                  {
                      return std::abs(a.distance_pct) < std::abs(b.distance_pct);
                  });
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  SMC Order Blocks
    // ─────────────────────────────────────────────────────────────────────────

    std::vector<OrderBlock> find_order_blocks(const std::vector<double> &opens,
                                              const std::vector<double> &highs,
                                              const std::vector<double> &lows,
                                              const std::vector<double> &closes,
                                              int lookback)
    {
        const int n = static_cast<int>(closes.size());
        const int start = std::max(3, n - lookback);
        std::vector<OrderBlock> result;

        // compute a rough body-size average to identify impulsive candles
        double avg_body = 0.0;
        int body_cnt = 0;
        for (int i = start; i < n; ++i)
        {
            avg_body += std::abs(closes[i] - opens[i]);
            ++body_cnt;
        }
        if (body_cnt == 0)
            return result;
        avg_body /= body_cnt;

        const double impulse_factor = 1.5;

        for (int i = start + 1; i < n - 1; ++i)
        {
            double body = closes[i] - opens[i];
            double body_size = std::abs(body);
            bool bullish = body > 0.0;
            bool impulsive = body_size >= impulse_factor * avg_body;

            if (!impulsive)
                continue;

            // bullish impulse → look for preceding bearish candle (demand zone)
            if (bullish && i >= 1)
            {
                int ob_idx = i - 1;
                if (closes[ob_idx] < opens[ob_idx])
                { // bearish candle
                    // check not already mitigated (price came back below OB low)
                    bool mitigated = false;
                    for (int j = i + 1; j < n; ++j)
                        if (lows[j] < lows[ob_idx])
                        {
                            mitigated = true;
                            break;
                        }
                    if (!mitigated)
                        result.push_back({highs[ob_idx], lows[ob_idx],
                                          (highs[ob_idx] + lows[ob_idx]) / 2.0,
                                          n - 1 - ob_idx, true});
                }
            }

            // bearish impulse → look for preceding bullish candle (supply zone)
            if (!bullish && i >= 1)
            {
                int ob_idx = i - 1;
                if (closes[ob_idx] > opens[ob_idx])
                { // bullish candle
                    bool mitigated = false;
                    for (int j = i + 1; j < n; ++j)
                        if (highs[j] > highs[ob_idx])
                        {
                            mitigated = true;
                            break;
                        }
                    if (!mitigated)
                        result.push_back({highs[ob_idx], lows[ob_idx],
                                          (highs[ob_idx] + lows[ob_idx]) / 2.0,
                                          n - 1 - ob_idx, false});
                }
            }
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Divergence
    // ─────────────────────────────────────────────────────────────────────────

    bool detect_bullish_divergence(const std::vector<double> &closes,
                                   const std::vector<double> &rsi_vals,
                                   int lookback)
    {
        const int n = static_cast<int>(closes.size());
        if (n < lookback + 2)
            return false;
        const int start = n - lookback;

        // find two pivot lows in price within the window
        double p1 = std::numeric_limits<double>::max(), p2 = std::numeric_limits<double>::max();
        double r1 = 0.0, r2 = 0.0;
        int p1i = -1, p2i = -1;

        for (int i = start; i < n; ++i)
        {
            if (i > start && i < n - 1 &&
                closes[i] < closes[i - 1] && closes[i] < closes[i + 1])
            {
                if (p1i == -1 || closes[i] < p1)
                {
                    p2 = p1;
                    p2i = p1i;
                    r2 = r1;
                    p1 = closes[i];
                    p1i = i;
                    r1 = rsi_vals[i];
                }
            }
        }

        if (p1i == -1 || p2i == -1)
            return false;
        // ensure p2 is earlier (smaller index)
        if (p1i < p2i)
        {
            std::swap(p1, p2);
            std::swap(p1i, p2i);
            std::swap(r1, r2);
        }

        // bullish divergence: price lower low, RSI higher low
        return (p1 < p2) && (r1 > r2) && !std::isnan(r1) && !std::isnan(r2);
    }

    bool detect_bearish_divergence(const std::vector<double> &highs,
                                   const std::vector<double> &rsi_vals,
                                   int lookback)
    {
        const int n = static_cast<int>(highs.size());
        if (n < lookback + 2)
            return false;
        const int start = n - lookback;

        double h1 = std::numeric_limits<double>::lowest(), h2 = std::numeric_limits<double>::lowest();
        double r1 = 0.0, r2 = 0.0;
        int h1i = -1, h2i = -1;

        for (int i = start + 1; i < n - 1; ++i)
        {
            if (highs[i] > highs[i - 1] && highs[i] > highs[i + 1])
            {
                if (h1i == -1 || highs[i] > h1)
                {
                    h2 = h1;
                    h2i = h1i;
                    r2 = r1;
                    h1 = highs[i];
                    h1i = i;
                    r1 = rsi_vals[i];
                }
            }
        }

        if (h1i == -1 || h2i == -1)
            return false;
        if (h1i < h2i)
        {
            std::swap(h1, h2);
            std::swap(h1i, h2i);
            std::swap(r1, r2);
        }

        // bearish divergence: price higher high, RSI lower high
        return (h1 > h2) && (r1 < r2) && !std::isnan(r1) && !std::isnan(r2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Position Sizing
    // ─────────────────────────────────────────────────────────────────────────

    SizeResult calc_size(double nav,
                         double max_position_pct,
                         double max_risk_pct,
                         double entry_price,
                         double stop_price,
                         double t1_price,
                         double min_t1_profit_usd,
                         double min_position_value)
    {
        SizeResult res{0, 0.0, 0.0, false, false, ""};

        if (entry_price <= 0.0 || stop_price <= 0.0 || t1_price <= 0.0)
            return res;

        double risk_per_share = std::abs(entry_price - stop_price);
        double t1_profit_per_share = std::abs(t1_price - entry_price);

        if (risk_per_share < 1e-6)
        {
            res.reason = "ZERO_RISK_PER_SHARE";
            return res;
        }

        double max_risk_usd = nav * max_risk_pct;
        double max_nav_usd = nav * max_position_pct;

        int risk_qty = static_cast<int>(std::floor(max_risk_usd / risk_per_share));
        int nav_qty = static_cast<int>(std::floor(max_nav_usd / entry_price));
        int qty = std::min(risk_qty, nav_qty);
        if (qty < 1)
            qty = 1;

        // bump up to meet MIN_POSITION_VALUE
        if (qty * entry_price < min_position_value)
        {
            qty = static_cast<int>(std::ceil(min_position_value / entry_price));
            // recheck nav limit
            if (qty * entry_price > max_nav_usd)
            {
                res.skip_min_value = true;
                res.reason = "SKIP_MIN_VALUE_EXCEEDS_NAV_LIMIT";
                return res;
            }
        }

        // check T1 profit gate
        double max_profit = qty * t1_profit_per_share;
        if (max_profit < min_t1_profit_usd)
        {
            res.skip_min_profit = true;
            res.reason = "SKIP_MIN_PROFIT";
            return res;
        }

        res.qty = qty;
        res.risk_usd = qty * risk_per_share;
        res.max_profit_usd = max_profit;
        return res;
    }

} // namespace trading
