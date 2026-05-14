/**
 * indicators.cpp — Technical indicator implementations
 * Part of trading_engine pybind11 module.
 */
#include "../include/trading_engine.h"

namespace trading
{

    // ── EMA ───────────────────────────────────────────────────────────────────
    std::vector<double> ema(const std::vector<double> &prices, int period)
    {
        const int n = static_cast<int>(prices.size());
        std::vector<double> result(n, nanval());
        if (n < period)
            return result;

        const double k = 2.0 / (period + 1.0);

        // seed with SMA of first `period` bars
        double sum = 0.0;
        for (int i = 0; i < period; ++i)
            sum += prices[i];
        result[period - 1] = sum / period;

        for (int i = period; i < n; ++i)
            result[i] = prices[i] * k + result[i - 1] * (1.0 - k);

        return result;
    }

    // ── SMA ───────────────────────────────────────────────────────────────────
    std::vector<double> sma(const std::vector<double> &prices, int period)
    {
        const int n = static_cast<int>(prices.size());
        std::vector<double> result(n, nanval());
        if (n < period)
            return result;

        double sum = 0.0;
        for (int i = 0; i < period; ++i)
            sum += prices[i];
        result[period - 1] = sum / period;

        for (int i = period; i < n; ++i)
        {
            sum += prices[i] - prices[i - period];
            result[i] = sum / period;
        }
        return result;
    }

    // ── RSI (Wilder's smoothing) ───────────────────────────────────────────────
    std::vector<double> rsi(const std::vector<double> &closes, int period)
    {
        const int n = static_cast<int>(closes.size());
        std::vector<double> result(n, nanval());
        if (n <= period)
            return result;

        // compute first average gain/loss over first `period` moves
        double avg_gain = 0.0, avg_loss = 0.0;
        for (int i = 1; i <= period; ++i)
        {
            double diff = closes[i] - closes[i - 1];
            if (diff > 0)
                avg_gain += diff;
            else
                avg_loss += -diff;
        }
        avg_gain /= period;
        avg_loss /= period;

        auto to_rsi = [](double ag, double al) -> double
        {
            if (al < 1e-10)
                return 100.0;
            return 100.0 - 100.0 / (1.0 + ag / al);
        };

        result[period] = to_rsi(avg_gain, avg_loss);

        for (int i = period + 1; i < n; ++i)
        {
            double diff = closes[i] - closes[i - 1];
            double gain = std::max(diff, 0.0);
            double loss = std::max(-diff, 0.0);
            avg_gain = (avg_gain * (period - 1) + gain) / period;
            avg_loss = (avg_loss * (period - 1) + loss) / period;
            result[i] = to_rsi(avg_gain, avg_loss);
        }
        return result;
    }

    // ── MACD ──────────────────────────────────────────────────────────────────
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    macd(const std::vector<double> &closes, int fast, int slow, int signal_period)
    {
        const int n = static_cast<int>(closes.size());
        auto fast_ema = ema(closes, fast);
        auto slow_ema = ema(closes, slow);

        std::vector<double> macd_line(n, nanval());
        for (int i = slow - 1; i < n; ++i)
            macd_line[i] = fast_ema[i] - slow_ema[i];

        // compute signal EMA only over valid macd values
        std::vector<double> macd_valid;
        int valid_start = slow - 1;
        for (int i = valid_start; i < n; ++i)
            macd_valid.push_back(macd_line[i]);

        auto sig_raw = ema(macd_valid, signal_period);

        std::vector<double> signal_line(n, nanval());
        std::vector<double> histogram(n, nanval());
        for (int i = 0; i < static_cast<int>(sig_raw.size()); ++i)
        {
            int idx = valid_start + i;
            if (!std::isnan(sig_raw[i]))
            {
                signal_line[idx] = sig_raw[i];
                histogram[idx] = macd_line[idx] - sig_raw[i];
            }
        }
        return {macd_line, signal_line, histogram};
    }

    // ── ATR (Wilder) ──────────────────────────────────────────────────────────
    std::vector<double> atr(const std::vector<double> &highs,
                            const std::vector<double> &lows,
                            const std::vector<double> &closes,
                            int period)
    {
        const int n = static_cast<int>(closes.size());
        std::vector<double> result(n, nanval());
        if (n < period + 1)
            return result;

        // true ranges
        std::vector<double> tr(n, nanval());
        tr[0] = highs[0] - lows[0];
        for (int i = 1; i < n; ++i)
            tr[i] = std::max({highs[i] - lows[i],
                              std::abs(highs[i] - closes[i - 1]),
                              std::abs(lows[i] - closes[i - 1])});

        // seed
        double sum = 0.0;
        for (int i = 0; i < period; ++i)
            sum += tr[i];
        result[period - 1] = sum / period;

        for (int i = period; i < n; ++i)
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period;

        return result;
    }

    // ── VWAP (intraday, single day) ───────────────────────────────────────────
    std::vector<double> vwap(const std::vector<double> &highs,
                             const std::vector<double> &lows,
                             const std::vector<double> &closes,
                             const std::vector<double> &volumes)
    {
        const int n = static_cast<int>(closes.size());
        std::vector<double> result(n);
        double cum_tp_vol = 0.0, cum_vol = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double tp = (highs[i] + lows[i] + closes[i]) / 3.0;
            cum_tp_vol += tp * volumes[i];
            cum_vol += volumes[i];
            result[i] = (cum_vol > 0.0) ? cum_tp_vol / cum_vol : tp;
        }
        return result;
    }

    // ── Bollinger Bands ────────────────────────────────────────────────────────
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    bollinger(const std::vector<double> &closes, int period, double std_devs)
    {
        const int n = static_cast<int>(closes.size());
        auto middle = sma(closes, period);
        std::vector<double> upper(n, nanval()), lower(n, nanval());
        for (int i = period - 1; i < n; ++i)
        {
            double sum_sq = 0.0;
            for (int j = i - period + 1; j <= i; ++j)
                sum_sq += (closes[j] - middle[i]) * (closes[j] - middle[i]);
            double stddev = std::sqrt(sum_sq / period);
            upper[i] = middle[i] + std_devs * stddev;
            lower[i] = middle[i] - std_devs * stddev;
        }
        return {upper, middle, lower};
    }

    // ── Rvol ──────────────────────────────────────────────────────────────────
    double calc_rvol(const std::vector<double> &volumes, int period)
    {
        const int n = static_cast<int>(volumes.size());
        if (n < 2)
            return 1.0;
        double latest = volumes[n - 1];
        int start = std::max(0, n - 1 - period);
        double sum = 0.0;
        int cnt = 0;
        for (int i = start; i < n - 1; ++i)
        {
            sum += volumes[i];
            ++cnt;
        }
        if (cnt == 0)
            return 1.0;
        double avg = sum / cnt;
        return (avg > 0.0) ? latest / avg : 1.0;
    }

    // ── RSI rising ────────────────────────────────────────────────────────────
    bool rsi_rising(const std::vector<double> &rsi_vals, int n)
    {
        const int sz = static_cast<int>(rsi_vals.size());
        if (sz < n + 1)
            return false;
        for (int i = sz - n; i < sz; ++i)
        {
            if (std::isnan(rsi_vals[i]) || std::isnan(rsi_vals[i - 1]))
                return false;
            if (rsi_vals[i] <= rsi_vals[i - 1])
                return false;
        }
        return true;
    }

    // ── HOD / LOD ─────────────────────────────────────────────────────────────
    double get_hod(const std::vector<double> &highs)
    {
        if (highs.empty())
            return 0.0;
        return *std::max_element(highs.begin(), highs.end());
    }
    double get_lod(const std::vector<double> &lows)
    {
        if (lows.empty())
            return 0.0;
        return *std::min_element(lows.begin(), lows.end());
    }

    // ── Trend day filter ──────────────────────────────────────────────────────
    bool is_trend_day(const std::vector<double> &closes,
                      const std::vector<double> &vwap_vals,
                      int bars, double threshold)
    {
        const int n = static_cast<int>(closes.size());
        if (n < bars)
            return false;
        int above = 0, below = 0;
        for (int i = n - bars; i < n; ++i)
        {
            if (std::isnan(vwap_vals[i]))
                continue;
            if (closes[i] > vwap_vals[i])
                ++above;
            else
                ++below;
        }
        double frac = static_cast<double>(std::max(above, below)) / bars;
        return frac >= threshold;
    }

    // ── VWAP deviation expanding ──────────────────────────────────────────────
    bool vwap_deviation_expanding(const std::vector<double> &closes,
                                  const std::vector<double> &vwap_vals)
    {
        const int n = static_cast<int>(closes.size());
        if (n < 2)
            return false;
        double dev_prev = std::abs(closes[n - 2] - vwap_vals[n - 2]);
        double dev_curr = std::abs(closes[n - 1] - vwap_vals[n - 1]);
        return dev_curr > dev_prev; // deviation still growing → skip entry
    }

} // namespace trading
