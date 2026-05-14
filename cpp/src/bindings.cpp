/**
 * bindings.cpp — pybind11 bindings for trading_engine module
 *
 * Build: ./build.sh
 * Import in Python: import trading_engine as te
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/trading_engine.h"

namespace py = pybind11;
using namespace trading;

PYBIND11_MODULE(trading_engine, m) {
    m.doc() = "C++ trading engine — indicators, S/R, sizing (US Algo Trader)";

    // ── Structs ──────────────────────────────────────────────────────────

    py::class_<Level>(m, "Level")
        .def_readonly("price",      &Level::price)
        .def_readonly("touches",    &Level::touches)
        .def_readonly("is_support", &Level::is_support)
        .def("__repr__", [](const Level& l){
            return "<Level price=" + std::to_string(l.price)
                 + " touches=" + std::to_string(l.touches)
                 + " support=" + (l.is_support ? "true" : "false") + ">";
        });

    py::class_<RoundLevel>(m, "RoundLevel")
        .def_readonly("price",        &RoundLevel::price)
        .def_readonly("magnitude",    &RoundLevel::magnitude)
        .def_readonly("distance_pct", &RoundLevel::distance_pct)
        .def("__repr__", [](const RoundLevel& r){
            return "<RoundLevel price=" + std::to_string(r.price)
                 + " mag=" + std::to_string(r.magnitude)
                 + " dist=" + std::to_string(r.distance_pct * 100) + "%>";
        });

    py::class_<OrderBlock>(m, "OrderBlock")
        .def_readonly("high",      &OrderBlock::high)
        .def_readonly("low",       &OrderBlock::low)
        .def_readonly("mid",       &OrderBlock::mid)
        .def_readonly("bar_index", &OrderBlock::bar_index)
        .def_readonly("bullish",   &OrderBlock::bullish)
        .def("__repr__", [](const OrderBlock& ob){
            return "<OrderBlock " + std::string(ob.bullish ? "DEMAND" : "SUPPLY")
                 + " [" + std::to_string(ob.low) + "-" + std::to_string(ob.high)
                 + "] bars_ago=" + std::to_string(ob.bar_index) + ">";
        });

    py::class_<SizeResult>(m, "SizeResult")
        .def_readonly("qty",              &SizeResult::qty)
        .def_readonly("risk_usd",         &SizeResult::risk_usd)
        .def_readonly("max_profit_usd",   &SizeResult::max_profit_usd)
        .def_readonly("skip_min_profit",  &SizeResult::skip_min_profit)
        .def_readonly("skip_min_value",   &SizeResult::skip_min_value)
        .def_readonly("reason",           &SizeResult::reason)
        .def("__repr__", [](const SizeResult& s){
            return "<SizeResult qty=" + std::to_string(s.qty)
                 + " risk=$" + std::to_string(s.risk_usd)
                 + " profit=$" + std::to_string(s.max_profit_usd)
                 + " reason=" + s.reason + ">";
        });

    // ── Indicators ───────────────────────────────────────────────────────

    m.def("ema",  &ema,  "Exponential Moving Average",
          py::arg("prices"), py::arg("period"));

    m.def("sma",  &sma,  "Simple Moving Average",
          py::arg("prices"), py::arg("period"));

    m.def("rsi",  &rsi,  "RSI (Wilder's smoothing)",
          py::arg("closes"), py::arg("period") = 14);

    m.def("macd", &macd, "MACD — returns (macd, signal, histogram)",
          py::arg("closes"),
          py::arg("fast")          = 12,
          py::arg("slow")          = 26,
          py::arg("signal_period") = 9);

    m.def("atr",  &atr,  "Wilder ATR",
          py::arg("highs"), py::arg("lows"), py::arg("closes"),
          py::arg("period") = 14);

    m.def("vwap", &vwap, "Intraday VWAP (provide current-day bars only)",
          py::arg("highs"), py::arg("lows"),
          py::arg("closes"), py::arg("volumes"));

    m.def("bollinger", &bollinger,
          "Bollinger Bands — returns (upper, middle, lower)",
          py::arg("closes"),
          py::arg("period")   = 20,
          py::arg("std_devs") = 2.0);

    m.def("calc_rvol", &calc_rvol,
          "Relative volume vs. period-bar average (scalar for latest bar)",
          py::arg("volumes"), py::arg("period") = 20);

    m.def("rsi_rising", &rsi_rising,
          "True if RSI has risen for last n bars",
          py::arg("rsi_vals"), py::arg("n") = 3);

    m.def("get_hod", &get_hod, "High of day", py::arg("highs"));
    m.def("get_lod", &get_lod, "Low of day",  py::arg("lows"));

    m.def("is_trend_day", &is_trend_day,
          "VWAP trend-day filter (≥threshold fraction on same VWAP side)",
          py::arg("closes"), py::arg("vwap_vals"),
          py::arg("bars") = 30, py::arg("threshold") = 26.0/30.0);

    m.def("vwap_deviation_expanding", &vwap_deviation_expanding,
          "True if price deviation from VWAP grew vs. prior bar",
          py::arg("closes"), py::arg("vwap_vals"));

    // ── Support / Resistance ─────────────────────────────────────────────

    m.def("find_sr_levels", &find_sr_levels,
          "Find S/R levels by pivot clustering",
          py::arg("highs"), py::arg("lows"), py::arg("closes"),
          py::arg("lookback")      = 60,
          py::arg("tolerance_pct") = 0.003);

    // ── Round Numbers ────────────────────────────────────────────────────

    m.def("find_round_levels", &find_round_levels,
          "Find nearby round-number price levels",
          py::arg("price"),
          py::arg("search_range_pct") = 0.05);

    // ── Order Blocks ─────────────────────────────────────────────────────

    m.def("find_order_blocks", &find_order_blocks,
          "Detect SMC order blocks (demand/supply zones)",
          py::arg("opens"), py::arg("highs"),
          py::arg("lows"),  py::arg("closes"),
          py::arg("lookback") = 30);

    // ── Divergence ───────────────────────────────────────────────────────

    m.def("detect_bullish_divergence", &detect_bullish_divergence,
          "Price lower low + RSI higher low",
          py::arg("closes"), py::arg("rsi_vals"),
          py::arg("lookback") = 20);

    m.def("detect_bearish_divergence", &detect_bearish_divergence,
          "Price higher high + RSI lower high",
          py::arg("highs"), py::arg("rsi_vals"),
          py::arg("lookback") = 20);

    // ── Position Sizing ──────────────────────────────────────────────────

    m.def("calc_size", &calc_size,
          "Position sizing with T1 profit gate and min-value bump",
          py::arg("nav"),
          py::arg("max_position_pct"),
          py::arg("max_risk_pct"),
          py::arg("entry_price"),
          py::arg("stop_price"),
          py::arg("t1_price"),
          py::arg("min_t1_profit_usd"),
          py::arg("min_position_value"));
}
