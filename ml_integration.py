"""
ml_integration.py — ML model for trade-outcome prediction.

Model: GradientBoostingClassifier trained on historical trade records.
Features: price context + technical indicators at trade entry.
Labels:   WIN / LOSE (based on actual exit P&L).
Auto-retrains every ML_RETRAIN_INTERVAL hours from trades.db.
"""
from __future__ import annotations
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)

# Feature columns used at prediction time (must match training)
FEATURE_COLS = [
    "rsi14", "rvol", "atr_pct",
    "macd_hist", "bb_pct",      # position within Bollinger band (0=lower, 1=upper)
    "vwap_dev_pct",              # (close - vwap) / vwap
    "hour_et",                   # hour of day in ET
    "day_of_week",
]


class MLTradingIntegration:
    """Manages ML model lifecycle: load, predict, auto-retrain."""

    def __init__(self) -> None:
        self._model:   Optional[GradientBoostingClassifier] = None
        self._scaler:  Optional[StandardScaler]             = None
        self._labels:  Optional[list[str]]                  = None
        self._lock     = threading.RLock()
        self._last_trained: Optional[float] = None

        self._load()
        self._start_retrain_thread()

    # ── Load ──────────────────────────────────────────────────────────────

    def _load(self) -> bool:
        paths = (config.ML_MODEL_PATH, config.ML_SCALER_PATH, config.ML_LABELS_PATH)
        if not all(os.path.exists(p) for p in paths):
            logger.info("[ML] Model files not found — will train on first retrain cycle")
            return False
        try:
            with self._lock:
                self._model  = joblib.load(config.ML_MODEL_PATH)
                self._scaler = joblib.load(config.ML_SCALER_PATH)
                self._labels = joblib.load(config.ML_LABELS_PATH)
            mtime = os.path.getmtime(config.ML_MODEL_PATH)
            self._last_trained = mtime
            age_h = (time.time() - mtime) / 3600
            logger.info("[ML] Model loaded (age %.1fh, classes=%s)", age_h, self._labels)
            return True
        except Exception as exc:
            logger.error("[ML] Load failed: %s", exc)
            return False

    # ── Predict ───────────────────────────────────────────────────────────

    def predict_proba(self, features: dict) -> float:
        """
        Return WIN probability in [0.0, 1.0].
        Returns 0.5 (neutral) if model unavailable or features incomplete.
        """
        with self._lock:
            if self._model is None or self._scaler is None or self._labels is None:
                return 0.5
            try:
                row = [features.get(col, 0.0) for col in FEATURE_COLS]
                x   = np.array(row, dtype=float).reshape(1, -1)
                x_s = self._scaler.transform(x)
                proba = self._model.predict_proba(x_s)[0]
                win_idx = self._labels.index("WIN") if "WIN" in self._labels else 1
                return float(proba[win_idx])
            except Exception as exc:
                logger.warning("[ML] predict_proba failed: %s", exc)
                return 0.5

    # ── Train ─────────────────────────────────────────────────────────────

    def retrain(self) -> bool:
        """Train on all closed trades in trades.db. Returns True on success."""
        try:
            con = sqlite3.connect(config.TRADES_DB)
            df  = pd.read_sql_query(
                """SELECT rsi14, rvol, atr_pct, macd_hist, bb_pct,
                          vwap_dev_pct, hour_et, day_of_week, pnl
                   FROM trades
                   WHERE status='CLOSED' AND pnl IS NOT NULL
                     AND rsi14 IS NOT NULL""",
                con,
            )
            con.close()
        except Exception as exc:
            logger.error("[ML] DB read failed during retrain: %s", exc)
            return False

        if len(df) < config.ML_MIN_SAMPLES:
            logger.info("[ML] Only %d samples, need %d — skipping retrain",
                        len(df), config.ML_MIN_SAMPLES)
            return False

        df["label"] = df["pnl"].apply(lambda x: "WIN" if x > 0 else "LOSE")
        X = df[FEATURE_COLS].fillna(0.0).values
        y = df["label"].values

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2,
                                                   random_state=42, stratify=y)
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42,
        )
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        labels = list(model.classes_)

        with self._lock:
            joblib.dump(model,  config.ML_MODEL_PATH)
            joblib.dump(scaler, config.ML_SCALER_PATH)
            joblib.dump(labels, config.ML_LABELS_PATH)
            self._model  = model
            self._scaler = scaler
            self._labels = labels
            self._last_trained = time.time()

        logger.info("[ML] Retrained on %d samples, test acc=%.3f", len(df), acc)
        return True

    # ── Auto-retrain thread ───────────────────────────────────────────────

    def _start_retrain_thread(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(config.ML_RETRAIN_INTERVAL * 3600)
                logger.info("[ML] Auto-retrain triggered")
                self.retrain()

        t = threading.Thread(target=_loop, daemon=True, name="ml-retrain")
        t.start()

    # ── Feature builder ───────────────────────────────────────────────────

    @staticmethod
    def build_features(
        rsi14:       float,
        rvol:        float,
        atr_pct:     float,
        macd_hist:   float,
        bb_pct:      float,
        vwap_dev_pct: float,
        entry_time:  datetime,
    ) -> dict:
        """Construct feature dict from raw indicator values at entry."""
        from zoneinfo import ZoneInfo
        et = entry_time.astimezone(ZoneInfo("America/New_York"))
        return {
            "rsi14":        rsi14,
            "rvol":         rvol,
            "atr_pct":      atr_pct,
            "macd_hist":    macd_hist,
            "bb_pct":       bb_pct,
            "vwap_dev_pct": vwap_dev_pct,
            "hour_et":      et.hour + et.minute / 60.0,
            "day_of_week":  et.weekday(),  # 0=Mon, 4=Fri
        }
