"""
setup_db.py — initialise trades.db and market.db schemas.
Run once before first use: python3 setup_db.py
"""
import sqlite3
import config

# ── trades.db ─────────────────────────────────────────────────────────────
def init_trades_db() -> None:
    con = sqlite3.connect(config.TRADES_DB)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT    NOT NULL,
            side         TEXT    NOT NULL,  -- LONG / SHORT
            qty          INTEGER NOT NULL,
            entry_price  REAL    NOT NULL,
            exit_price   REAL,
            entry_time   TEXT    NOT NULL,  -- ISO-8601 UTC
            exit_time    TEXT,
            strategy     TEXT    NOT NULL,
            confidence   REAL,
            stop         REAL,
            target1      REAL,
            target2      REAL,
            pnl          REAL,
            pct_chg      REAL,
            reason       TEXT,  -- STOP / TARGET1 / TARGET2 / FORCE_CLOSE / MANUAL
            status       TEXT    NOT NULL DEFAULT 'OPEN',  -- OPEN / CLOSED
            ib_order_id  INTEGER,
            commission   REAL    DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_entry  ON trades(entry_time);

        CREATE TABLE IF NOT EXISTS daily_stats (
            date              TEXT PRIMARY KEY,  -- YYYY-MM-DD ET
            starting_nav      REAL,
            ending_nav        REAL,
            total_pnl         REAL,
            total_commission  REAL,
            num_trades        INTEGER,
            num_wins          INTEGER,
            num_losses        INTEGER,
            consecutive_losses INTEGER DEFAULT 0
        );
    """)
    con.commit()
    con.close()
    print(f"[setup_db] trades.db initialised at {config.TRADES_DB}")


# ── market.db ─────────────────────────────────────────────────────────────
def init_market_db() -> None:
    con = sqlite3.connect(config.MARKET_DB)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS market_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT    NOT NULL,
            timestamp TEXT    NOT NULL,  -- ISO-8601 UTC (minute bars)
            open      REAL    NOT NULL,
            high      REAL    NOT NULL,
            low       REAL    NOT NULL,
            close     REAL    NOT NULL,
            volume    INTEGER NOT NULL,
            UNIQUE(symbol, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_md_sym_ts ON market_data(symbol, timestamp);
    """)
    con.commit()
    con.close()
    print(f"[setup_db] market.db initialised at {config.MARKET_DB}")


if __name__ == "__main__":
    init_trades_db()
    init_market_db()
    print("[setup_db] Done.")
