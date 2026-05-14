"""
test_connection.py — verify IB TWS connectivity and basic account data.

Usage: python3 test_connection.py
Expected output: "IB connection OK" + account data > 0
"""
from __future__ import annotations
import sys
import config

try:
    from ib_insync import IB, util
except ImportError:
    print("ib_insync not installed. Run: pip3 install ib_insync")
    sys.exit(1)


def main() -> int:
    ib = IB()
    print(f"Connecting to IB on {config.IB_HOST}:{config.IB_PORT} (clientId=99)…")
    try:
        ib.connect(config.IB_HOST, config.IB_PORT, clientId=99, timeout=10)
    except Exception as exc:
        print(f"❌ Connection FAILED: {exc}")
        print("   Is TWS/Gateway running on port 7497 with API enabled?")
        return 1

    if not ib.isConnected():
        print("❌ IB connection FAILED (not connected after connect())")
        return 1

    print("✅ IB connection OK")

    vals = ib.accountValues()
    nav  = next((float(v.value) for v in vals
                 if v.tag == "NetLiquidation" and v.currency == "USD"), None)

    if nav is None or nav <= 0:
        print("❌ Account NAV is zero or unavailable")
        ib.disconnect()
        return 1

    print(f"✅ Account data returns — NAV: ${nav:,.2f}")

    acct = next((v.account for v in vals), "unknown")
    print(f"   Account ID: {acct}")

    ib.disconnect()
    print("\ntest_connection.py PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
