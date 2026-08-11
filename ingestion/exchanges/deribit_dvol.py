"""
Fetch Deribit's DVOL index — the market's own 30-day volatility forecast.

DVOL is Deribit's BTC implied-volatility index: annualized, in percent, looking
30 days forward. It is the number that decides whether a volatility model is
tradeable, because it is what the options market already charges for exactly
the thing we are forecasting.

Beating HAR is a research result. Beating DVOL is a trading result. They are
different claims, and only this data can settle the second.

Usage
-----
    python ingestion/exchanges/deribit_dvol.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
ROOT = Path(__file__).resolve().parents[2]

# DVOL history begins in 2021; earlier requests return an empty series.
START = datetime(2021, 3, 24, tzinfo=timezone.utc)
RESOLUTION = 3600          # hourly, matching our bar grid
CHUNK_DAYS = 60            # the endpoint caps how much it returns per call


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def fetch_window(start: datetime, end: datetime, currency: str) -> list[list]:
    url = (f"{API}?currency={currency}"
           f"&start_timestamp={int(start.timestamp() * 1000)}"
           f"&end_timestamp={int(end.timestamp() * 1000)}"
           f"&resolution={RESOLUTION}")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ForeSight/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read().decode())
            return payload.get("result", {}).get("data", [])
        except Exception as e:
            if attempt == 3:
                raise
            print(f"      retry {attempt + 1}/3 ({e})")
            time.sleep(2 ** attempt)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    out = data_root() / "raw" / "deribit"
    dest = out / f"dvol_{args.currency}_1h.json"
    end_all = datetime.now(timezone.utc)

    print(f"DVOL {args.currency}, hourly, {START:%Y-%m-%d} .. {end_all:%Y-%m-%d}")
    if not args.apply:
        print("DRY RUN -- re-run with --apply")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rows: list[list] = []
    cur = START
    while cur < end_all:
        stop = min(cur + timedelta(days=CHUNK_DAYS), end_all)
        got = fetch_window(cur, stop, args.currency)
        rows.extend(got)
        print(f"  {cur:%Y-%m-%d} .. {stop:%Y-%m-%d}  {len(got):>5} points")
        cur = stop
        time.sleep(0.25)          # be polite to a free public endpoint

    # de-duplicate on timestamp; chunk boundaries overlap by one point
    seen, clean = set(), []
    for r in sorted(rows, key=lambda x: x[0]):
        if r[0] not in seen:
            seen.add(r[0])
            clean.append(r)

    dest.write_text(json.dumps({
        "source": "deribit get_volatility_index_data",
        "currency": args.currency,
        "resolution_s": RESOLUTION,
        "columns": ["timestamp_ms", "open", "high", "low", "close"],
        "note": "annualized implied volatility in percent, 30-day horizon",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": clean,
    }, indent=None), encoding="utf-8")

    first = datetime.fromtimestamp(clean[0][0] / 1000, timezone.utc)
    last = datetime.fromtimestamp(clean[-1][0] / 1000, timezone.utc)
    print(f"\n{len(clean):,} hourly points  {first:%Y-%m-%d} .. {last:%Y-%m-%d}")
    print(f"wrote {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
