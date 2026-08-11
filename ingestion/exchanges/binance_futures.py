"""
Fetch Binance USD-M futures positioning data — funding, open interest, and the
long/short ratios.

This is the information a price-history model structurally cannot see, and it
is free. Where klines tell you what happened, these tell you how the crowd was
positioned when it happened, which is the closest thing to a directional signal
Binance publishes.

    fundingRate   monthly files, 8-hourly. Perpetual funding: positive means
                  longs pay shorts, i.e. crowded long. Extremes precede
                  squeezes.
    metrics       daily files, 5-minutely. Open interest, the TOP-TRADER
                  long/short ratio (by account and by position), the overall
                  long/short account ratio, and the taker buy/sell volume
                  ratio. Top-trader positioning is the interesting column:
                  it is the closest public proxy for informed flow.

Usage
-----
    python ingestion/exchanges/binance_futures.py --what funding --apply
    python ingestion/exchanges/binance_futures.py --what metrics --apply
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 60


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def get_zip(url: str) -> pd.DataFrame | None:
    """Return the single CSV inside a zip, or None on 404."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ForeSight/0.1"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                blob = r.read()
            zf = zipfile.ZipFile(io.BytesIO(blob))
            with zf.open(zf.namelist()[0]) as f:
                return pd.read_csv(f)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 2:
                raise
        except Exception:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    return None


def months(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--what", choices=["funding", "metrics"], default="funding")
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    out = data_root() / "raw" / "binance_futures"
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    if not args.apply:
        print(f"DRY RUN -- would fetch {args.what} for {args.symbol} "
              f"from {args.start}. Re-run with --apply.")
        return 0

    # ------------------------------------------------------------ funding --
    if args.what == "funding":
        y0, m0 = map(int, args.start.split("-"))
        prev = now.replace(day=1) - timedelta(days=1)
        frames, missing = [], 0
        for y, m in months((y0, m0), (prev.year, prev.month)):
            url = f"{BASE}/monthly/fundingRate/{args.symbol}/{args.symbol}-fundingRate-{y}-{m:02d}.zip"
            df = get_zip(url)
            if df is None:
                missing += 1
                continue
            frames.append(df)
            print(f"  {y}-{m:02d}  {len(df):>4} rows", flush=True)
        if not frames:
            print("nothing fetched")
            return 1
        d = pd.concat(frames, ignore_index=True)
        d["ts"] = pd.to_datetime(d["calc_time"], unit="ms", utc=True)
        d = d[["ts", "funding_interval_hours", "last_funding_rate"]] \
            .drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        dest = out / f"funding_{args.symbol}.parquet"
        d.to_parquet(dest, index=False)
        print(f"\n{len(d):,} funding points  {d.ts.min():%Y-%m-%d} .. "
              f"{d.ts.max():%Y-%m-%d}  ({missing} months unavailable)")
        print(f"wrote {dest}")
        return 0

    # ------------------------------------------------------------ metrics --
    # Daily files only. Binance keeps roughly the last few years.
    dest = out / f"metrics_{args.symbol}.parquet"
    have = set()
    if dest.exists():
        old = pd.read_parquet(dest)
        have = set(pd.to_datetime(old["ts"]).dt.date)
        print(f"resuming: {len(old):,} rows already stored")
    else:
        old = None

    d0 = date.fromisoformat(args.start + "-01") if len(args.start) == 7 \
        else date.fromisoformat(args.start)
    frames, missing, got = [], 0, 0
    day = d0
    last = (now - timedelta(days=1)).date()
    while day <= last:
        if day in have:
            day += timedelta(days=1)
            continue
        url = (f"{BASE}/daily/metrics/{args.symbol}/"
               f"{args.symbol}-metrics-{day:%Y-%m-%d}.zip")
        df = get_zip(url)
        if df is None:
            missing += 1
        else:
            frames.append(df)
            got += 1
            if got % 100 == 0:
                print(f"  {day}  ({got} days fetched)", flush=True)
        day += timedelta(days=1)

    if not frames and old is None:
        print(f"nothing available (tried from {d0}); "
              "Binance retains only recent metrics files")
        return 1
    new = pd.concat(frames, ignore_index=True) if frames else None
    if new is not None:
        new["ts"] = pd.to_datetime(new["create_time"], utc=True)
        new = new.drop(columns=["create_time", "symbol"], errors="ignore")
    d = pd.concat([x for x in (old, new) if x is not None], ignore_index=True)
    d = d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    d.to_parquet(dest, index=False)
    print(f"\n{len(d):,} metric points  {d.ts.min()} .. {d.ts.max()}")
    print(f"({got} days newly fetched, {missing} unavailable)")
    print(f"columns: {[c for c in d.columns if c != 'ts']}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
