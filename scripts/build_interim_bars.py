"""
raw/ -> interim/ : parse Binance kline zips into one validated bar table.

Reads the verified zips fetched by ingestion/exchanges/binance.py, normalizes
them, asserts a long list of invariants, and writes a single Parquet file.

Nothing here is a judgement call: every transformation is mechanical and every
assertion is a fact that must hold or the run stops.

The three things this file exists to get right
----------------------------------------------
1. TIMESTAMP UNITS CHANGE MID-ARCHIVE. Binance spot switched from milliseconds
   to microseconds on 2025-01-01. Detected per-row by magnitude rather than by
   date, so it is robust to Binance changing it again.
2. GAPS ARE NOT ZEROS. Minutes with no trades are simply absent from the file.
   They are reported, never forward-filled. A gap is absence of data; a zero
   return is an observation of no movement, and conflating them invents
   non-events a model will happily learn.
3. THE TARGET IS NOT BUILT HERE. This produces bars only. Building y_t =
   ln(C_{t+1}/C_t) belongs downstream, where the train/test boundary is known,
   so a label can never leak across it.

Usage
-----
    python scripts/build_interim_bars.py                # dry run: validate only
    python scripts/build_interim_bars.py --apply        # write the Parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Binance kline CSVs are headerless with 12 columns, verified against the
# 2017-08 and 2025-03 files.
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote"]

# ms timestamps for this era are ~1.5e12; us are ~1.5e15. Nothing lands near
# the threshold, so magnitude is a safe discriminator.
US_THRESHOLD = 1e14


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def to_utc(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Normalize mixed ms/us epoch integers to tz-aware UTC. Returns the
    converted series and a count of how many rows used each unit."""
    is_us = series >= US_THRESHOLD
    counts = {"microseconds": int(is_us.sum()), "milliseconds": int((~is_us).sum())}
    micros = series.where(is_us, series * 1000)   # promote ms -> us
    return pd.to_datetime(micros, unit="us", utc=True), counts


def load_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)

    # A few 2025+ files ship a header row; drop it if present.
    if not str(df.iloc[0]["open_time"]).replace(".", "").isdigit():
        df = df.iloc[1:].reset_index(drop=True)

    df["open_time"] = pd.to_numeric(df["open_time"], errors="raise").astype("int64")
    for c in KEEP:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df[["open_time", *KEEP]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--apply", action="store_true", help="write the Parquet file")
    args = ap.parse_args()

    src = data_root() / "raw" / "binance" / args.symbol / args.interval
    zips = sorted(src.glob(f"{args.symbol}-{args.interval}-*.zip"))
    if not zips:
        print(f"no zips in {src}\nRun ingestion/exchanges/binance.py --apply first.")
        return 1

    print(f"reading {len(zips)} monthly file(s) from {src}\n")

    frames, unit_counts = [], {"microseconds": 0, "milliseconds": 0}
    for z in zips:
        df = load_zip(z)
        ts, counts = to_utc(df["open_time"])
        for k, v in counts.items():
            unit_counts[k] += v
        df.insert(0, "ts", ts)
        frames.append(df.drop(columns=["open_time"]))

    bars = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)

    # ------------------------------------------- sub-minute open times -----
    # Binance's early kline generation sometimes stamped a bar's open_time a
    # few seconds into the minute (e.g. 2017-12-04 06:00:20.799). Verified
    # against the archive: 2017-12 has 20,401 such bars and 2018-02 has 1,201,
    # and 21,600 of those 21,602 are the ONLY bar for their minute. They are
    # real bars with real OHLCV, not duplicates.
    #
    # So they are FLOORED to the minute, not dropped. Dropping would discard
    # nearly half of December 2017 -- the blow-off top, and the single most
    # informative month in the series.
    #
    # The handful that collide with a properly aligned bar for the same minute
    # are resolved in favour of the aligned one, which is the canonically
    # generated bar.
    was_aligned = (bars["ts"].dt.second == 0) & (bars["ts"].dt.microsecond == 0)
    n_floored = int((~was_aligned).sum())
    bars["ts"] = bars["ts"].dt.floor("min")

    n_collisions = 0
    if n_floored:
        bars["_aligned"] = was_aligned.astype(int)
        before = len(bars)
        # sort so aligned bars win the duplicate drop, then restore time order
        bars = (bars.sort_values(["ts", "_aligned"], ascending=[True, False])
                    .drop_duplicates(subset="ts", keep="first")
                    .drop(columns="_aligned")
                    .sort_values("ts")
                    .reset_index(drop=True))
        n_collisions = before - len(bars)

    # -------------------------------------------------------- invariants ---
    problems: list[str] = []

    dupes = int(bars["ts"].duplicated().sum())
    if dupes:
        problems.append(f"{dupes} duplicate timestamps")

    if not bars["ts"].is_monotonic_increasing:
        problems.append("timestamps not monotonic after sort")

    if (bars["ts"].dt.second != 0).any() or (bars["ts"].dt.microsecond != 0).any():
        problems.append("bars still not minute-aligned after flooring")

    for c in ["open", "high", "low", "close"]:
        if (bars[c] <= 0).any():
            problems.append(f"non-positive prices in {c}")

    bad_hl = int((bars["high"] < bars["low"]).sum())
    if bad_hl:
        problems.append(f"{bad_hl} bars with high < low")

    bad_range = int((
        (bars["open"] > bars["high"]) | (bars["open"] < bars["low"])
        | (bars["close"] > bars["high"]) | (bars["close"] < bars["low"])
    ).sum())
    if bad_range:
        problems.append(f"{bad_range} bars with open/close outside [low, high]")

    if (bars["volume"] < 0).any():
        problems.append("negative volume")

    nulls = int(bars.isna().sum().sum())
    if nulls:
        problems.append(f"{nulls} null values")

    # ------------------------------------------------------------- gaps ----
    # Distinguish two very different things: months that were never downloaded
    # (an ingestion gap -- go fetch them) and minutes absent inside a month
    # that IS present (a real exchange outage). Reporting them together makes
    # a partial download look like catastrophic data loss.
    have_months = {tuple(map(int, z.stem.split("-")[-2:])) for z in zips}
    first, last = bars["ts"].iloc[0], bars["ts"].iloc[-1]
    span_months = {(p.year, p.month)
                   for p in pd.period_range(first.tz_localize(None),
                                            last.tz_localize(None), freq="M")}
    absent_months = sorted(span_months - have_months)

    # Expected minutes inside downloaded months only, and clipped to
    # [first, last]. The clip matters: BTCUSDT listed mid-2017-08, so without
    # it the pre-listing part of that month is reported as a permanent
    # 23,280-minute "outage" that can never be filled because the pair did not
    # exist yet. Absence before listing is not missing data.
    exp_parts = []
    for y, m in sorted(have_months):
        p = pd.Period(year=y, month=m, freq="M")
        lo = max(p.start_time.tz_localize("UTC"), first)
        hi = min(p.end_time.tz_localize("UTC"), last)
        if lo <= hi:
            exp_parts.append(pd.date_range(lo, hi, freq="min", tz="UTC"))
    expected = (pd.DatetimeIndex([]).append(exp_parts) if exp_parts
                else pd.DatetimeIndex([]))
    missing = expected.difference(pd.DatetimeIndex(bars["ts"]))

    # --------------------------------------------------------- reporting ---
    span_days = (last - first).total_seconds() / 86400
    print(f"bars              {len(bars):,}")
    print(f"first             {first}")
    print(f"last              {last}")
    print(f"span              {span_days:,.1f} days")
    print(f"timestamp units   ms {unit_counts['milliseconds']:,} | "
          f"us {unit_counts['microseconds']:,}")
    if n_floored:
        print(f"sub-minute opens  {n_floored:,} floored to the minute "
              f"({n_collisions:,} collided with an aligned bar and were dropped)")
    print(f"months downloaded {len(have_months)}")

    if absent_months:
        print(f"months NOT downloaded {len(absent_months)}  <- ingestion gap, "
              f"not an exchange outage")
        shown = ", ".join(f"{y}-{m:02d}" for y, m in absent_months[:6])
        more = f" ... (+{len(absent_months) - 6} more)" if len(absent_months) > 6 else ""
        print(f"  {shown}{more}")
        print("  Fetch them with ingestion/exchanges/binance.py --apply")

    print(f"expected bars     {len(expected):,}  (within downloaded months only)")
    print(f"MISSING bars      {len(missing):,} "
          f"({100 * len(missing) / max(len(expected), 1):.4f}%)")
    print(f"zero-volume       {int((bars['volume'] == 0).sum()):,} "
          f"({100 * (bars['volume'] == 0).mean():.2f}%)")
    print(f"zero-trade        {int((bars['trades'] == 0).sum()):,}")

    if len(missing):
        print("\n  Largest gaps inside downloaded months "
              "(real exchange outages, not a bug):")
        s = pd.Series(missing)
        runs = (s.diff() != pd.Timedelta("1min")).cumsum()
        top = s.groupby(runs).agg(["min", "max", "count"]).nlargest(5, "count")
        for _, r in top.iterrows():
            print(f"    {r['min']} -> {r['max']}  ({r['count']:,} minutes)")
        print("  Gaps are LEFT ABSENT, never forward-filled. See module docstring.")

    if problems:
        print("\nFAILED invariants:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall invariants passed")

    # ------------------------------------------------------------ write ----
    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0

    out_dir = data_root() / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"binance_{args.symbol}_{args.interval}.parquet"
    bars.to_parquet(out, index=False, compression="zstd")
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
