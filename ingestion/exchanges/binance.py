"""
Ingest Binance spot klines from the official public archive.

    https://data.binance.vision/data/spot/monthly/klines/<SYMBOL>/<INTERVAL>/

This script does exactly one job: fetch the published zip files and prove they
are intact. It does NOT parse, clean or reshape anything -- `data/raw/` is
immutable by contract (see data/README.md), so what lands there is the bytes
Binance served, unmodified, plus provenance.

Every file is verified against the `.CHECKSUM` Binance publishes beside it.
A mismatch is a hard failure, never a warning.

Standard library only, so it runs before any venv exists.

Usage
-----
    # dry run: show what would be fetched
    python ingestion/exchanges/binance.py --start 2017-08

    # actually fetch
    python ingestion/exchanges/binance.py --start 2017-08 --apply

    # resume / top up (skips files already present and verified)
    python ingestion/exchanges/binance.py --apply

    # re-verify everything on disk without downloading
    python ingestion/exchanges/binance.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/monthly/klines"
ROOT = Path(__file__).resolve().parents[2]

# BTCUSDT's first month on Binance spot. Earlier months simply 404.
FIRST_MONTH = (2017, 8)

CHUNK = 1 << 16
RETRIES = 4
BACKOFF = 2.0
TIMEOUT = 120


# ----------------------------------------------------------------- paths ----
def data_root() -> Path:
    """Where raw bytes land.

    Override with FORESIGHT_DATA_ROOT to keep multi-GB archives off the
    Google-Drive-synced sandbox. Git ignores data/ either way, but git ignoring
    a file does not stop Drive from syncing it.
    """
    import os

    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def raw_dir(symbol: str, interval: str) -> Path:
    return data_root() / "raw" / "binance" / symbol / interval


# ------------------------------------------------------------- utilities ----
def months(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def last_complete_month(today: datetime | None = None) -> tuple[int, int]:
    """Binance publishes a month's file after the month ends; never ask for
    the current month."""
    now = today or datetime.now(timezone.utc)
    first_of_this = now.replace(day=1)
    prev = first_of_this - timedelta(days=1)
    return prev.year, prev.month


def parse_month(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def expected_sha(checksum_path: Path) -> str:
    """Binance .CHECKSUM format:  '<sha256>  <filename>'"""
    text = checksum_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty checksum file: {checksum_path}")
    return text.split()[0].lower()


# ------------------------------------------------------------ networking ----
def fetch(url: str, dest: Path) -> bool:
    """Download to <dest>.part then rename, so a killed run never leaves a
    truncated file that a later run would trust. Returns False on 404."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ForeSight/0.1"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, tmp.open("wb") as f:
                while chunk := r.read(CHUNK):
                    f.write(chunk)
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            if e.code == 404:
                return False
            last_err = e
        except Exception as e:  # network flake, timeout, partial read
            tmp.unlink(missing_ok=True)
            last_err = e

        if attempt < RETRIES:
            wait = BACKOFF ** attempt
            print(f"      retry {attempt}/{RETRIES - 1} in {wait:.0f}s ({last_err})")
            time.sleep(wait)

    raise RuntimeError(f"failed after {RETRIES} attempts: {url}\n  last error: {last_err}")


# -------------------------------------------------------------- manifest ----
def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": "data.binance.vision", "files": {}}


def save_manifest(path: Path, man: dict) -> None:
    path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main ----
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", default=None, help="YYYY-MM (default: first listed month)")
    ap.add_argument("--end", default=None, help="YYYY-MM (default: last complete month)")
    ap.add_argument("--apply", action="store_true", help="actually download")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-verify files already on disk; download nothing")
    args = ap.parse_args()

    out = raw_dir(args.symbol, args.interval)
    man_path = out / "manifest.json"
    start = parse_month(args.start) if args.start else FIRST_MONTH
    end = parse_month(args.end) if args.end else last_complete_month()

    if start > end:
        print(f"nothing to do: start {start} is after end {end}")
        return 0

    print(f"symbol   {args.symbol} {args.interval}")
    print(f"range    {start[0]}-{start[1]:02d} .. {end[0]}-{end[1]:02d}")
    print(f"dest     {out}")
    if out.resolve().is_relative_to(ROOT.resolve()):
        print("  note   this path is inside the Drive-synced sandbox; set "
              "FORESIGHT_DATA_ROOT to move it out")
    print()

    # ---------------------------------------------------------- verify only
    if args.verify_only:
        bad = ok = 0
        for z in sorted(out.glob(f"{args.symbol}-{args.interval}-*.zip")):
            c = z.with_suffix(".zip.CHECKSUM")
            if not c.exists():
                print(f"  MISSING CHECKSUM  {z.name}")
                bad += 1
                continue
            if sha256(z) == expected_sha(c):
                ok += 1
            else:
                print(f"  CORRUPT           {z.name}")
                bad += 1
        print(f"\nverified {ok} file(s), {bad} problem(s)")
        return 1 if bad else 0

    # -------------------------------------------------------------- fetch
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(man_path)

    planned = list(months(start, end))
    have = sum(1 for y, m in planned
               if (out / f"{args.symbol}-{args.interval}-{y}-{m:02d}.zip").exists())
    print(f"{len(planned)} month(s) in range, {have} already on disk\n")

    if not args.apply:
        print("DRY RUN -- nothing downloaded. Re-run with --apply.")
        return 0

    fetched = skipped = absent = 0
    for y, m in planned:
        name = f"{args.symbol}-{args.interval}-{y}-{m:02d}.zip"
        zpath, cpath = out / name, out / (name + ".CHECKSUM")

        if zpath.exists() and cpath.exists():
            if sha256(zpath) == expected_sha(cpath):
                skipped += 1
                continue
            print(f"  {name}: on disk but checksum FAILED -- refetching")
            zpath.unlink(missing_ok=True)

        url = f"{BASE}/{args.symbol}/{args.interval}/{name}"
        print(f"  {name} ... ", end="", flush=True)

        if not fetch(url + ".CHECKSUM", cpath):
            print("not published (404)")
            cpath.unlink(missing_ok=True)
            absent += 1
            continue
        if not fetch(url, zpath):
            print("zip missing though checksum exists (404)")
            cpath.unlink(missing_ok=True)
            absent += 1
            continue

        got, want = sha256(zpath), expected_sha(cpath)
        if got != want:
            zpath.unlink(missing_ok=True)
            cpath.unlink(missing_ok=True)
            raise SystemExit(
                f"\nCHECKSUM MISMATCH for {name}\n  expected {want}\n  got      {got}\n"
                "Refusing to keep unverified bytes in data/raw/."
            )

        man["files"][name] = {
            "url": url,
            "sha256": got,
            "bytes": zpath.stat().st_size,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_manifest(man_path, man)
        fetched += 1
        print(f"ok  {zpath.stat().st_size:,} bytes  sha256 verified")

    print(f"\nfetched {fetched}, already had {skipped}, not published {absent}")
    print(f"manifest: {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
