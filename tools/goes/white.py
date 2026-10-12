"""Climatological white points: the brightest each chosen band gets over the disk around the summer solstice, so each channel's full scale can be fixed once, in absolute radiance.

Samples GOES-18 full-disk frames hourly over 16–02 UTC (09–19 PDT, every daytime peak) for June 14–28 2026, reprojects bands exactly as bands.py does (same disk, same scale), and pools every in-disk pixel per band. Reports high percentiles and the maximum in each band's own radiance unit, with the reflectance or brightness-temperature equivalent, plus when the maximum happened. Per-frame results are cached in /tmp/goes-preview/white/, so a rerun only fetches what's missing.

Usage: .venv/bin/python white.py [bands, default 6,7,9] [lat lon scale, default 47.6 −124.5 3.9]
"""

import datetime as dt
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bands import BUCKET, N, NAMES, equivalent, list_keys, read_band

CACHE = Path("/tmp/goes-preview/white")
DAYS = [dt.date(2026, 6, 14) + dt.timedelta(days=d) for d in range(15)]
HOURS = list(range(16, 24)) + [0, 1, 2]  # 0–2 UTC belong to the next UTC day


def frames():
    """(stamp, {band: key}) for the first full-disk scan of each sampled hour."""
    out = []
    for day in DAYS:
        for h in HOURS:
            t = dt.datetime(day.year, day.month, day.day, h, tzinfo=dt.timezone.utc) + (dt.timedelta(days=1) if h < 16 else dt.timedelta())
            by_start = {}
            for k in list_keys(f"ABI-L1b-RadF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"):
                m = re.search(r"M\dC(\d\d)_G18_s(\d+)_", k)
                if m:
                    by_start.setdefault(m[2], {})[int(m[1])] = k
            if by_start:
                first = min(by_start)
                out.append((dt.datetime.strptime(first[:13], "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc), by_start[first]))
    return out


def one(job):
    """One band of one frame: cached in-disk radiances (float32), fetched if missing."""
    stamp, band, key, lat, lon, scale = job
    path = CACHE / f"{stamp:%Y%m%d-%H%M}-b{band:02}.npy"
    if not path.exists():
        _, img, cal, _ = read_band((band, key, lat, lon, scale))
        yy, xx = np.mgrid[0:N, 0:N] + 0.5
        inside = (xx - N / 2) ** 2 + (yy - N / 2) ** 2 <= (N / 2) ** 2
        np.save(path, img[inside].astype(np.float32))
        np.save(path.with_suffix(".cal.npy"), np.array([cal], dtype=object), allow_pickle=True)
    return stamp, band, np.load(path), np.load(path.with_suffix(".cal.npy"), allow_pickle=True)[0]


def main():
    args = sys.argv[1:]
    want = [int(b) for b in args[0].split(",")] if args else [6, 7, 9]
    lat, lon, scale = (float(args[1]), float(args[2]), float(args[3])) if len(args) >= 4 else (47.6, -124.5, 3.9)
    CACHE.mkdir(parents=True, exist_ok=True)
    fr = frames()
    print(f"{len(fr)} frames, {fr[0][0]:%Y-%m-%d %H:%M} … {fr[-1][0]:%Y-%m-%d %H:%M} UTC; bands {want}")

    pooled = {b: [] for b in want}
    peaks = {b: [] for b in want}
    cal = {}
    jobs = [(stamp, b, keys[b], lat, lon, scale) for stamp, keys in fr for b in want if b in keys]
    with ProcessPoolExecutor(12) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            stamp, band, v, c = f.result()
            pooled[band].append(v)
            peaks[band].append((float(np.nanmax(v)), stamp))
            cal[band] = c
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    lines = [f"White points over the disk (centre {lat}° {lon}°, {scale} km/px), hourly 16–02 UTC, {DAYS[0]} … {DAYS[-1]}, {len(fr)} frames\n"]
    for b in want:
        v = np.concatenate(pooled[b])
        v = v[np.isfinite(v)]
        c = cal[b]
        top, when = max(peaks[b])
        lines.append(f"band {b} ({c['wl']:.2f} µm, {NAMES[b]}), {c['unit']}, {v.size:,} pixels")
        for q in (99.0, 99.9, 99.99):
            p = float(np.percentile(v, q))
            lines.append(f"  p{q:<6} {p:10.4f}  ({equivalent(c, p)})")
        lines.append(f"  max     {top:10.4f}  ({equivalent(c, top)})  at {when:%Y-%m-%d %H:%M} UTC")
    report = "\n".join(lines)
    (CACHE / "report.txt").write_text(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
