"""The satellite face's PC side, first slice: fetch a GOES-18 frame, map bands straight onto the panel primaries, and put it on the panel.

Mapping: band 9 → red, 7 → green, 6 → blue, each linear in absolute radiance between that band's black and white, drive = (L − black) / (white − black), clipped independently. The subtraction is on radiance, which is linear in light; the panel model applies the gamma afterwards. Both ends ratchet on robust per-frame statistics over the disk (so a lone fire pixel or dead sample can't swing a channel): white only ever rises, to a frame's 99.99th percentile when that beats it; black only ever falls, to a frame's 0.01th percentile. Whites start from the June 2026 solstice sample (white.py: blue at its p99.9, green at 330 K, red at 260 K); black starts at the first frame's. State and every move are kept in state/white.json (repo root, git-ignored); each frame's radiances are cached in cache/frames/ while the board's ring holds that frame.

Usage:
  .venv/bin/python client.py [--at 2026-10-12T01:20] [--depth 565|888] [port]
  .venv/bin/python client.py loop [--frames 47] [--fps 15] [port]      keep the board playing a loop of the newest frames, one flash slot written per new frame
  .venv/bin/python client.py replay      rebuild the state from scratch by running every cached frame (white.py's solstice sample, then the frame cache) through the ratchets in time order
"""

import datetime as dt
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from blake3 import blake3

from bands import MEASURE, N, equivalent, find_frame, list_keys, read_band, write_png16

MAP = {"R": 9, "G": 7, "B": 6}
# Everything persistent lives on /mnt/Harbor, never in ~ (wiped daily): state and cache in the repo (git-ignored), the key with the other keys.
REPO = Path(__file__).resolve().parents[2]
STATE = REPO / "state/white.json"
RING = REPO / "state/ring.json"
CACHE = REPO / "cache"
FRAMES = CACHE / "frames"
WORK = CACHE / "work"
# The encoder, built ahead (`cargo build --release` in tools/measure), so the service doesn't need cargo.
SHOW = MEASURE / "target/x86_64-unknown-linux-gnu/release/show"
USB = "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_20:6E:F1:16:E9:B0-if00"
PORT = "tcp://goes-puck.local:7766"
HOLD_MS = 2000
CENTRE, SCALE = (47.6, -124.5), 3.9
TOP, BOTTOM = 99.99, 0.01


def planck(cal, kelvin):
    return cal["planck_fk1"] / (np.exp(cal["planck_fk2"] / (cal["planck_bc1"] + cal["planck_bc2"] * kelvin)) - 1)


def seed(cal):
    """Starting whites from the solstice sample, as radiance in each band's own unit."""
    return {6: 9.5136, 7: float(planck(cal[7], 330.0)), 9: float(planck(cal[9], 260.0))}


def fresh(cal):
    return {"white": {str(b): v for b, v in seed(cal).items()}, "black": {}, "moves": []}


def ratchet(state, when, band, pixels, cal):
    """Move band's white up / black down if this frame's robust extremes pass them; returns the frame's (p0.01, p99.99)."""
    v = pixels[np.isfinite(pixels)]
    lo, hi = (float(x) for x in np.percentile(v, [BOTTOM, TOP]))
    for end, value, beyond in (("white", hi, lambda new, old: new > old), ("black", lo, lambda new, old: new < old)):
        old = state[end].get(str(band))
        if old is None or beyond(value, old):
            state["moves"].append({"frame": when.isoformat(), "band": band, "end": end, "from": old, "to": value})
            state[end][str(band)] = value
    return lo, hi


def replay():
    """Run every cached frame through the ratchets in time order and save the resulting state."""
    import glob

    frames = {}
    for f in glob.glob("/tmp/goes-preview/white/*-b??.npy"):
        if f.endswith(".cal.npy"):
            continue
        name = Path(f).name
        when = dt.datetime.strptime(name[:13], "%Y%m%d-%H%M").replace(tzinfo=dt.timezone.utc)
        band = int(name[15:17])
        frames.setdefault(when, {})[band] = (np.load(f), np.load(f[:-4] + ".cal.npy", allow_pickle=True)[0])
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    inside = (xx - N / 2) ** 2 + (yy - N / 2) ** 2 <= (N / 2) ** 2
    for f in glob.glob(str(FRAMES / "*.npz")):
        z = np.load(f)
        cal = {int(k): v for k, v in json.loads(str(z["cal"])).items()}
        frames[dt.datetime.fromisoformat(str(z["when"]))] = {b: (z[f"rad{b}"][inside], cal[b]) for b in MAP.values()}
    order = sorted(frames)
    first = frames[order[0]]
    state = fresh({b: first[b][1] for b in MAP.values()})
    for when in order:
        for band in MAP.values():
            ratchet(state, when, band, *frames[when][band])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))
    cal = frames[order[-1]]
    print(f"replayed {len(order)} frames, {order[0]:%Y-%m-%d %H:%M} … {order[-1]:%Y-%m-%d %H:%M} UTC; {len(state['moves'])} moves")
    for ch, band in MAP.items():
        c = cal[band][1]
        b, w = state["black"][str(band)], state["white"][str(band)]
        last = [m for m in state["moves"] if m["band"] == band]
        lb = max((m["frame"] for m in last if m["end"] == "black"))
        lw = max((m["frame"] for m in last if m["end"] == "white"))
        print(f"  {ch} = band {band:2}: black {b:8.4f} ({equivalent(c, b):>8}, last moved {lb[:16]})   white {w:8.4f} ({equivalent(c, w):>8}, last moved {lw[:16]})   {c['unit']}")


def frame_list(n):
    """(time, {band: key}) of the newest `n` full-disk frames that have all our bands, oldest first."""
    now = dt.datetime.now(dt.timezone.utc)
    found = {}
    for back in range(n // 6 + 3):
        t = now - dt.timedelta(hours=back)
        for k in list_keys(f"ABI-L1b-RadF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"):
            m = re.search(r"M\dC(\d\d)_G18_s(\d+)_", k)
            if m and int(m[1]) in MAP.values():
                found.setdefault(m[2], {})[int(m[1])] = k
    stamp = lambda s: dt.datetime.strptime(s[:13], "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc)
    complete = sorted(s for s, b in found.items() if len(b) == len(MAP))
    return [(stamp(s), found[s]) for s in complete[-n:]]


def tag_of(when):
    return f"{when:%Y%m%d-%H%M}"


def load_frames(frames):
    """{time: (radiances, calibration)} for each (time, keys): from the cache in frames/, fetching what's missing in parallel."""
    out, jobs = {}, []
    FRAMES.mkdir(parents=True, exist_ok=True)
    for when, keys in frames:
        path = FRAMES / f"{tag_of(when)}.npz"
        if path.exists():
            z = np.load(path)
            out[when] = ({b: z[f"rad{b}"] for b in MAP.values()}, {int(k): v for k, v in json.loads(str(z["cal"])).items()})
        else:
            jobs += [(when, b, keys[b]) for b in MAP.values()]
    if jobs:
        print(f"fetching {len(jobs) // len(MAP)} frames…", flush=True)
        got = {}
        with ProcessPoolExecutor(12) as pool:
            for (when, _, _), (band, img, c, _) in zip(jobs, pool.map(read_band, [(b, k, *CENTRE, SCALE) for _, b, k in jobs])):
                got.setdefault(when, ({}, {}))
                got[when][0][band], got[when][1][band] = img, c
        for when, (rad, cal) in got.items():
            np.savez_compressed(FRAMES / f"{tag_of(when)}.npz", **{f"rad{b}": rad[b] for b in rad}, cal=json.dumps(cal), when=when.isoformat())
            out[when] = (rad, cal)
    return out


INSIDE = (lambda yy, xx: (xx - N / 2) ** 2 + (yy - N / 2) ** 2 <= (N / 2) ** 2)(*(np.mgrid[0:N, 0:N] + 0.5))


def drive_of(when, rad, cal, state, verbose=True):
    """Ratchet the ends on this frame, then its linear panel drive: per channel (L − black) / (white − black), clipped; outside the disk black."""
    drive = np.zeros((N, N, 3))
    for ch, band in MAP.items():
        moves = len(state["moves"])
        lo, hi = ratchet(state, when, band, rad[band][INSIDE], cal[band])
        for m in state["moves"][moves:]:
            print(f"  band {band}: {m['end']} moved {m['from']} → {m['to']:.4f} ({equivalent(cal[band], m['to'])})")
        b, w = state["black"][str(band)], state["white"][str(band)]
        if verbose:
            v = rad[band][INSIDE]
            v = v[np.isfinite(v)]
            print(f"  {ch} = band {band:2}: black {b:8.4f} ({equivalent(cal[band], b):>8})  white {w:8.4f} ({equivalent(cal[band], w):>8})  {cal[band]['unit']:22}  frame spans {(lo - b) / (w - b) * 100:5.1f}–{(hi - b) / (w - b) * 100:5.1f} % of drive, {np.mean(v < b) * 100:.3f} % below black, {np.mean(v > w) * 100:.3f} % above white")
        # Linear in radiance, so linear in light: subtract black, divide by the span.
        drive[..., "RGB".index(ch)] = np.nan_to_num(np.clip((rad[band] - b) / (w - b), 0, 1))
    drive[~INSIDE] = 0
    return drive


def encode565(when, drive):
    """Big-endian RGB565 bytes of the disk's pixels, row by row (the board's stored-frame format): the measure crate's blue-noise dither against the panel model, then each bit-replicated code back to its 5- or 6-bit value."""
    from PIL import Image

    WORK.mkdir(parents=True, exist_ok=True)
    write_png16(WORK / "face.png", drive)
    subprocess.run([SHOW, WORK / "face.png", "panel-linear", "1", "stretch", "-", "565"], cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    c = np.asarray(Image.open(WORK / "face.codes.png")).astype(np.uint16)
    return (((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3))[INSIDE].astype(">u2").tobytes()


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def loop(args):
    """Keep the board's flash ring holding the newest `count` frames and playing them: each new frame is fetched, ratcheted, encoded and written into the one slot holding the oldest frame (only that slot is erased and written; frames already on the board keep the ends they were encoded with), then playback restarts from the new oldest. Polls for new frames every minute."""
    import time

    from panel import Panel

    count, fps, port = 47, 15, PORT
    while args and args[0].startswith("--"):
        if args[0] == "--frames":
            count = int(args[1])
        elif args[0] == "--fps":
            fps = float(args[1])
        args = args[2:]
    port = args[0] if args else port
    panel = Panel(port)
    ring = json.loads(RING.read_text()) if RING.exists() else None
    if not ring or ring["count"] != count:
        ring = {"count": count, "next": 0, "slots": {}}
    period = round(1000 / fps)

    def play():
        stored = len(ring["slots"])
        if stored:
            start, n = (ring["next"], count) if stored == count else (0, stored)
            panel.send(f"p {start} {n} {period} {HOLD_MS}")
            print(f"playing {n} frames at {1000 / period:.1f} fps, newest held {HOLD_MS / 1000:g} s", flush=True)

    def poll():
        nonlocal panel
        if panel is None:
            panel = Panel(port)
            play()
        newest = max(ring["slots"].values(), default="")
        new = [(w, k) for w, k in frame_list(count) if tag_of(w) > newest]
        if not new:
            return
        frames = load_frames(new)
        state = json.loads(STATE.read_text()) if STATE.exists() else fresh(frames[new[0][0]][1])
        for when, _ in new:
            rad, cal = frames[when]
            slot = ring["next"]
            t = time.time()
            data = encode565(when, drive_of(when, rad, cal, state, verbose=False))
            save_state(state)
            panel.store(slot, data)
            # The ring only records a slot once the board has confirmed it, so a failed store is simply redone next poll.
            ring["slots"][str(slot)] = tag_of(when)
            ring.setdefault("hashes", {})[str(slot)] = blake3(data).hexdigest()
            ring["next"] = (slot + 1) % count
            RING.write_text(json.dumps(ring, indent=1))
            print(f"{when:%Y-%m-%d %H:%M} UTC → slot {slot} ({time.time() - t:.1f} s)", flush=True)
        # Keep only the cached radiances the ring still holds.
        oldest = min(ring["slots"].values())
        for f in FRAMES.glob("*.npz"):
            if f.stem < oldest:
                f.unlink()
        play()

    play()
    while True:
        try:
            poll()
        except Exception as e:
            # A network hiccup, an S3 error, a board that stopped answering: log it, keep the ring as it is, and try again next poll, with a fresh link and playback restarted.
            print(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC: poll failed ({type(e).__name__}: {e}); retrying in a minute", flush=True)
            if panel is not None:
                panel.close()
            panel = None
            try:
                panel = Panel(port)
                play()
            except Exception as e2:
                panel = None
                print(f"  board not answering ({type(e2).__name__}: {e2})", flush=True)
        time.sleep(60)


def main():
    if sys.argv[1:2] == ["replay"]:
        return replay()
    if sys.argv[1:2] == ["loop"]:
        return loop(sys.argv[2:])
    args, at, depth = sys.argv[1:], None, "565"
    while args and args[0].startswith("--"):
        if args[0] == "--at":
            at = dt.datetime.fromisoformat(args[1]).replace(tzinfo=dt.timezone.utc)
        elif args[0] == "--depth":
            depth = args[1]
        args = args[2:]
    port = args[0] if args else PORT

    when, keys = find_frame(at)
    rad, cal = load_frames([(when, keys)])[when]
    state = json.loads(STATE.read_text()) if STATE.exists() else fresh(cal)
    print(f"frame {when:%Y-%m-%d %H:%M} UTC")
    drive = drive_of(when, rad, cal, state)
    save_state(state)
    WORK.mkdir(parents=True, exist_ok=True)
    write_png16(WORK / "face.png", drive)
    subprocess.run([SHOW, WORK / "face.png", "panel-linear", "1", "stretch", port, depth], cwd=WORK, check=True)


if __name__ == "__main__":
    main()
